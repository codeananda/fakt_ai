import base64
import json
import os
import time

import pytest

from fakt_ai import llm

SCHEMA = {"type": "object"}
SLOT_COUNT = llm._SLOTS._value
PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"


def _finished(*events, stderr=""):
    return "".join(json.dumps(event) + "\n" for event in events), stderr


def _result(**fields):
    return {"type": "result", "is_error": False} | fields


def _flag(command, name):
    return command[command.index(name) + 1]


def test_image_travels_inline_and_the_cli_gets_no_tools(mocker, tmp_path):
    picture = tmp_path / "shot.png"
    picture.write_bytes(PNG)
    run = mocker.patch.object(
        llm,
        "_finish_within_deadline",
        return_value=_finished(_result(structured_output={"a": 1})),
    )

    answer = llm._run_cli.__wrapped__("Read it.", SCHEMA, "system", "haiku", picture)

    assert answer == {"a": 1}
    command, message, *_ = run.call_args.args
    assert _flag(command, "--tools") == ""
    assert _flag(command, "--input-format") == "stream-json"
    assert "--add-dir" not in command
    assert "--allowedTools" not in command
    content = json.loads(message)["message"]["content"]
    assert content[0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(PNG).decode(),
    }
    assert content[1] == {"type": "text", "text": "Read it."}
    assert str(tmp_path) not in " ".join(command) + message


def test_answer_is_taken_from_the_result_event(mocker):
    events = [{"type": "system"}, {"type": "assistant"}, _result(structured_output={"a": 2})]
    mocker.patch.object(llm, "_finish_within_deadline", return_value=_finished(*events))
    assert llm._run_cli.__wrapped__("p", SCHEMA, "s", "haiku", None) == {"a": 2}


def test_output_without_a_result_means_claude_is_unavailable(mocker):
    reply = _finished({"type": "system"}, stderr="limit reached")
    mocker.patch.object(llm, "_finish_within_deadline", return_value=reply)
    with pytest.raises(llm.ClaudeUnavailable, match="limit reached"):
        llm._run_cli.__wrapped__("p", SCHEMA, "s", "haiku", None)


@pytest.mark.parametrize(
    ("start", "kind"),
    [
        (PNG, "image/png"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"GIF89a", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8", "image/webp"),
    ],
)
def test_picture_type_comes_from_the_bytes(start, kind):
    assert llm._picture_type(start) == kind


def test_unknown_picture_is_refused():
    with pytest.raises(ValueError, match="PNG"):
        llm._picture_type(b"<svg/>")


def _script_that_hangs(tmp_path):
    """A stand-in CLI whose hung child would keep running if only the parent were killed."""
    script = tmp_path / "claude"
    marker = tmp_path / "child.pid"
    script.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {marker}\nwait\n")
    script.chmod(0o755)
    return script, marker


def _is_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_hung_cli_is_killed_with_its_children_and_frees_the_slot(mocker, tmp_path):
    script, marker = _script_that_hangs(tmp_path)
    mocker.patch.object(llm.shutil, "which", return_value=str(script))
    mocker.patch.object(llm, "_CLI_TIMEOUT_SECONDS", 3)

    with pytest.raises(llm.ClaudeUnavailable, match="ran past"):
        llm._run_cli.__wrapped__("p", SCHEMA, "s", "haiku", None)

    assert llm._SLOTS._value == SLOT_COUNT
    child = int(marker.read_text())
    deadline = time.monotonic() + 5
    while _is_running(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _is_running(child)


def test_timeout_is_retried_without_holding_a_slot_during_backoff(mocker, tmp_path):
    script, _ = _script_that_hangs(tmp_path)
    mocker.patch.object(llm.shutil, "which", return_value=str(script))
    mocker.patch.object(llm, "_CLI_TIMEOUT_SECONDS", 0.2)
    slots_while_waiting = []
    sleep = mocker.patch.object(
        llm._run_cli.retry,
        "sleep",
        side_effect=lambda _: slots_while_waiting.append(llm._SLOTS._value),
    )

    with pytest.raises(llm.ClaudeUnavailable):
        llm._run_cli("p", SCHEMA, "s", "haiku", None)

    assert sleep.call_count == 5
    assert slots_while_waiting == [SLOT_COUNT] * 5
