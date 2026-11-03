import base64
import json
import subprocess

import pytest

from fakt_ai import llm

SCHEMA = {"type": "object"}
PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"


def _finished(*events, stderr=""):
    stdout = "".join(json.dumps(event) + "\n" for event in events)
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr)


def _result(**fields):
    return {"type": "result", "is_error": False} | fields


def _flag(command, name):
    return command[command.index(name) + 1]


def test_image_travels_inline_and_the_cli_gets_no_tools(mocker, tmp_path):
    picture = tmp_path / "shot.png"
    picture.write_bytes(PNG)
    run = mocker.patch.object(
        llm.subprocess, "run", return_value=_finished(_result(structured_output={"a": 1}))
    )

    answer = llm._run_cli.__wrapped__("Read it.", SCHEMA, "system", "haiku", picture)

    assert answer == {"a": 1}
    command = run.call_args.args[0]
    assert _flag(command, "--tools") == ""
    assert _flag(command, "--input-format") == "stream-json"
    assert "--add-dir" not in command
    assert "--allowedTools" not in command
    content = json.loads(run.call_args.kwargs["input"])["message"]["content"]
    assert content[0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(PNG).decode(),
    }
    assert content[1] == {"type": "text", "text": "Read it."}
    assert str(tmp_path) not in " ".join(command) + run.call_args.kwargs["input"]


def test_answer_is_taken_from_the_result_event(mocker):
    events = [{"type": "system"}, {"type": "assistant"}, _result(structured_output={"a": 2})]
    mocker.patch.object(llm.subprocess, "run", return_value=_finished(*events))
    assert llm._run_cli.__wrapped__("p", SCHEMA, "s", "haiku", None) == {"a": 2}


def test_output_without_a_result_means_claude_is_unavailable(mocker):
    reply = _finished({"type": "system"}, stderr="limit reached")
    mocker.patch.object(llm.subprocess, "run", return_value=reply)
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
