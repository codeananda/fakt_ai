"""Reasoning model calls through the Claude Code CLI, billed to the user's subscription."""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fakt_ai import config
from fakt_ai.cache import cached

Output = TypeVar("Output", bound=BaseModel)

# ponytail: fixed cap on parallel CLI processes, tune if the subscription rate limit bites
_SLOTS = threading.BoundedSemaphore(int(os.environ.get("FAKT_CLAUDE_SLOTS", "4")))
_API_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK")


class ClaudeUnavailable(RuntimeError):
    """The Claude Code CLI failed, usually because a usage limit was reached."""


def ask(
    prompt: str,
    schema: type[Output],
    *,
    system: str = "You are a meticulous, politically neutral research analyst.",
    model: str | None = None,
    image: Path | None = None,
) -> Output:
    """
    Get a schema-validated answer from Claude, caching identical requests.

    Parameters
    ----------
    prompt : str
        The task, including any evidence.
    schema : type[Output]
        Pydantic model the answer must satisfy.
    system : str
        System prompt.
    model : str | None
        Claude Code model alias. Defaults to `config.DEEP_MODEL`.
    image : Path | None
        Image for Claude to read, for screenshots.

    Returns
    -------
    Output
        The validated answer.
    """
    model = model or config.DEEP_MODEL
    json_schema = schema.model_json_schema()
    key = [prompt, system, model, json_schema, image and _file_digest(image)]
    payload = cached(
        "claude", key, lambda: _run_cli(prompt, json_schema, system, model, image)
    )
    return schema.model_validate(payload)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@retry(
    retry=retry_if_exception_type(ClaudeUnavailable),
    wait=wait_exponential(min=20, max=900),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _run_cli(
    prompt: str, json_schema: dict, system: str, model: str, image: Path | None
) -> dict:
    binary = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
    command = [
        binary,
        "-p",
        "--model",
        model,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--json-schema",
        json.dumps(json_schema),
        "--system-prompt",
        system,
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--tools",
        "",
    ]
    env = {k: v for k, v in os.environ.items() if k not in _API_ENV_VARS}
    message = _user_message(prompt, image)
    with _SLOTS, tempfile.TemporaryDirectory() as workdir:
        finished = subprocess.run(
            command, input=message, capture_output=True, text=True, env=env, cwd=workdir
        )
    reply = _final_reply(finished.stdout)
    if reply is None:
        raise ClaudeUnavailable(finished.stderr[:300] or finished.stdout[:300])
    if reply.get("is_error") or "structured_output" not in reply:
        logger.warning("claude -p failed: {}", str(reply.get("result"))[:200])
        raise ClaudeUnavailable(str(reply.get("result"))[:300])
    return reply["structured_output"]


def _user_message(prompt: str, image: Path | None) -> str:
    """Build the one stdin message, carrying any image inline so the CLI needs no file tools."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image:
        picture = image.read_bytes()
        source = {
            "type": "base64",
            "media_type": _picture_type(picture),
            "data": base64.b64encode(picture).decode(),
        }
        content.insert(0, {"type": "image", "source": source})
    return json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"


def _picture_type(picture: bytes) -> str:
    """Read the type from the bytes, because the API rejects a label that does not match."""
    if picture.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if picture.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if picture.startswith(b"GIF8"):
        return "image/gif"
    if picture[:4] == b"RIFF" and picture[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Expected a PNG, JPEG, GIF or WebP image")


def _final_reply(stream: str) -> dict | None:
    for line in reversed(stream.splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return event
    return None
