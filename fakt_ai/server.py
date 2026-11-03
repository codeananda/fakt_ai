"""Local API and web app, reachable only from this machine."""

import asyncio
import base64
import binascii
import json
import re
import tempfile
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit
from uuid import uuid4

import fire
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.responses import Response

from fakt_ai import ledger, pipeline, store
from fakt_ai.models import Report

HOST = "127.0.0.1"
PORT = 8741

_UI_DIR = Path(__file__).resolve().parent.parent / "extension" / "ui"
_ORIGIN_PATTERN = r"(chrome-extension://[a-p]{32}|https?://(localhost|127\.0\.0\.1)(:\d+)?)"
_PAGE_POLICY = (
    "default-src 'self'; img-src 'self' data:; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'"
)
_EVENT_POLL_SECONDS = 0.1
_MAX_IMAGE_BASE64 = 20_000_000
_MAX_TEXT = 200_000
_EXCLUDED_FROM_REPORT = {"checks": {"__all__": {"sources": {"__all__": {"full_text"}}}}}
_IMAGE_SIGNATURES = {b"\x89PNG": ".png", b"\xff\xd8\xff": ".jpg", b"RIFF": ".webp"}

ReportId = Annotated[str, PathParam(pattern=r"^[\w-]{1,64}$")]


class CheckRequest(BaseModel):
    """What the user wants checked: a link, text, a screenshot or a mix."""

    url: str | None = None
    text: str | None = Field(None, max_length=_MAX_TEXT)
    image_base64: str | None = Field(None, max_length=_MAX_IMAGE_BASE64)
    max_claims: int = Field(5, ge=1, le=10)

    @field_validator("url")
    @classmethod
    def _only_web_links(cls, url: str | None) -> str | None:
        if url and urlsplit(url).scheme not in ("http", "https"):
            raise ValueError("Only http and https links can be checked")
        return url

    @model_validator(mode="after")
    def _something_to_check(self) -> "CheckRequest":
        if not (self.url or self.text or self.image_base64):
            raise ValueError("Send a url, text or image_base64")
        return self


class Question(BaseModel):
    """A follow-up about a report, optionally about one source."""

    question: str = Field(min_length=1, max_length=4000)
    source_id: str | None = None


@dataclass
class _Job:
    """A running check. Events are appended by the worker thread and read by streams."""

    events: list[dict] = field(default_factory=list)
    report_id: str | None = None
    error: str | None = None


# ponytail: finished jobs stay in memory until the process exits, fine for one local user.
_jobs: dict[str, _Job] = {}

app = FastAPI(title="Fakt AI", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_ORIGIN_PATTERN,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Last-Event-ID"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[HOST, "localhost"])


@app.middleware("http")
async def _refuse_foreign_origins(request: Request, call_next) -> Response:
    # CORS only hides responses; a web page could still fire a POST and spend money.
    origin = request.headers.get("origin")
    if origin and not re.fullmatch(_ORIGIN_PATTERN, origin):
        return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = _PAGE_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.post("/checks", status_code=202)
def start_check(check: CheckRequest) -> dict:
    """Start a check in a background thread and return its job id."""
    image = _decode_image(check.image_base64) if check.image_base64 else None
    job_id = uuid4().hex
    job = _jobs[job_id] = _Job()
    threading.Thread(target=_run_check, args=(job, check, image), daemon=True).start()
    return {"id": job_id}


@app.get("/checks/{job_id}/events")
def check_events(
    job_id: str, last_event_id: Annotated[int | None, Header()] = None
) -> StreamingResponse:
    """Stream progress as server-sent events, ending with `done` or `failed`."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such check. It may predate a server restart.")
    first = 0 if last_event_id is None else last_event_id + 1
    return StreamingResponse(
        _stream_events(job, first),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/reports")
def list_reports(limit: int = 30) -> list[dict]:
    """Return short descriptions of recent reports, newest first."""
    return [_describe(report) for report in store.recent(limit=limit)]


@app.get("/reports/{report_id}")
def read_report(report_id: ReportId) -> dict:
    """Return one report, without the full text of its sources."""
    return _load(report_id).model_dump(mode="json", exclude=_EXCLUDED_FROM_REPORT)


@app.post("/reports/{report_id}/ask")
def ask_report(report_id: ReportId, ask: Question) -> dict:
    """Answer a follow-up question about a report."""
    report = _load(report_id)
    known = {source.id for check in report.checks for source in check.sources}
    if ask.source_id and ask.source_id not in known:
        raise HTTPException(404, "That source is not part of this report.")
    try:
        answer = pipeline.follow_up(report, ask.question, source_id=ask.source_id)
    except Exception as error:
        logger.exception("Follow-up failed for report {}", report_id)
        raise HTTPException(502, f"The question could not be answered: {error}") from error
    return {"answer": answer}


@app.get("/spend")
def read_spend() -> dict:
    """Return estimated spend per paid service in US dollars."""
    return ledger.summary()


app.mount("/", StaticFiles(directory=_UI_DIR, html=True), name="ui")


def serve(port: int = PORT) -> None:
    """
    Run the server on the loopback interface only.

    Parameters
    ----------
    port : int
        Port to listen on. The extension only reaches the default.
    """
    uvicorn.run(app, host=HOST, port=port)


def _run_check(job: _Job, check: CheckRequest, image: bytes | None) -> None:
    try:
        with tempfile.TemporaryDirectory(prefix="fakt-") as folder:
            image_path = None
            if image:
                image_path = Path(folder) / f"screenshot{_image_suffix(image)}"
                image_path.write_bytes(image)
            report = pipeline.run(
                url=check.url,
                text=check.text,
                image=image_path,
                on_event=job.events.append,
                max_claims=check.max_claims,
            )
        job.report_id = report.id
    except Exception as error:  # noqa: BLE001 - any failure must reach the waiting stream
        logger.exception("Check failed")
        job.error = str(error) or type(error).__name__


async def _stream_events(job: _Job, first: int) -> AsyncIterator[str]:
    sent = first
    while True:
        # Read the outcome first: the worker sets it only after its last event.
        report_id, error = job.report_id, job.error
        for index in range(sent, len(job.events)):
            yield f"id: {index}\ndata: {json.dumps(job.events[index])}\n\n"
            sent = index + 1
        if report_id:
            yield f"event: done\ndata: {json.dumps({'report_id': report_id})}\n\n"
            return
        if error:
            yield f"event: failed\ndata: {json.dumps({'message': error})}\n\n"
            return
        await asyncio.sleep(_EVENT_POLL_SECONDS)


def _decode_image(encoded: str) -> bytes:
    payload = encoded.split(",", 1)[-1] if encoded.startswith("data:") else encoded
    try:
        image = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(422, "image_base64 is not valid base64") from error
    if _image_suffix(image) is None:
        raise HTTPException(422, "image_base64 must be a PNG, JPEG or WebP image")
    return image


def _image_suffix(image: bytes) -> str | None:
    for signature, suffix in _IMAGE_SIGNATURES.items():
        if image.startswith(signature):
            return suffix
    return None


def _load(report_id: str) -> Report:
    try:
        return store.load(report_id)
    except (FileNotFoundError, KeyError) as error:
        raise HTTPException(404, "No such report") from error


def _describe(report: Report) -> dict:
    return {
        "id": report.id,
        "created": report.created,
        "platform": report.capture.platform,
        "url": report.capture.url,
        "author": report.capture.author,
        "excerpt": report.capture.content[:160],
        "claims": [
            {
                "text": check.claim.text,
                "label": check.verdict.label if check.verdict else None,
                "probability_true": (
                    check.verdict.probability_true if check.verdict else None
                ),
            }
            for check in report.checks
        ],
    }


if __name__ == "__main__":
    fire.Fire(serve)
