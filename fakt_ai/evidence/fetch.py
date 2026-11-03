"""Polite, cached HTTP for the evidence sources."""

import os
import threading
import time
from collections.abc import Mapping
from typing import Any

import httpx
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from fakt_ai.cache import cached

_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
_USER_AGENT = "fakt-ai/0.2 (evidence-graded fact-checking)"
_CLIENT = httpx.Client(
    timeout=httpx.Timeout(30, connect=15),
    follow_redirects=True,
    headers={"User-Agent": _USER_AGENT},
)
_NCBI_GAP_SECONDS = 0.34
_NCBI_LOCK = threading.Lock()
_ncbi_last_call = 0.0


class DownloadRefused(RuntimeError):
    """A download broke its size or time limit."""


def contact_params() -> dict[str, str]:
    """Return the `mailto` query parameter when an operator has set a contact address."""
    email = os.environ.get("FAKT_CONTACT_EMAIL")
    return {"mailto": email} if email else {}


def get_json(
    namespace: str,
    url: str,
    params: Mapping[str, Any] | None = None,
    *,
    attempts: int = 4,
) -> Any:
    """
    Fetch JSON once per distinct request, retrying transient failures.

    Parameters
    ----------
    namespace : str
        Cache namespace, usually the service name.
    url : str
        Endpoint.
    params : Mapping[str, Any] | None
        Query parameters.
    attempts : int
        Total tries before giving up.

    Returns
    -------
    Any
        The decoded JSON.
    """
    params = dict(params or {})
    return cached(
        namespace,
        [url, params],
        lambda: send("GET", url, params=params | contact_params(), attempts=attempts).json(),
    )


def get_ncbi_text(url: str, params: Mapping[str, Any]) -> str:
    """Fetch text from NCBI once per distinct request, at most three calls a second."""
    params = dict(params)
    return cached("ncbi", [url, params], lambda: _send_paced(url, params))


def send(method: str, url: str, *, attempts: int = 4, **request: Any) -> httpx.Response:
    """
    Send one request, backing off on 429, 5xx and connection failures.

    Parameters
    ----------
    method : str
        HTTP method.
    url : str
        Target.
    attempts : int
        Total tries before giving up.
    **request : Any
        Passed to `httpx.Client.request`.

    Returns
    -------
    httpx.Response
        The successful response.
    """
    retrying = Retrying(
        retry=retry_if_exception(_is_transient),
        wait=wait_exponential(min=1, max=20),
        stop=stop_after_attempt(attempts),
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            response = _CLIENT.request(method, url, **request)
            response.raise_for_status()
    return response


def download(url: str, *, max_bytes: int, max_seconds: float) -> tuple[bytes, str]:
    """
    Stream a URL into memory, refusing anything too large or too slow.

    Parameters
    ----------
    url : str
        Public http or https address.
    max_bytes : int
        Size limit.
    max_seconds : float
        Limit on the whole transfer.

    Returns
    -------
    tuple[bytes, str]
        The body and its Content-Type.
    """
    if not url.startswith(("http://", "https://")):
        raise DownloadRefused(f"unsupported address: {url[:60]}")
    deadline = time.monotonic() + max_seconds
    with _CLIENT.stream("GET", url) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_bytes():
            body += chunk
            if len(body) > max_bytes:
                raise DownloadRefused(f"over {max_bytes} bytes: {url[:60]}")
            if time.monotonic() > deadline:
                raise DownloadRefused(f"over {max_seconds}s: {url[:60]}")
        return bytes(body), response.headers.get("content-type", "")


def _send_paced(url: str, params: dict) -> str:
    global _ncbi_last_call
    with _NCBI_LOCK:
        pause = _ncbi_last_call + _NCBI_GAP_SECONDS - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        _ncbi_last_call = time.monotonic()
    email = contact_params().get("mailto")
    identity = {"tool": "fakt-ai"} | ({"email": email} if email else {})
    return send("GET", url, params=params | identity).text


def _is_transient(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in _TRANSIENT_STATUSES
    return isinstance(error, httpx.TransportError)
