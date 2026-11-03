"""Polite, cached HTTP for capture and the evidence sources, refusing private addresses."""

import ipaddress
import os
import socket
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
    headers={"User-Agent": _USER_AGENT},
)
_MAX_REDIRECTS = 5
_NCBI_GAP_SECONDS = 0.34
_NCBI_LOCK = threading.Lock()
_ncbi_last_call = 0.0


class DownloadRefused(RuntimeError):
    """A fetch broke the address, size or time rules."""


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
        Passed to `httpx.Client.build_request`.

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
            response = _open(method, url, **request)
            try:
                response.read()
            finally:
                response.close()
            response.raise_for_status()
    return response


def download(
    url: str, *, max_bytes: int, max_seconds: float, headers: Mapping[str, str] | None = None
) -> tuple[bytes, str]:
    """
    Stream a URL into memory, refusing anything too large, too slow or not public.

    Parameters
    ----------
    url : str
        Public http or https address.
    max_bytes : int
        Size limit, after decompression.
    max_seconds : float
        Limit on the whole transfer, redirects included.
    headers : Mapping[str, str] | None
        Request headers, sent to every hop.

    Returns
    -------
    tuple[bytes, str]
        The body and its Content-Type.

    Raises
    ------
    DownloadRefused
        When the address, size or time rules are broken.
    """
    deadline = time.monotonic() + max_seconds
    response = _open("GET", url, headers=headers)
    try:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_bytes():
            body += chunk
            if len(body) > max_bytes:
                raise DownloadRefused(f"over {max_bytes} bytes: {url[:60]}")
            if time.monotonic() > deadline:
                raise DownloadRefused(f"over {max_seconds}s: {url[:60]}")
        return bytes(body), response.headers.get("content-type", "")
    finally:
        response.close()


def final_url(url: str, *, headers: Mapping[str, str] | None = None) -> str:
    """Return where a link ends up after its redirects, without reading the page."""
    response = _open("GET", url, headers=headers)
    response.close()
    return str(response.url)


def require_public(url: str) -> None:
    """
    Refuse anything but an http or https address on the public internet.

    Parameters
    ----------
    url : str
        Address about to be fetched.

    Raises
    ------
    DownloadRefused
        When the scheme is wrong or the host resolves to a non-public address.
    httpx.ConnectError
        When the host does not resolve.
    """
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise DownloadRefused(f"unsupported address: {url[:60]}")
    # ponytail: httpx resolves the name again when it connects, so a host that rebinds its
    # DNS between the two lookups gets through; closing that means pinning the checked IP.
    addresses = _resolve(parsed.raw_host.decode(), parsed.port or 443)
    if not all(_is_public(address) for address in addresses):
        raise DownloadRefused(f"not a public address: {url[:60]}")


def _open(method: str, url: str, **request: Any) -> httpx.Response:
    """
    Start a request, checking the address of every hop. The caller closes the response.

    Only GET follows redirects, so an API key on a POST never reaches a redirect target.
    """
    for _ in range(_MAX_REDIRECTS + 1):
        require_public(url)
        response = _CLIENT.send(_CLIENT.build_request(method, url, **request), stream=True)
        if method != "GET" or not response.has_redirect_location:
            return response
        response.close()
        url = str(response.url.join(response.headers["location"]))
        request.pop("params", None)
    raise DownloadRefused(f"too many redirects: {url[:60]}")


def _resolve(host: str, port: int) -> list[str]:
    try:
        found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise httpx.ConnectError(f"cannot resolve {host}") from error
    return [entry[4][0] for entry in found]


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_global and not ip.is_multicast


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
