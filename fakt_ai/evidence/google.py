"""Google results through Serper, for local and niche pages other indexes miss."""

from loguru import logger

from fakt_ai import config, ledger
from fakt_ai.cache import cached
from fakt_ai.evidence.fetch import send
from fakt_ai.models import Source

_ENDPOINT = "https://google.serper.dev/search"
_USD_PER_SEARCH = 0.001
_RESULTS = 8


def search_google(query: str, found_by: str) -> list[Source]:
    """
    Search Google. Without a key the search is skipped.

    Parameters
    ----------
    query : str
        What to look for.
    found_by : str
        Provenance label.

    Returns
    -------
    list[Source]
        Results with the snippet in `abstract`. Page text comes later, with full-text fetching.
    """
    api_key = config.secret("SERPER_API_KEY", "serper-api-key")
    if not api_key:
        logger.warning("no Serper key, skipping Google search")
        return []
    body = cached("serper", [query], lambda: _search(query, api_key))
    return [
        Source(
            title=item.get("title") or item["link"],
            url=item["link"],
            abstract=item.get("snippet", ""),
            found_by=found_by,
        )
        for item in body.get("organic", [])
        if item.get("link")
    ]


def _search(query: str, api_key: str) -> dict:
    ledger.charge("serper", _USD_PER_SEARCH)
    payload = {"q": query, "num": _RESULTS}
    return send("POST", _ENDPOINT, json=payload, headers={"X-API-KEY": api_key}).json()
