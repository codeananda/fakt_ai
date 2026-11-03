"""Web, news and paper search through Exa, a paid source that needs a key."""

from datetime import date

from loguru import logger

from fakt_ai import config, ledger
from fakt_ai.cache import cached
from fakt_ai.evidence.classify import FACT_CHECK_DOMAINS, doi_in_url, pmcid_in_url
from fakt_ai.evidence.fetch import send
from fakt_ai.models import Source, SourceKind

_ENDPOINT = "https://api.exa.ai/search"
_USD_PER_SEARCH = 0.01
_RESULTS = 8
_TEXT_CHARS = 60_000
_SNIPPET_CHARS = 1_500


def search_exa(
    query: str,
    found_by: str,
    *,
    category: str | None = None,
    fact_checkers_only: bool = False,
) -> list[Source]:
    """
    Search Exa, keeping page text. Without a key the search is skipped.

    Parameters
    ----------
    query : str
        What to look for.
    found_by : str
        Provenance label.
    category : str | None
        Exa category, such as "research paper" or "news".
    fact_checkers_only : bool
        Restrict results to known fact-checking sites.

    Returns
    -------
    list[Source]
        Results with text in `full_text` and a leading snippet in `abstract`.
    """
    api_key = config.secret("EXA_API_KEY", "exa-api-key")
    if not api_key:
        logger.warning("no Exa key, skipping Exa search")
        return []
    payload = {
        "query": query,
        "type": "auto",
        "numResults": _RESULTS,
        "contents": {"text": {"maxCharacters": _TEXT_CHARS}},
    }
    if category:
        payload["category"] = category
    if fact_checkers_only:
        payload["includeDomains"] = sorted(FACT_CHECK_DOMAINS)
    freshness = date.today().isoformat() if category == "news" else None
    body = cached("exa", [payload, freshness], lambda: _search(payload, api_key))
    return [_from_result(item, found_by, category) for item in body.get("results", [])]


def _search(payload: dict, api_key: str) -> dict:
    ledger.charge("exa", _USD_PER_SEARCH)
    return send("POST", _ENDPOINT, json=payload, headers={"x-api-key": api_key}).json()


def _from_result(item: dict, found_by: str, category: str | None) -> Source:
    url = item.get("url", "")
    text = item.get("text") or ""
    published = item.get("publishedDate") or ""
    return Source(
        kind=SourceKind.PAPER if category == "research paper" else SourceKind.WEB,
        title=item.get("title") or url,
        url=url,
        doi=doi_in_url(url),
        pmcid=pmcid_in_url(url),
        year=int(published[:4]) if published[:4].isdigit() else None,
        authors=[item["author"]] if item.get("author") else [],
        abstract=text[:_SNIPPET_CHARS],
        full_text=text,
        found_by=found_by,
    )
