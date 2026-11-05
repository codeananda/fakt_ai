"""Decide what kind of thing a source is, and merge duplicate records of it."""

import hashlib
import re
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit

from fakt_ai.models import Source, SourceKind

FACT_CHECK_DOMAINS = frozenset(
    {
        "snopes.com",
        "politifact.com",
        "factcheck.org",
        "fullfact.org",
        "leadstories.com",
        "healthfeedback.org",
        "climatefeedback.org",
        "science.feedback.org",
        "factcheck.afp.com",
        "checkyourfact.com",
        "africacheck.org",
        "truthorfiction.com",
    }
)
_FACT_CHECK_PATHS = (
    "reuters.com/fact-check",
    "apnews.com/hub/ap-fact-check",
    "apnews.com/article/fact-check",
    "bbc.co.uk/news/reality_check",
    "bbc.com/news/reality_check",
    "washingtonpost.com/politics/fact-checker",
    "usatoday.com/story/news/factcheck",
)
_DATASET_HOSTS = (
    "data.giss.nasa.gov",
    "ncei.noaa.gov",
    "psl.noaa.gov",
    "data.gov",
    "data.gov.uk",
    "data.cdc.gov",
    "data.who.int",
    "data.worldbank.org",
    "ourworldindata.org",
    "berkeleyearth.org",
    "crudata.uea.ac.uk",
    "metoffice.gov.uk/hadobs",
    "ons.gov.uk/datasets",
    "clinicaltrials.gov",
    "zenodo.org",
    "datadryad.org",
    "ec.europa.eu/eurostat",
)
_PAPER_HOSTS = (
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov/pmc",
    "pmc.ncbi.nlm.nih.gov",
    "europepmc.org",
    "doi.org",
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "ssrn.com",
    "sciencedirect.com",
    "link.springer.com",
    "onlinelibrary.wiley.com",
    "tandfonline.com",
    "journals.plos.org",
    "thelancet.com",
    "nejm.org",
    "bmj.com",
    "jamanetwork.com",
    "pnas.org",
    "researchgate.net",
    "semanticscholar.org",
)
_OFFICIAL_SUFFIXES = (
    ".gov",
    ".mil",
    ".int",
    ".nhs.uk",
    ".europa.eu",
    ".gc.ca",
    ".judiciary.uk",
)
_OFFICIAL_MARKERS = re.compile(r"\.gov\.[a-z]{2}$|\.gouv\.[a-z]{2}$|\.go\.[a-z]{2}$")
_OFFICIAL_HOSTS = (
    "gov.uk",
    "un.org",
    "oecd.org",
    "worldbank.org",
    "imf.org",
    "supremecourt.uk",
)
_NEWS_HOSTS = (
    "reuters.com",
    "apnews.com",
    "bbc.co.uk",
    "bbc.com",
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "ft.com",
    "theguardian.com",
    "economist.com",
    "bloomberg.com",
    "npr.org",
    "cnn.com",
    "nbcnews.com",
    "abcnews.go.com",
    "cbsnews.com",
    "foxnews.com",
    "politico.com",
    "axios.com",
    "statnews.com",
    "telegraph.co.uk",
    "thetimes.co.uk",
    "independent.co.uk",
    "aljazeera.com",
)
_DOI_IN_URL = re.compile(r"\b10\.\d{4,9}/[^\s?#]+")
_DOI_PAGE_SUFFIX = re.compile(r"(\.pdf|/(pdf|full|abstract|fulltext|epdf))+$", re.IGNORECASE)
_PMCID_IN_URL = re.compile(r"\bPMC\d+\b", re.IGNORECASE)
_RETRACTION_PREFIX = re.compile(r"^\W*(retracted|withdrawn)\W+", re.IGNORECASE)
_MIN_TITLE_KEY_CHARS = 25


def doi_in_url(url: str) -> str | None:
    """Extract a DOI from a link to a paper."""
    found = _DOI_IN_URL.search(url)
    return _DOI_PAGE_SUFFIX.sub("", found.group()).rstrip("/").lower() if found else None


def pmcid_in_url(url: str) -> str | None:
    """Extract a PubMed Central id from a link."""
    found = _PMCID_IN_URL.search(url)
    return found.group().upper() if found else None


def classify(source: Source) -> SourceKind:
    """
    Decide a source's kind from its address, trusting a scholarly index's own kind.

    Parameters
    ----------
    source : Source
        Candidate whose `kind` is what its search reported.

    Returns
    -------
    SourceKind
        The kind.
    """
    host, path = _host_and_path(source.url)
    address = f"{host}{path}"
    if _at_any(address, FACT_CHECK_DOMAINS) or _at_any(address, _FACT_CHECK_PATHS):
        return SourceKind.FACT_CHECK
    if source.kind is not SourceKind.WEB:
        return source.kind
    if _at_any(address, _DATASET_HOSTS):
        return SourceKind.DATASET
    if source.doi or _at_any(address, _PAPER_HOSTS):
        return SourceKind.PAPER
    if (
        host.endswith(_OFFICIAL_SUFFIXES)
        or _OFFICIAL_MARKERS.search(host)
        or _at_any(host, _OFFICIAL_HOSTS)
    ):
        return SourceKind.OFFICIAL
    if ":news" in source.found_by or _at_any(host, _NEWS_HOSTS):
        return SourceKind.NEWS
    return SourceKind.WEB


def merge_duplicates(sources: Iterable[Source]) -> list[Source]:
    """
    Collapse records sharing a DOI, normalised title or URL into one.

    Records with the same title but different DOIs stay apart, because replies and letters often reuse the title of the study they answer. A retraction banner overrides this, as a notice can carry its own DOI.

    Parameters
    ----------
    sources : Iterable[Source]
        Candidates from every search.

    Returns
    -------
    list[Source]
        One record per work, with a stable id, in first-seen order.
    """
    owners: dict[str, Source] = {}
    merged: list[Source] = []
    for source in sources:
        keys = identity_keys(source)
        matches = _distinct(
            owners[key]
            for key in keys
            if key in owners and not _title_clash(key, source, owners[key])
        )
        keeper = matches[0] if matches else source
        if not matches:
            merged.append(source)
        for other in [*matches[1:], source]:
            if other is not keeper:
                _absorb(keeper, other)
        if len(matches) > 1:
            absorbed = {id(other) for other in matches[1:]}
            merged = [m for m in merged if id(m) not in absorbed]
            owners = {k: keeper if id(o) in absorbed else o for k, o in owners.items()}
        owners.update(dict.fromkeys(keys + identity_keys(keeper), keeper))
    for source in merged:
        stable = source.doi or source.url
        source.id = hashlib.sha256(stable.encode()).hexdigest()[:8]
    return merged


def identity_keys(source: Source) -> list[str]:
    """Return every key that marks two records as one work: DOI, normalised URL and title."""
    keys = [f"url:{_normalised_url(source.url)}"] if source.url else []
    if source.doi:
        keys.append(f"doi:{source.doi.lower()}")
    title = normalised_title(source.title)
    if len(title) >= _MIN_TITLE_KEY_CHARS:
        keys.append(f"title:{title}")
    return keys


def normalised_title(title: str) -> str:
    """Reduce a title to lowercase letters and digits, ignoring any retraction banner."""
    return re.sub(r"[\W_]+", "", _RETRACTION_PREFIX.sub("", title).casefold())


def _title_clash(key: str, source: Source, owner: Source) -> bool:
    """Return whether a title match joins two works that have different DOIs."""
    has_banner = _RETRACTION_PREFIX.match(source.title) or _RETRACTION_PREFIX.match(
        owner.title
    )
    return bool(
        key.startswith("title:")
        and source.doi
        and owner.doi
        and source.doi.lower() != owner.doi.lower()
        and not has_banner
    )


def _distinct(sources: Iterable[Source]) -> list[Source]:
    return list({id(s): s for s in sources}.values())


def _absorb(keeper: Source, other: Source) -> None:
    keeper.doi = keeper.doi or other.doi
    keeper.pmcid = keeper.pmcid or other.pmcid
    keeper.open_access_url = keeper.open_access_url or other.open_access_url
    keeper.year = keeper.year or other.year
    keeper.venue = keeper.venue or other.venue
    keeper.authors = keeper.authors or other.authors
    keeper.citations = max(keeper.citations or 0, other.citations or 0) or None
    keeper.retracted = keeper.retracted or other.retracted
    keeper.abstract = max(keeper.abstract, other.abstract, key=len)
    keeper.full_text = max(keeper.full_text, other.full_text, key=len)
    keeper.publication_types = list(
        dict.fromkeys(keeper.publication_types + other.publication_types)
    )
    keeper.found_by = _combined_provenance(keeper.found_by, other.found_by)
    if keeper.kind is SourceKind.WEB:
        keeper.kind = other.kind


def _combined_provenance(first: str, second: str) -> str:
    """Join search labels; a cited work stays exactly "cited"."""
    if "cited" in (first, second):
        return "cited"
    labels = dict.fromkeys(f"{first},{second}".split(","))
    return ",".join(label for label in labels if label)


def _normalised_url(url: str) -> str:
    host, path = _host_and_path(url)
    kept = [(k, v) for k, v in parse_qsl(urlsplit(url).query) if not k.startswith("utm_")]
    return f"{host}{path}?{urlencode(kept)}" if kept else f"{host}{path}"


def _host_and_path(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().removeprefix("www.")
    return host, parts.path.rstrip("/")


def _at_any(address: str, places: Iterable[str]) -> bool:
    """Match a host and path against domains or domain/path prefixes, subdomains included."""
    host, _, path = address.partition("/")
    for place in places:
        domain, _, prefix = place.partition("/")
        on_domain = host == domain or host.endswith(f".{domain}")
        if on_domain and path.startswith(prefix):
            return True
    return False
