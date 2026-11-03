"""Fetch open-access text for the shortlist and trim it to the sections that matter."""

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

import pymupdf
import trafilatura
from loguru import logger

from fakt_ai.cache import cached
from fakt_ai.evidence.fetch import download, send
from fakt_ai.models import Source, SourceKind

_WORKERS = 8
_MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
_MAX_DOWNLOAD_SECONDS = 60
_CHAR_LIMIT = 40_000
_SECTION_SHARE = _CHAR_LIMIT // 5
_TAIL_SHARE = 0.3
_MIN_PAPER_CHARS = 2_000
_MIN_ABSTRACT_RATIO = 1.5
_EUROPE_PMC_XML = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
_CUT = "\n[...]\n"
_DROPPED = re.compile(
    r"^(references?|bibliography|works cited|literature cited)\b", re.IGNORECASE
)
_SECTION_RANKS = (
    (re.compile(r"abstract|summary|conclusion|interpretation", re.IGNORECASE), 0),
    (
        re.compile(
            r"conflict|competing|disclosure|declaration|funding|financial|sponsor|acknowledg",
            re.IGNORECASE,
        ),
        0,
    ),
    (
        re.compile(
            r"method|materials|design|participants|patients|result|finding", re.IGNORECASE
        ),
        1,
    ),
    (re.compile(r"discussion|limitation", re.IGNORECASE), 2),
)
_OTHER_RANK = 3
_PLAIN_HEADING = re.compile(
    r"^[ \t]*(?:\d+(?:\.\d+)*\.?[ \t]+)?"
    r"(abstract|summary|introduction|background|materials? and methods|patients and methods"
    r"|methods?|methodology|study design|results and discussion|results?|findings|discussion"
    r"|conclusions?|interpretation|limitations|funding|financial support|financial disclosures?"
    r"|acknowledge?ments?|conflicts? of interests?|competing interests?"
    r"|declarations? of (?:competing )?interests?|disclosures?|references|bibliography)"
    r"[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

pymupdf.TOOLS.mupdf_display_errors(False)


def attach_full_text(sources: Iterable[Source]) -> int:
    """
    Fill `full_text` for each source that has open text, trimmed to about 40k characters.

    Parameters
    ----------
    sources : Iterable[Source]
        Shortlisted sources, changed in place.

    Returns
    -------
    int
        How many sources ended up with full text.
    """
    sources = list(sources)
    with ThreadPoolExecutor(_WORKERS) as pool:
        list(pool.map(_fill, sources))
    return sum(1 for source in sources if source.full_text)


def fit_sections(sections: Sequence[tuple[str, str]], limit: int = _CHAR_LIMIT) -> str:
    """
    Fit titled sections into `limit` characters, favouring what a reviewer needs.

    Conclusions, funding and conflict-of-interest sections are funded first, then methods
    and results, then discussion, then anything else. References are dropped.

    Parameters
    ----------
    sections : Sequence[tuple[str, str]]
        Section titles and bodies, in document order.
    limit : int
        Target length of the bodies.

    Returns
    -------
    str
        The kept sections in document order.
    """
    kept = [(t, b) for t, b in sections if b.strip() and not _DROPPED.match(t.strip())]
    if sum(len(body) for _, body in kept) <= limit:
        return _render(kept)
    if len(kept) == 1:
        return _render([(kept[0][0], _head_and_tail(kept[0][1], limit))])
    allowances = [0] * len(kept)
    remaining = limit
    by_rank = sorted(range(len(kept)), key=lambda i: _section_rank(kept[i][0]))
    for ceiling in (_SECTION_SHARE, limit):
        for i in by_rank:
            grant = min(len(kept[i][1]), ceiling) - allowances[i]
            grant = max(min(grant, remaining), 0)
            allowances[i] += grant
            remaining -= grant
    return _render(
        (title, body if allowed == len(body) else body[:allowed] + _CUT)
        for (title, body), allowed in zip(kept, allowances, strict=True)
        if allowed
    )


def fit_plain_text(text: str, limit: int = _CHAR_LIMIT) -> str:
    """Split extracted text on recognisable headings, then fit it into `limit` characters."""
    return fit_sections(_split_on_headings(text), limit)


def parse_jats(xml: str) -> list[tuple[str, str]]:
    """Read titled sections out of a JATS article, including back matter and funding."""
    root = ET.fromstring(xml)
    sections: list[tuple[str, str]] = []
    for abstract in root.iterfind(".//front//abstract"):
        sections.append(("Abstract", _node_text(abstract)))
    for tag, title in (("funding-group", "Funding"), ("author-notes", "Author notes")):
        sections += [(title, _node_text(n)) for n in root.iterfind(f".//front//{tag}")]
    for part in ("body", "back"):
        for child in root.iterfind(f".//{part}/*"):
            if child.tag != "ref-list":
                heading = child.findtext("title") or child.tag.replace("-", " ")
                sections.append((heading, _node_text(child)))
    return sections


def _fill(source: Source) -> None:
    source.full_text = _best_text(source)


def _best_text(source: Source) -> str:
    is_paper = source.kind is SourceKind.PAPER
    if source.pmcid and (text := _attempt(_europe_pmc_text, source.pmcid)):
        return text
    if source.open_access_url:
        text = _attempt(_page_text, source.open_access_url)
        if _is_substantial(text, source):
            return fit_plain_text(text)
    if source.full_text and (not is_paper or _is_substantial(source.full_text, source)):
        return fit_plain_text(source.full_text)
    return "" if is_paper else fit_plain_text(_attempt(_page_text, source.url))


def _is_substantial(text: str, source: Source) -> bool:
    """Tell a real paper from a landing page repeating the abstract."""
    return len(text) >= max(_MIN_PAPER_CHARS, _MIN_ABSTRACT_RATIO * len(source.abstract))


def _attempt(fetch: Callable[[str], str], target: str) -> str:
    try:
        return fetch(target)
    except Exception as e:  # noqa: BLE001 - publishers fail in every way; each is a miss
        logger.debug("fetch failed for {}: {}", target[:100], e)
        return ""


def _europe_pmc_text(pmcid: str) -> str:
    url = _EUROPE_PMC_XML.format(pmcid=pmcid)
    return cached(
        "fulltext",
        [url],
        lambda: fit_sections(parse_jats(send("GET", url, attempts=2).text)),
    )


def _page_text(url: str) -> str:
    return cached("fulltext", [url], lambda: _extract_download(url))


def _extract_download(url: str) -> str:
    body, content_type = download(
        url, max_bytes=_MAX_DOWNLOAD_BYTES, max_seconds=_MAX_DOWNLOAD_SECONDS
    )
    if body[:5] == b"%PDF-":
        return _pdf_text(body)
    if "html" not in content_type and "xml" not in content_type:
        return ""
    return trafilatura.extract(body, include_tables=True, favor_recall=True) or ""


def _pdf_text(body: bytes) -> str:
    with pymupdf.open(stream=body, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document)


def _split_on_headings(text: str) -> list[tuple[str, str]]:
    marks = list(_PLAIN_HEADING.finditer(text))
    if not marks:
        return [("", text)]
    sections = [("", text[: marks[0].start()])]
    for mark, following in zip(marks, [*marks[1:], None], strict=True):
        end = following.start() if following else len(text)
        sections.append((mark.group(1), text[mark.end() : end].strip()))
    return sections


def _section_rank(title: str) -> int:
    return next((rank for words, rank in _SECTION_RANKS if words.search(title)), _OTHER_RANK)


def _head_and_tail(text: str, limit: int) -> str:
    """Keep both ends, since funding and conflict statements usually close a document."""
    tail = int(limit * _TAIL_SHARE)
    return text[: limit - tail] + _CUT + text[-tail:]


def _render(sections: Iterable[tuple[str, str]]) -> str:
    parts = [f"{title}\n{body}".strip() for title, body in sections]
    return "\n\n".join(parts)


def _node_text(node: ET.Element) -> str:
    return " ".join("".join(node.itertext()).split())
