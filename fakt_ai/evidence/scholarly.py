"""Searches over the free scholarly indexes, each mapped to `Source` records."""

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from urllib.parse import quote

import httpx
from loguru import logger

from fakt_ai.evidence.fetch import get_json, get_ncbi_text
from fakt_ai.models import Source, SourceKind

_PER_QUERY = 15
_OPENALEX_FIELDS = (
    "id,doi,title,publication_year,type,is_retracted,cited_by_count,primary_location,"
    "open_access,ids,authorships,abstract_inverted_index"
)
_OPENALEX = "https://api.openalex.org/works"
_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_CROSSREF = "https://api.crossref.org/works"
_SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/search"
_CLINICAL_TRIALS = "https://clinicaltrials.gov/api/v2/studies"
_RETRACTED_TYPES = {"retraction", "withdrawal", "removal"}
_CONCERN_TYPES = {"expression_of_concern", "expression-of-concern"}
_RETRACTED_TITLE = re.compile(r"^\W*(retracted|withdrawn)\b", re.IGNORECASE)
_MARKUP = re.compile(r"<[^>]+>")
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>]+")
_CITED_NOISE = re.compile(r"\b(the|study|paper|report|review|et|al|and|of|in|from|by|on)\b")
_DOI_SAFE = "/:()"
_MIN_CITED_MATCH = 0.6


def search_openalex(query: str, found_by: str) -> list[Source]:
    """Search OpenAlex works by relevance."""
    params = {"search": query, "per-page": _PER_QUERY, "select": _OPENALEX_FIELDS}
    payload = get_json("openalex", _OPENALEX, params, attempts=3)
    return [_from_openalex(work, found_by) for work in payload.get("results", [])]


def search_pubmed(query: str, found_by: str, *, reviews_only: bool = False) -> list[Source]:
    """Search PubMed through esearch and efetch."""
    term = f"({query}) AND systematic[sb]" if reviews_only else query
    params = {"db": "pubmed", "term": term, "retmax": _PER_QUERY, "retmode": "xml"}
    found = ET.fromstring(get_ncbi_text(f"{_EUTILS}/esearch.fcgi", params))
    ids = [node.text for node in found.iterfind("IdList/Id") if node.text]
    if not ids:
        return []
    params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
    return parse_pubmed(get_ncbi_text(f"{_EUTILS}/efetch.fcgi", params), found_by)


def search_europe_pmc(query: str, found_by: str) -> list[Source]:
    """Search Europe PMC, which also covers preprints."""
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": _PER_QUERY,
    }
    payload = get_json("europepmc", _EUROPE_PMC, params)
    results = payload.get("resultList", {}).get("result", [])
    return [_from_europe_pmc(item, found_by) for item in results if item.get("title")]


def search_semantic_scholar(query: str, found_by: str) -> list[Source]:
    """Search Semantic Scholar once without a key, returning nothing when rate limited."""
    params = {
        "query": query,
        "limit": _PER_QUERY,
        "fields": "title,abstract,year,venue,citationCount,externalIds,publicationTypes,"
        "openAccessPdf,authors,url",
    }
    try:
        payload = get_json("semanticscholar", _SEMANTIC_SCHOLAR, params, attempts=1)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 429:
            raise
        logger.info("Semantic Scholar rate limited, skipping")
        return []
    return [_from_semantic_scholar(p, found_by) for p in payload.get("data") or []]


def search_clinical_trials(query: str, found_by: str) -> list[Source]:
    """Search trial registrations on ClinicalTrials.gov."""
    params = {
        "query.term": query,
        "pageSize": 10,
        "fields": "NCTId,BriefTitle,BriefSummary,OverallStatus,StartDate,HasResults,"
        "LeadSponsorName",
    }
    payload = get_json("clinicaltrials", _CLINICAL_TRIALS, params)
    return [_from_trial(study, found_by) for study in payload.get("studies", [])]


def resolve_cited_work(citation: str, likely_reference: str = "") -> list[Source]:
    """
    Find the paper a speaker cited, with every record of it flagged `found_by="cited"`.

    Parameters
    ----------
    citation : str
        The speaker's words, such as "Wakefield et al., Lancet 1998", or a DOI.
    likely_reference : str
        A fuller reference to try first. Crossref must confirm it.

    Returns
    -------
    list[Source]
        Records of the paper from Crossref, OpenAlex and Europe PMC, to be merged by DOI.
        Empty when no credible match exists.
    """
    attempts = (reference for reference in (likely_reference, citation) if reference)
    work = next(filter(None, map(_match_on_crossref, attempts)), None)
    if work is None:
        logger.warning("could not resolve cited work: {}", citation[:120])
        return []
    resolved = _from_crossref(work, "cited")
    return [resolved, *_records_for_doi(resolved.doi or "")]


def note_editorial_updates(source: Source) -> None:
    """Mark a source retracted, or of concern, from Crossref's `updated-by` records."""
    message = get_json("crossref", _crossref_work_url(source.doi)).get("message", {})
    updates = {u.get("type", "").lower() for u in message.get("updated-by", [])}
    source.retracted = source.retracted or bool(updates & _RETRACTED_TYPES)
    if updates & _CONCERN_TYPES:
        source.publication_types = [*source.publication_types, "Expression of Concern"]


def parse_pubmed(xml: str, found_by: str) -> list[Source]:
    """Map a PubMed efetch document to sources."""
    root = ET.fromstring(xml)
    return [_from_pubmed(article, found_by) for article in root.iterfind("PubmedArticle")]


def rebuild_abstract(inverted_index: Mapping[str, Iterable[int]] | None) -> str:
    """Rebuild plain text from OpenAlex's word-to-positions abstract index."""
    if not inverted_index:
        return ""
    placed = [
        (position, word)
        for word, positions in inverted_index.items()
        for position in positions
    ]
    return " ".join(word for _, word in sorted(placed))


def plain_text(markup: str | None) -> str:
    """Strip tags such as JATS and collapse whitespace."""
    return " ".join(_MARKUP.sub(" ", markup or "").split())


def _records_for_doi(doi: str) -> list[Source]:
    records: list[Source] = []
    if not doi:
        return records
    try:
        address = f"{_OPENALEX}/doi:{quote(doi, safe=_DOI_SAFE)}"
        work = get_json("openalex", address, {"select": _OPENALEX_FIELDS})
        records.append(_from_openalex(work, "cited"))
    except httpx.HTTPError as e:
        logger.info("OpenAlex has no record of {}: {}", doi, e)
    try:
        found = search_europe_pmc(f'DOI:"{doi}"', "cited")
        records += [record for record in found if record.doi == doi]
    except httpx.HTTPError as e:
        logger.info("Europe PMC has no record of {}: {}", doi, e)
    return records


def _match_on_crossref(citation: str) -> dict | None:
    if doi := _DOI.search(citation):
        try:
            bare = doi.group().rstrip(".,;")
            return get_json("crossref", _crossref_work_url(bare)).get("message")
        except httpx.HTTPStatusError:
            logger.info("cited DOI not on Crossref: {}", doi.group())
    params = {"query.bibliographic": citation, "rows": 5}
    items = get_json("crossref", _CROSSREF, params).get("message", {}).get("items", [])
    return _best_cited_match(citation, items)


def _best_cited_match(citation: str, items: Iterable[dict]) -> dict | None:
    """Prefer the record covering most of the citation, then the most cited one."""
    scored = [(_citation_coverage(citation, item), item) for item in items]
    credible = [(cover, item) for cover, item in scored if cover >= _MIN_CITED_MATCH]
    if not credible:
        return None
    return max(
        credible, key=lambda pair: (pair[0], pair[1].get("is-referenced-by-count") or 0)
    )[1]


def _citation_coverage(citation: str, item: dict) -> float:
    """Return the share of the citation's meaningful words found in a Crossref record."""
    year = _crossref_year(item)
    years = re.findall(r"\b(?:1[89]|20)\d{2}\b", citation)
    if year and years and all(abs(int(y) - year) > 1 for y in years):
        return 0.0
    wanted = set(re.findall(r"[a-z]{3,}", _CITED_NOISE.sub(" ", citation.lower())))
    if not wanted:
        return 0.0
    surnames = " ".join(a.get("family", "") for a in item.get("author", []))
    record = " ".join([*item.get("title", []), *item.get("container-title", []), surnames])
    return len(wanted & set(re.findall(r"[a-z]{3,}", record.lower()))) / len(wanted)


def _crossref_work_url(doi: str) -> str:
    return f"{_CROSSREF}/{quote(doi, safe=_DOI_SAFE)}"


def _crossref_year(item: dict) -> int | None:
    parts = (item.get("issued") or {}).get("date-parts") or [[None]]
    return parts[0][0] if parts[0] else None


def _from_crossref(item: dict, found_by: str) -> Source:
    doi = item.get("DOI", "").lower()
    title = plain_text(" ".join(item.get("title", []))) or doi
    updates = {u.get("type", "").lower() for u in item.get("updated-by", [])}
    return Source(
        kind=SourceKind.PAPER,
        title=title,
        url=item.get("URL") or f"https://doi.org/{doi}",
        doi=doi or None,
        year=_crossref_year(item),
        venue=next(iter(item.get("container-title", [])), None),
        authors=[
            " ".join(filter(None, [a.get("given"), a.get("family")]))
            for a in item.get("author", [])
        ],
        citations=item.get("is-referenced-by-count"),
        abstract=plain_text(item.get("abstract")),
        retracted=bool(updates & _RETRACTED_TYPES) or bool(_RETRACTED_TITLE.match(title)),
        publication_types=[item["type"]] if item.get("type") else [],
        found_by=found_by,
    )


def _from_openalex(work: dict, found_by: str) -> Source:
    ids = work.get("ids") or {}
    location = work.get("primary_location") or {}
    title = plain_text(work.get("title")) or work.get("id", "untitled")
    doi = _bare_doi(work.get("doi"))
    return Source(
        kind=SourceKind.DATASET if work.get("type") == "dataset" else SourceKind.PAPER,
        title=title,
        url=location.get("landing_page_url") or work.get("doi") or work.get("id", ""),
        doi=doi,
        pmcid=_bare_pmcid(ids.get("pmcid")),
        open_access_url=(work.get("open_access") or {}).get("oa_url"),
        year=work.get("publication_year"),
        venue=(location.get("source") or {}).get("display_name"),
        authors=[
            (a.get("author") or {}).get("display_name", "")
            for a in work.get("authorships", [])
        ],
        citations=work.get("cited_by_count"),
        abstract=rebuild_abstract(work.get("abstract_inverted_index")),
        retracted=bool(work.get("is_retracted")) or bool(_RETRACTED_TITLE.match(title)),
        publication_types=[work["type"]] if work.get("type") else [],
        found_by=found_by,
    )


def _from_pubmed(article: ET.Element, found_by: str) -> Source:
    citation = article.find("MedlineCitation")
    pmid = citation.findtext("PMID", "")
    title = _element_text(citation.find("Article/ArticleTitle"))
    types = [_element_text(t) for t in citation.iterfind("Article/PublicationTypeList/*")]
    ids = {
        node.get("IdType"): node.text
        for node in article.iterfind("PubmedData/ArticleIdList/ArticleId")
    }
    sections = [
        (node.get("Label"), _element_text(node))
        for node in citation.iterfind("Article/Abstract/AbstractText")
    ]
    pub_date = citation.find("Article/Journal/JournalIssue/PubDate")
    year_text = pub_date.findtext("Year") or pub_date.findtext("MedlineDate") or ""
    authors = [
        f"{a.findtext('ForeName', '')} {a.findtext('LastName', '')}".strip()
        for a in citation.iterfind("Article/AuthorList/Author")
    ]
    return Source(
        kind=SourceKind.PAPER,
        title=title or f"PubMed {pmid}",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        doi=(ids.get("doi") or "").lower() or None,
        pmcid=_bare_pmcid(ids.get("pmc")),
        year=_first_year(year_text),
        venue=citation.findtext("Article/Journal/Title"),
        authors=[a for a in authors if a],
        abstract=" ".join(f"{k}: {v}" if k else v for k, v in sections),
        retracted="Retracted Publication" in types,
        publication_types=types,
        found_by=found_by,
    )


def _from_europe_pmc(item: dict, found_by: str) -> Source:
    types = (item.get("pubTypeList") or {}).get("pubType") or []
    title = plain_text(item.get("title"))
    links = (item.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    is_open = item.get("isOpenAccess") == "Y"
    open_links = [link["url"] for link in links if link.get("availabilityCode") in ("OA", "F")]
    doi = (item.get("doi") or "").lower() or None
    pmid = item.get("pmid")
    fallback = f"https://europepmc.org/article/{item.get('source')}/{item.get('id')}"
    journal = (item.get("journalInfo") or {}).get("journal") or {}
    return Source(
        kind=SourceKind.PAPER,
        title=title,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else fallback,
        doi=doi,
        pmcid=_bare_pmcid(item.get("pmcid")),
        open_access_url=next(iter(open_links), None) if is_open else None,
        year=_first_year(str(item.get("pubYear", ""))),
        venue=journal.get("title"),
        authors=[a.strip() for a in (item.get("authorString") or "").split(",") if a.strip()],
        citations=item.get("citedByCount"),
        abstract=plain_text(item.get("abstractText")),
        retracted="Retracted Publication" in types or bool(_RETRACTED_TITLE.match(title)),
        publication_types=types,
        found_by=found_by,
    )


def _from_semantic_scholar(paper: dict, found_by: str) -> Source:
    external = paper.get("externalIds") or {}
    doi = (external.get("DOI") or "").lower() or None
    return Source(
        kind=SourceKind.PAPER,
        title=paper.get("title") or "untitled",
        url=paper.get("url") or (f"https://doi.org/{doi}" if doi else ""),
        doi=doi,
        pmcid=_bare_pmcid(external.get("PubMedCentral")),
        open_access_url=(paper.get("openAccessPdf") or {}).get("url") or None,
        year=paper.get("year"),
        venue=paper.get("venue") or None,
        authors=[a.get("name", "") for a in paper.get("authors") or []],
        citations=paper.get("citationCount"),
        abstract=paper.get("abstract") or "",
        publication_types=paper.get("publicationTypes") or [],
        found_by=found_by,
    )


def _from_trial(study: dict, found_by: str) -> Source:
    protocol = study.get("protocolSection", {})
    nct_id = protocol.get("identificationModule", {}).get("nctId", "")
    status = protocol.get("statusModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    summary = protocol.get("descriptionModule", {}).get("briefSummary", "")
    facts = [
        f"Status: {status.get('overallStatus', 'unknown')}.",
        f"Results posted: {'yes' if study.get('hasResults') else 'no'}.",
        f"Sponsor: {sponsor.get('name', 'unknown')}.",
    ]
    return Source(
        kind=SourceKind.DATASET,
        title=protocol.get("identificationModule", {}).get("briefTitle", nct_id),
        url=f"https://clinicaltrials.gov/study/{nct_id}",
        year=_first_year(status.get("startDateStruct", {}).get("date", "")),
        venue="ClinicalTrials.gov",
        abstract=" ".join([*facts, summary]),
        publication_types=["Clinical Trial Registration"],
        found_by=found_by,
    )


def _element_text(node: ET.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def _bare_doi(doi: str | None) -> str | None:
    return doi.lower().removeprefix("https://doi.org/") if doi else None


def _bare_pmcid(pmcid: str | None) -> str | None:
    if not pmcid:
        return None
    found = re.search(r"(?:PMC)?(\d+)/?$", pmcid, re.IGNORECASE)
    return f"PMC{found.group(1)}" if found else None


def _first_year(text: str) -> int | None:
    found = re.search(r"\b(?:1[89]|20)\d{2}\b", text)
    return int(found.group()) if found else None
