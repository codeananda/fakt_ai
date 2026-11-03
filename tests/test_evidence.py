import httpx
import pytest
import tenacity

from fakt_ai import config, evidence, ledger
from fakt_ai.evidence import exa, fetch, fulltext, plan, rank, scholarly
from fakt_ai.evidence.classify import (
    classify,
    doi_in_url,
    merge_duplicates,
    normalised_title,
)
from fakt_ai.models import Claim, Source, SourceKind

PUBMED_XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID>9500320</PMID>
<Article><Journal><JournalIssue><PubDate><Year>1998</Year></PubDate></JournalIssue>
<Title>Lancet</Title></Journal>
<ArticleTitle>Ileal-lymphoid-nodular <i>hyperplasia</i>.</ArticleTitle>
<Abstract><AbstractText Label="METHODS">12 children.</AbstractText></Abstract>
<AuthorList><Author><LastName>Wakefield</LastName><ForeName>A J</ForeName></Author></AuthorList>
<PublicationTypeList><PublicationType>Journal Article</PublicationType>
<PublicationType>Retracted Publication</PublicationType></PublicationTypeList>
</Article></MedlineCitation>
<PubmedData><ArticleIdList><ArticleId IdType="doi">10.1016/S0140-6736(97)11096-0</ArticleId>
<ArticleId IdType="pmc">PMC123</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle></PubmedArticleSet>"""

JATS = """<article><front><article-meta><abstract><p>Short abstract.</p></abstract>
<funding-group><funding-statement>Funded by Acme.</funding-statement></funding-group>
</article-meta></front>
<body><sec><title>Methods</title><p>We counted.</p></sec></body>
<back><fn-group><fn><p>No competing interests.</p></fn></fn-group>
<ref-list><ref>Smith 2000</ref></ref-list></back></article>"""


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)


def _source(title="A sufficiently long study title here", **fields):
    return Source(title=title, url=fields.pop("url", f"https://x.org/{title}"), **fields)


def test_rebuild_abstract_orders_words():
    index = {"world": [1, 3], "hello": [0], "again": [2]}
    assert scholarly.rebuild_abstract(index) == "hello world again world"
    assert scholarly.rebuild_abstract(None) == ""


def test_pubmed_article_maps_retraction_doi_and_pmcid():
    source = scholarly.parse_pubmed(PUBMED_XML, "pubmed:support")[0]
    assert source.title == "Ileal-lymphoid-nodular hyperplasia."
    assert source.retracted
    assert source.doi == "10.1016/s0140-6736(97)11096-0"
    assert source.pmcid == "PMC123"
    assert source.year == 1998
    assert source.abstract == "METHODS: 12 children."
    assert source.authors == ["A J Wakefield"]


def test_openalex_work_mapping():
    work = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1/ABC",
        "title": "RETRACTED: Some paper",
        "publication_year": 2001,
        "type": "article",
        "cited_by_count": 9,
        "is_retracted": False,
        "ids": {"pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/456"},
        "open_access": {"oa_url": "https://x.org/a.pdf"},
        "primary_location": {"landing_page_url": "https://x.org/a"},
        "abstract_inverted_index": {"hi": [0]},
    }
    source = scholarly._from_openalex(work, "openalex:review")
    assert source.doi == "10.1/abc"
    assert source.pmcid == "PMC456"
    assert source.retracted
    assert source.abstract == "hi"
    assert source.open_access_url == "https://x.org/a.pdf"


def test_semantic_scholar_rate_limit_is_tolerated(mocker):
    response = httpx.Response(429, request=httpx.Request("GET", "https://s2"))
    error = httpx.HTTPStatusError("busy", request=response.request, response=response)
    mocker.patch.object(scholarly, "get_json", side_effect=error)
    assert scholarly.search_semantic_scholar("q", "s2") == []


WAKEFIELD = {
    "DOI": "10.1016/S0140-6736(97)11096-0",
    "title": ["RETRACTED: Ileal-lymphoid-nodular hyperplasia"],
    "container-title": ["The Lancet"],
    "author": [{"family": "Wakefield", "given": "A"}],
    "issued": {"date-parts": [[1998, 2]]},
    "is-referenced-by-count": 2046,
    "updated-by": [{"type": "retraction"}],
}
BYSTANDER = {
    "DOI": "10.1/other",
    "title": ["Wakefield and the Lancet in context"],
    "container-title": ["Letters"],
    "author": [{"family": "Jones"}],
    "issued": {"date-parts": [[1998]]},
    "is-referenced-by-count": 1,
}


def test_cited_work_resolves_to_the_most_cited_match_and_stays_flagged(mocker):
    def answer(namespace, url, params=None, **_):
        if namespace == "crossref":
            return {"message": {"items": [BYSTANDER, WAKEFIELD]}}
        if namespace == "openalex":
            raise httpx.ConnectError("down")
        return {"resultList": {"result": []}}

    mocker.patch.object(scholarly, "get_json", side_effect=answer)
    [source] = scholarly.resolve_cited_work("Wakefield et al., Lancet 1998")
    assert source.found_by == "cited"
    assert source.doi == "10.1016/s0140-6736(97)11096-0"
    assert source.retracted


def test_cited_work_with_no_credible_match_is_logged_not_invented(mocker):
    unrelated = {"title": ["Bird migration"], "issued": {"date-parts": [[1990]]}}
    answer = {"message": {"items": [unrelated]}}
    mocker.patch.object(scholarly, "get_json", return_value=answer)
    assert scholarly.resolve_cited_work("Smith et al. vaccine study") == []


def test_cited_work_tries_the_likely_reference_first_then_the_speakers_words(mocker):
    def answer(namespace, url, params=None, **_):
        if namespace != "crossref":
            return {}
        found = "Ileal" in params["query.bibliographic"]
        return {"message": {"items": [WAKEFIELD] if found else []}}

    mocker.patch.object(scholarly, "get_json", side_effect=answer)
    likely = "Wakefield 1998 Ileal-lymphoid-nodular hyperplasia, The Lancet"
    assert scholarly.resolve_cited_work("the Lancet paper", likely)[0].found_by == "cited"
    assert scholarly.resolve_cited_work("the Lancet paper") == []


def test_cited_work_rejects_wrong_year():
    assert scholarly._best_cited_match("Wakefield Lancet 2015", [WAKEFIELD]) is None


def test_editorial_updates_mark_retraction_and_concern(mocker):
    updates = [{"type": "Retraction"}, {"type": "expression_of_concern"}]
    answer = {"message": {"updated-by": updates}}
    mocker.patch.object(scholarly, "get_json", return_value=answer)
    source = _source(doi="10.1/x")
    scholarly.note_editorial_updates(source)
    assert source.retracted
    assert "Expression of Concern" in source.publication_types


@pytest.mark.parametrize(
    ("url", "reported", "expected"),
    [
        ("https://www.snopes.com/fact-check/x", SourceKind.WEB, SourceKind.FACT_CHECK),
        ("https://www.reuters.com/fact-check/x", SourceKind.WEB, SourceKind.FACT_CHECK),
        ("https://www.reuters.com/world/x", SourceKind.WEB, SourceKind.NEWS),
        ("https://www.cdc.gov/vaccines", SourceKind.WEB, SourceKind.OFFICIAL),
        ("https://www.gov.uk/guidance", SourceKind.WEB, SourceKind.OFFICIAL),
        ("https://www.who.int/news", SourceKind.WEB, SourceKind.OFFICIAL),
        ("https://www.health.gov.au/a", SourceKind.WEB, SourceKind.OFFICIAL),
        ("https://data.giss.nasa.gov/gistemp/", SourceKind.WEB, SourceKind.DATASET),
        ("https://pubmed.ncbi.nlm.nih.gov/1/", SourceKind.WEB, SourceKind.PAPER),
        ("https://ntrs.nasa.gov/citations/1", SourceKind.PAPER, SourceKind.PAPER),
        ("https://someblog.net/post", SourceKind.WEB, SourceKind.WEB),
        ("https://notsnopes.com/a", SourceKind.WEB, SourceKind.WEB),
    ],
)
def test_classify(url, reported, expected):
    assert classify(_source(url=url, kind=reported)) is expected


def test_news_search_results_count_as_news():
    assert classify(_source(url="https://local.example/a", found_by="exa:news")) is (
        SourceKind.NEWS
    )


def test_doi_in_url_drops_page_suffix():
    assert doi_in_url("https://x.org/doi/10.1234/AbC.5/full?x=1") == "10.1234/abc.5"
    assert doi_in_url("https://x.org/a") is None


def test_merge_collapses_doi_title_and_url_and_keeps_best_fields():
    first = _source(
        "Measles vaccine and autism cohort", doi="10.1/A", found_by="pubmed:support"
    )
    by_doi = _source(
        "Other title", doi="10.1/a", abstract="longer", citations=5, url="https://b/"
    )
    by_title = _source(
        "RETRACTED: Measles vaccine and autism, cohort.",
        retracted=True,
        found_by="openalex:challenge",
    )
    by_url = _source(url="https://b/?utm_source=x", full_text="words")
    distinct = _source("Measles vaccine and autism in adults", url="https://c/")
    merged = merge_duplicates([first, by_doi, by_title, by_url, distinct])
    assert len(merged) == 2
    keeper = merged[0]
    assert keeper.abstract == "longer"
    assert keeper.citations == 5
    assert keeper.retracted
    assert keeper.full_text == "words"
    assert keeper.found_by == "pubmed:support,openalex:challenge"


def test_merge_bridges_two_records_joined_by_a_third():
    by_doi = _source(
        "First distinct study title of many words", doi="10.1/a", url="https://a/"
    )
    by_url = _source("Second distinct study title of many words", url="https://b/")
    bridge = _source(
        "Third distinct study title of many words", doi="10.1/a", url="https://b/"
    )
    later = _source("Fourth distinct study title of many words", url="https://b/")
    assert len(merge_duplicates([by_doi, by_url, bridge, later])) == 1


def test_merge_keeps_cited_flag_and_gives_stable_ids():
    make = lambda found_by: _source(doi="10.1/a", found_by=found_by)
    [merged] = merge_duplicates([make("pubmed:support"), make("cited")])
    assert merged.found_by == "cited"
    [again] = merge_duplicates([make("cited")])
    assert again.id == merged.id


def test_normalised_title_ignores_retraction_banner_and_punctuation():
    assert normalised_title("RETRACTED: Foo, bar!") == normalised_title("foo bar")


def _candidate(i, **fields):
    return _source(f"Candidate number {i} about things", url=f"https://x.org/{i}", **fields)


def test_shortlist_protects_cited_reviews_sides_and_kinds(mocker):
    weak_cited = _candidate(100, found_by="cited")
    review = _candidate(101, publication_types=["Meta-Analysis"], found_by="pubmed:review")
    retracted_review = _candidate(102, publication_types=["Meta-Analysis"], retracted=True)
    dissent = _candidate(103, found_by="exa:support")
    fact_check = _candidate(104, kind=SourceKind.FACT_CHECK)
    junk = _candidate(105)
    papers = [_candidate(i, kind=SourceKind.PAPER) for i in range(40)]
    pool = [weak_cited, review, retracted_review, dissent, fact_check, junk, *papers]
    scores = {s.id: 0.9 for s in papers} | {
        review.id: 0.5,
        retracted_review.id: 0.95,
        dissent.id: 0.4,
        fact_check.id: 0.3,
        junk.id: 0.01,
    }
    mocker.patch.object(config, "secret", return_value="key")
    mocker.patch.object(rank, "_judge", side_effect=lambda _, s: scores[s.id])
    kept = rank.shortlist(Claim(text="c"), pool, limit=10)
    ids = {s.id for s in kept}
    assert len(kept) == 10
    assert {weak_cited.id, review.id, dissent.id, fact_check.id} <= ids
    assert junk.id not in ids
    assert kept[0] is weak_cited


def test_shortlist_keeps_every_cited_work_past_the_limit(mocker):
    mocker.patch.object(config, "secret", return_value=None)
    cited = [_candidate(i, found_by="cited") for i in range(4)]
    assert len(rank.shortlist(Claim(text="c"), cited, limit=2)) == 4


def test_ranking_without_a_jev_key_uses_shared_words(mocker):
    mocker.patch.object(config, "secret", return_value=None)
    judge = mocker.patch.object(rank.jev, "judge")
    on_topic = _source("Measles vaccine autism risk in children")
    off_topic = _source("Soil chemistry of river deltas in spring")
    claim = Claim(text="Measles vaccine causes autism")
    assert rank.shortlist(claim, [off_topic, on_topic]) == [on_topic]
    judge.assert_not_called()


def test_jev_failure_stops_further_calls_and_falls_back(mocker):
    mocker.patch.object(config, "secret", return_value="key")
    judge = mocker.patch.object(rank.jev, "judge", side_effect=RuntimeError("down"))
    mocker.patch.object(rank, "_WORKERS", 1)
    claim = Claim(text="Measles vaccine causes autism")
    kept = rank.shortlist(claim, [_source(f"Measles vaccine autism {i}") for i in range(5)])
    assert len(kept) == 5
    assert judge.call_count == 1


def test_fit_sections_keeps_funding_and_drops_references():
    sections = [
        ("Introduction", "i" * 30_000),
        ("Methods", "m" * 30_000),
        ("Results", "r" * 30_000),
        ("Funding", "paid by Acme"),
        ("References", "x" * 5_000),
    ]
    fitted = fulltext.fit_sections(sections, limit=40_000)
    assert len(fitted) < 41_000
    assert "paid by Acme" in fitted
    assert "x" * 100 not in fitted
    assert fitted.count("m") > fitted.count("i")
    assert fitted.index("Methods") < fitted.index("Funding")


def test_fit_plain_text_without_headings_keeps_both_ends():
    fitted = fulltext.fit_plain_text("start " + "w" * 100_000 + " funded by Acme", limit=1_000)
    assert fitted.startswith("start")
    assert fitted.endswith("funded by Acme")
    assert len(fitted) < 1_100


def test_fit_plain_text_finds_numbered_headings():
    text = "Title\n\n2. Methods\nmmm\n\nConflicts of interest:\nnone\n\nReferences\n1. A"
    fitted = fulltext.fit_plain_text(text)
    assert "mmm" in fitted
    assert "none" in fitted
    assert "1. A" not in fitted


def test_parse_jats_reads_front_body_and_back_without_references():
    fitted = fulltext.fit_sections(fulltext.parse_jats(JATS))
    assert "Short abstract." in fitted
    assert "Funded by Acme." in fitted
    assert "We counted." in fitted
    assert "No competing interests." in fitted
    assert "Smith 2000" not in fitted


def test_full_text_prefers_europe_pmc_then_open_access(mocker):
    xml = mocker.patch.object(fulltext, "_europe_pmc_text", return_value="xml text")
    page = mocker.patch.object(fulltext, "_page_text", return_value="p" * 3_000)
    paper = _source(pmcid="PMC1", open_access_url="https://x/a.pdf", kind=SourceKind.PAPER)
    fulltext.attach_full_text([paper])
    assert paper.full_text == "xml text"
    page.assert_not_called()
    xml.return_value = ""
    fulltext.attach_full_text([paper])
    assert paper.full_text == "p" * 3_000


def test_full_text_rejects_a_landing_page_and_tolerates_failures(mocker):
    mocker.patch.object(fulltext, "_page_text", return_value="sign in to read")
    paper = _source(open_access_url="https://x/a", kind=SourceKind.PAPER, abstract="a" * 500)
    down = _source(url="https://y/b", kind=SourceKind.NEWS)
    assert fulltext.attach_full_text([paper]) == 0
    assert paper.full_text == ""
    mocker.patch.object(fulltext, "_page_text", side_effect=OSError("boom"))
    assert fulltext.attach_full_text([down]) == 0


def test_full_text_keeps_exa_text_for_web_pages(mocker):
    page = mocker.patch.object(fulltext, "_page_text")
    web = _source(full_text="already here", kind=SourceKind.WEB)
    assert fulltext.attach_full_text([web]) == 1
    page.assert_not_called()


def test_pdf_download_is_read_and_oversize_refused(mocker):
    import pymupdf

    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Vaccine findings")
    mocker.patch.object(fulltext, "download", return_value=(document.tobytes(), "x/y"))
    assert "Vaccine findings" in fulltext._extract_download("https://x/a.pdf")


def test_download_refuses_oversize_bodies(mocker):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 2_000))
    mocker.patch.object(fetch, "_CLIENT", httpx.Client(transport=transport))
    with pytest.raises(fetch.DownloadRefused):
        fetch.download("https://x/big", max_bytes=1_000, max_seconds=5)
    with pytest.raises(fetch.DownloadRefused):
        fetch.download("file:///etc/passwd", max_bytes=1_000, max_seconds=5)


def test_transient_status_is_retried_and_cached(mocker):
    replies = [httpx.Response(429), httpx.Response(200, json={"ok": 1})]
    calls = []

    def serve(request):
        calls.append(request)
        return replies[len(calls) - 1]

    mocker.patch.object(fetch, "_CLIENT", httpx.Client(transport=httpx.MockTransport(serve)))
    mocker.patch.object(tenacity.nap.time, "sleep")
    assert fetch.get_json("ns", "https://x/a", {"q": 1}) == {"ok": 1}
    assert fetch.get_json("ns", "https://x/a", {"q": 1}) == {"ok": 1}
    assert len(calls) == 2


def test_contact_email_goes_in_mailto_only_when_set(monkeypatch):
    monkeypatch.delenv("FAKT_CONTACT_EMAIL", raising=False)
    assert fetch.contact_params() == {}
    monkeypatch.setenv("FAKT_CONTACT_EMAIL", "someone@example.org")
    assert fetch.contact_params() == {"mailto": "someone@example.org"}


def test_exa_charges_once_per_search_and_keeps_key_in_a_header(mocker):
    mocker.patch.object(config, "secret", return_value="secret-key")
    charge = mocker.spy(ledger, "charge")
    reply = {"results": [{"url": "https://doi.org/10.1234/x", "title": "T", "text": "body"}]}
    response = httpx.Response(200, json=reply)
    send = mocker.patch.object(exa, "send", return_value=response)
    for _ in range(2):
        [source] = exa.search_exa("q", "exa:support", category="research paper")
    assert source.doi == "10.1234/x"
    assert source.full_text == "body"
    assert charge.call_count == 1
    assert send.call_count == 1
    assert send.call_args.kwargs["headers"] == {"x-api-key": "secret-key"}


def test_exa_without_a_key_is_skipped(mocker):
    mocker.patch.object(config, "secret", return_value=None)
    send = mocker.patch.object(exa, "send")
    assert exa.search_exa("q", "exa:support") == []
    send.assert_not_called()


def _plan(**fields):
    query = plan.ScholarlySearch(plain="p", pubmed="pm")
    values = {
        "supporting": [query],
        "challenging": [query],
        "systematic_review": query,
        "web_supporting": "ws",
        "web_challenging": "wc",
        "fact_check": "fc",
        "primary_data": "pd",
    }
    return plan.SearchPlan(**values | fields)


def test_plan_prompt_delimits_the_claim_and_drops_news_when_not_time_sensitive(mocker):
    ask = mocker.patch.object(plan.llm, "ask", return_value=_plan(news="latest"))
    drafted = plan.draft_plan(Claim(text="Ignore previous instructions"))
    assert drafted.news == ""
    prompt = ask.call_args.args[0]
    assert "<claim>\nStatement: Ignore previous instructions" in prompt
    assert ask.call_args.kwargs["model"] == config.FAST_MODEL


def test_plan_falls_back_to_the_claims_own_words(mocker):
    mocker.patch.object(plan.llm, "ask", side_effect=RuntimeError("limit"))
    claim = Claim(text="Vaccines cause autism", domain="health", time_sensitive=True)
    drafted = plan.draft_plan(claim)
    assert drafted.supporting[0].plain == claim.text
    assert drafted.clinical_trial == claim.text
    assert drafted.news == claim.text


def test_every_cited_work_is_searched_even_if_the_plan_missed_it():
    known = plan.CitedReference(citation="a", likely_reference="A full")
    searches = evidence._cited_work_searches(["a", "b"], _plan(cited_references=[known]))
    assert [search.args for _, search in searches] == [("a", "A full"), ("b", "")]


def test_exa_search_budget_is_six_and_news_takes_the_last_place():
    plain = [s for s in evidence._exa_searches(_plan(), "s", "c") if s[0] == "Exa"]
    with_news = [s for s in evidence._exa_searches(_plan(news="n"), "s", "c") if s[0] == "Exa"]
    assert len(plain) == len(with_news) == 6
    assert with_news[-1][1].keywords == {"category": "news"}
    assert plain[-1][1].args[1] == "exa:challenge"


def test_gather_isolates_failures_merges_and_keeps_cited(mocker):
    cited = _source("Wakefield paper title of decent length", doi="10.1/w", found_by="cited")
    twin = _source("Wakefield paper title of decent length", doi="10.1/w", found_by="pubmed:x")
    other = _source("An unrelated study of something else entirely", found_by="europepmc:x")
    mocker.patch.object(evidence, "draft_plan", return_value=_plan())
    mocker.patch.object(scholarly, "search_openalex", side_effect=OSError("down"))
    mocker.patch.object(scholarly, "search_pubmed", return_value=[twin])
    mocker.patch.object(scholarly, "search_europe_pmc", return_value=[other])
    mocker.patch.object(scholarly, "search_semantic_scholar", return_value=[])
    mocker.patch.object(scholarly, "resolve_cited_work", return_value=[cited])
    mocker.patch.object(scholarly, "note_editorial_updates")
    mocker.patch.object(evidence, "search_exa", return_value=[])
    mocker.patch.object(evidence, "attach_full_text", return_value=0)
    mocker.patch.object(rank, "_judge", return_value=0.5)
    mocker.patch.object(config, "secret", return_value="key")
    events = []
    claim = Claim(text="MMR causes autism", cited_works=["Wakefield 1998"])
    sources = evidence.gather(claim, events.append)
    assert [s.found_by for s in sources] == ["cited", "europepmc:x"]
    assert sources[0].kind is SourceKind.PAPER
    assert any("Found" in event for event in events)
