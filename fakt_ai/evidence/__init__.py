"""Find the evidence on a claim: papers first, then data, official pages and journalism."""

from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from loguru import logger

from fakt_ai import config
from fakt_ai.evidence import scholarly
from fakt_ai.evidence.classify import classify, merge_duplicates
from fakt_ai.evidence.exa import search_exa
from fakt_ai.evidence.fulltext import attach_full_text
from fakt_ai.evidence.google import search_google
from fakt_ai.evidence.plan import SearchPlan, draft_plan
from fakt_ai.evidence.rank import shortlist
from fakt_ai.models import Claim, Source, SourceKind

_SEARCH_WORKERS = 12
_UPDATE_WORKERS = 5
_MAX_SCHOLARLY_PER_SIDE = 3

Search = tuple[str, Callable[[], list[Source]]]


def noop(_message: str) -> None:
    """Ignore a progress message."""


def gather(
    claim: Claim, on_event: Callable[[str], None] = noop, *, max_sources: int = 30
) -> list[Source]:
    """
    Search every source for evidence on a claim, for and against.

    Parameters
    ----------
    claim : Claim
        The claim to research.
    on_event : Callable[[str], None]
        Receives progress messages.
    max_sources : int
        Size of the shortlist, not counting cited works beyond it.

    Returns
    -------
    list[Source]
        Deduplicated sources, most relevant first, with abstracts and open-access full text.
    """
    on_event("Planning balanced searches")
    plan = draft_plan(claim)
    searches = _planned_searches(plan) + _cited_work_searches(claim.cited_works, plan)
    on_event(f"Running {len(searches)} searches")
    found = _run_searches(searches, on_event)
    candidates = _classified(merge_duplicates(found))
    on_event(f"Ranking {len(candidates)} unique sources")
    kept = shortlist(claim, candidates, limit=max_sources)
    _note_all_editorial_updates(kept)
    on_event(f"Fetching full text for {len(kept)} sources")
    with_text = attach_full_text(kept)
    on_event(f"Kept {len(kept)} sources, {with_text} with full text: {_kind_counts(kept)}")
    return kept


def search_more(
    claim: Claim, queries: Iterable[str], known: Iterable[Source], *, max_sources: int = 8
) -> list[Source]:
    """
    Run follow-up searches for what the first round left open.

    Parameters
    ----------
    claim : Claim
        The claim.
    queries : Iterable[str]
        Web searches to run.
    known : Iterable[Source]
        Sources already held, which are left out of the results.
    max_sources : int
        Most new sources to return.

    Returns
    -------
    list[Source]
        New sources, most relevant first, with text attached.
    """
    searches: list[Search] = []
    for query in queries:
        searches += [
            ("Exa", partial(search_exa, query, "exa:follow_up")),
            ("Google", partial(search_google, query, "google:follow_up")),
        ]
    held = {s.url for s in known}
    fresh = [s for s in merge_duplicates(_run_searches(searches, noop)) if s.url not in held]
    candidates = _classified(fresh)
    kept = shortlist(claim, candidates, limit=max_sources)
    attach_full_text(kept)
    return kept


def _classified(sources: Iterable[Source]) -> list[Source]:
    candidates = list(sources)
    for source in candidates:
        source.kind = classify(source)
    if config.HIDE_FACT_CHECKS:
        return [s for s in candidates if s.kind is not SourceKind.FACT_CHECK]
    return candidates


def _planned_searches(plan: SearchPlan) -> list[Search]:
    searches: list[Search] = []
    sides = {
        "support": plan.supporting[:_MAX_SCHOLARLY_PER_SIDE],
        "challenge": plan.challenging[:_MAX_SCHOLARLY_PER_SIDE],
        "review": [plan.systematic_review],
    }
    for side, queries in sides.items():
        for query in queries:
            searches += [
                (
                    "OpenAlex",
                    partial(scholarly.search_openalex, query.plain, f"openalex:{side}"),
                ),
                (
                    "PubMed",
                    partial(
                        scholarly.search_pubmed,
                        query.pubmed,
                        f"pubmed:{side}",
                        reviews_only=side == "review",
                    ),
                ),
                (
                    "Europe PMC",
                    partial(scholarly.search_europe_pmc, query.plain, f"europepmc:{side}"),
                ),
            ]
    support = next(iter(plan.supporting), plan.systematic_review).plain
    challenge = next(iter(plan.challenging), plan.systematic_review).plain
    searches.append(
        (
            "Semantic Scholar",
            partial(scholarly.search_semantic_scholar, support, "semanticscholar:support"),
        )
    )
    if plan.clinical_trial:
        searches.append(
            (
                "ClinicalTrials.gov",
                partial(
                    scholarly.search_clinical_trials,
                    plan.clinical_trial,
                    "clinicaltrials:trial",
                ),
            )
        )
    return searches + _exa_searches(plan, support, challenge)


def _exa_searches(plan: SearchPlan, support: str, challenge: str) -> list[Search]:
    """Spend at most six paid searches, with news taking the last place when it applies."""
    last = (
        partial(search_exa, plan.news, "exa:news", category="news")
        if plan.news
        else partial(search_exa, challenge, "exa:challenge", category="research paper")
    )
    exa = [
        ("Exa", search)
        for search in (
            partial(search_exa, support, "exa:support", category="research paper"),
            partial(search_exa, plan.web_supporting, "exa:support"),
            partial(search_exa, plan.web_challenging, "exa:challenge"),
            partial(search_exa, plan.fact_check, "exa:fact_check", fact_checkers_only=True),
            partial(search_exa, plan.primary_data, "exa:primary"),
            last,
        )
    ]
    google_queries = [plan.web_supporting, plan.web_challenging, plan.news]
    google = [
        ("Google", partial(search_google, query, "google"))
        for query in google_queries
        if query
    ]
    return exa + google


def _cited_work_searches(cited_works: Iterable[str], plan: SearchPlan) -> list[Search]:
    """Resolve every cited work, whether or not the plan recognised it."""
    recognised = {r.citation: r.likely_reference for r in plan.cited_references}
    return [
        ("Cited works", partial(scholarly.resolve_cited_work, c, recognised.get(c, "")))
        for c in cited_works
    ]


def _run_searches(searches: Iterable[Search], on_event: Callable[[str], None]) -> list[Source]:
    """Run searches concurrently, so that one failing source costs only its own results."""
    found: list[Source] = []
    tally: Counter[str] = Counter()
    with ThreadPoolExecutor(_SEARCH_WORKERS) as pool:
        running = {pool.submit(search): name for name, search in searches}
        for future in as_completed(running):
            name = running[future]
            try:
                results = future.result()
            except Exception as e:  # noqa: BLE001 - one source failing must not stop the rest
                logger.warning("{} search failed: {}", name, e)
                continue
            found += results
            tally[name] += len(results)
    on_event("Found " + ", ".join(f"{name}: {n}" for name, n in tally.most_common()))
    return found


def _note_all_editorial_updates(sources: Iterable[Source]) -> None:
    papers = [s for s in sources if s.doi and s.kind is SourceKind.PAPER and not s.retracted]

    def check(source: Source) -> None:
        try:
            scholarly.note_editorial_updates(source)
        except Exception as e:  # noqa: BLE001 - an optional enrichment
            logger.info("no Crossref update record for {}: {}", source.doi, e)

    with ThreadPoolExecutor(_UPDATE_WORKERS) as pool:
        list(pool.map(check, papers))


def _kind_counts(sources: Iterable[Source]) -> str:
    counts = Counter(source.kind.value for source in sources)
    return ", ".join(f"{kind}: {n}" for kind, n in counts.most_common())
