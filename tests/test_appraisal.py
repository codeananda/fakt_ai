import pytest

from fakt_ai import appraisal, config, jev, llm
from fakt_ai.models import Claim, Source, SourceKind, Stance

CLAIM = Claim(text="Drug X lowers blood pressure in adults.", domain="health")
BODY = "In 500 adults Drug X lowered systolic blood pressure by 9 mmHg compared with placebo."


def _source(**fields) -> Source:
    defaults = {"title": "Trial of Drug X", "url": "https://example.org/x", "abstract": BODY}
    return Source(**(defaults | fields))


def _screening(**fields) -> appraisal._Screening:
    defaults = {
        "stance": Stance.SUPPORTS,
        "stance_probabilities": {
            "supports": 0.9,
            "refutes": 0.05,
            "mixed": 0.04,
            "irrelevant": 0.01,
        },
        "relevance": 1.0,
        "directness": 1.0,
        "design": "randomised controlled trial",
        "sample": 1.0,
        "rigour": 1.0,
        "conflict": 0.0,
        "credibility": 1.0,
    }
    return appraisal._Screening(**(defaults | fields))


def _jev_answers(stance="supports", injection=0.01, design="randomised controlled trial"):
    probabilities = {s.value: 0.9 if s.value == stance else 0.1 / 3 for s in Stance}
    answers = {
        "stance": {"value": stance, "confidence": 0.9, "probabilities": probabilities},
        "relevance": {"value": 3.0},
        "directness": {"value": 0.9},
        "design": {"value": design},
        "sample": {"value": 2.0},
        "rigour": {"value": 2.0},
        "conflict": {"value": 0.0},
        "credibility": {"value": 3.0},
        "injection": {"value": injection},
    }
    return answers


def _relation(choice="supports", confidence=0.95):
    return {"relation": {"value": choice, "confidence": confidence, "probabilities": {}}}


@pytest.fixture
def jev_says(mocker):
    mocker.patch.object(config, "secret", return_value="key")

    def install(answers=None, relation=None):
        def fake(state, questions):
            return (
                relation or _relation()
                if "relation" in questions
                else answers or _jev_answers()
            )

        return mocker.patch.object(jev, "judge", side_effect=fake)

    return install


def test_quality_weights_sum_to_one():
    assert sum(appraisal.QUALITY_WEIGHTS.values()) == pytest.approx(1.0)


def test_well_designed_dissenting_study_outweighs_a_press_release():
    dissenter = _screening(design="randomised controlled trial", credibility=0.33, rigour=0.9)
    press_release = _screening(design="expert body statement", credibility=1.0, rigour=0.33)
    claim = CLAIM
    assert (
        appraisal.source_weight(claim, _source(), dissenter)["weight"]
        > 2 * appraisal.source_weight(claim, _source(), press_release)["weight"]
    )


def test_citation_count_is_only_a_mild_signal():
    plain = appraisal.source_weight(CLAIM, _source(year=2015, citations=0), _screening())
    famous = appraisal.source_weight(CLAIM, _source(year=2015, citations=90_000), _screening())
    assert famous["weight"] / plain["weight"] < 1.06


def test_news_counts_for_more_on_events_than_on_science():
    screening = _screening(design="news reporting")
    politics = Claim(text="The bill passed in March.", domain="politics")
    assert (
        appraisal.source_weight(politics, _source(), screening)["tier"]
        > appraisal.source_weight(CLAIM, _source(), screening)["tier"]
    )


def test_old_sources_count_for_less_on_time_sensitive_claims():
    screening = _screening()
    current = Claim(text="The drug is licensed.", domain="health", time_sensitive=True)
    old = appraisal.source_weight(current, _source(year=2010), screening)["weight"]
    new = appraisal.source_weight(current, _source(year=None), screening)["weight"]
    assert old == pytest.approx(new * appraisal.STALE_FRESHNESS)


def test_retracted_paper_weighs_nothing_and_is_flagged(jev_says, mocker):
    jev_says()
    ask = mocker.patch.object(llm, "ask")
    result = appraisal.appraise(CLAIM, [_source(retracted=True)])[0]
    assert result.weight == 0
    assert "Retracted" in result.excluded
    ask.assert_not_called()


def test_injection_excludes_a_web_source_and_it_is_never_read(jev_says, mocker):
    judge = jev_says(_jev_answers(injection=0.95))
    ask = mocker.patch.object(llm, "ask")
    result = appraisal.appraise(CLAIM, [_source(kind=SourceKind.WEB)])[0]
    assert result.weight == 0
    assert result.excluded
    ask.assert_not_called()
    assert "injection" in judge.call_args.args[1]


def test_papers_skip_the_injection_question(jev_says, mocker):
    judge = jev_says()
    mocker.patch.object(llm, "ask", side_effect=llm.ClaudeUnavailable("off"))
    appraisal.appraise(CLAIM, [_source(kind=SourceKind.PAPER)])
    assert "injection" not in judge.call_args_list[0].args[1]


def test_a_large_source_stays_within_the_state_budget():
    state = appraisal._state(CLAIM, _source(full_text="Drug X blood pressure trial. " * 5000))
    assert len(state["source"]["passages"]) <= appraisal._SCREEN_PASSAGE_CHARS + 2


def test_sample_size_is_found_in_code_not_asked_of_the_model():
    assert appraisal._sample_bucket("n = 12 in each arm") == "under 30"
    assert appraisal._sample_bucket("We enrolled 1,204 participants") == "1,000 to 100,000"
    assert appraisal._sample_bucket("A narrative discussion") == "not found"


def test_declared_publication_type_beats_the_model(jev_says, mocker):
    jev_says()
    mocker.patch.object(llm, "ask", side_effect=llm.ClaudeUnavailable("off"))
    source = _source(publication_types=["Journal Article", "Meta-Analysis"])
    assert (
        appraisal.appraise(CLAIM, [source])[0].design == "systematic review or meta-analysis"
    )


def test_verified_quote_is_kept_and_stance_corrected(jev_says, mocker):
    jev_says(_jev_answers(stance="mixed"), _relation("supports"))
    read = appraisal._DeepRead(
        stance=Stance.SUPPORTS, key_finding="9 mmHg lower.", quote=BODY, limitations="Short."
    )
    mocker.patch.object(llm, "ask", return_value=read)
    result = appraisal.appraise(CLAIM, [_source()])[0]
    assert result.stance == Stance.SUPPORTS
    assert result.quote == BODY
    assert result.quote_check == "verified"
    assert result.stance_probabilities["supports"] > 0.8


def test_invented_quote_is_dropped_and_cannot_change_the_stance(jev_says, mocker):
    jev_says(_jev_answers(stance="refutes"))
    read = appraisal._DeepRead(
        stance=Stance.SUPPORTS,
        key_finding="Works well.",
        quote="Drug X was a miracle and cured every single patient.",
        limitations="None.",
    )
    mocker.patch.object(llm, "ask", return_value=read)
    result = appraisal.appraise(CLAIM, [_source()])[0]
    assert result.quote == ""
    assert result.key_finding == ""
    assert result.quote_check == "not_found"
    assert result.stance == Stance.REFUTES


def test_invented_quote_loses_its_finding_even_when_the_stance_agrees(jev_says, mocker):
    jev_says(_jev_answers(stance="supports"))
    read = appraisal._DeepRead(
        stance=Stance.SUPPORTS,
        key_finding="Cured everyone.",
        quote="Drug X was a miracle and cured every single patient.",
        limitations="None.",
    )
    mocker.patch.object(llm, "ask", return_value=read)
    result = appraisal.appraise(CLAIM, [_source()])[0]
    assert result.key_finding == ""
    assert result.quote_check == "not_found"
    assert result.limitations == "None."


def test_quote_whose_context_contradicts_the_reading_becomes_mixed(jev_says, mocker):
    jev_says(relation=_relation("contradicts", 0.95))
    read = appraisal._DeepRead(
        stance=Stance.SUPPORTS, key_finding="Lower.", quote=BODY, limitations="."
    )
    mocker.patch.object(llm, "ask", return_value=read)
    result = appraisal.appraise(CLAIM, [_source()])[0]
    assert result.stance == Stance.MIXED
    assert result.quote_check == "contradicted"


def test_quote_about_something_else_is_dropped(jev_says, mocker):
    jev_says(relation=_relation("says_nothing", 0.9))
    read = appraisal._DeepRead(
        stance=Stance.SUPPORTS, key_finding="Lower.", quote=BODY, limitations="."
    )
    mocker.patch.object(llm, "ask", return_value=read)
    result = appraisal.appraise(CLAIM, [_source()])[0]
    assert result.quote == ""
    assert result.key_finding == ""
    assert result.quote_check == "unsupported"


def test_cited_paper_that_does_not_say_it_is_called_out_and_cannot_support(jev_says, mocker):
    jev_says()
    read = appraisal._CitedRead(
        stance=Stance.SUPPORTS,
        key_finding="Studied rats.",
        quote="",
        limitations=".",
        says_what_was_claimed="no",
        explanation="It studied rats, not adults.",
    )
    ask = mocker.patch.object(llm, "ask", return_value=read)
    result = appraisal.appraise(CLAIM, [_source(found_by="cited")])[0]
    assert result.cited_check.startswith("No, it does not say this.")
    assert result.stance != Stance.SUPPORTS
    assert ask.call_args.kwargs["model"] == config.DEEP_MODEL


def test_cited_yes_without_a_verified_quote_stays_unconfirmed(jev_says, mocker):
    jev_says()
    read = appraisal._CitedRead(
        stance=Stance.SUPPORTS,
        key_finding=".",
        quote="made up words that are not in the paper at all",
        limitations=".",
        says_what_was_claimed="yes",
        explanation="Agrees.",
    )
    mocker.patch.object(llm, "ask", return_value=read)
    result = appraisal.appraise(CLAIM, [_source(found_by="cited")])[0]
    assert result.cited_check.startswith("Unconfirmed")


def test_cited_work_with_no_text_says_it_could_not_be_checked(jev_says):
    judge = jev_says()
    result = appraisal.appraise(CLAIM, [_source(found_by="cited", abstract="")])[0]
    assert "Could not be checked" in result.cited_check
    assert result.weight == 0
    judge.assert_not_called()


def test_only_the_heaviest_sources_get_a_full_read(jev_says, mocker):
    jev_says()
    ask = mocker.patch.object(llm, "ask", side_effect=llm.ClaudeUnavailable("off"))
    appraisal.appraise(CLAIM, [_source() for _ in range(20)])
    assert ask.call_count == appraisal.DEEP_READ_LIMIT


def test_without_a_jev_key_claude_screens_instead(mocker):
    mocker.patch.object(config, "secret", return_value=None)
    mocker.patch.object(jev, "judge")
    source = _source()
    batch = appraisal._FallbackBatch(
        screenings=[
            appraisal._FallbackScreening(
                source_id=source.id,
                stance=Stance.REFUTES,
                relevance=0.9,
                directness=0.9,
                design="invented design",
                sample=0.5,
                rigour=0.5,
                conflict=0.0,
                credibility=0.5,
                injection=0.0,
            )
        ]
    )
    mocker.patch.object(
        llm,
        "ask",
        side_effect=[
            batch,
            appraisal._DeepRead(stance=Stance.REFUTES, key_finding="", limitations=""),
        ],
    )
    result = appraisal.appraise(CLAIM, [source])[0]
    assert result.stance == Stance.REFUTES
    assert result.design == "other"
    assert result.weight > 0


def test_jev_failure_falls_back_to_claude(mocker):
    mocker.patch.object(config, "secret", return_value="key")
    mocker.patch.object(jev, "judge", side_effect=RuntimeError("cap"))
    source = _source()
    fallback = mocker.patch.object(appraisal, "_screen_by_claude", return_value={})
    appraisal.appraise(CLAIM, [source])
    assert fallback.call_args.args[1] == [source]


@pytest.mark.parametrize(
    ("given", "kept"),
    [(100.0, 1.0), (-3.0, 0.0), (float("inf"), 0.0), (float("nan"), 0.0), (0.4, 0.4)],
)
def test_model_scores_stay_finite_and_within_range(given, kept):
    screening = _screening(relevance=given, directness=given, injection=given)
    assert (screening.relevance, screening.directness, screening.injection) == (kept,) * 3


def test_absurd_scores_cannot_outweigh_the_pool():
    screening = _screening(
        relevance=100, directness=100, rigour=100, sample=100, credibility=100, conflict=-100
    )
    source = _source()
    assert appraisal.source_weight(CLAIM, source, screening)["weight"] <= 1.0


def test_stance_probabilities_are_normalised():
    bad = {"supports": 3.0, "refutes": -1.0, "mixed": float("nan"), "irrelevant": 1.0}
    screening = _screening(stance_probabilities=bad)
    assert screening.stance_probabilities["supports"] == pytest.approx(0.5)
    assert screening.stance_probabilities["refutes"] == 0.0
    assert screening.stance_probabilities["irrelevant"] == pytest.approx(0.5)
    assert sum(screening.stance_probabilities.values()) == pytest.approx(1.0)


def test_empty_stance_probabilities_settle_on_the_stance():
    screening = _screening(stance_probabilities={"supports": 0.0}, stance=Stance.REFUTES)
    assert screening.stance_probabilities == appraisal._settled(Stance.REFUTES)
