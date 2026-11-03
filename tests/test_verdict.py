import pytest

from fakt_ai import llm, verdict
from fakt_ai.models import Appraisal, Claim, Source, Stance, VerdictLabel


def _appraisal(stance=Stance.SUPPORTS, weight=0.5, source_id="aaaaaaaa", **fields):
    share = {s.value: 0.9 if s == stance else 0.1 / 3 for s in Stance}
    return Appraisal(
        source_id=source_id, stance=stance, stance_probabilities=share, weight=weight, **fields
    )


def _ruling(**fields):
    defaults = {
        "label": VerdictLabel.MOSTLY_TRUE,
        "probability_true": 0.8,
        "consensus": "Agreed.",
        "summary": "Yes.",
        "reasoning": "See [aaaaaaaa] and also [deadbeef].",
        "strongest_for": "",
        "strongest_against": "",
        "what_would_change_this": "",
    }
    return verdict._Ruling(**(defaults | fields))


SOURCE = Source(id="aaaaaaaa", title="T", url="https://x.org")


def test_no_evidence_is_exactly_a_half():
    assert verdict.evidence_balance([]) == 0.5


def test_irrelevant_sources_do_not_move_the_balance():
    assert verdict.evidence_balance([_appraisal(Stance.IRRELEVANT, 1.0)]) == pytest.approx(
        0.5, abs=0.02
    )


def test_little_evidence_stays_near_a_half_and_more_moves_further():
    little = verdict.evidence_balance([_appraisal(Stance.REFUTES, 0.05)])
    much = verdict.evidence_balance([_appraisal(Stance.REFUTES, 0.8) for _ in range(4)])
    assert 0.4 < little < 0.5
    assert much < 0.2


def test_balance_treats_both_directions_alike():
    assert verdict.evidence_balance([_appraisal(Stance.SUPPORTS)]) == pytest.approx(
        1 - verdict.evidence_balance([_appraisal(Stance.REFUTES)])
    )


def test_even_evidence_is_a_half_however_much_there_is():
    evenly = [_appraisal(Stance.SUPPORTS, 1.0), _appraisal(Stance.REFUTES, 1.0)]
    assert verdict.evidence_balance(evenly) == pytest.approx(0.5)


def test_one_strong_study_beats_many_weak_mentions():
    pile = [_appraisal(Stance.SUPPORTS, 0.02) for _ in range(10)]
    assert verdict.evidence_balance(pile + [_appraisal(Stance.REFUTES, 0.6)]) < 0.4


def test_split_reports_a_sixty_forty_case():
    share_for, share_against = verdict.evidence_split(
        [_appraisal(Stance.SUPPORTS, 0.6), _appraisal(Stance.REFUTES, 0.4)]
    )
    assert share_for == pytest.approx(0.58, abs=0.03)
    assert share_for + share_against == pytest.approx(1.0)


def test_no_usable_evidence_is_unproven_without_spending_a_call(mocker):
    ask = mocker.patch.object(llm, "ask")
    found = verdict.judge(
        Claim(text="X"), [SOURCE], [_appraisal(weight=0.0, excluded="Retracted")]
    )
    assert found.label == VerdictLabel.UNPROVEN
    assert found.probability_true == 0.5
    ask.assert_not_called()


def test_label_and_probability_cannot_contradict_each_other():
    found = verdict._sane_verdict(
        _ruling(label=VerdictLabel.TRUE, probability_true=0.4), 0.5, 2.0, {}
    )
    assert found.probability_true == 0.90
    found = verdict._sane_verdict(
        _ruling(label=VerdictLabel.FALSE, probability_true=0.7), 0.5, 2.0, {}
    )
    assert found.probability_true == 0.10


def test_certainty_is_never_absolute():
    found = verdict._sane_verdict(
        _ruling(label=VerdictLabel.TRUE, probability_true=1.0), 0.9, 5.0, {}
    )
    assert found.probability_true == 0.99


def test_firm_label_on_thin_evidence_becomes_unproven():
    found = verdict._sane_verdict(
        _ruling(label=VerdictLabel.FALSE, probability_true=0.02), 0.45, 0.05, {}
    )
    assert found.label == VerdictLabel.UNPROVEN
    assert found.probability_true == 0.30


def test_misleading_can_be_literally_likely():
    found = verdict._sane_verdict(
        _ruling(label=VerdictLabel.MISLEADING, probability_true=0.85), 0.8, 2.0, {}
    )
    assert found.label == VerdictLabel.MISLEADING
    assert found.probability_true == 0.85


def test_citations_of_unseen_sources_are_removed():
    found = verdict._sane_verdict(_ruling(), 0.5, 2.0, {"aaaaaaaa": SOURCE})
    assert "[aaaaaaaa]" in found.reasoning
    assert "deadbeef" not in found.reasoning


def test_judge_runs_both_sides_then_the_deep_model(mocker):
    from fakt_ai import config

    seen = []

    def fake(prompt, schema, **kwargs):
        seen.append((schema, kwargs.get("model"), prompt))
        return (
            _ruling() if schema is verdict._Ruling else verdict._Case(argument=schema.__name__)
        )

    mocker.patch.object(llm, "ask", side_effect=fake)
    appraisals = [_appraisal(key_finding="Lower.")]
    found = verdict.judge(Claim(text="X"), [SOURCE], appraisals)
    assert [s[0] for s in seen].count(verdict._Case) == 2
    assert seen[-1][0] is verdict._Ruling
    assert seen[-1][1] == config.DEEP_MODEL
    assert found.evidence_balance == pytest.approx(verdict.evidence_balance(appraisals))
    assert "<case_for>" in seen[-1][2]


def test_unfound_cited_work_reaches_the_judge(mocker):
    block = verdict._evidence_block(
        Claim(text="X", cited_works=["The Lancet 1998 Wakefield paper"]),
        {"aaaaaaaa": SOURCE},
        [_appraisal()],
    )
    assert "could not be found" in block


def test_found_cited_work_is_not_reported_missing():
    cited = Source(
        id="aaaaaaaa", title="Wakefield et al, The Lancet", url="u", found_by="cited"
    )
    block = verdict._evidence_block(
        Claim(text="X", cited_works=["Wakefield Lancet paper"]),
        {"aaaaaaaa": cited},
        [_appraisal()],
    )
    assert "could not be found" not in block
