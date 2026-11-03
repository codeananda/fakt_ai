import fire
import pytest

from fakt_ai import (
    appraisal,
    capture,
    cli,
    evidence,
    ledger,
    llm,
    pipeline,
    report,
    store,
    verdict,
)
from fakt_ai import claims as claims_module
from fakt_ai.models import (
    Appraisal,
    Capture,
    Claim,
    ClaimCheck,
    Report,
    Source,
    Stance,
    Verdict,
    VerdictLabel,
)

SOURCE = Source(
    id="aaaaaaaa", title="A study", url="https://example.org/a", abstract="Finding."
)
VERDICT = Verdict(
    label=VerdictLabel.TRUE, probability_true=0.95, summary="Yes.", reasoning="Yes."
)


@pytest.fixture
def stages(mocker):
    mocker.patch.object(capture, "capture", return_value=Capture(text="A post."))
    mocker.patch.object(
        pipeline, "extract_claims", return_value=[Claim(text="One"), Claim(text="Two")]
    )
    mocker.patch.object(evidence, "gather", return_value=[SOURCE])
    mocker.patch.object(
        appraisal,
        "appraise",
        return_value=[Appraisal(source_id=SOURCE.id, stance=Stance.SUPPORTS, weight=0.5)],
    )
    mocker.patch.object(verdict, "judge", return_value=VERDICT)
    mocker.patch.object(report, "profile", return_value=Report(capture=Capture()).profile)
    mocker.patch.object(report, "one_page", return_value="page")
    mocker.patch.object(report, "detail", return_value="detail")


def test_run_reports_each_stage_and_saves(stages):
    events = []
    result = pipeline.run(text="A post.", on_event=events.append)
    assert [c.verdict for c in result.checks] == [VERDICT, VERDICT]
    assert {"capture", "claims", "evidence", "appraisal", "verdict", "report", "done"} <= {
        e["stage"] for e in events
    }
    assert store.load(result.id).one_page == "page"
    assert result.created


def test_one_failing_claim_does_not_kill_the_report(stages, mocker):
    def flaky(claim, sources, appraisals):
        if claim.text == "One":
            raise RuntimeError("boom")
        return VERDICT

    mocker.patch.object(verdict, "judge", side_effect=flaky)
    result = pipeline.run(text="A post.")
    failed, finished = result.checks
    assert failed.failure == "boom"
    assert failed.verdict is None
    assert failed.sources == [SOURCE]
    assert finished.verdict == VERDICT


def test_a_broken_progress_listener_is_harmless(stages):
    def listener(event):
        raise ValueError("gone")

    assert pipeline.run(text="A post.", on_event=listener).checks


def test_spending_cap_at_capture_falls_back_to_supplied_text(stages, mocker):
    fetch = mocker.patch.object(
        capture,
        "capture",
        side_effect=[ledger.SpendCapReached("cap"), Capture(text="Pasted.")],
    )
    pipeline.run(url="https://x.org", text="Pasted.")
    assert fetch.call_args.kwargs == {"text": "Pasted.", "image": None}


def test_spending_cap_at_capture_with_only_a_link_is_an_error(stages, mocker):
    mocker.patch.object(capture, "capture", side_effect=ledger.SpendCapReached("cap"))
    with pytest.raises(ledger.SpendCapReached):
        pipeline.run(url="https://x.org")


def test_spending_cap_during_evidence_marks_only_that_claim(stages, mocker):
    mocker.patch.object(evidence, "gather", side_effect=ledger.SpendCapReached("cap"))
    result = pipeline.run(text="A post.")
    assert all("cap" in c.failure for c in result.checks)


def _saved_report() -> Report:
    check = pipeline.ClaimCheck(claim=Claim(text="One"), sources=[SOURCE])
    return Report(capture=Capture(text="x"), checks=[check], one_page="page", detail="detail")


def test_follow_up_links_citations(mocker):
    mocker.patch.object(llm, "ask", return_value=pipeline._Reply(answer="It does [aaaaaaaa]."))
    answer = pipeline.follow_up(_saved_report(), "Does it?")
    assert answer == "It does [A study](https://example.org/a)."


def test_follow_up_is_told_the_ids_it_can_cite(mocker):
    ask = mocker.patch.object(llm, "ask", return_value=pipeline._Reply(answer="Yes."))
    pipeline.follow_up(_saved_report(), "Does it?")
    assert "[aaaaaaaa] A study" in ask.call_args.args[0]


def test_follow_up_on_a_source_reads_its_text(mocker):
    ask = mocker.patch.object(llm, "ask", return_value=pipeline._Reply(answer="Yes."))
    pipeline.follow_up(_saved_report(), "What did it find?", source_id="aaaaaaaa")
    assert "Finding." in ask.call_args.args[0]


def test_follow_up_searches_when_the_report_cannot_answer(mocker):
    extra = Source(id="cccccccc", title="New", url="https://example.org/c", abstract="More.")
    gather = mocker.patch.object(evidence, "gather", return_value=[extra])
    ask = mocker.patch.object(
        llm,
        "ask",
        side_effect=[
            pipeline._Reply(
                answer="Unknown.", needs_new_evidence=True, search_for="Tea helps"
            ),
            pipeline._Reply(answer="New work says so [cccccccc]."),
        ],
    )
    answer = pipeline.follow_up(_saved_report(), "What about tea?")
    assert gather.called
    assert "More." in ask.call_args.args[0]
    assert "(https://example.org/c)" in answer


def test_cli_tells_links_from_text():
    assert cli._split_target("https://x.org/a?b=1") == ("https://x.org/a?b=1", None)
    assert cli._split_target("Coffee is good https://x.org") == (
        None,
        "Coffee is good https://x.org",
    )


def test_cli_keeps_ids_that_look_like_numbers_as_text(mocker):
    shown = mocker.Mock(return_value="shown")
    fire.Fire({"show": shown}, command=cli._kept_as_text(["show", "1e234567"]))
    shown.assert_called_once_with("1e234567")


def test_cli_check_writes_markdown_and_prints_the_one_pager(
    mocker, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    run = mocker.patch.object(
        pipeline,
        "run",
        return_value=Report(capture=Capture(), one_page="PAGE", detail="DETAIL"),
    )
    cli.check("Coffee is good", max_claims=1)
    assert run.call_args.kwargs["text"] == "Coffee is good"
    assert "PAGE" in capsys.readouterr().out
    assert "DETAIL" in next((tmp_path / "reports").glob("*.md")).read_text()


def test_claims_module_is_wired_in():
    assert pipeline.extract_claims is claims_module.extract_claims


def _open_verdict(probability: float, questions: list[str]) -> Verdict:
    return Verdict(
        label=VerdictLabel.UNPROVEN,
        probability_true=probability,
        summary="s",
        reasoning="r",
        open_questions=questions,
    )


def test_research_follows_up_once_on_open_verdict(mocker):
    source = Source(title="a", url="https://a.example")
    extra = Source(title="b", url="https://b.example")
    mocker.patch.object(pipeline.evidence, "gather", return_value=[source])
    more = mocker.patch.object(pipeline.evidence, "search_more", return_value=[extra])
    mocker.patch.object(pipeline.appraisal_stage, "appraise", return_value=[])
    first, final = _open_verdict(0.5, ["who said it"]), _open_verdict(0.8, ["again"])
    judge = mocker.patch.object(pipeline.verdict, "judge", side_effect=[first, final])

    check = ClaimCheck(claim=Claim(text="c"))
    pipeline.research(check, lambda stage, message: None)

    assert check.verdict is final
    assert [s.url for s in check.sources] == [source.url, extra.url]
    assert judge.call_count == 2
    more.assert_called_once()


def test_research_skips_follow_up_when_settled(mocker):
    mocker.patch.object(pipeline.evidence, "gather", return_value=[])
    more = mocker.patch.object(pipeline.evidence, "search_more")
    mocker.patch.object(pipeline.appraisal_stage, "appraise", return_value=[])
    mocker.patch.object(pipeline.verdict, "judge", return_value=_open_verdict(0.03, ["more"]))

    pipeline.research(ClaimCheck(claim=Claim(text="c")), lambda stage, message: None)

    more.assert_not_called()
