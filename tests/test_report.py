from fakt_ai import jev, llm, report
from fakt_ai.models import (
    Appraisal,
    Capture,
    Claim,
    ClaimCheck,
    GlossaryEntry,
    PostProfile,
    Report,
    Source,
    Stance,
    Verdict,
    VerdictLabel,
)


def _check() -> ClaimCheck:
    claim = Claim(id="c1", text="Coffee causes cancer.", quote="coffee gives you cancer")
    source = Source(
        id="aaaaaaaa",
        title="Coffee [and] cancer | a review",
        url="https://example.org/a",
        year=2020,
        retracted=False,
    )
    pulled = Source(
        id="bbbbbbbb", title="Pulled paper", url="https://example.org/b", retracted=True
    )
    return ClaimCheck(
        claim=claim,
        sources=[source, pulled],
        appraisals=[
            Appraisal(
                source_id="aaaaaaaa",
                stance=Stance.REFUTES,
                weight=0.6,
                quality=0.7,
                design="systematic review or meta-analysis",
                key_finding="No link found.",
                quote="No link was found.",
                limitations="Observational.",
            ),
            Appraisal(
                source_id="bbbbbbbb",
                stance=Stance.SUPPORTS,
                excluded="Retracted: this paper has been withdrawn.",
            ),
        ],
        verdict=Verdict(
            label=VerdictLabel.MOSTLY_FALSE,
            probability_true=0.17,
            evidence_balance=0.2,
            summary="Probably not.",
            reasoning="A review found nothing [aaaaaaaa].",
        ),
    )


def _report(*checks) -> Report:
    return Report(
        capture=Capture(text="Coffee gives you cancer!!"),
        checks=list(checks),
        profile=PostProfile(
            signal_vs_hype=0.2, fact_vs_fiction=0.17, tactics=["false urgency"]
        ),
    )


def _plain(why="Big reviews found no link.") -> report._PlainSummary:
    return report._PlainSummary(
        bottom_line="Do not worry about your coffee.",
        post_in_a_sentence="It says coffee causes cancer.",
        claims=[
            report._PlainClaim(
                claim_id="c1",
                plain_claim="Coffee gives you cancer",
                why=why,
                good_evidence="Studies of studies find no link.",
            )
        ],
    )


def test_one_page_has_code_numbers_bottom_line_first_and_links(mocker):
    mocker.patch.object(llm, "ask", return_value=_plain())
    page = report.one_page(_report(_check()))
    assert page.startswith("# Bottom line\n\nDo not worry")
    assert "**Mostly false.** Chance the statement is true as worded: about 17%." in page
    assert "(https://example.org/a)" in page
    assert "retracted" in page
    assert "|" not in page.replace("\\|", "")


def test_generated_likelihoods_are_struck_out(mocker):
    mocker.patch.object(
        llm,
        "ask",
        return_value=_plain("Reviews found no link. This is 95% likely to be false."),
    )
    page = report.one_page(_report(_check()))
    assert "95%" not in page
    assert "Reviews found no link." in page


def test_study_figures_may_stay(mocker):
    mocker.patch.object(llm, "ask", return_value=_plain("Risk fell by 12% in the trial."))
    assert "12%" in report.one_page(_report(_check()))


def test_one_page_survives_the_writer_failing(mocker):
    mocker.patch.object(llm, "ask", side_effect=llm.ClaudeUnavailable("limit"))
    page = report.one_page(_report(_check()))
    assert "Probably not." in page
    assert "17%" in page


def test_failed_claim_is_shown_not_fatal(mocker):
    mocker.patch.object(llm, "ask", return_value=_plain())
    failed = ClaimCheck(claim=Claim(text="Second"), failure="search broke")
    page = report.one_page(_report(_check(), failed))
    assert "could not check this one" in page
    assert "could not be checked" in report.detail(_report(failed))


def test_nothing_to_check_needs_no_writer(mocker):
    ask = mocker.patch.object(llm, "ask")
    assert "no checkable" in report.one_page(_report())
    ask.assert_not_called()


def test_unread_speech_warns_even_when_no_claims_are_found():
    unheard = _report()
    unheard.capture.speech_unread = True
    assert "could not transcribe" in report.one_page(unheard)


def test_detail_links_citations_and_shows_table_quote_and_warnings():
    text = report.detail(_report(_check()))
    assert "](https://example.org/a)" in text
    assert "[aaaaaaaa]" not in text
    assert "| Source | Design |" in text
    assert "> No link was found." in text
    assert "has been retracted" in text
    assert "Method notes" in text


def test_citations_given_as_urls_become_links():
    linked = report.link_citations("Yes [https://example.org/a].", [])
    assert linked == "Yes [source](https://example.org/a)."


def test_hostile_titles_cannot_break_out_of_markdown():
    source = Source(
        id="s", title="x](javascript:alert(1)) <script>", url="javascript:alert(1)"
    )
    link = report._source_link(source)
    assert "](" not in link.replace("\\]", "")
    assert "\\<script" in link


def test_unfound_cited_work_is_reported():
    check = _check()
    check.claim.cited_works = ["Smith 2019 Nature paper"]
    assert "Cited but not found" in report.detail(_report(check))


def test_profile_uses_thresholds_and_computes_fact_vs_fiction(mocker):
    answers = {name: {"value": 0.1} for name in report._TACTICS}
    answers["selling"] = {"value": 0.9}
    answers["signal"] = {"value": 1.5}
    mocker.patch.object(jev, "judge", return_value=answers)
    check = _check()
    found = report.profile(Capture(text="Buy now"), [check, check])
    assert found.signal_vs_hype == 0.5
    assert found.tactics == ["selling something"]
    assert found.fact_vs_fiction == 0.17


def test_misleading_claims_count_against_fact_vs_fiction(mocker):
    check = _check()
    check.verdict.label = VerdictLabel.MISLEADING
    check.verdict.probability_true = 0.85
    assert report._fact_vs_fiction([check]) == 0.5


def test_profile_falls_back_to_claude_without_jev(mocker):
    mocker.patch.object(jev, "judge", side_effect=RuntimeError("no key"))
    mocker.patch.object(
        llm,
        "ask",
        return_value=report._PostReading(
            signal_vs_hype=0.4, tactics=["false urgency", "made up"]
        ),
    )
    found = report.profile(Capture(text="Act now"), [])
    assert found.tactics == ["false urgency"]


def _glossary(*entries: tuple[str, str]) -> report._Glossary:
    return report._Glossary(
        entries=[GlossaryEntry(term=term, meaning=meaning) for term, meaning in entries]
    )


def test_glossary_keeps_only_terms_the_reader_meets(mocker):
    ask = mocker.patch.object(
        llm,
        "ask",
        return_value=_glossary(
            ("Systematic  review", " A study that pools every good study on a question. "),
            ("systematic review", "A duplicate."),
            ("odds ratio", "Not written anywhere in this report."),
            ("meta-analysis", ""),
        ),
    )
    found = report.glossary(_report(_check()))
    assert [(entry.term, entry.meaning) for entry in found] == [
        ("Systematic review", "A study that pools every good study on a question.")
    ]
    assert "systematic review or meta-analysis: No link found." in ask.call_args.args[0]


def test_glossary_needs_no_writer_without_a_verdict(mocker):
    ask = mocker.patch.object(llm, "ask")
    failed = ClaimCheck(claim=Claim(text="Second"), failure="search broke")
    assert report.glossary(_report(failed)) == []
    ask.assert_not_called()


def test_glossary_survives_the_writer_failing(mocker):
    mocker.patch.object(llm, "ask", side_effect=llm.ClaudeUnavailable("limit"))
    assert report.glossary(_report(_check())) == []


def test_reports_stored_without_a_glossary_still_load():
    stored = _report(_check()).model_dump(mode="json")
    del stored["glossary"]
    assert Report.model_validate(stored).glossary == []
