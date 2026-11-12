from fakt_ai import config, jev, llm, report
from fakt_ai.models import (
    Appraisal,
    Capture,
    Claim,
    ClaimCheck,
    ClaimPart,
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
    assert report.glossary(_report(_check())) is None


def test_a_malformed_glossary_reply_does_not_sink_the_report(mocker):
    mocker.patch.object(llm, "ask", side_effect=ValueError("not the schema"))
    assert report.glossary(_report(_check())) is None


def test_reports_stored_without_a_glossary_still_load():
    stored = _report(_check()).model_dump(mode="json")
    del stored["glossary"]
    assert Report.model_validate(stored).glossary is None


def test_glossary_drops_terms_the_page_cannot_mark():
    text = "The cohort study in the current review found no effect for most children, the authors said."
    entries = [
        GlossaryEntry(term="RR", meaning="Only inside the word current."),
        GlossaryEntry(term="or", meaning="Only inside the word for."),
        GlossaryEntry(term="Cohort Study", meaning="Case differs past the first letter."),
        GlossaryEntry(term="Cohort study", meaning="Research that follows a group over time."),
        GlossaryEntry(term="author", meaning="Matches the plural."),
    ]
    assert [entry.term for entry in report._terms_used_in(text, entries)] == [
        "Cohort study",
        "author",
    ]


def test_glossary_drops_entries_that_are_not_a_term_and_a_sentence():
    text = (
        "The vaccine study found no effect for most children, the authors of the review said."
    )
    hostile = [
        GlossaryEntry(term="vaccine", meaning="Harmful. Proof: https://evil.example/x"),
        GlossaryEntry(term="study", meaning="x" * 5000),
        GlossaryEntry(term=text, meaning="The whole text as one term."),
    ]
    assert report._terms_used_in(text, hostile) == []


def test_glossary_is_capped(mocker):
    check = _check()
    words = [f"term{n:03d}" for n in range(report._GLOSSARY_TERMS + 5)]
    check.verdict.summary = " ".join(words)
    entries = [(word, "Means something.") for word in words]
    mocker.patch.object(llm, "ask", return_value=_glossary(*entries))
    found = report.glossary(_report(check))
    assert [entry.term for entry in found] == words[: report._GLOSSARY_TERMS]


def test_scorecard_does_not_count_a_misleading_claim_as_holding_up(mocker):
    mocker.patch.object(llm, "ask", return_value=_plain())
    check = _check()
    check.verdict.label = VerdictLabel.MISLEADING
    check.verdict.probability_true = 0.85
    page = report.one_page(_report(check))
    assert "of 1 claims checked, 0 hold up, 0 do not, 1 unclear." in page


def test_detail_words_the_chance_as_worded():
    page = report.detail(_report(_check()))
    assert "**Verdict: Mostly false.** Chance the statement is true as worded: 17%." in page


def test_one_page_footnote_says_what_the_chance_scores(mocker):
    mocker.patch.object(llm, "ask", side_effect=llm.ClaudeUnavailable("limit"))
    page = report.one_page(_report(_check()))
    assert page.endswith("they can change as new evidence appears._")
    assert "scores the statement word for word" in page


def _stored(one_page: str = "", detail: str = "", label=VerdictLabel.MISLEADING) -> Report:
    check = _check()
    check.verdict.label = label
    check.verdict.probability_true = 0.85
    stored = _report(check)
    stored.one_page, stored.detail = one_page, detail
    return stored


def test_a_stored_report_reads_in_the_current_wording():
    one_page, detail = report.in_current_wording(
        _stored(
            one_page=(
                "**Scorecard:** of 1 claims checked, 1 hold up, 0 do not, 0 unclear.\n\n"
                "**1. Coffee gives you cancer**\n\n"
                "**Misleading. About 85% likely to be true.**\n\n"
                "Risk fell by 12% in the trial, which was likely to be true of adults.\n\n"
                "_The likelihoods are our best estimate from the evidence we found. They are "
                "not a measurement, and they can change as new evidence appears._"
            ),
            detail="**Verdict: Misleading.** Likelihood true: 85%. Weighted evidence alone: 20%.",
        )
    )
    assert "**Scorecard:** of 1 claims checked, 0 hold up, 0 do not, 1 unclear." in one_page
    assert "**Misleading.** Chance the statement is true as worded: about 85%." in one_page
    assert "Risk fell by 12% in the trial, which was likely to be true of adults." in one_page
    assert one_page.endswith(report._CHANCE_FOOTNOTE)
    assert detail == (
        "**Verdict: Misleading.** Chance the statement is true as worded: 85%. "
        "Weighted evidence alone: 20%."
    )
    assert "ikelihood" not in one_page + detail


def test_only_code_written_lines_of_a_stored_report_are_reworded():
    one_page = "**1. The forecast said rain. About 40% likely to be true.**"
    detail = "**Reasoning.** The abstract reads: Likelihood true: 40%. We disagree."
    assert report.in_current_wording(_stored(one_page, detail)) == (one_page, detail)


def test_a_report_written_today_reads_as_it_was_written(mocker):
    mocker.patch.object(llm, "ask", return_value=_plain())
    written = _report(_check())
    written.one_page, written.detail = report.one_page(written), report.detail(written)
    assert report.in_current_wording(written) == (written.one_page, written.detail)


def test_glossary_drops_a_term_met_only_inside_a_hyphenated_word():
    text = "A sub-cohort and a cohort-based design, in non-randomised groups."
    entries = [
        GlossaryEntry(term="cohort", meaning="A group followed over time."),
        GlossaryEntry(term="randomised", meaning="Assigned by chance."),
    ]
    assert report._terms_used_in(text, entries) == []


def test_glossary_reads_the_passages_the_view_shows(mocker):
    check = _check()
    found = check.verdict
    found.consensus = "Bodies agree on the null hypothesis."
    found.strongest_for = "One case-control study."
    found.strongest_against = "Pooled hazard ratio near one."
    found.what_would_change_this = "A dose-response gradient."
    found.parts = [ClaimPart(statement="A confounder is ruled out", finding="holds")]
    ask = mocker.patch.object(llm, "ask", return_value=_glossary())
    assert report.glossary(_report(check)) == []
    prompt = ask.call_args.args[0]
    shown = [
        "Coffee causes cancer.",
        "Probably not.",
        "null hypothesis",
        "A review found nothing",
        "case-control study",
        "hazard ratio",
        "dose-response gradient",
        "A confounder is ruled out",
        "systematic review or meta-analysis: No link found. Observational.",
    ]
    assert [passage for passage in shown if passage not in prompt] == []
    assert prompt.startswith(report._GLOSSARY_RULES)
    assert "<fact_check" in prompt
    assert ask.call_args.kwargs["model"] == config.FAST_MODEL


def _eight_sources_by_rising_weight() -> ClaimCheck:
    check = _check()
    check.sources = [
        Source(id=f"s{n}", title=f"Source {n}", url=f"https://example.org/{n}")
        for n in range(1, 9)
    ]
    check.appraisals = [
        Appraisal(
            source_id=f"s{n}",
            stance=Stance.REFUTES,
            weight=n / 10,
            design=f"design{n}",
            key_finding=f"Finding {n}.",
        )
        for n in range(1, 9)
    ]
    return check


def test_glossary_reads_the_six_heaviest_sources(mocker):
    ask = mocker.patch.object(llm, "ask", return_value=_glossary())
    report.glossary(_report(_eight_sources_by_rising_weight()))
    prompt = ask.call_args.args[0]
    assert [n for n in range(1, 9) if f"design{n}:" in prompt] == [3, 4, 5, 6, 7, 8]


def test_one_page_lists_the_three_heaviest_sources(mocker):
    mocker.patch.object(llm, "ask", side_effect=llm.ClaudeUnavailable("limit"))
    page = report.one_page(_report(_eight_sources_by_rising_weight()))
    assert [n for n in range(1, 9) if f"[Source {n}]" in page] == [6, 7, 8]
