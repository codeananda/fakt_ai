# 🔍 Fakt AI

See something online and wonder if it's true? Point Fakt AI at it. You get each claim graded against the evidence, academic papers first, with a verdict, the chance it is true as worded, a one-page plain-English summary and a full report you can question.

## What it does

- **Takes almost anything:** an Instagram reel, an X post, a YouTube video or a three-hour podcast, a news article, pasted text or a screenshot. Audio is transcribed and burned-in captions are read.
- **Finds the claims that matter**, including the papers a speaker cites.
- **Searches for both sides:** OpenAlex, PubMed, Europe PMC, Crossref, ClinicalTrials.gov, Exa and Google, with a second round for anything left open.
- **Reads the sources:** every source is graded on study design, sample, rigour and conflicts of interest. The strongest are read in full, and every quote is checked against the source text.
- **Checks citations:** when someone cites a paper, it tests whether the paper says what they claimed.
- **Argues it out:** an advocate and a sceptic make the strongest honest cases, then a judge rules, marking which parts hold up. Retracted papers count for nothing.
- **Stays transparent:** the score from the weighted evidence is shown beside the judge's probability, so you can see when they differ.

## Install

You need [uv](https://docs.astral.sh/uv/), `ffmpeg`, and the [Claude Code CLI](https://claude.com/claude-code) logged in. All reasoning runs through `claude -p` on your Claude subscription, so no Anthropic API key is used.

```bash
git clone https://github.com/codeananda/fakt_ai && cd fakt_ai
uv sync
```

Optional keys add sources; see `.env.example`. Without any, it still runs on the free scholarly databases, with Claude doing the grading.

## Use it

```bash
uv run fakt serve                                    # app at http://127.0.0.1:8741
uv run fakt check "https://www.youtube.com/watch?v=..." --max_claims=3
uv run fakt check "Sugar makes children hyperactive"
uv run fakt ask <report id> "Is the sample size in that study big enough?"
uv run fakt spend                                    # estimated paid API spend
```

### Reading a report

Each claim gets a verdict with what that verdict means, the claim taken piece by piece, the chance that the statement is true as worded, and the case for and against it with the strongest evidence on each side. The verdict and the chance answer different questions: the chance scores the wording alone, while the verdict also judges the impression the wording leaves, so a Misleading claim can score high.

Hover over, tap or tab to a numbered mark to see the sources behind a statement, each with the quote found in it where the source was read in full. A word with a dotted underline explains itself in one sentence the same way. A report whose terms have not been explained yet offers an **Explain the technical terms** button.

### Chrome side panel

1. Run `uv run fakt serve`.
2. Open `chrome://extensions`, turn on Developer mode, choose **Load unpacked** and pick the `extension/` folder.
3. Click the Fakt button in the toolbar, or press `Alt+Shift+F`, while a post is in view. Right-click selected text for **Fact-check this**.

The panel talks only to your own machine. In a feed it picks the post nearest the middle of the screen. If that post has no link, it uses the visible text and a screenshot.

## How it's scored

`evals/` holds two labelled sets. `recent_claims.jsonl` comes from fact-checks published after the model's knowledge cutoff, with fact-checking sites hidden from the search, plus true counterparts, so it measures research rather than recall.

Latest results (lower Brier is better; "direction" counts whether it landed on the right side of 50%):

| Test set | Fakt AI Brier | Claude alone Brier | Fakt AI direction | Claude alone direction |
|---|---|---|---|---|
| 80 recent fact-checked claims | 0.030 | 0.103 | 97.3% | 89.0% |
| 36 true and distorted claims about papers from Aug 2026 on | 0.004 | 0.077 | 100% | 91.7% |
| 42 long-known claims | 0.004 | 0.004 | 100% | 100% |

Research adds nothing on long-known claims, as expected. Its value is in what the model cannot already know.

The first set measures agreement with professional fact-checkers on recent news, so most of its claims are political. `recent_papers.jsonl` holds the paper pairs.

```bash
FAKT_HIDE_FACT_CHECKS=1 uv run python evals/score.py --claims=recent_claims.jsonl
uv run python evals/score.py --claims=recent_claims.jsonl --closed_book=True  # baseline
```

## Costs

Exa, Jev, Google via Serper, Groq and Apify are paid. Every paid call goes through `fakt_ai/ledger.py`, which stops at `FAKT_SPEND_CAP_USD` (default $40). A typical claim costs about 10 cents. Results are cached, so repeats are free.

## Limits

- The chance given for each claim is an estimate from what was found, not a measurement.
- A reel's on-screen text is read for videos under three minutes. Longer videos rely on speech.
- Instagram and Facebook change their pages often, so the extension's post detection may need updating.
- Reports are kept locally in `.fakt/` and `reports/`, which git ignores.
