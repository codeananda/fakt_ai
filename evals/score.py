"""Score verdicts against the labelled claim set."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import fire
from loguru import logger
from pydantic import BaseModel

from fakt_ai import appraisal, config, evidence, ledger, llm, verdict
from fakt_ai.models import Claim

_FOLDER = Path(__file__).parent
_SCALE = ["False", "Mostly false", "Misleading", "Unproven", "Contested", "Mostly true", "True"]


class _Guess(BaseModel):
    probability_true: float


def run(
    claims: str = "claims.jsonl",
    only: str = "",
    every: int = 1,
    workers: int = 3,
    closed_book: bool = False,
) -> dict:
    """
    Check labelled claims and print accuracy and calibration.

    Parameters
    ----------
    claims : str
        Labelled file in this folder.
    only : str
        Keep claims whose text or domain contains this.
    every : int
        Keep every nth claim, for a quick spread across the set.
    workers : int
        Claims checked at once.
    closed_book : bool
        Answer from model knowledge alone, as a baseline.

    Returns
    -------
    dict
        Summary metrics.
    """
    rows = [json.loads(line) for line in (_FOLDER / claims).read_text().splitlines()]
    rows = [r for r in rows if only.lower() in (r["claim"] + r["domain"]).lower()][::every]
    check = _closed_book if closed_book else _check
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [r for r in pool.map(check, rows) if r]
    metrics = _metrics(results)
    _save(results, metrics, closed_book)
    for r in sorted(results, key=lambda r: abs(r["p"] - r["truth"]), reverse=True):
        print(
            f"{r['p']:.2f} vs {r['truth']:.2f} {r['predicted']:>12} / {r['label']:<12} "
            f"{r['claim'][:64]}"
        )
    print(json.dumps(metrics), "paid spend so far", round(ledger.spent(), 2))
    return metrics


def _check(row: dict) -> dict | None:
    claim = Claim(
        text=row["claim"],
        quote=row["claim"],
        domain=row["domain"],
        time_sensitive="published" in row or row["domain"] == "law",
    )
    try:
        sources = evidence.gather(claim)
        appraisals = appraisal.appraise(claim, sources)
        result = verdict.judge(claim, sources, appraisals)
    except Exception as error:  # noqa: BLE001 - keep scoring the rest
        logger.exception("failed: {} {}", row["claim"], error)
        return None
    return row | {
        "p": result.probability_true,
        "balance": result.evidence_balance,
        "predicted": result.label.value,
        "sources": len(sources),
        "used": sum(a.weight > 0 for a in appraisals),
        "summary": result.summary,
    }


def _closed_book(row: dict) -> dict:
    guess = llm.ask(
        f"How likely is this claim to be true as stated? Give a probability.\n\n{row['claim']}",
        _Guess,
        model=config.FAST_MODEL,
    )
    return row | {"p": guess.probability_true, "balance": 0.5, "predicted": "-", "summary": ""}


def _metrics(results: list[dict]) -> dict:
    n = len(results)
    decided = [r for r in results if abs(r["truth"] - 0.5) > 0.2]
    near = [
        r
        for r in results
        if r["predicted"] in _SCALE
        and abs(_SCALE.index(r["predicted"]) - _SCALE.index(r["label"])) <= 1
    ]
    return {
        "n": n,
        "brier": round(sum((r["p"] - r["truth"]) ** 2 for r in results) / n, 4),
        "direction_accuracy": round(
            sum((r["p"] > 0.5) == (r["truth"] > 0.5) for r in decided) / len(decided), 3
        ),
        "label_within_one": round(len(near) / n, 3),
        "balance_brier": round(sum((r["balance"] - r["truth"]) ** 2 for r in results) / n, 4),
    }


def _save(results: list[dict], metrics: dict, closed_book: bool) -> None:
    folder = config.STATE_DIR / "evals"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = "closed" if closed_book else "open"
    (folder / f"{stamp}-{suffix}.json").write_text(
        json.dumps({"metrics": metrics, "results": results}, indent=1)
    )


if __name__ == "__main__":
    fire.Fire(run)
