"""Command line for Fakt AI."""

import re
import sys
from pathlib import Path

import fire
from loguru import logger

from fakt_ai import config, ledger, store
from fakt_ai.models import Report

_REPORT_DIR = Path("reports")
_URL = re.compile(r"(https?://|www\.)\S+$", re.IGNORECASE)


def main() -> None:
    """Run the `fakt` command."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    commands = {"check": check, "show": show, "ask": ask, "spend": spend, "serve": serve}
    fire.Fire(commands, command=_kept_as_text(sys.argv[1:]))


def check(target: str = "", image: str | None = None, max_claims: int = 3) -> None:
    """
    Check a link, pasted text or a screenshot and print the one-page summary.

    Parameters
    ----------
    target : str
        A URL or the text to check.
    image : str | None
        Path to a screenshot.
    max_claims : int
        Most claims to check.
    """
    from fakt_ai import pipeline

    url, text = _split_target(str(target))
    report = pipeline.run(
        url=url,
        text=text,
        image=Path(image) if image else None,
        on_event=_print_progress,
        max_claims=int(max_claims),
    )
    path = _write_markdown(report)
    print(report.one_page)
    print(f"\nReport {report.id}. Full version: {path}")


def show(report_id: str, full: bool = False) -> None:
    """
    Print a saved report.

    Parameters
    ----------
    report_id : str
        The report's id.
    full : bool
        Print the full report as well as the one-pager.
    """
    from fakt_ai.report import in_current_wording

    report = store.load(str(report_id))
    print(in_current_wording(report.one_page))
    if full:
        print(f"\n---\n\n{in_current_wording(report.detail)}")


def ask(report_id: str, question: str, source: str | None = None) -> None:
    """
    Ask a question about a saved report.

    Parameters
    ----------
    report_id : str
        The report's id.
    question : str
        The question.
    source : str | None
        Id of one source to focus on.
    """
    from fakt_ai import pipeline

    print(pipeline.follow_up(store.load(str(report_id)), str(question), source_id=source))


def spend() -> None:
    """Print estimated spend per paid service against the cap."""
    for service, usd in sorted(ledger.summary().items()):
        print(f"{service}: ${usd:.2f}")
    print(f"total: ${ledger.spent():.2f} of ${config.SPEND_CAP_USD:.2f}")


def serve() -> None:
    """Start the local web app."""
    from fakt_ai import server

    server.serve()


def _kept_as_text(arguments: list[str]) -> list[str]:
    """Quote positional values, or fire reads an id like 1e234567 as a number."""
    command, *rest = arguments or [""]
    return [command, *(a if a.startswith("-") else repr(a) for a in rest)]


def _split_target(target: str) -> tuple[str | None, str | None]:
    target = target.strip()
    if not target:
        return None, None
    return (target, None) if _URL.fullmatch(target) else (None, target)


def _print_progress(event: dict) -> None:
    print(f"[{event['stage']}] {event['message']}", file=sys.stderr)


def _write_markdown(report: Report) -> Path:
    _REPORT_DIR.mkdir(exist_ok=True)
    path = _REPORT_DIR / f"{report.id}.md"
    path.write_text(f"{report.one_page}\n\n---\n\n{report.detail}\n")
    return path
