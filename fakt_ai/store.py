"""Finished reports as JSON files."""

import os
import re
import tempfile
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from fakt_ai import config
from fakt_ai.models import Report

_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")


def save(report: Report) -> Path:
    """
    Write a report atomically, so readers never see half a file.

    A report written again keeps its file time, because the history lists reports in that
    order.

    Parameters
    ----------
    report : Report
        The report.

    Returns
    -------
    Path
        Where it was stored.
    """
    path = _path(report.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    first_written = path.stat().st_mtime if path.exists() else None
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(handle, "w") as f:
        f.write(report.model_dump_json(indent=2))
    Path(temporary).replace(path)
    if first_written is not None:
        os.utime(path, (first_written, first_written))
    return path


def load(report_id: str) -> Report:
    """
    Read one report.

    Parameters
    ----------
    report_id : str
        The report's id.

    Returns
    -------
    Report
        The report.

    Raises
    ------
    FileNotFoundError
        When no report has this id.
    """
    if not _SAFE_ID.fullmatch(report_id):
        raise FileNotFoundError(f"no report {report_id!r}")
    path = _path(report_id)
    if not path.exists():
        raise FileNotFoundError(f"no report {report_id!r}")
    return Report.model_validate_json(path.read_text())


def recent(limit: int = 30) -> list[Report]:
    """
    Read the latest reports, newest first.

    Parameters
    ----------
    limit : int
        Most reports to return.

    Returns
    -------
    list[Report]
        Readable reports. Damaged files are skipped.
    """
    folder = config.STATE_DIR / "reports"
    if not folder.exists():
        return []
    paths = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    reports = []
    for path in paths:
        if len(reports) == limit:
            break
        try:
            reports.append(Report.model_validate_json(path.read_text()))
        except (ValidationError, OSError) as error:
            logger.warning("skipping unreadable report {}: {}", path.name, error)
    return reports


def _path(report_id: str) -> Path:
    return config.STATE_DIR / "reports" / f"{report_id}.json"
