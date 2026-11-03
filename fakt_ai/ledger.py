"""Estimated spend on paid services, with a hard stop before the cap is crossed."""

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from loguru import logger

from fakt_ai import config


class SpendCapReached(RuntimeError):
    """A paid call would take the estimated total past the cap."""


def charge(service: str, usd: float) -> None:
    """
    Record a paid call, refusing it when the cap would be crossed.

    Parameters
    ----------
    service : str
        Paid service name, such as "exa" or "jev".
    usd : float
        Estimated cost of the call in US dollars.

    Raises
    ------
    SpendCapReached
        When the estimated total would exceed `config.SPEND_CAP_USD`.
    """
    with _exclusive_access():
        total = spent()
        if total + usd > config.SPEND_CAP_USD:
            raise SpendCapReached(
                f"{service}: ${total:.2f} spent, cap ${config.SPEND_CAP_USD}"
            )
        entry = {"at": datetime.now(UTC).isoformat(), "service": service, "usd": usd}
        with _path().open("a") as f:
            f.write(json.dumps(entry) + "\n")
    logger.debug("spend {} ${:.4f}", service, usd)


def spent() -> float:
    """Return the estimated total spend so far in US dollars."""
    return sum(summary().values())


def summary() -> dict[str, float]:
    """Return estimated spend per service in US dollars."""
    path = _path()
    if not path.exists():
        return {}
    totals: dict[str, float] = {}
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        totals[entry["service"]] = totals.get(entry["service"], 0.0) + entry["usd"]
    return totals


def _path():
    return config.STATE_DIR / "spend.jsonl"


@contextmanager
def _exclusive_access() -> Iterator[None]:
    """Hold the ledger against every thread and process until the block ends."""
    # ponytail: flock is POSIX only, and a process not taking it can still append.
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name("spend.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
