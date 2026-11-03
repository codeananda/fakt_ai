"""Persistent JSON cache so repeat runs cost no subscription quota or credits."""

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from typing import Any

from fakt_ai import config

_LOCK = threading.Lock()
_WEEK_SECONDS = 7 * 24 * 60 * 60
# Searches and editorial status go stale. Other namespaces keep their entries for good:
# the keys of `claude` and `jev` contain the evidence they judged.
_MAX_AGE_SECONDS = {"exa": _WEEK_SECONDS, "serper": _WEEK_SECONDS, "crossref": _WEEK_SECONDS}


def cached(namespace: str, key: Any, compute: Callable[[], Any]) -> Any:
    """
    Return the stored JSON value for `key`, computing and storing it on a miss or expiry.

    Parameters
    ----------
    namespace : str
        Separates callers, such as "claude" or "openalex".
    key : Any
        JSON-serialisable identity of the request.
    compute : Callable[[], Any]
        Produces a JSON-serialisable value on a miss. Exceptions are not cached.

    Returns
    -------
    Any
        The cached or freshly computed value.
    """
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
    with _LOCK, _connect() as db:
        row = db.execute(
            "select value from cache where ns=? and key=?", (namespace, digest)
        ).fetchone()
        fresh = row is not None and _is_fresh(db, namespace, digest)
    if fresh:
        return json.loads(row[0])
    value = compute()
    with _LOCK, _connect() as db:
        db.execute(
            "insert or replace into cache values (?,?,?)",
            (namespace, digest, json.dumps(value)),
        )
        db.execute(
            "insert or replace into cache_times values (?,?,?)",
            (namespace, digest, time.time()),
        )
    return value


def _is_fresh(db: sqlite3.Connection, namespace: str, digest: str) -> bool:
    """Judge an entry's age, starting the clock on one that has no recorded time."""
    max_age = _MAX_AGE_SECONDS.get(namespace)
    if max_age is None:
        return True
    row = db.execute(
        "select at from cache_times where ns=? and key=?", (namespace, digest)
    ).fetchone()
    if row is None:
        db.execute(
            "insert or ignore into cache_times values (?,?,?)",
            (namespace, digest, time.time()),
        )
        return True
    return time.time() - row[0] < max_age


def _connect() -> sqlite3.Connection:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.STATE_DIR / "cache.sqlite", timeout=30)
    db.execute(
        "create table if not exists cache (ns text, key text, value text, primary key (ns, key))"
    )
    db.execute(
        "create table if not exists cache_times "
        "(ns text, key text, at real, primary key (ns, key))"
    )
    return db
