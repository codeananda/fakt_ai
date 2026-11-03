"""Persistent JSON cache so repeat runs cost no subscription quota or credits."""

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable
from typing import Any

from fakt_ai import config

_LOCK = threading.Lock()


def cached(namespace: str, key: Any, compute: Callable[[], Any]) -> Any:
    """
    Return the stored JSON value for `key`, computing and storing it on a miss.

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
    if row:
        return json.loads(row[0])
    value = compute()
    with _LOCK, _connect() as db:
        db.execute(
            "insert or replace into cache values (?,?,?)",
            (namespace, digest, json.dumps(value)),
        )
    return value


def _connect() -> sqlite3.Connection:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.STATE_DIR / "cache.sqlite", timeout=30)
    db.execute(
        "create table if not exists cache (ns text, key text, value text, primary key (ns, key))"
    )
    return db
