import hashlib
import json

import pytest

from fakt_ai import cache, config, ledger


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)


def test_cache_computes_once():
    calls = []
    compute = lambda: calls.append(1) or {"a": 1}
    assert cache.cached("ns", ["k"], compute) == {"a": 1}
    assert cache.cached("ns", ["k"], compute) == {"a": 1}
    assert len(calls) == 1


def test_ledger_stops_at_cap(monkeypatch):
    monkeypatch.setattr(config, "SPEND_CAP_USD", 1.0)
    ledger.charge("exa", 0.6)
    ledger.charge("jev", 0.3)
    with pytest.raises(ledger.SpendCapReached):
        ledger.charge("exa", 0.2)
    assert ledger.summary() == {"exa": 0.6, "jev": 0.3}


WEEK = 7 * 24 * 60 * 60
LEGACY_INSERT = "insert or replace into cache values (?,?,?)"


def _store_without_a_time(namespace, key, value):
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
    with cache._connect() as db:
        db.execute(LEGACY_INSERT, (namespace, digest, json.dumps(value)))


def _at(mocker, seconds):
    mocker.patch.object(cache.time, "time", return_value=seconds)


@pytest.mark.parametrize("namespace", ["exa", "serper", "crossref"])
def test_fresh_entries_expire_after_a_week(mocker, namespace):
    _at(mocker, 1_000.0)
    assert cache.cached(namespace, ["k"], lambda: "old") == "old"
    _at(mocker, 1_000.0 + WEEK - 1)
    assert cache.cached(namespace, ["k"], lambda: "new") == "old"
    _at(mocker, 1_000.0 + WEEK + 1)
    assert cache.cached(namespace, ["k"], lambda: "new") == "new"
    assert cache.cached(namespace, ["k"], lambda: "newer") == "new"


@pytest.mark.parametrize("namespace", ["claude", "jev", "fulltext", "openalex"])
def test_other_namespaces_keep_their_entries(mocker, namespace):
    _at(mocker, 1_000.0)
    cache.cached(namespace, ["k"], lambda: "old")
    _at(mocker, 1_000.0 + 100 * WEEK)
    assert cache.cached(namespace, ["k"], lambda: "new") == "old"


def test_entry_with_no_time_counts_as_written_at_first_sight(mocker):
    _store_without_a_time("exa", ["k"], "legacy")
    _at(mocker, 1_000.0)
    assert cache.cached("exa", ["k"], lambda: "new") == "legacy"
    _at(mocker, 1_000.0 + WEEK - 1)
    assert cache.cached("exa", ["k"], lambda: "new") == "legacy"
    _at(mocker, 1_000.0 + WEEK + 1)
    assert cache.cached("exa", ["k"], lambda: "new") == "new"


def test_a_failed_refresh_is_not_cached(mocker):
    _at(mocker, 1_000.0)
    cache.cached("exa", ["k"], lambda: "old")
    _at(mocker, 1_000.0 + 2 * WEEK)

    def fail():
        raise RuntimeError

    with pytest.raises(RuntimeError):
        cache.cached("exa", ["k"], fail)
    assert cache.cached("exa", ["k"], lambda: "new") == "new"
