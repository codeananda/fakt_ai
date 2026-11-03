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
