import pytest

from fakt_ai import config
from fakt_ai.evidence import fetch


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """Answer every lookup with a public address, so tests stay off the network."""
    monkeypatch.setattr(fetch, "_resolve", lambda _host, _port: ["93.184.216.34"])
