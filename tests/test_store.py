import os

import pytest

from fakt_ai import config, store
from fakt_ai.models import Capture, Report


def test_save_then_load_round_trips():
    report = Report(capture=Capture(text="hello"))
    store.save(report)
    assert store.load(report.id) == report


def test_recent_returns_newest_first_within_the_limit():
    first, second = Report(capture=Capture()), Report(capture=Capture())
    store.save(first)
    store.save(second)
    assert len(store.recent(limit=1)) == 1
    assert {r.id for r in store.recent()} == {first.id, second.id}


def test_a_report_saved_again_keeps_its_place_in_the_history():
    older, newer = Report(capture=Capture()), Report(capture=Capture())
    os.utime(store.save(older), (1_000, 1_000))
    os.utime(store.save(newer), (2_000, 2_000))
    older.one_page = "Amended."
    store.save(older)
    assert [r.id for r in store.recent()] == [newer.id, older.id]
    assert store.load(older.id).one_page == "Amended."


def test_recent_skips_damaged_files():
    store.save(Report(capture=Capture()))
    (config.STATE_DIR / "reports" / "bad.json").write_text("{not json")
    assert len(store.recent()) == 1


def test_recent_with_nothing_saved():
    assert store.recent() == []


@pytest.mark.parametrize("report_id", ["../secret", "missing", "a/b", ""])
def test_load_refuses_unknown_and_unsafe_ids(report_id):
    with pytest.raises(FileNotFoundError):
        store.load(report_id)
