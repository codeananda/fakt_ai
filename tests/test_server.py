import base64
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakt_ai import ledger, pipeline, server, store
from fakt_ai.models import (
    Capture,
    Claim,
    ClaimCheck,
    Report,
    Source,
    Verdict,
    VerdictLabel,
)

BASE_URL = "http://127.0.0.1:8765"
EXTENSION = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n pixels").decode()


@pytest.fixture
def client():
    server._jobs.clear()
    return TestClient(server.app, base_url=BASE_URL)


def _report(**overrides) -> Report:
    source = Source(id="s1", title="A trial", url="https://example.org", full_text="x" * 500)
    verdict = Verdict(
        label=VerdictLabel.MOSTLY_FALSE, probability_true=0.2, summary="No.", reasoning="[s1]"
    )
    check = ClaimCheck(claim=Claim(text="Water is dry"), sources=[source], verdict=verdict)
    fields = {
        "id": "r1",
        "capture": Capture(text="Water is dry, says a post"),
        "checks": [check],
    }
    return Report(**(fields | overrides))


def _events(client: TestClient, job_id: str, **headers) -> list[tuple[str, dict]]:
    with client.stream("GET", f"/checks/{job_id}/events", headers=headers) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.read().decode()
    parsed = []
    for block in filter(None, body.split("\n\n")):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        parsed.append((lines.get("event", "message"), json.loads(lines["data"])))
    return parsed


def _fake_run(on_event, **_):
    on_event({"stage": "capture", "message": "Reading the post"})
    on_event({"stage": "verdict", "message": "Weighing evidence"})
    return _report()


def test_check_streams_progress_then_report_id(client, mocker):
    run = mocker.patch.object(pipeline, "run", side_effect=_fake_run)

    response = client.post(
        "/checks", json={"url": "https://x.com/a/status/1", "max_claims": 3}
    )

    assert response.status_code == 202
    assert _events(client, response.json()["id"]) == [
        ("message", {"stage": "capture", "message": "Reading the post"}),
        ("message", {"stage": "verdict", "message": "Weighing evidence"}),
        ("done", {"report_id": "r1"}),
    ]
    assert run.call_args.kwargs["url"] == "https://x.com/a/status/1"
    assert run.call_args.kwargs["max_claims"] == 3


def test_reconnecting_stream_resumes_after_last_event(client, mocker):
    mocker.patch.object(pipeline, "run", side_effect=_fake_run)

    job_id = client.post("/checks", json={"text": "claim"}).json()["id"]

    events = _events(client, job_id, **{"Last-Event-ID": "0"})
    assert events == [
        ("message", {"stage": "verdict", "message": "Weighing evidence"}),
        ("done", {"report_id": "r1"}),
    ]


def test_pipeline_failure_reaches_the_stream(client, mocker):
    error = ledger.SpendCapReached("exa: $40 spent")
    mocker.patch.object(pipeline, "run", side_effect=error)

    job_id = client.post("/checks", json={"text": "claim"}).json()["id"]

    assert _events(client, job_id) == [("failed", {"message": "exa: $40 spent"})]


def test_screenshot_exists_during_capture_and_is_removed_after(client, mocker):
    seen = {}

    def inspect_image(image, **_):
        seen.update(path=Path(image), content=Path(image).read_bytes())
        return _report()

    mocker.patch.object(pipeline, "run", side_effect=inspect_image)

    data_url = f"data:image/png;base64,{PNG}"
    job_id = client.post("/checks", json={"image_base64": data_url}).json()["id"]
    _events(client, job_id)

    assert seen["content"].startswith(b"\x89PNG")
    assert seen["path"].suffix == ".png"
    assert not seen["path"].exists()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"url": "file:///etc/passwd"},
        {"text": "claim", "max_claims": 0},
        {"image_base64": "not base64!"},
        {"image_base64": base64.b64encode(b"<svg>").decode()},
    ],
)
def test_bad_check_requests_are_refused(client, mocker, body):
    run = mocker.patch.object(pipeline, "run")

    assert client.post("/checks", json=body).status_code == 422
    run.assert_not_called()


def test_unknown_job_is_404(client):
    assert client.get("/checks/nope/events").status_code == 404


def test_report_omits_source_full_text(client, mocker):
    mocker.patch.object(store, "load", return_value=_report())

    report = client.get("/reports/r1").json()

    source = report["checks"][0]["sources"][0]
    assert source["title"] == "A trial"
    assert "full_text" not in source


def test_missing_report_is_404(client, mocker):
    mocker.patch.object(store, "load", side_effect=FileNotFoundError)

    assert client.get("/reports/absent").status_code == 404


def test_report_id_cannot_climb_directories(client, mocker):
    load = mocker.patch.object(store, "load")

    assert client.get("/reports/..%2F..%2Fsecret").status_code in (404, 422)
    load.assert_not_called()


def test_report_list_describes_each_check(client, mocker):
    recent = mocker.patch.object(store, "recent", return_value=[_report()])

    described = client.get("/reports", params={"limit": 2}).json()

    recent.assert_called_once_with(limit=2)
    assert described[0]["id"] == "r1"
    assert described[0]["claims"][0]["label"] == "Mostly false"


def test_ask_passes_source_to_follow_up(client, mocker):
    report = _report()
    mocker.patch.object(store, "load", return_value=report)
    follow_up = mocker.patch.object(pipeline, "follow_up", return_value="Because.")

    response = client.post("/reports/r1/ask", json={"question": "Why?", "source_id": "s1"})

    assert response.json() == {"answer": "Because."}
    follow_up.assert_called_once_with(report, "Why?", source_id="s1")


def test_ask_about_unknown_source_is_404(client, mocker):
    mocker.patch.object(store, "load", return_value=_report())
    follow_up = mocker.patch.object(pipeline, "follow_up")

    response = client.post("/reports/r1/ask", json={"question": "Why?", "source_id": "zz"})

    assert response.status_code == 404
    follow_up.assert_not_called()


def test_failed_follow_up_is_reported(client, mocker):
    mocker.patch.object(store, "load", return_value=_report())
    mocker.patch.object(pipeline, "follow_up", side_effect=RuntimeError("quota"))

    response = client.post("/reports/r1/ask", json={"question": "Why?"})

    assert response.status_code == 502
    assert "quota" in response.json()["detail"]


def test_spend_returns_ledger_summary(client, mocker):
    mocker.patch.object(ledger, "summary", return_value={"exa": 0.04})

    assert client.get("/spend").json() == {"exa": 0.04}


@pytest.mark.parametrize(
    "origin", [EXTENSION, "http://localhost:3000", "http://127.0.0.1:8765"]
)
def test_local_origins_are_allowed(client, origin):
    response = client.get("/spend", headers={"Origin": origin})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "http://localhost.evil.example",
        "http://127.0.0.1.evil.example",
        "chrome-extension://short",
        "null",
    ],
)
def test_foreign_origins_cannot_start_checks(client, mocker, origin):
    run = mocker.patch.object(pipeline, "run")

    response = client.post("/checks", json={"text": "x"}, headers={"Origin": origin})

    assert response.status_code == 403
    run.assert_not_called()


def test_foreign_host_header_is_refused(client):
    assert client.get("/spend", headers={"Host": "evil.example"}).status_code == 400


def test_serve_binds_to_loopback_only(mocker):
    run = mocker.patch.object(server.uvicorn, "run")

    server.serve()

    assert run.call_args.kwargs == {"host": "127.0.0.1", "port": 8765}


def test_ui_is_served_at_root_with_a_content_policy(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_worker_runs_off_the_request_thread(client, mocker):
    threads = []

    def record_thread(**_):
        threads.append(threading.current_thread())
        return _report()

    mocker.patch.object(pipeline, "run", side_effect=record_thread)

    job_id = client.post("/checks", json={"text": "claim"}).json()["id"]
    _events(client, job_id)

    assert threads[0].daemon
