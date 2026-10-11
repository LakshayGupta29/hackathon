"""End-to-end API tests with fake models and live feeds disabled."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    from src.modules.module_a import rebalancer
    from src.modules.module_b import stress_tester

    rebalancer.reset()
    stress_tester.reset()
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/").json()
    assert body["status"] == "ok"
    assert body["portfolio_value"] == 100_000_000
    assert body["live_feeds_enabled"] is False


def test_process_endpoint_returns_structured_signal(client):
    body = client.post("/process", json={
        "text": "Federal Reserve hikes interest rates by 75 basis points as inflation fears grow",
        "source": "news:bloomberg",
    }).json()
    assert body["event_class"] == "Macroeconomic"
    assert -1 <= body["sentiment_score"] <= 1
    assert 1 <= body["impact_score"] <= 10


def test_black_swan_triggers_stress_test_and_module_a(client):
    body = client.post("/black-swan", json={
        "headline": "Russia missile strikes escalate war; TSMC and NVIDIA shares fall as crisis deepens",
        "source": "news:reuters",
    }).json()
    event = body["event"]
    assert event["event_class"] == "Geopolitical"
    assert event["impact_score"] >= 7
    assert event["stress_test_triggered"] is True
    assert any(u["ticker"] in {"TSM", "NVDA"} for u in event["module_a_updates"])

    latest = client.get("/module-b/latest").json()
    assert latest["trigger"] == "auto"
    assert latest["total_loss"] < 0
    history = client.get("/module-a/history").json()["history"]
    assert len(history) >= 2

    # The exact same headline again is a duplicate: no new stress test.
    repeat = client.post("/black-swan", json={
        "headline": "Russia missile strikes escalate war; TSMC and NVIDIA shares fall as crisis deepens",
    }).json()["event"]
    assert repeat["is_duplicate"] is True
    assert repeat["stress_test_triggered"] is False


def test_manual_stress_test_and_validation(client):
    ok = client.post("/stress-test", json={
        "event_class": "Credit Event", "impact_score": 8, "affected_tickers": ["jpm"],
    })
    assert ok.status_code == 200
    assert ok.json()["affected_tickers"] == ["JPM"]
    bad = client.post("/stress-test", json={"event_class": "Alien Invasion", "impact_score": 8})
    assert bad.status_code == 422


def test_events_and_weights(client):
    events = client.get("/events?limit=10").json()
    assert events["count"] >= 2
    weights = client.get("/module-a/weights").json()["weights"]
    assert abs(sum(w["weight"] for w in weights.values()) - 1.0) < 1e-4
    portfolio = client.get("/portfolio").json()
    assert portfolio["position_count"] == 13


def test_resets(client):
    assert client.post("/module-b/reset").json()["status"] == "reset"
    assert client.get("/module-b/latest").status_code == 404
    assert client.post("/module-a/reset").json()["status"] == "reset"
