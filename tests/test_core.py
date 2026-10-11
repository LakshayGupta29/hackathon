"""Unit tests for the NLP engine, Module A and Module B (offline)."""

import json
import math
import sys
import types
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.config import EVENT_CLASSES, data_path

# ─── Impact scoring ───────────────────────────────────────────


def test_impact_weights_sum_to_one_and_score_in_range():
    from src.engine.impact_scorer import IMPACT_WEIGHTS, compute_impact

    assert math.isclose(sum(IMPACT_WEIGHTS.values()), 1.0)
    result = compute_impact(1.0, "Geopolitical", 1.0, ["NVDA"], ["news:reuters"])
    assert 1.0 <= result["impact_score"] <= 10.0
    assert math.isclose(sum(result["contributions"].values()), result["impact_score"], abs_tol=0.02)


def test_source_authority_resolves_feed_ids():
    from src.engine.impact_scorer import source_authority

    assert source_authority("news:reuters") == 1.0
    assert source_authority("news:Financial Times") == 0.9
    assert source_authority("news:cnbc_markets") == 0.75
    assert source_authority("news:marketwatch_rt") == 0.70
    assert source_authority("social:twitter") == 0.40
    assert source_authority("news:some_blog") == 0.50


def test_every_event_class_has_priors_and_shocks():
    from src.engine.impact_scorer import EVENT_SEVERITY

    templates = json.loads(data_path("shock_templates.json").read_text())
    graph = json.loads(data_path("risk_graph_static.json").read_text())["event_sector_impact"]
    for cls in EVENT_CLASSES:
        assert cls in EVENT_SEVERITY
        assert cls in templates
        assert cls in graph


# ─── Event classification ─────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ("Federal Reserve raises rates by 50 basis points", "Macroeconomic"),
    ("Moody's downgrades Boeing credit rating to junk", "Credit Event"),
    ("Russia launches full-scale invasion of Ukraine", "Geopolitical"),
    ("Amazon agrees to acquire iRobot in 1.7 billion dollar deal", "Merger/Acquisition"),
    ("Apple unveils new iPhone with titanium design", "Product Launch"),
    ("Nike earnings beat estimates as China sales rebound", "Earnings/Corporate"),
    ("Tesla recalls 27,000 Cybertrucks", "Earnings/Corporate"),
    ("Oil prices surge as OPEC extends production cuts", "Geopolitical"),
])
def test_keyword_classifier(text, expected):
    from src.engine.event_classifier import classify_event

    result = classify_event(text)
    assert result["event_class"] == expected
    assert result["method"] == "keyword"


def test_keyword_rules_avoid_false_positives():
    from src.engine.event_classifier import _keyword_match

    assert _keyword_match("software platform revenue grows") is not None  # revenue -> corporate
    assert _keyword_match("software platform update") is None             # 'war' not in 'software'
    match = _keyword_match("Apple sets default privacy settings in new iOS")
    assert match is None or match["category"] != "Credit Event"
    match = _keyword_match("Tesla cuts prices as price war with rivals intensifies")
    assert match["category"] == "Earnings/Corporate"
    assert _keyword_match("Fed officials debut new communication strategy")["category"] == "Macroeconomic"


def test_tied_keywords_defer_to_zero_shot_among_tied_classes():
    from src.engine.event_classifier import classify_event

    result = classify_event("Company unveils quarterly update")  # Product Launch vs Earnings tie
    assert result["method"] == "keyword+zero-shot"
    assert set(result["all_scores"]) == {"Product Launch", "Earnings/Corporate"}


def test_low_confidence_zero_shot_abstains_to_other(monkeypatch):
    from src.engine import event_classifier

    class Unsure:
        def __call__(self, text, candidate_labels, **kwargs):
            n = len(candidate_labels)
            return {"labels": list(candidate_labels), "scores": [0.30] + [0.70 / (n - 1)] * (n - 1)}

    monkeypatch.setattr(event_classifier, "_load_classifier", lambda: Unsure())
    result = event_classifier.classify_event("How to downsize like a pro in retirement")
    assert result["event_class"] == "Other"
    assert result["method"] == "zero-shot+abstain"
    assert result["low_confidence"] is True


def test_credit_rules_need_credit_context():
    from src.engine.event_classifier import _keyword_match

    assert _keyword_match("Moody's predicts aging populations will strain public finances") is None
    assert _keyword_match("Moody's downgrades Italy's sovereign rating")["category"] == "Credit Event"
    assert _keyword_match("Germany flash services PMI falls to 9-month low")["category"] == "Macroeconomic"


def test_rss_drops_stale_items(monkeypatch):
    from src.ingestion import rss_loader

    now = rss_loader.datetime.now(rss_loader.timezone.utc)
    fresh = now.strftime("%a, %d %b %Y %H:%M:%S +0000")
    feed = f"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
      <item><title>Fresh headline about markets today</title><pubDate>{fresh}</pubDate></item>
      <item><title>Hurricane Isaias hits US</title><pubDate>Tue, 04 Aug 2020 10:00:00 +0000</pubDate></item>
      <item><title>Undated headline is kept</title></item>
    </channel></rss>"""

    class Resp:
        content = feed.encode()
        def raise_for_status(self):
            return None

    monkeypatch.setattr(rss_loader.httpx, "get", lambda *a, **k: Resp())
    _, articles, error = rss_loader._fetch_feed("news:test", "http://x", 10)
    titles = [a["text"] for a in articles]
    assert error is None
    assert "Fresh headline about markets today" in titles
    assert "Undated headline is kept" in titles
    assert all("Isaias" not in t for t in titles)


def test_unmatched_text_uses_zero_shot_over_all_classes():
    from src.engine.event_classifier import classify_event

    result = classify_event("Something happened somewhere today")
    assert result["method"] == "zero-shot"
    assert set(result["all_scores"]) == set(EVENT_CLASSES)


# ─── Entity linking ───────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ("$NVDA absolutely crushing it this quarter", ["NVDA"]),
    ("Taiwan tensions hit TSMC and ASML hard", ["TSM", "ASML"]),
    ("Google and Facebook face antitrust scrutiny", ["GOOGL", "META"]),
    ("Base metals rally as China stimulus lifts demand", []),
    ("US tightens H-1B visa rules; Visa restrictions hit students", []),
    ("Visa Inc beats estimates on cross-border volumes", ["V"]),
    ("The S&P 500 fell 2 percent", []),
    ("Shares of NVDA and AMD slid; IBM flat", ["NVDA", "AMD", "IBM"]),
    ("Goldman Sachs and JP Morgan lead bank rally", ["GS", "JPM"]),
    ("Greg Abbott signs bill", []),
])
def test_ticker_extraction(text, expected):
    from src.engine.ner_linker import extract_tickers

    assert extract_tickers(text) == expected


# ─── Sentiment ────────────────────────────────────────────────


def test_sentiment_score_is_prob_difference():
    from src.engine.sentiment import analyze

    neg = analyze("Stocks plunge as crisis deepens", mode="finetuned")
    assert neg["label"] == "negative"
    assert math.isclose(neg["score"], 0.04 - 0.90, abs_tol=1e-4)
    neutral = analyze("Company holds annual meeting", mode="finetuned")
    assert neutral["label"] == "neutral" and abs(neutral["score"]) < 0.05


def test_sentiment_handles_flat_pipeline_output(monkeypatch):
    from src.engine import sentiment

    flat = lambda texts, **kw: [  # noqa: E731
        {"label": "Bullish", "score": 0.7}, {"label": "Bearish", "score": 0.1}, {"label": "Neutral", "score": 0.2}
    ]
    monkeypatch.setattr(sentiment, "_load_finbert", lambda: flat)
    result = sentiment.analyze("anything", mode="finbert")
    assert result["label"] == "positive"
    assert math.isclose(result["score"], 0.6, abs_tol=1e-4)


# ─── Pipeline, novelty, fusion ────────────────────────────────


def test_pipeline_output_schema_and_duplicate_flag():
    from src.engine.pipeline import process, reset_novelty_memory

    reset_novelty_memory()
    text = "Taiwan tensions escalate as missile drills hit TSMC and NVIDIA shares"
    first = process(text, "news:reuters")
    for key in ["sentiment_score", "event_class", "impact_score", "affected_tickers",
                "affected_sectors", "novelty_score", "is_duplicate", "impact_contributions"]:
        assert key in first
    assert -1.0 <= first["sentiment_score"] <= 1.0
    assert 1.0 <= first["impact_score"] <= 10.0
    assert first["event_class"] == "Geopolitical"
    assert {"TSM", "NVDA"} <= set(first["affected_tickers"])
    assert first["is_duplicate"] is False
    repeat = process(text, "news:cnbc")
    assert repeat["is_duplicate"] is True
    assert repeat["novelty_score"] < 0.05


def test_fusion_uses_complete_linkage_no_chaining(monkeypatch):
    from src.engine import event_fusion

    # A~B and B~C are similar (0.6) but A and C are not (0.2): DBSCAN would chain them.
    a = np.array([1.0, 0.0, 0.0])
    c = np.array([0.2, 0.9797959, 0.0])
    b = (a + c) / np.linalg.norm(a + c)
    monkeypatch.setattr(event_fusion, "embed_batch", lambda texts: np.vstack([a, b, c]))
    result = event_fusion.fuse_events(
        [{"event_id": i, "headline": h} for i, h in enumerate("abc")],
        similarity_threshold=0.5,
    )
    ids = [e["event_cluster_id"] for e in result["events"]]
    assert ids[0] != ids[2]


# ─── Risk graph ───────────────────────────────────────────────


def test_risk_graph_propagates_event():
    from src.risk.risk_graph import traverse

    result = traverse("Geopolitical", ["TSM", "NVDA"])
    assert "Technology" in result["affected_sectors"]
    assert "TSM" in result["sector_tickers"]["Technology"]
    corporate = traverse("Earnings/Corporate", ["BA"])
    assert corporate["affected_sectors"] == ["Industrials"]


# ─── Module A ─────────────────────────────────────────────────


def test_rebalancer_bounds_sum_and_turnover():
    from src.modules.module_a import MAX_TURNOVER, MAX_WEIGHT, MIN_WEIGHT, IndexRebalancer

    r = IndexRebalancer()
    for _ in range(60):
        r.update("NVDA", 1.0)
        r.update("BA", -1.0)
    weights = r.get_weights()
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)
    assert min(weights.values()) >= MIN_WEIGHT - 1e-9
    assert max(weights.values()) <= MAX_WEIGHT + 1e-9
    assert r.update("AAPL", 1.0)["portfolio_turnover"] <= MAX_TURNOVER + 1e-9


def test_rebalancer_deadband_and_decay():
    from src.modules.module_a import EQUAL_WEIGHT, HALF_LIFE_DAYS, IndexRebalancer

    r = IndexRebalancer()
    assert r.update("AAPL", 0.05)["skipped"] == "deadband"
    r.update("NVDA", 1.0)
    tilt = r.get_weights()["NVDA"] - EQUAL_WEIGHT
    assert tilt > 0
    r.advance(r._clock + timedelta(days=HALF_LIFE_DAYS))
    decayed = r.get_weights()["NVDA"] - EQUAL_WEIGHT
    assert math.isclose(decayed, tilt / 2, rel_tol=1e-6)
    assert math.isclose(sum(r.get_weights().values()), 1.0, abs_tol=1e-9)


def test_backtest_runs_with_stub_prices(monkeypatch):
    from src.modules import module_a

    tickers = list(module_a.INDEX_STOCKS)
    idx = pd.bdate_range("2023-01-02", "2023-12-29")
    rng = np.random.default_rng(1)
    opens = pd.DataFrame(100 * np.exp(rng.normal(0, 0.01, (len(idx), len(tickers))).cumsum(0)),
                         index=idx, columns=tickers)
    closes = opens * (1 + rng.normal(0.0002, 0.01, opens.shape))
    prices = pd.concat({"Close": closes, "Open": opens}, axis=1)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: prices))

    signals = [
        {"event_id": "a", "date": "2023-03-10", "ticker": "JPM", "sentiment_score": -0.9},
        {"event_id": "b", "date": "2023-05-24", "ticker": "NVDA", "sentiment_score": 0.95},
        {"event_id": "c", "date": "2023-05-24", "ticker": "NVDA", "sentiment_score": 0.6},
    ]
    result = module_a.run_backtest(sentiment_events=signals, save=False)
    assert result["trading_days"] > 200
    assert result["signals_applied"] == 2  # same-day NVDA signals are aggregated
    assert len(result["weights_history"]) == result["trading_days"]
    assert result["sentiment_final"] <= result["sentiment_final_gross"] + 1e-12
    assert result["transaction_costs_included"] is True


# ─── Module B ─────────────────────────────────────────────────


def _portfolio():
    return json.loads(data_path("synthetic_portfolio.json").read_text())


def test_rate_shock_hits_every_bond_systemically():
    from src.risk.stress_tester import run_stress_test

    result = run_stress_test([], [], "Macroeconomic", 10.0, portfolio=_portfolio())
    bonds = [p for p in result["positions"] if p["type"] == "bond"]
    assert bonds and all(p["pnl"] < 0 for p in bonds)


def test_bond_pnl_formula():
    from src.risk.stress_tester import run_stress_test

    result = run_stress_test(["AAPL"], ["Technology"], "Macroeconomic", 10.0, portfolio=_portfolio())
    aapl = next(p for p in result["positions"] if p["ticker"] == "AAPL")
    # Macro @10: +50bp rates, +80bp spreads; direct x1.5; AAA rating beta 0.6; D = SD = 6.5
    expected = 12_000_000 * (-6.5 * 0.005 - 6.5 * 0.008 * 1.5 * 0.6)
    assert math.isclose(aapl["pnl"], round(expected, 2), abs_tol=0.01)
    irs = next(p for p in result["positions"] if p["id"] == "D001")
    assert math.isclose(irs["pnl"], -4900 * 50, abs_tol=0.01)  # receive-fixed loses when rates rise
    cds = next(p for p in result["positions"] if p["id"] == "D002")
    assert cds["pnl"] > 0  # protection buyer gains when spreads widen


def test_company_specific_events_have_small_spillover():
    from src.risk.stress_tester import run_stress_test

    result = run_stress_test(["BA"], [], "Earnings/Corporate", 8.0, portfolio=_portfolio(), sentiment_score=-0.8)
    by_id = {p["id"]: p for p in result["positions"]}
    assert by_id["L001"]["exposure_type"] == "direct"
    assert by_id["E001"]["exposure_multiplier"] == pytest.approx(0.10)
    assert abs(by_id["L001"]["pnl"]) > abs(by_id["E001"]["pnl"]) / 15_000_000 * 8_000_000


def test_idiosyncratic_shock_direction_follows_sentiment():
    from src.risk.shock_generator import generate_shocks

    bad = generate_shocks("Product Launch", 8.0, sentiment_score=-0.7)["shocks"]
    good = generate_shocks("Product Launch", 8.0, sentiment_score=0.7)["shocks"]
    assert bad["equity_shock"] < 0 < good["equity_shock"]
    systemic = generate_shocks("Geopolitical", 8.0, sentiment_score=0.9)["shocks"]
    assert systemic["equity_shock"] < 0  # stress scenarios stay adverse


def test_baseline_mode_does_not_compound_but_cumulative_does():
    from src.modules.module_b import PortfolioStressTester

    tester = PortfolioStressTester()
    first = tester.run_scenario("Geopolitical", 8.0, ["TSM"], [])
    second = tester.run_scenario("Geopolitical", 8.0, ["TSM"], [])
    assert first["portfolio_before"] == second["portfolio_before"] == tester.initial_value
    assert first["total_loss"] == second["total_loss"] < 0

    cumulative = PortfolioStressTester()
    a = cumulative.run_scenario("Geopolitical", 8.0, ["TSM"], [], mode="cumulative")
    b = cumulative.run_scenario("Geopolitical", 8.0, ["TSM"], [], mode="cumulative")
    assert b["portfolio_before"] == pytest.approx(a["portfolio_after"])
    assert cumulative.get_portfolio_summary()["current_value"] < cumulative.initial_value


def test_trigger_rule():
    from src.modules.module_b import STRESS_TRIGGER_IMPACT, PortfolioStressTester

    t = PortfolioStressTester()
    assert STRESS_TRIGGER_IMPACT == 7.0
    assert t.should_trigger(7.0, "Geopolitical", 0.0)
    assert t.should_trigger(7.0, "Geopolitical", -0.9)
    assert not t.should_trigger(7.1, "Geopolitical", 0.9)        # e.g. "LMT wins NATO contract"
    assert not t.should_trigger(9.0, "Other", -0.9)               # non-events never trigger
    assert not t.should_trigger(6.99, "Geopolitical", -0.9)
    assert not t.should_trigger(7.7, "Product Launch", 0.9)      # good news never "stresses"
    assert t.should_trigger(7.5, "Earnings/Corporate", -0.8)     # bad company news can
    assert t.should_trigger(7.0)                                  # legacy signature


def test_attribution_has_dashboard_keys():
    from src.modules.module_b import PortfolioStressTester

    result = PortfolioStressTester().run_scenario("Credit Event", 8.0, ["JPM"], [])
    assert result["by_sector"]
    for row in result["by_sector"]:
        assert {"sector", "loss", "pct_of_gross_loss", "pct_of_total"} <= set(row)
    assert result["value_by_asset_type_before"].keys() == result["value_by_asset_type_after"].keys()


# ─── Fresh test-set scorer ────────────────────────────────────


def test_fresh_test_summary_metrics():
    sys.path.insert(0, str(data_path("x").parents[1] / "scripts"))
    import fresh_test_set as fts

    rows = [
        {"label": "Geopolitical", "predicted": "Geopolitical", "tickers": "TSM", "predicted_tickers": ["TSM"]},
        {"label": "Other", "predicted": "Other", "tickers": "", "predicted_tickers": []},
        {"label": "Other", "predicted": "Merger/Acquisition", "tickers": "", "predicted_tickers": []},
        {"label": "Earnings/Corporate", "predicted": "Earnings/Corporate", "tickers": "AAPL;MSFT",
         "predicted_tickers": ["AAPL"]},
    ]
    summary = fts.summarise(rows)
    assert summary["accuracy"] == 0.75
    assert summary["majority_class"] == "Other" and summary["majority_baseline_accuracy"] == 0.5
    assert summary["accuracy_excluding_other"] == 1.0
    assert summary["ticker_recall"] == round(2 / 3, 4)
    lo, hi = summary["accuracy_95ci"]
    assert 0 < lo < 0.75 < hi <= 1
