"""
FastAPI application: ingestion -> NLP risk engine -> Module A / Module B.

Background tasks started on launch:
  1. processor     pulls raw items from the queue, runs the NLP pipeline in a
                   worker thread (the event loop never blocks), then feeds
                   Module A and, for qualifying events, Module B
  2. news stream   seeds ~20 live RSS headlines, then polls every 120 s
                   (falls back to bundled historical headlines if no feed is
                   reachable)
  3. social replay real financial posts every 15 s
  4. warm-up       loads all models once so the first event is fast

Set DISABLE_LIVE_FEEDS=1 to run without RSS/social streams (tests, offline).

Endpoints
  GET  /                      health + stats
  GET  /events                recent structured risk signals
  POST /process               analyse any text (optionally trigger modules)
  POST /black-swan            inject a headline; processed immediately
  GET  /risk/{ticker}         recent events and exposure for one ticker
  GET  /portfolio             synthetic portfolio positions
  POST /stress-test           manual Module B scenario
  GET  /stress-result         latest Module B scenario
  GET  /module-a/weights      current index weights
  GET  /module-a/history      weight history (time series)
  GET  /module-a/backtest     historical event replay vs equal weight
  POST /module-a/reset
  GET  /module-b/portfolio    stress summary
  GET  /module-b/history      all scenarios this session
  GET  /module-b/latest       latest scenario with full attribution
  POST /module-b/reset
  WS   /ws/events             live event stream
"""

import asyncio
import json
import logging
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.event_store import store
from src.api.models import (
    BlackSwanRequest,
    HealthResponse,
    PortfolioResponse,
    ProcessRequest,
    StressTestRequest,
)
from src.config import PIPELINE_VERSION, env_flag, env_float, output_path
from src.engine.pipeline import process, warmup
from src.engine.sentiment import model_status
from src.ingestion.queue_manager import raw_queue
from src.ingestion.rss_loader import fetch_rss_articles, load_offline_news
from src.ingestion.social_loader import stream_tweets_to_queue
from src.modules.module_a import EQUAL_WEIGHT, INDEX_STOCKS, rebalancer, run_backtest
from src.modules.module_b import STRESS_TRIGGER_IMPACT, stress_tester

logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

LIVE_FEEDS = not env_flag("DISABLE_LIVE_FEEDS")
RSS_INTERVAL_SECONDS = env_float("RSS_INTERVAL_SECONDS", 120.0)
SEED_SIZE = 20
STRESS_TRIGGER_THRESHOLD = STRESS_TRIGGER_IMPACT

_state = {"news_source": "disabled" if not LIVE_FEEDS else "starting"}


class SeenHeadlines:
    """Bounded insertion-ordered set of normalised headline keys."""

    def __init__(self, max_size: int = 2000):
        self._keys: OrderedDict[str, None] = OrderedDict()
        self._max = max_size

    @staticmethod
    def key(text: str) -> str:
        return re.sub(r"\W+", " ", str(text)[:80].lower()).strip()

    def add(self, text: str) -> bool:
        """Return True if the headline is new."""
        k = self.key(text)
        if not k or k in self._keys:
            return False
        self._keys[k] = None
        while len(self._keys) > self._max:
            self._keys.popitem(last=False)
        return True


# ─── Event handling ───────────────────────────────────────────

async def _run_and_store_stress_test(
    event_class: str,
    impact_score: float,
    affected_tickers: list,
    affected_sectors: list,
    headline: str = "",
    sentiment_score: float | None = None,
    mode: str = "baseline",
    trigger: str = "manual",
) -> dict:
    result = stress_tester.run_scenario(
        event_class=event_class,
        impact_score=impact_score,
        affected_tickers=affected_tickers,
        affected_sectors=affected_sectors,
        headline=headline,
        sentiment_score=sentiment_score,
        mode=mode,
        trigger=trigger,
    )
    await store.set_stress_result(result)
    return result


async def _apply_downstream(event: dict) -> dict:
    """Feed Module A and (if the trigger rule fires) Module B, then store."""
    event["module_a_updates"] = []
    event["stress_test_triggered"] = False

    if not event.get("is_duplicate"):
        for ticker in event.get("affected_tickers", []):
            if ticker in INDEX_STOCKS:
                update = rebalancer.update(ticker, event["sentiment_score"], source=event["source"])
                if update:
                    event["module_a_updates"].append(update)

        if stress_tester.should_trigger(
            event["impact_score"], event["event_class"], event["sentiment_score"]
        ):
            log.info("High-impact %s event (%.2f/10): running stress test",
                     event["event_class"], event["impact_score"])
            result = await _run_and_store_stress_test(
                event_class=event["event_class"],
                impact_score=event["impact_score"],
                affected_tickers=event["affected_tickers"],
                affected_sectors=event["affected_sectors"],
                headline=event["headline"],
                sentiment_score=event["sentiment_score"],
                trigger="auto",
            )
            event["stress_test_triggered"] = True
            event["stress_scenario_id"] = result["scenario_id"]
            event["stress_loss_pct"] = result["loss_pct"]

    await store.add_event(event)
    return event


async def _analyse(text: str, source: str, published_at: str | None = None) -> dict:
    return await asyncio.to_thread(process, text, source, published_at)


async def _processor():
    log.info("Processor started")
    while True:
        item = await raw_queue.get()
        try:
            event = await _analyse(item["text"], item.get("source", "unknown"), item.get("published_at"))
            await _apply_downstream(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("Processor error on %r: %s", str(item.get("text"))[:80], exc)


async def _news_stream():
    """Seed recent headlines, then poll RSS for unseen ones."""
    seen = SeenHeadlines()
    articles = await asyncio.to_thread(fetch_rss_articles, 15)
    if articles:
        _state["news_source"] = "rss"
    else:
        articles = load_offline_news()
        _state["news_source"] = "offline-replay"
        log.warning("No RSS feed reachable; replaying bundled historical headlines")

    seeded = 0
    for article in articles[:SEED_SIZE]:
        if seen.add(article["text"]):
            raw_queue.put_nowait({"text": article["text"], "source": article["source"],
                                  "published_at": article.get("publishedAt")})
            seeded += 1
    log.info("Seeded %d headlines (%s)", seeded, _state["news_source"])

    while True:
        await asyncio.sleep(RSS_INTERVAL_SECONDS)
        try:
            fresh = await asyncio.to_thread(fetch_rss_articles, 10, False)
            if fresh:
                _state["news_source"] = "rss"
            added = 0
            for article in fresh:
                if seen.add(article["text"]):
                    raw_queue.put_nowait({"text": article["text"], "source": article["source"],
                                          "published_at": article.get("publishedAt")})
                    added += 1
            log.info("RSS refresh: %d new headlines", added)
        except Exception as exc:
            log.error("RSS stream error: %s", exc)


async def _warmup():
    try:
        await asyncio.to_thread(warmup)
        log.info("Models warmed up (sentiment model: %s)", model_status()["model"])
    except Exception as exc:
        log.error("Model warm-up failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    raw_queue.rebind()
    store.rebind()
    tasks = [
        asyncio.create_task(_processor()),
        asyncio.create_task(_warmup()),
    ]
    if LIVE_FEEDS:
        tasks.append(asyncio.create_task(_news_stream()))
        tasks.append(asyncio.create_task(stream_tweets_to_queue()))
    log.info("Background tasks started (live feeds: %s)", LIVE_FEEDS)
    yield
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    log.info("Background tasks stopped")


app = FastAPI(
    title="Financial Risk Intelligence API",
    description="Real-time NLP risk engine with index rebalancing and stress testing",
    version=PIPELINE_VERSION,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Core endpoints ───────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def health():
    status = model_status()
    return HealthResponse(
        **store.stats(),
        portfolio_value=stress_tester.initial_value,
        sentiment_model=status["model"],
        sentiment_model_fallback=status["fallback"],
        live_feeds_enabled=LIVE_FEEDS,
        news_source=_state["news_source"],
        queue_size=raw_queue.qsize(),
    )


@app.get("/events")
async def get_events(limit: int = Query(50, ge=1, le=500)):
    events = await store.get_events(limit=limit)
    return {"count": len(events), "events": events}


@app.post("/process")
async def process_text(req: ProcessRequest):
    """Analyse any text and return the structured risk signal."""
    event = await _analyse(req.text, req.source)
    if req.trigger_downstream:
        event = await _apply_downstream(event)
    return event


@app.post("/black-swan")
async def black_swan(req: BlackSwanRequest):
    """Inject a headline for the live demo; processed immediately (not queued)."""
    log.info("BLACK SWAN injected: %s", req.headline)
    event = await _analyse(req.headline, req.source)
    event = await _apply_downstream(event)
    return {
        "status": "processed",
        "headline": req.headline,
        "event": event,
        "message": (
            "Stress test triggered." if event["stress_test_triggered"]
            else "Processed; impact or event type below the stress-test trigger."
        ),
    }


@app.get("/risk/{ticker}")
async def get_ticker_risk(ticker: str):
    ticker = ticker.upper()
    events = await store.get_events(limit=500)
    relevant = [e for e in events if ticker in e.get("affected_tickers", [])]
    stress = await store.get_stress_result()
    exposed = [p for p in (stress or {}).get("positions", []) if p.get("ticker") == ticker]
    return {
        "ticker": ticker,
        "recent_events": relevant[:10],
        "event_count": len(relevant),
        "index_weight": rebalancer.get_weights().get(ticker),
        "exposed_positions": exposed,
    }


@app.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio_state(mode: str = Query("baseline", pattern="^(baseline|cumulative)$")):
    positions = stress_tester.get_positions(mode)
    return PortfolioResponse(
        portfolio_name=stress_tester.portfolio_name,
        total_value=sum(float(p["value"]) for p in positions),
        mode=mode,
        position_count=len(positions),
        positions=positions,
    )


@app.post("/stress-test")
async def manual_stress_test(req: StressTestRequest):
    return await _run_and_store_stress_test(
        event_class=req.event_class,
        impact_score=req.impact_score,
        affected_tickers=req.affected_tickers,
        affected_sectors=req.affected_sectors,
        headline="Manual stress test",
        sentiment_score=req.sentiment_score,
        mode=req.mode,
        trigger="manual",
    )


@app.get("/stress-result")
async def get_stress_result():
    result = await store.get_stress_result()
    if not result:
        raise HTTPException(status_code=404, detail="No stress test has been run yet.")
    return result


# ─── Module A ─────────────────────────────────────────────────

@app.get("/module-a/weights")
async def get_index_weights():
    weights = rebalancer.get_weights()
    return {
        "weights": {
            t: {
                "weight": round(w, 6),
                "weight_pct": round(w * 100, 3),
                "sector": INDEX_STOCKS.get(t, "Unknown"),
            }
            for t, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)
        },
        "equal_weight_pct": round(EQUAL_WEIGHT * 100, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/module-a/history")
async def get_index_history(limit: int = Query(500, ge=1, le=2000)):
    history = rebalancer.get_history(limit=limit)
    return {"count": len(history), "history": history}


@app.post("/module-a/reset")
async def reset_index():
    rebalancer.reset()
    return {"status": "reset"}


@app.get("/module-a/backtest")
async def get_backtest(days: int = Query(0, ge=0), refresh: bool = False):
    """Historical event replay vs equal weight (cached in data/backtest_results.json)."""
    cache = output_path("backtest_results.json")
    if cache.exists() and not refresh:
        try:
            cached = json.loads(cache.read_text())
            if (cached.get("calendar_days_requested") or 0) == days and \
                    cached.get("pipeline_version") == PIPELINE_VERSION:
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    try:
        return await asyncio.to_thread(run_backtest, days)
    except Exception as exc:
        log.exception("Backtest failed")
        raise HTTPException(status_code=503, detail=f"Backtest unavailable: {exc}") from exc


# ─── Module B ─────────────────────────────────────────────────

@app.get("/module-b/portfolio")
async def get_stress_portfolio():
    return stress_tester.get_portfolio_summary()


@app.get("/module-b/history")
async def get_scenario_history():
    scenarios = stress_tester.get_history()
    return {"scenarios": scenarios, "count": len(scenarios)}


@app.get("/module-b/latest")
async def get_latest_scenario():
    if not stress_tester.last_result:
        raise HTTPException(status_code=404, detail="No stress scenarios run yet.")
    return stress_tester.last_result


@app.post("/module-b/reset")
async def reset_stress_tester():
    stress_tester.reset()
    await store.reset_stress_state()
    return {"status": "reset", "portfolio_value": stress_tester.initial_value}


# ─── WebSocket ────────────────────────────────────────────────

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await websocket.accept()
    q = store.subscribe()
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                await websocket.send_text(json.dumps(event, default=str))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        store.unsubscribe(q)
