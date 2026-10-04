"""
FastAPI Application
Central service connecting ingestion → NLP engine → risk engine → dashboard.

On startup, three background tasks run concurrently:
  1. news_loader:    fetches articles every 30s → raw_queue
  2. social_loader:  streams tweets every 15s  → raw_queue
  3. processor:      pulls from raw_queue → pipeline → store → WebSocket

Endpoints:
  GET  /                    health check
  GET  /events              recent processed events
  GET  /portfolio           current portfolio state
  GET  /risk/{ticker}       exposure for a specific ticker
  POST /stress-test         run manual stress test
  POST /black-swan          inject a fake headline (demo button)
  WS   /ws/events           live event stream
"""

import asyncio
import json
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.event_store import store
from src.api.models import (
    StressTestRequest, BlackSwanRequest,
    HealthResponse, StressTestResponse, PortfolioResponse
)
from src.ingestion.queue_manager import raw_queue
from src.ingestion.news_loader   import stream_news_to_queue, load_news_batch
from src.ingestion.social_loader import stream_tweets_to_queue
from src.engine.pipeline         import process, reset_novelty_memory
from src.risk.risk_graph         import traverse
from src.risk.stress_tester      import run_stress_test
from src.risk.loss_attribution   import attribute_losses
from src.risk.exposure_calc      import get_portfolio
from src.modules.module_a        import rebalancer, run_backtest, INDEX_STOCKS
from src.modules.module_b        import stress_tester, STRESS_TRIGGER_IMPACT

logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# Impact threshold to auto-trigger stress test
STRESS_TRIGGER_THRESHOLD = 4.5


async def _processor():
    """
    Background task: pull items from raw_queue, run through
    NLP pipeline, store result, trigger stress test if needed.
    """
    log.info("Processor started")
    # Process seeded articles first, then reset novelty
    # so live events get fresh novelty scores
    seeding_done = False
    items_processed = 0
    while True:
        try:
            item = await asyncio.wait_for(raw_queue.get(), timeout=1.0)
            event = process(item["text"], item.get("source", "unknown"))
            await store.add_event(event)
            items_processed += 1
            # Reset novelty after seeding batch (first 15 items)
            if not seeding_done and items_processed >= 15:
                reset_novelty_memory()
                seeding_done = True
                log.info("Novelty memory reset - live events now scored fresh")

            # Feed sentiment into Module A rebalancer
            for ticker in event.get("affected_tickers", []):
                rebalancer.update(ticker, event["sentiment_score"])

            # Feed high-impact events into Module B stress tester
            if event["impact_score"] >= STRESS_TRIGGER_THRESHOLD:
                stress_tester.run_scenario(
                    event_class=event["event_class"],
                    impact_score=event["impact_score"],
                    affected_tickers=event["affected_tickers"],
                    affected_sectors=event["affected_sectors"],
                    headline=event["headline"],
                )

            # Auto-trigger stress test for high-impact events
            if event["impact_score"] >= STRESS_TRIGGER_THRESHOLD:
                log.info(f"High impact event: {event['impact_score']}/10 "
                         f"- running stress test")
                await _run_and_store_stress_test(
                    event_class=event["event_class"],
                    impact_score=event["impact_score"],
                    affected_tickers=event["affected_tickers"],
                    affected_sectors=event["affected_sectors"],
                )
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            log.error(f"Processor error: {e}")
            await asyncio.sleep(1)


async def _run_and_store_stress_test(
    event_class: str,
    impact_score: float,
    affected_tickers: list,
    affected_sectors: list,
) -> dict:
    """Run stress test and store result. Returns full result."""
    # Get sectors from risk graph if none provided
    if not affected_sectors:
        graph_result = traverse(event_class, affected_tickers)
        affected_sectors = graph_result["affected_sectors"]

    stress = run_stress_test(
        affected_tickers=affected_tickers,
        affected_sectors=affected_sectors,
        event_class=event_class,
        impact_score=impact_score,
    )
    attribution = attribute_losses(stress)

    full_result = {
        "event_class":    event_class,
        "impact_score":   impact_score,
        **stress,
        **attribution,
    }
    await store.set_stress_result(full_result)
    return full_result


async def _seed_initial_events():
    """
    Load a batch of news on startup so the dashboard
    isn't empty when the demo starts.
    """
    log.info("Seeding initial events...")
    articles = load_news_batch()
    for article in articles[:10]:  # seed with 10 articles
        title = article.get("title", "")
        desc  = article.get("description", "")
        text  = f"{title}. {desc}" if desc else title
        if text.strip():
            source = f"news:{article.get('source',{}).get('name','unknown')}"
            raw_queue.put_nowait(
                {"text": text, "source": source}
            )
    log.info(f"Seeded {min(10, len(articles))} articles")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tasks on startup, cancel on shutdown."""
    await _seed_initial_events()

    tasks = [
        asyncio.create_task(_processor()),
        asyncio.create_task(stream_news_to_queue(interval_seconds=3600)),
        asyncio.create_task(stream_tweets_to_queue(interval_seconds=15)),
    ]
    log.info("All background tasks started")
    yield
    for task in tasks:
        task.cancel()
    log.info("Background tasks cancelled")


app = FastAPI(
    title="Financial Risk Intelligence API",
    description="Real-time NLP-driven financial risk engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST Endpoints ───────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def health():
    """Health check and system stats."""
    return HealthResponse(**store.stats())


@app.get("/events")
async def get_events(limit: int = 50):
    """Get most recent processed events, newest first."""
    events = await store.get_events(limit=limit)
    return {"count": len(events), "events": events}


@app.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio_state():
    """Current portfolio state."""
    portfolio = get_portfolio()
    return PortfolioResponse(
        total_value=await store.get_portfolio_value(),
        position_count=len(portfolio["positions"]),
        positions=portfolio["positions"],
    )


@app.get("/risk/{ticker}")
async def get_ticker_risk(ticker: str):
    """Get recent events and exposure for a specific ticker."""
    ticker = ticker.upper()
    events = await store.get_events(limit=200)
    relevant = [
        e for e in events
        if ticker in e.get("affected_tickers", [])
    ]
    stress = await store.get_stress_result()
    exposed_positions = []
    if stress:
        exposed_positions = [
            p for p in stress.get("positions", [])
            if p.get("ticker") == ticker
        ]
    return {
        "ticker":            ticker,
        "recent_events":     relevant[:10],
        "event_count":       len(relevant),
        "exposed_positions": exposed_positions,
    }


@app.post("/stress-test")
async def manual_stress_test(req: StressTestRequest):
    """Manually trigger a stress test with custom parameters."""
    result = await _run_and_store_stress_test(
        event_class=req.event_class,
        impact_score=req.impact_score,
        affected_tickers=req.affected_tickers,
        affected_sectors=req.affected_sectors,
    )
    return result


@app.post("/black-swan")
async def black_swan(req: BlackSwanRequest):
    """
    Inject a fake high-impact headline into the pipeline.
    Used for live demo: shows entire system reacting in real time.
    """
    log.info(f"BLACK SWAN injected: {req.headline}")
    item = {"text": req.headline, "source": req.source}
    await raw_queue.put(item)
    return {
        "status":   "injected",
        "headline": req.headline,
        "message":  "Event injected into pipeline. "
                    "Watch the dashboard for system response."
    }


@app.get("/stress-result")
async def get_stress_result():
    """Get the most recent stress test result."""
    result = await store.get_stress_result()
    if not result:
        raise HTTPException(
            status_code=404,
            detail="No stress test has been run yet."
        )
    return result




# ─── Module B Endpoints ───────────────────────────────────────

@app.get("/module-b/portfolio")
async def get_stress_portfolio():
    """Current portfolio state after all stress scenarios."""
    return stress_tester.get_portfolio_summary()


@app.get("/module-b/history")
async def get_scenario_history():
    """All stress test scenarios run this session."""
    return {
        "scenarios": stress_tester.get_history(),
        "count":     len(stress_tester.get_history()),
    }


@app.get("/module-b/latest")
async def get_latest_scenario():
    """Most recent stress test result with full attribution."""
    if not stress_tester.last_result:
        raise HTTPException(
            status_code=404,
            detail="No stress scenarios run yet."
        )
    return stress_tester.last_result


@app.post("/module-b/reset")
async def reset_stress_tester():
    """Reset portfolio to initial state. Used between demo sessions."""
    stress_tester.reset()
    return {"status": "reset", "portfolio_value": stress_tester.initial_value}

# ─── WebSocket ────────────────────────────────────────────────

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    Live event stream. Dashboard connects here to receive
    new events in real time as they are processed.
    """
    await websocket.accept()
    q = store.subscribe()
    log.info("WebSocket client connected")

    try:
        while True:
            # Wait for next event (with timeout to detect disconnects)
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                await websocket.send_text(json.dumps(event, default=str))
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    finally:
        store.unsubscribe(q)


# ─── Module A Endpoints ───────────────────────────────────────

@app.get("/module-a/weights")
async def get_index_weights():
    """Current index weights for all 15 stocks."""
    weights = rebalancer.get_weights()
    return {
        "weights": {
            t: {
                "weight":     round(w, 6),
                "weight_pct": round(w * 100, 2),
                "sector":     INDEX_STOCKS.get(t, "Unknown"),
            }
            for t, w in sorted(
                weights.items(),
                key=lambda x: x[1],
                reverse=True
            )
        },
        "equal_weight_pct": round(100 / len(INDEX_STOCKS), 2),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/module-a/backtest")
async def get_backtest(days: int = 90):
    """
    Run or retrieve backtest results.
    Compares sentiment-weighted vs equal-weight index.
    """
    from pathlib import Path

    # Use cached results if available and days match
    cache = Path("data/backtest_results.json")
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("days") == days:
            return cached

    # Otherwise run fresh backtest
    result = run_backtest(days=days)
    return result
