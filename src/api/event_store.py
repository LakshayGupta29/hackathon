"""
Event Store.

In-memory store for processed events and the latest stress result, plus a
WebSocket fan-out. Every processed signal is also appended to a JSON Lines
file (data/signals_live.jsonl by default) so downstream consumers can read
the engine's output from disk as well as through the API.

Why in-memory: zero setup, fast enough for demo scale and resets cleanly
between sessions. A production deployment would use Redis/Kafka + Postgres.
"""

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from src.config import env_flag, output_path

log = logging.getLogger(__name__)
SIGNAL_LOG_NAME = "signals_live.jsonl"


class EventStore:
    def __init__(self, max_events: int = 500):
        self._events = deque(maxlen=max_events)
        self._stress_result: Optional[dict] = None
        self._subscribers: set = set()
        self._lock = asyncio.Lock()
        self._total_processed = 0
        self._duplicates_suppressed = 0
        self._stress_triggers = 0
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._write_signals = not env_flag("DISABLE_SIGNAL_LOG")

    def rebind(self):
        """Fresh asyncio primitives for a new event loop."""
        self._lock = asyncio.Lock()
        self._subscribers = set()

    async def add_event(self, event: dict):
        """Store an event, append it to the signal log, notify subscribers."""
        async with self._lock:
            self._events.appendleft(event)
            self._total_processed += 1
            if event.get("is_duplicate"):
                self._duplicates_suppressed += 1
            if event.get("stress_test_triggered"):
                self._stress_triggers += 1
        self._append_signal(event)
        await self._broadcast(event)

    def _append_signal(self, event: dict) -> None:
        if not self._write_signals:
            return
        try:
            with output_path(SIGNAL_LOG_NAME).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        except OSError as exc:
            log.warning("Could not append to signal log: %s", exc)

    async def get_events(self, limit: int = 50) -> list:
        async with self._lock:
            return list(self._events)[:max(0, int(limit))]

    async def set_stress_result(self, result: dict):
        async with self._lock:
            self._stress_result = result

    async def get_stress_result(self) -> Optional[dict]:
        async with self._lock:
            return self._stress_result

    async def reset_stress_state(self):
        async with self._lock:
            self._stress_result = None

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    async def _broadcast(self, event: dict):
        dead = set()
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.add(q)  # slow subscriber: drop it
        for q in dead:
            self._subscribers.discard(q)

    def stats(self) -> dict:
        last = self._stress_result or {}
        return {
            "total_processed": self._total_processed,
            "events_in_store": len(self._events),
            "duplicates_suppressed": self._duplicates_suppressed,
            "stress_tests_triggered": self._stress_triggers,
            "active_subscribers": len(self._subscribers),
            "started_at": self._started_at,
            "last_stress_loss": last.get("total_loss", 0.0),
            "last_stress_loss_pct": last.get("loss_pct", 0.0),
        }


store = EventStore()
