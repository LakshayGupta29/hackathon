"""
Event Store
In-memory store for all processed events and stress test results.

Why in-memory over a database:
  - Zero setup, zero dependencies
  - Fast enough for demo scale (100s of events)
  - State resets cleanly between demo sessions
  - A real production system would use Redis or Postgres

Thread safety: asyncio.Lock protects all writes since
FastAPI runs in an async context.
"""

import asyncio
from collections import deque
from datetime import datetime
from typing import Optional


class EventStore:
    def __init__(self, max_events: int = 500):
        # Recent processed events (ring buffer)
        self._events = deque(maxlen=max_events)

        # Latest stress test result
        self._stress_result: Optional[dict] = None

        # Latest portfolio state
        self._portfolio_value: float = 100_000_000.0

        # WebSocket subscribers (set of queues)
        self._subscribers: set = set()

        # Async lock for thread safety
        self._lock = asyncio.Lock()

        # Stats
        self._total_processed = 0
        self._started_at = datetime.utcnow().isoformat()

    async def add_event(self, event: dict):
        """Add a processed event and notify all WebSocket subscribers."""
        async with self._lock:
            self._events.appendleft(event)  # newest first
            self._total_processed += 1

        # Notify subscribers (non-blocking)
        await self._broadcast(event)

    async def get_events(self, limit: int = 50) -> list:
        """Get most recent events."""
        async with self._lock:
            return list(self._events)[:limit]

    async def set_stress_result(self, result: dict):
        """Store latest stress test result."""
        async with self._lock:
            self._stress_result = result
            if result:
                self._portfolio_value = result.get(
                    "portfolio_after",
                    self._portfolio_value
                )

    async def get_stress_result(self) -> Optional[dict]:
        async with self._lock:
            return self._stress_result

    async def get_portfolio_value(self) -> float:
        async with self._lock:
            return self._portfolio_value

    def subscribe(self) -> asyncio.Queue:
        """Register a new WebSocket subscriber. Returns its queue."""
        q = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove a WebSocket subscriber."""
        self._subscribers.discard(q)

    async def _broadcast(self, event: dict):
        """Push event to all connected WebSocket clients."""
        dead = set()
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.add(q)  # slow subscriber, drop it
        for q in dead:
            self._subscribers.discard(q)

    def stats(self) -> dict:
        return {
            "total_processed": self._total_processed,
            "events_in_store": len(self._events),
            "active_subscribers": len(self._subscribers),
            "started_at": self._started_at,
            "portfolio_value": self._portfolio_value,
        }


# Global singleton
store = EventStore()
