import asyncio
from collections import deque
from datetime import datetime, timezone
import uuid

class EventQueue:
    def __init__(self, maxsize: int = 1000):
        self._maxsize = maxsize
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._history = deque(maxlen=500)

    def rebind(self):
        """Fresh queue for a new event loop (app restart / test client)."""
        self._queue = asyncio.Queue(maxsize=self._maxsize)

    async def put(self, item: dict):
        await self._queue.put(item)
        self._history.append(item)

    async def get(self) -> dict:
        return await self._queue.get()

    def put_nowait(self, item: dict):
        try:
            self._queue.put_nowait(item)
            self._history.append(item)
        except asyncio.QueueFull:
            pass

    def qsize(self) -> int:
        return self._queue.qsize()

    def get_history(self) -> list:
        return list(self._history)

def make_raw_item(text: str, source: str, url: str = "",
                  published_at: str = "") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "source": source,
        "text": text,
        "url": url,
        "published_at": published_at or datetime.now(timezone.utc).isoformat(),
        "raw_entities": []
    }

# Global queue instance
raw_queue = EventQueue()
