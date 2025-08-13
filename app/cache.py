# app/cache.py
from __future__ import annotations
import time
import asyncio
from typing import Any, Callable, Dict, Tuple, Optional

class TTLCache:
    """
    Simple async-safe TTL cache with best-effort LRU eviction.
    Keys must be hashable. Values are arbitrary JSON-serializable or strings.
    """
    def __init__(self, max_items: int = 1024, ttl_seconds: int = 600):
        self.max_items = max_items
        self.ttl = ttl_seconds
        self._store: Dict[Any, Tuple[float, Any]] = {}
        self._touch: Dict[Any, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: Any) -> Optional[Any]:
        async with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            exp, val = item
            if exp < time.time():
                self._store.pop(key, None)
                self._touch.pop(key, None)
                return None
            self._touch[key] = time.time()
            return val

    async def set(self, key: Any, val: Any):
        async with self._lock:
            if len(self._store) >= self.max_items:
                # evict stale or least recently touched
                victim = min(self._touch.items(), key=lambda kv: kv[1])[0] if self._touch else None
                if victim in self._store:
                    self._store.pop(victim, None)
                    self._touch.pop(victim, None)
            self._store[key] = (time.time() + self.ttl, val)
            self._touch[key] = time.time()

# singleton cache instances
default_cache = TTLCache(max_items=2048, ttl_seconds=600)  # 10 minutes
short_cache = TTLCache(max_items=512, ttl_seconds=120)     # 2 minutes for volatile calls
