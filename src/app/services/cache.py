"""Caching service.

Simple in-memory TTL cache — drop-in placeholder for Redis / Valkey.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class TTLCache:
    """Fixed-capacity in-memory cache with per-key TTL.

    Usage::

        cache = TTLCache(maxsize=256, default_ttl=300)
        cache.set("key", value)
        value = cache.get("key")  # None if expired or missing
    """

    def __init__(self, maxsize: int = 256, default_ttl: float = 300) -> None:
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._data[key]
            return None
        # Move to end (LRU-order)
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        else:
            # Evict oldest if at capacity
            while len(self._data) >= self.maxsize:
                popped = self._data.popitem(last=False)
                logger.debug("Cache evicted %s", popped[0])

        expires_at = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        self._data[key] = (expires_at, value)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    @property
    def size(self) -> int:
        return len(self._data)


# Global default instance — import this for a quick shared cache.
default_cache = TTLCache()
