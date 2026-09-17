"""Redis-backed query cache for RAG-Bench.

Usage
-----
    from src.cache.redis_cache import cache

    cached = cache.get(question)
    if cached:
        return cached

    result = pipeline.query(question)
    cache.set(question, result)
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

import redis

from src.config import settings

logger = logging.getLogger(__name__)

# TTL for cached answers (24 hours)
_DEFAULT_TTL_SEC: int = 86_400


class RedisCache:
    """Simple key-value cache wrapping Redis.  Serialises values as JSON."""

    def __init__(self, url: str = settings.REDIS_URL, ttl: int = _DEFAULT_TTL_SEC):
        self._ttl = ttl
        try:
            self._client = redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
            self._client.ping()
            self._available = True
            logger.info("Redis cache connected at %s", url)
        except Exception as exc:
            self._available = False
            logger.warning("Redis unavailable (%s) — caching disabled.", exc)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _make_key(self, query: str) -> str:
        """Stable cache key from the raw query string."""
        digest = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
        return f"rag_bench:query:{digest}"

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Return cached answer dict or None if not found / Redis unavailable."""
        if not self._available:
            return None
        try:
            raw = self._client.get(self._make_key(query))
            if raw:
                logger.debug("Cache HIT for query: %.60s…", query)
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis GET failed: %s", exc)
        return None

    def set(self, query: str, value: Dict[str, Any]) -> None:
        """Store answer dict in Redis with TTL.  Silently skips if unavailable."""
        if not self._available:
            return
        try:
            self._client.set(self._make_key(query), json.dumps(value), ex=self._ttl)
            logger.debug("Cache SET for query: %.60s…", query)
        except Exception as exc:
            logger.warning("Redis SET failed: %s", exc)

    def delete(self, query: str) -> None:
        """Evict a specific query from the cache."""
        if not self._available:
            return
        try:
            self._client.delete(self._make_key(query))
        except Exception as exc:
            logger.warning("Redis DELETE failed: %s", exc)

    def flush(self) -> None:
        """Clear the entire rag_bench namespace (dev/test utility)."""
        if not self._available:
            return
        try:
            keys = self._client.keys("rag_bench:*")
            if keys:
                self._client.delete(*keys)
                logger.info("Flushed %d cache keys.", len(keys))
        except Exception as exc:
            logger.warning("Redis FLUSH failed: %s", exc)

    def ping(self) -> bool:
        """Return True if Redis responds."""
        try:
            return self._available and bool(self._client.ping())
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------
cache = RedisCache()
