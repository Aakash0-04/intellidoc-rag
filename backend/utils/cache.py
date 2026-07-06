"""Simple in-memory query cache with TTL."""
from __future__ import annotations

import hashlib
import time
from typing import Optional, Any


class QueryCache:
    """
    Cache frequent queries to avoid redundant LLM calls.
    TTL = Time To Live in seconds.
    """
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 100):
        self.cache: dict[str, dict] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
    
    def _key(self, query: str, session_id: str = "") -> str:
        """Hash the query for cache key."""
        raw = f"{session_id}:{query.lower().strip()}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def get(self, query: str, session_id: str = "") -> Optional[Any]:
        """Get cached result if not expired."""
        key = self._key(query, session_id)
        entry = self.cache.get(key)
        
        if not entry:
            return None
        
        if time.time() - entry["timestamp"] > self.ttl:
            del self.cache[key]
            return None
        
        entry["hits"] += 1
        return entry["result"]
    
    def set(self, query: str, result: Any, session_id: str = ""):
        """Cache a query result."""
        key = self._key(query, session_id)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest]
        
        self.cache[key] = {
            "result": result,
            "timestamp": time.time(),
            "hits": 0,
        }
    
    def get_stats(self) -> dict:
        """Return cache statistics."""
        total = len(self.cache)
        hits = sum(v["hits"] for v in self.cache.values())
        return {
            "total_entries": total,
            "total_hits": hits,
            "ttl_seconds": self.ttl,
            "max_size": self.max_size,
        }
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()