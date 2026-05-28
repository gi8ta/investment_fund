"""
SQLite-based API response cache for Investment Fund.

Caches vendor API responses by MD5 hash of (method + args).
Reduces redundant API calls, especially important for Alpha Vantage (25 req/day).

TTL defaults:
  - news: 1 hour
  - fundamentals: 24 hours
  - indicators: 24 hours
  - stock_data: 1 hour
"""

import hashlib
import json
import os
import sqlite3
import time
from functools import wraps


_DB_PATH = None
_CONN = None

# TTL in seconds per data category
DEFAULT_TTL = {
    "news_data": 3600,           # 1 hour
    "fundamental_data": 86400,   # 24 hours
    "technical_indicators": 86400,  # 24 hours
    "core_stock_apis": 3600,     # 1 hour
}


def _get_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        cache_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
            "data_cache",
        )
        os.makedirs(cache_dir, exist_ok=True)
        _DB_PATH = os.path.join(cache_dir, "api_cache.db")
    return _DB_PATH


def _get_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(_get_db_path(), check_same_thread=False)
        _CONN.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        _CONN.execute("CREATE INDEX IF NOT EXISTS idx_category ON cache(category)")
        _CONN.commit()
    return _CONN


def _make_key(method: str, args: tuple, kwargs: dict) -> str:
    """Create a deterministic cache key from method name + arguments."""
    # Serialize args in a stable way
    key_data = json.dumps({"method": method, "args": list(args), "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


def cache_get(key: str, category: str = "news_data") -> str | None:
    """Get a cached value if it exists and hasn't expired."""
    ttl = DEFAULT_TTL.get(category, 3600)
    conn = _get_conn()
    cursor = conn.execute(
        "SELECT value, created_at FROM cache WHERE key = ?",
        (key,),
    )
    row = cursor.fetchone()
    if row is None:
        return None

    value, created_at = row
    if time.time() - created_at > ttl:
        # Expired — delete and return None
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        return None

    return value


def cache_set(key: str, value: str, category: str = "news_data") -> None:
    """Store a value in cache."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, value, category, created_at) VALUES (?, ?, ?, ?)",
        (key, value, category, time.time()),
    )
    conn.commit()


def cache_clear(category: str = None) -> int:
    """Clear cache entries. If category is specified, only clear that category."""
    conn = _get_conn()
    if category:
        cursor = conn.execute("DELETE FROM cache WHERE category = ?", (category,))
    else:
        cursor = conn.execute("DELETE FROM cache")
    conn.commit()
    return cursor.rowcount


def cache_stats() -> dict:
    """Return cache statistics."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    by_category = {}
    for row in conn.execute("SELECT category, COUNT(*) FROM cache GROUP BY category"):
        by_category[row[0]] = row[1]
    return {"total_entries": total, "by_category": by_category}
