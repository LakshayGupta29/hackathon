"""
News Cache
Persists fetched articles to disk so API restarts don't
waste rate limit quota refetching the same articles.

Strategy:
  - On fetch: save new articles to cache file
  - On restart: load cache first, then only fetch articles
    newer than the most recent cached article
  - Cache is a simple JSON file, max 500 articles (ring buffer)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

CACHE_FILE = Path("data/news_cache.json")
MAX_CACHE_SIZE = 500


def load_cache() -> list:
    """Load cached articles from disk."""
    if not CACHE_FILE.exists():
        return []
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return []


def save_cache(articles: list):
    """Save articles to disk, keeping only the most recent MAX_CACHE_SIZE."""
    # Sort by date, keep newest
    try:
        articles = sorted(
            articles,
            key=lambda x: x.get("publishedAt", ""),
            reverse=True
        )[:MAX_CACHE_SIZE]
        CACHE_FILE.write_text(json.dumps(articles, indent=2))
    except Exception as e:
        print(f"[NewsCache] Save error: {e}")


def get_latest_cached_date() -> Optional[str]:
    """
    Returns the publishedAt date of the most recent cached article.
    Used to avoid refetching articles we already have.
    """
    articles = load_cache()
    if not articles:
        return None
    dates = [a.get("publishedAt", "") for a in articles if a.get("publishedAt")]
    return max(dates) if dates else None


def merge_and_save(new_articles: list) -> list:
    """
    Merge new articles with cached ones.
    Deduplicates by URL.
    Returns only the genuinely new articles.
    """
    cached = load_cache()
    cached_urls = {a.get("url", "") for a in cached}

    # Filter to only articles not already in cache
    genuinely_new = [
        a for a in new_articles
        if a.get("url", "") not in cached_urls
    ]

    if genuinely_new:
        all_articles = cached + genuinely_new
        save_cache(all_articles)
        print(f"[NewsCache] +{len(genuinely_new)} new articles "
              f"(cache size: {min(len(all_articles), MAX_CACHE_SIZE)})")
    else:
        print(f"[NewsCache] No new articles (all {len(new_articles)} already cached)")

    return genuinely_new


def cache_stats() -> dict:
    articles = load_cache()
    return {
        "cached_articles": len(articles),
        "latest_date":     get_latest_cached_date(),
        "cache_file":      str(CACHE_FILE),
    }
