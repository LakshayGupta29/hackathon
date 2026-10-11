"""
RSS Loader.

Pulls live financial headlines from free public RSS feeds (no API key, no
daily quota). Feeds are fetched in parallel with httpx (per-request timeout
and a browser-like User-Agent, which several publishers require), then
parsed with feedparser. A failing feed is skipped, never fatal.
"""

from calendar import timegm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import os
import re

import feedparser
import httpx

RSS_FEEDS = {
    "news:cnbc": "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "news:cnbc_economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "news:cnbc_finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "news:marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "news:marketwatch_rt": "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
    "news:investing": "https://www.investing.com/rss/news_301.rss",
    "news:investing_stocks": "https://www.investing.com/rss/news_25.rss",
    "news:seeking_alpha": "https://seekingalpha.com/market_currents.xml",
    "news:ft_companies": "https://www.ft.com/companies?format=rss",
    "news:wsj_markets": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
}

REQUEST_TIMEOUT = 8.0
# Some feeds keep years-old items; anything older than this is not "live" news.
MAX_AGE_HOURS = float(os.getenv("RSS_MAX_AGE_HOURS", "72"))
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}
TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    text = html.unescape(TAG_RE.sub(" ", str(text or "")))
    return re.sub(r"\s+", " ", text).strip()


def _entry_time(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return datetime.fromtimestamp(timegm(parsed), tz=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                continue
    return None


def _fetch_feed(source: str, url: str, max_per_feed: int) -> tuple[str, list[dict], str | None]:
    try:
        response = httpx.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception as exc:
        return source, [], str(exc)

    articles = []
    now = datetime.now(timezone.utc)
    for entry in feed.entries[:max_per_feed]:
        title = _clean(entry.get("title", ""))
        if not title:
            continue
        published = _entry_time(entry)
        if published is not None and (now - published).total_seconds() > MAX_AGE_HOURS * 3600:
            continue  # stale item still served by the feed
        summary = _clean(entry.get("summary", ""))
        text = f"{title}. {summary}" if summary and summary.lower() != title.lower() else title
        articles.append({
            "text": text[:500],
            "source": source,
            "url": entry.get("link", ""),
            "publishedAt": (published or now).isoformat(),
        })
    return source, articles, None


def fetch_rss_articles(max_per_feed: int = 25, verbose: bool = True) -> list:
    """Fetch the latest articles from all feeds, de-duplicated by headline."""
    unique_feeds = {}
    for source, url in RSS_FEEDS.items():
        unique_feeds.setdefault(url, source)

    all_articles, working = [], 0
    with ThreadPoolExecutor(max_workers=len(unique_feeds)) as pool:
        futures = [
            pool.submit(_fetch_feed, source, url, max_per_feed)
            for url, source in unique_feeds.items()
        ]
        for future in futures:
            source, articles, error = future.result()
            if error:
                if verbose:
                    print(f"[RSSLoader] {source} failed: {error[:120]}")
                continue
            if articles:
                working += 1
            all_articles.extend(articles)

    seen, unique_articles = set(), []
    for article in all_articles:
        key = re.sub(r"\W+", " ", article["text"][:80].lower()).strip()
        if key not in seen:
            seen.add(key)
            unique_articles.append(article)

    if verbose:
        print(f"[RSSLoader] {working}/{len(unique_feeds)} feeds working, "
              f"{len(all_articles)} articles, {len(unique_articles)} after dedup")
    return unique_articles


def load_offline_news(limit: int = 40) -> list:
    """
    Offline fallback when no RSS feed is reachable (blocked network at a
    venue, publisher outage): replay real historical headlines bundled in
    data/ so the live dashboard is never empty.
    """
    import json
    from src.config import data_path

    articles = []
    for name in ("gdelt_curated_events.json", "master_tuning_set.json", "master_holdout_set.json"):
        path = data_path(name)
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        events = payload["events"] if isinstance(payload, dict) else payload
        for event in events:
            articles.append({
                "text": event["headline"],
                "source": event.get("source", "news:unknown"),
                "url": "",
                "publishedAt": event.get("date", ""),
            })
    seen, unique = set(), []
    for article in articles:
        if article["text"] not in seen:
            seen.add(article["text"])
            unique.append(article)
    return unique[:limit]


if __name__ == "__main__":
    articles = fetch_rss_articles()
    print("\nSample headlines:")
    for a in articles[:15]:
        print(f"  [{a['source']}] {a['text'][:90]}")
