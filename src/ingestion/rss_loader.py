"""
RSS Loader
Pulls live financial headlines from free public RSS feeds.
No API key needed, no rate limits like NewsAPI's 100/day cap.
"""

import feedparser
import socket
from datetime import datetime

# Global timeout for feedparser's underlying urllib calls
socket.setdefaulttimeout(8)

RSS_FEEDS = {
    "news:cnbc":              "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "news:cnbc_markets":      "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "news:cnbc_economy":      "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "news:marketwatch":       "http://feeds.marketwatch.com/marketwatch/topstories/",
    "news:marketwatch_rt":    "http://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
    "news:investing":         "https://www.investing.com/rss/news_301.rss",
    "news:investing_stocks":  "https://www.investing.com/rss/news_25.rss",
    "news:seeking_alpha":     "https://seekingalpha.com/market_currents.xml",
    "news:ft_companies":      "https://www.ft.com/companies?format=rss",
    "news:wsj_markets":       "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
}


def fetch_rss_articles(max_per_feed: int = 25) -> list:
    """
    Fetch latest articles from all RSS feeds.
    Returns list of {text, source, url, publishedAt} dicts.
    Skips feeds that fail or return nothing — doesn't break the batch.
    """
    all_articles = []
    working_feeds = 0

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:max_per_feed]
            if not entries:
                print(f"[RSSLoader] {source}: 0 entries, skipping")
                continue

            working_feeds += 1
            for entry in entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = f"{title}. {summary}" if summary else title

                all_articles.append({
                    "text":        text[:500],
                    "source":      source,
                    "url":         entry.get("link", ""),
                    "publishedAt": entry.get("published",
                                              datetime.utcnow().isoformat()),
                })
            print(f"[RSSLoader] {source}: {len(entries)} entries")
        except Exception as e:
            print(f"[RSSLoader] {source} failed: {e}")
            continue

    # Deduplicate by title (some feeds overlap on syndicated stories)
    seen_titles = set()
    unique_articles = []
    for a in all_articles:
        title_key = a["text"][:60].lower()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(a)

    print(f"\n[RSSLoader] {working_feeds}/{len(RSS_FEEDS)} feeds working")
    print(f"[RSSLoader] {len(all_articles)} total, "
          f"{len(unique_articles)} after dedup")
    return unique_articles


if __name__ == "__main__":
    articles = fetch_rss_articles()
    print(f"\nSample headlines:")
    for a in articles[:15]:
        print(f"  [{a['source']}] {a['text'][:70]}")
