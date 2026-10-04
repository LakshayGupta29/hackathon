import os
import asyncio
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from .queue_manager import raw_queue, make_raw_item

from src.ingestion.news_cache import merge_and_save, load_cache, cache_stats

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

FINANCIAL_QUERIES = [
    "stock market Federal Reserve",
    "earnings merger acquisition",
    "geopolitical oil supply chain",
    "bank credit default inflation"
]

SAMPLE_ARTICLES = [
    {
        "title": "Federal Reserve signals potential rate cuts amid cooling inflation",
        "description": "Fed Chair Powell indicated the central bank may reduce interest rates as inflation data shows sustained decline toward the 2% target.",
        "url": "https://example.com/fed-rate-cuts",
        "publishedAt": datetime.utcnow().isoformat(),
        "source": {"name": "Reuters"}
    },
    {
        "title": "NVIDIA reports record quarterly earnings driven by AI chip demand",
        "description": "NVIDIA Corporation posted record revenue beating analyst expectations as demand for AI training chips continues to surge globally.",
        "url": "https://example.com/nvidia-earnings",
        "publishedAt": datetime.utcnow().isoformat(),
        "source": {"name": "Bloomberg"}
    },
    {
        "title": "Geopolitical tensions in Taiwan Strait rattle semiconductor stocks",
        "description": "Military exercises near Taiwan sparked fears of supply chain disruptions, sending shares of TSMC, ASML and NVIDIA lower.",
        "url": "https://example.com/taiwan-crisis",
        "publishedAt": datetime.utcnow().isoformat(),
        "source": {"name": "Financial Times"}
    },
    {
        "title": "JPMorgan warns of credit market stress as corporate defaults rise",
        "description": "JPMorgan analysts flagged rising corporate default rates in leveraged loan markets, warning of potential credit spread widening.",
        "url": "https://example.com/jpmorgan-credit",
        "publishedAt": datetime.utcnow().isoformat(),
        "source": {"name": "Wall Street Journal"}
    },
    {
        "title": "Apple announces major acquisition of AI startup for 3.2 billion",
        "description": "Apple confirmed the acquisition expanding its artificial intelligence capabilities ahead of new iPhone launch.",
        "url": "https://example.com/apple-acquisition",
        "publishedAt": datetime.utcnow().isoformat(),
        "source": {"name": "TechCrunch"}
    },
    {
        "title": "Oil prices surge as OPEC extends production cuts through year end",
        "description": "Crude oil prices jumped after OPEC and allies agreed to extend supply cuts, boosting shares of Chevron and energy sector broadly.",
        "url": "https://example.com/opec-cuts",
        "publishedAt": datetime.utcnow().isoformat(),
        "source": {"name": "Reuters"}
    },
    {
        "title": "Boeing faces fresh scrutiny over manufacturing defects and safety concerns",
        "description": "Federal investigators opened new probe into Boeing manufacturing processes following reports of quality control failures on 737 MAX aircraft.",
        "url": "https://example.com/boeing-safety",
        "publishedAt": datetime.utcnow().isoformat(),
        "source": {"name": "Bloomberg"}
    },
    {
        "title": "Microsoft Azure cloud revenue grows 35 percent beating estimates",
        "description": "Microsoft reported strong quarterly results driven by Azure cloud and AI services, with commercial bookings hitting record levels.",
        "url": "https://example.com/msft-earnings",
        "publishedAt": datetime.utcnow().isoformat(),
        "source": {"name": "CNBC"}
    },
    {
        "title": "Goldman Sachs warns of recession risk as yield curve inverts further",
        "description": "Goldman Sachs economists raised recession probability to 35 percent citing persistent yield curve inversion and tightening credit conditions.",
        "url": "https://example.com/goldman-recession",
        "publishedAt": datetime.utcnow().isoformat(),
        "source": {"name": "Financial Times"}
    },
    {
        "title": "Taiwan semiconductor exports hit record high despite geopolitical risks",
        "description": "TSMC reported record monthly revenue as global chip demand remains strong despite ongoing tensions in the Taiwan Strait region.",
        "url": "https://example.com/tsmc-exports",
        "publishedAt": datetime.utcnow().isoformat(),
        "source": {"name": "Reuters"}
    }
]

def fetch_news_articles(query: str = "", days_back: int = 7) -> list:
    """
    Fetch articles from NewsAPI with cache fallback.
    On rate limit or error: returns cached articles so
    the system keeps running without burning quota.
    """
    if not NEWSAPI_KEY or NEWSAPI_KEY == "your_newsapi_key_here":
        print("[NewsLoader] No API key, using cache + sample data")
        cached = load_cache()
        return cached if cached else SAMPLE_ARTICLES

    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=NEWSAPI_KEY)
        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        response = client.get_everything(
            q=query or "financial markets",
            from_param=from_date,
            language="en",
            sort_by="publishedAt",
            page_size=20
        )
        articles = response.get("articles", [])

        if articles:
            # Merge with cache, get only genuinely new ones
            new_articles = merge_and_save(articles)
            print(f"[NewsLoader] {len(new_articles)} new / {len(articles)} fetched")
            return articles  # return all for processing
        else:
            # API returned nothing, use cache
            cached = load_cache()
            print(f"[NewsLoader] Empty response, using {len(cached)} cached articles")
            return cached if cached else SAMPLE_ARTICLES

    except Exception as e:
        # Rate limited or error: use cache silently
        cached = load_cache()
        if cached:
            print(f"[NewsLoader] Rate limited, using {len(cached)} cached articles")
            return cached
        print(f"[NewsLoader] Error + no cache: {e}, using samples")
        return SAMPLE_ARTICLES

async def stream_news_to_queue(interval_seconds: int = 3600):
    print("[NewsLoader] Starting news stream...")
    while True:
        for query in FINANCIAL_QUERIES:
            articles = fetch_news_articles(query, days_back=1)
            for article in articles:
                title = article.get("title", "")
                desc = article.get("description", "")
                if not title:
                    continue
                text = f"{title}. {desc}" if desc else title
                item = make_raw_item(
                    text=text,
                    source=f"news:{article.get('source',{}).get('name','unknown')}",
                    url=article.get("url", ""),
                    published_at=article.get("publishedAt", "")
                )
                raw_queue.put_nowait(item)
        await asyncio.sleep(interval_seconds)

def load_news_batch() -> list:
    """
    Load a batch for initial seeding.
    Uses cache first to avoid burning API quota on restart.
    """
    # Cache disabled temporarily - causes novelty scorer pollution
    # TODO: re-enable with novelty-aware seeding strategy
    # No cache, fetch fresh
    articles = []
    for query in FINANCIAL_QUERIES:
        articles.extend(fetch_news_articles(query, days_back=7))
    return articles
