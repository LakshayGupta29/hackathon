"""
Social stream (replay).

There is no free live X/Twitter API, so the social source replays real
financial tweets from the project's training corpus (data/train_sentiment.csv,
~8.4k posts) in a shuffled order, without repeats until the corpus is
exhausted. A short curated list of S&P-100 cashtag posts is played first so
the demo touches the Module A index early. Labels are never read.

An optional data/financial_tweets.csv (column text/tweet/sentence) takes
precedence if present.
"""

import asyncio
import random
import re
from datetime import datetime, timezone

from src.config import data_path, env_float
from .queue_manager import make_raw_item, raw_queue

SAMPLE_TWEETS = [
    "$NVDA absolutely crushing it this quarter, AI demand is insane",
    "$TSLA deliveries miss again, this is getting concerning for investors",
    "Credit markets flashing red. $JPM $GS exposure to commercial real estate is scary",
    "Taiwan tensions escalating fast. $TSM $NVDA all going to feel this hard",
    "$CVX earnings beat on higher oil prices, strong cash flow generation continues",
    "Boeing $BA quality issues won't go away. Another whistleblower coming forward today",
    "$MSFT Azure growth reaccelerating strongly. AI monetization ahead of schedule",
    "$GS trading desk kills it in volatile quarter. Fixed income revenue up 40 percent",
    "$AMZN AWS reaccelerating, cloud spend recovering across enterprise customers",
    "$META ad revenue surging, Reels monetization exceeding all internal targets",
    "$UNH guidance cut spooks healthcare investors, medical costs running hot",
    "$LMT wins major defense contract as NATO allies boost spending",
]

URL_RE = re.compile(r"https?://\S+")
SEED = 42


def _clean(text: str) -> str:
    text = URL_RE.sub("", str(text))
    return re.sub(r"\s+", " ", text).strip(" -…")


def load_tweet_dataset() -> list[dict]:
    """Return [{"text": ...}] in replay order (curated first, then corpus)."""
    import pandas as pd

    corpus: list[str] = []
    for name in ("financial_tweets.csv", "train_sentiment.csv"):
        path = data_path(name)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            col = next((c for c in df.columns if c.lower() in {"text", "tweet", "sentence"}), None)
            if col:
                corpus = [_clean(t) for t in df[col].dropna().astype(str)]
                corpus = [t for t in corpus if len(t) >= 25]
                print(f"[SocialLoader] Replaying {len(corpus)} posts from {name}")
                break
        except Exception as exc:
            print(f"[SocialLoader] Could not read {name}: {exc}")

    random.Random(SEED).shuffle(corpus)
    ordered = list(SAMPLE_TWEETS) + corpus
    return [{"text": t} for t in dict.fromkeys(ordered)]


async def stream_tweets_to_queue(interval_seconds: float | None = None, batch_size: int = 3):
    interval = interval_seconds or env_float("SOCIAL_INTERVAL_SECONDS", 15.0)
    tweets = load_tweet_dataset()
    print(f"[SocialLoader] Social replay started ({len(tweets)} posts, every {interval:.0f}s)")
    position = 0
    while True:
        for _ in range(batch_size):
            tweet = tweets[position % len(tweets)]
            position += 1
            raw_queue.put_nowait(make_raw_item(
                text=tweet["text"],
                source="social:twitter",
                published_at=datetime.now(timezone.utc).isoformat(),
            ))
        await asyncio.sleep(interval)


def load_social_batch(n: int = 50) -> list:
    return load_tweet_dataset()[:n]
