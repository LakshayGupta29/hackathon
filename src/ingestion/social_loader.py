import asyncio
import random
from datetime import datetime
from pathlib import Path
from .queue_manager import raw_queue, make_raw_item

SAMPLE_TWEETS = [
    {"text": "$NVDA absolutely crushing it this quarter, AI demand is insane"},
    {"text": "$TSLA deliveries miss again, this is getting concerning for investors"},
    {"text": "Hearing $AAPL M&A deal could be announced this week, big news incoming"},
    {"text": "Credit markets flashing red. $JPM $GS exposure to commercial real estate is scary"},
    {"text": "Fed pivot coming? Futures pricing in 3 cuts this year. Bullish for equities"},
    {"text": "Taiwan tensions escalating fast. $TSM $ASML $NVDA all going to feel this hard"},
    {"text": "$CVX earnings beat on higher oil prices, strong cash flow generation continues"},
    {"text": "Boeing $BA quality issues won't go away. Another whistleblower coming forward today"},
    {"text": "Inflation data hotter than expected. Rate cuts delayed, markets selling off hard"},
    {"text": "$MSFT Azure growth reaccelerating strongly. AI monetization ahead of schedule"},
    {"text": "Oil supply cut extended by OPEC. $CVX massive beneficiary of this decision"},
    {"text": "Semiconductor shortage returning? Lead times stretching again $AMAT $ASML $TSM"},
    {"text": "$GS trading desk kills it in volatile quarter. Fixed income revenue up 40 percent"},
    {"text": "Consumer spending slowing sharply. $AMZN $HD both flagging very weak guidance"},
    {"text": "$MCD same store sales beat globally. Value menu driving traffic in tough economy"},
    {"text": "Emergency Fed meeting rumors circulating. Market volatility spiking $VIX"},
    {"text": "$AMZN AWS reaccelerating, cloud spend recovering across enterprise customers"},
    {"text": "China property market contagion fears spreading to global credit markets"},
    {"text": "$META ad revenue surging, Reels monetization exceeding all internal targets"},
    {"text": "Yield curve inversion deepening. Recession probability rising say economists"}
]

def load_tweet_dataset() -> list:
    tweets_path = Path("data/financial_tweets.csv")
    if tweets_path.exists():
        import pandas as pd
        try:
            df = pd.read_csv(tweets_path)
            text_col = next((c for c in df.columns 
                           if c.lower() in ["text","tweet","sentence"]), None)
            if text_col:
                tweets = [{"text": str(row[text_col])} 
                         for _, row in df.iterrows()]
                print(f"[SocialLoader] Loaded {len(tweets)} tweets from dataset")
                return tweets
        except Exception as e:
            print(f"[SocialLoader] Dataset load error: {e}")
    print("[SocialLoader] Using sample tweets")
    return SAMPLE_TWEETS

async def stream_tweets_to_queue(interval_seconds: int = 15):
    print("[SocialLoader] Starting social stream...")
    tweets = load_tweet_dataset()
    while True:
        batch = random.sample(tweets, min(5, len(tweets)))
        for tweet in batch:
            item = make_raw_item(
                text=tweet["text"],
                source="social:twitter",
                url="",
                published_at=datetime.utcnow().isoformat()
            )
            raw_queue.put_nowait(item)
        await asyncio.sleep(interval_seconds)

def load_social_batch(n: int = 50) -> list:
    tweets = load_tweet_dataset()
    return random.sample(tweets, min(n, len(tweets)))
