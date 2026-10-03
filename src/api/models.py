"""
API Models
Pydantic schemas for all request and response bodies.

Benefits:
  - Automatic validation (wrong types → 422 error with clear message)
  - Auto-generated docs at /docs and /redoc
  - Clear contract between API and dashboard
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ─── Responses ────────────────────────────────────────────────

class ImpactBreakdown(BaseModel):
    sentiment_magnitude:  float
    event_severity_prior: float
    novelty_score:        float
    entity_count_factor:  float
    source_authority:     float


class EventResponse(BaseModel):
    event_id:            str
    timestamp:           str
    headline:            str
    source:              str
    sentiment_score:     float
    sentiment_label:     str
    event_class:         str
    event_confidence:    float
    novelty_score:       float
    novelty_label:       str
    impact_score:        float
    impact_breakdown:    ImpactBreakdown
    affected_tickers:    list[str]
    affected_sectors:    list[str]
    triggering_sentence: str
    latency_ms:          float


class PositionPnL(BaseModel):
    id:            str
    obligor:       str
    ticker:        Optional[str]
    sector:        Optional[str]
    type:          str
    value_before:  float
    value_after:   float
    pnl:           float
    exposure_type: str


class AttributionItem(BaseModel):
    sector:       Optional[str] = None
    type:         Optional[str] = None
    loss:         float
    pct_of_total: float


class StressTestResponse(BaseModel):
    event_class:      str
    impact_score:     float
    portfolio_before: float
    portfolio_after:  float
    total_loss:       float
    loss_pct:         float
    shocks_applied:   dict
    positions:        list[PositionPnL]
    by_sector:        list[dict]
    by_asset_type:    list[dict]
    top_positions:    list[dict]
    summary_sentence: str


class PortfolioResponse(BaseModel):
    total_value:    float
    currency:       str = "USD"
    position_count: int
    positions:      list[dict]


class HealthResponse(BaseModel):
    status:            str = "ok"
    total_processed:   int
    events_in_store:   int
    active_subscribers: int
    portfolio_value:   float
    started_at:        str


# ─── Requests ─────────────────────────────────────────────────

class StressTestRequest(BaseModel):
    event_class:      str = Field(
        default="Geopolitical",
        description="One of: Geopolitical, Macroeconomic, Credit Event, "
                    "Merger/Acquisition, Product Launch"
    )
    impact_score:     float = Field(
        default=7.5,
        ge=1.0, le=10.0,
        description="Impact severity 1-10"
    )
    affected_tickers: list[str] = Field(
        default=["TSM", "NVDA", "ASML"],
        description="List of S&P100 tickers affected"
    )
    affected_sectors: list[str] = Field(
        default=["Technology", "Energy", "Financials"],
        description="List of affected sectors"
    )


class BlackSwanRequest(BaseModel):
    headline: str = Field(
        default="Fed announces emergency rate hike of 100 basis points",
        description="Injected headline for black swan simulation"
    )
    source: str = Field(
        default="news:reuters",
        description="Source identifier"
    )
