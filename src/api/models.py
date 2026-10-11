"""
API Models: Pydantic request/response schemas.

Validation errors return HTTP 422 with a clear message, and the schemas
drive the auto-generated docs at /docs.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.config import EVENT_CLASSES


class HealthResponse(BaseModel):
    status: str = "ok"
    total_processed: int
    events_in_store: int
    duplicates_suppressed: int = 0
    stress_tests_triggered: int = 0
    active_subscribers: int
    started_at: str
    portfolio_value: float
    last_stress_loss: float = 0.0
    last_stress_loss_pct: float = 0.0
    sentiment_model: str = ""
    sentiment_model_fallback: bool = False
    live_feeds_enabled: bool = True
    news_source: str = ""
    queue_size: int = 0


class PortfolioResponse(BaseModel):
    portfolio_name: str = "Synthetic Wholesale Banking Portfolio"
    total_value: float
    currency: str = "USD"
    mode: str = "baseline"
    position_count: int
    positions: list[dict]


class StressTestRequest(BaseModel):
    event_class: str = Field(
        default="Geopolitical",
        description="One of: " + ", ".join(EVENT_CLASSES),
    )
    impact_score: float = Field(default=7.5, ge=1.0, le=10.0, description="Impact severity 1-10")
    affected_tickers: list[str] = Field(default=["TSM", "NVDA", "ASML"])
    affected_sectors: list[str] = Field(
        default=[],
        description="Leave empty to derive sectors from the risk graph",
    )
    sentiment_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    mode: Literal["baseline", "cumulative"] = "baseline"

    @field_validator("event_class")
    @classmethod
    def _valid_class(cls, value: str) -> str:
        if value not in EVENT_CLASSES:
            raise ValueError(f"event_class must be one of {EVENT_CLASSES}")
        return value

    @field_validator("affected_tickers")
    @classmethod
    def _upper(cls, value: list[str]) -> list[str]:
        return [t.strip().upper() for t in value if t and t.strip()]


class BlackSwanRequest(BaseModel):
    headline: str = Field(
        default="China blockade of Taiwan Strait sparks military standoff; TSMC, NVIDIA and Apple shares plunge as sanctions loom",
        min_length=5,
        max_length=1000,
    )
    source: str = Field(default="news:reuters")


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000)
    source: str = Field(default="news:unknown")
    trigger_downstream: bool = Field(
        default=False,
        description="If true, the event also updates Module A and may trigger Module B",
    )
