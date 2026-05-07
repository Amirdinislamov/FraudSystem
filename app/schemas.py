import math
from typing import Optional, List, Dict, Set
from datetime import datetime
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field, field_serializer


class DecisionStatus(str, Enum):
    APPROVED = "APPROVED"
    DECLINED = "DECLINED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class TransferPurpose(str, Enum):
    ME2ME = "ME2ME"
    FAMILY = "FAMILY"
    PURCHASE = "PURCHASE"
    DEBT_PAYOFF = "DEBT_PAYOFF"
    INVESTMENT = "INVESTMENT"
    CHARITY = "CHARITY"


class TransactionPayload(BaseModel):
    transaction_id: UUID
    client_id: str
    amount_usd: float = Field(..., gt=0)
    mcc_code: Optional[str] = None
    transfer_purpose: Optional[TransferPurpose] = None
    receiver_id: Optional[str] = None
    terminal_lat: Optional[float] = None
    terminal_lon: Optional[float] = None
    ip_address: Optional[str] = None
    device_id: Optional[str] = None
    timestamp: datetime


class MccStats(BaseModel):
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0


class ClientProfileState(BaseModel):
    client_id: str
    total_tx_count: int = 0
    first_tx_timestamp: Optional[datetime] = None
    last_tx_timestamp: Optional[datetime] = None
    last_geo_lat: Optional[float] = None
    last_geo_lon: Optional[float] = None
    # Время последней транзакции, у которой были GPS-координаты.
    # Используется GeoVelocityRule, чтобы корректно считать elapsed
    # между двумя geo-точками (а не до любой последней транзакции).
    last_geo_timestamp: Optional[datetime] = None
    recent_tx_hour_count: int = 0
    time_sin_sum: float = 0.0
    time_cos_sum: float = 0.0
    mcc_stats: Dict[str, MccStats] = Field(default_factory=dict)

    spent_last_24h: float = 0.0
    last_day_date: Optional[str] = None

    used_transfer_purposes: Set[str] = Field(default_factory=set)

    first_digit_counts: List[int] = Field(default_factory=lambda: [0] * 9)

    @field_serializer("used_transfer_purposes")
    def serialize_set(self, value: Set[str]) -> List[str]:
        return list(value)

    @property
    def mean_time_angle(self) -> Optional[float]:
        if self.total_tx_count == 0:
            return None
        return math.atan2(self.time_sin_sum, self.time_cos_sum)


class RuleResult(BaseModel):
    rule_name: str
    score: int
    raw_value: float = 0.0
    reason: Optional[str] = None


class FraudDecision(BaseModel):
    transaction_id: UUID
    decision: DecisionStatus
    risk_score: int
    triggered_rules: List[str]
    rule_scores: Dict[str, int] = Field(default_factory=dict)
    rule_raw_values: Dict[str, float] = Field(default_factory=dict)