"""Plain dataclasses shared by the engine, simulation and API layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional


def utcnow() -> datetime:
    """Naive UTC 'now' (the whole code base uses naive UTC datetimes)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class DonorProfile:
    donor_id: int
    blood_group: str
    sex: str                      # 'M', 'F' or 'O'
    age: int
    weight_kg: float
    lat: float
    lon: float
    availability_status: str = "available"
    last_donation_date: Optional[date] = None
    last_active_at: Optional[datetime] = None
    total_contacted: int = 0      # times the donor was asked to donate
    total_responded: int = 0      # times the donor replied (accept or decline)
    total_accepted: int = 0       # times the donor accepted
    total_donations: int = 0      # times the donor actually donated
    avg_response_minutes: Optional[float] = None
    is_active: bool = True
    display_name: str = ""
    city: str = ""


@dataclass(frozen=True)
class RequestContext:
    patient_blood_group: str
    lat: float
    lon: float
    urgency: str = "normal"
    now: datetime = field(default_factory=utcnow)
    max_radius_km: float = 50.0
