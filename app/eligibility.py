"""Basic donor eligibility screening (demo rules, not clinical advice)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .config import (
    ELIGIBILITY_MARGIN_DAYS,
    MAX_AGE,
    MIN_AGE,
    MIN_DONATION_INTERVAL_DAYS,
    MIN_WEIGHT_KG,
)
from .domain import DonorProfile


@dataclass
class EligibilityResult:
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    days_until_eligible: int = 0
    margin_score: float = 1.0   # 0 = just became eligible, 1 = comfortably rested


def check_eligibility(donor: DonorProfile, now: datetime) -> EligibilityResult:
    reasons: list[str] = []

    if not donor.is_active:
        reasons.append("Donor profile is inactive")
    if donor.availability_status == "unavailable":
        reasons.append("Donor has marked themselves unavailable")
    if donor.age < MIN_AGE or donor.age > MAX_AGE:
        reasons.append(f"Age {donor.age} outside {MIN_AGE}-{MAX_AGE}")
    if donor.weight_kg < MIN_WEIGHT_KG:
        reasons.append(f"Weight below {MIN_WEIGHT_KG:.0f} kg")

    days_until = 0
    margin = 1.0
    if donor.last_donation_date is not None:
        interval = MIN_DONATION_INTERVAL_DAYS.get(donor.sex, max(MIN_DONATION_INTERVAL_DAYS.values()))
        since = (now.date() - donor.last_donation_date).days
        if since < interval:
            days_until = interval - since
            margin = 0.0
            reasons.append(f"Donated {since} days ago; eligible again in {days_until} days")
        else:
            margin = min(1.0, (since - interval) / ELIGIBILITY_MARGIN_DAYS)

    return EligibilityResult(
        eligible=not reasons,
        reasons=reasons,
        days_until_eligible=days_until,
        margin_score=margin,
    )
