"""Trust-aware scoring: reliability, availability, proximity, compatibility, eligibility.

Every sub-score is in [0, 1]. The final score is a transparent weighted sum, so
each donor's rank can be explained component by component.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

from .compatibility import compatibility_score
from .config import (
    ACTIVITY_DECAY_DAYS,
    AVAILABILITY_WEIGHTS,
    DEFAULT_RESPONSE_MINUTES,
    EXPERIENCE_SATURATION,
    PROXIMITY_SCALE_KM,
    RECENT_ACTIVITY_DECAY_DAYS,
    RELIABILITY_WEIGHTS,
    RESPONSE_SPEED_DECAY_MIN,
    SCORE_WEIGHTS,
    STATUS_SCORE,
)
from .domain import DonorProfile, RequestContext
from .eligibility import EligibilityResult, check_eligibility
from .geo import haversine_km

COMPONENT_NAMES = ("reliability", "availability", "proximity", "compatibility", "eligibility")


@dataclass
class ScoreComponents:
    # reliability parts
    response_rate: float
    completion_rate: float
    activity: float
    experience: float
    reliability: float
    # availability parts
    status_score: float
    recent_activity: float
    speed: float
    availability: float
    # context parts
    distance_km: float
    proximity: float
    compatibility: float
    eligibility_margin: float

    def as_dict(self) -> dict:
        return asdict(self)


def _days_since(ts: Optional[datetime], now: datetime, default: float = 365.0) -> float:
    if ts is None:
        return default
    return max(0.0, (now - ts).total_seconds() / 86400.0)


def compute_components(
    donor: DonorProfile,
    ctx: RequestContext,
    elig: Optional[EligibilityResult] = None,
    distance_km: Optional[float] = None,
) -> ScoreComponents:
    now = ctx.now
    elig = elig or check_eligibility(donor, now)
    if distance_km is None:
        distance_km = haversine_km(donor.lat, donor.lon, ctx.lat, ctx.lon)

    # ---- reliability (historical behaviour) -------------------------------
    # Laplace-smoothed rates so that donors with no history sit at 0.5 instead
    # of 0 or 1 (a cold-start prior).
    response_rate = (donor.total_responded + 1) / (donor.total_contacted + 2)
    completion_rate = (donor.total_donations + 1) / (donor.total_accepted + 2)
    days_inactive = _days_since(donor.last_active_at, now)
    activity = math.exp(-days_inactive / ACTIVITY_DECAY_DAYS)
    experience = min(1.0, math.log1p(donor.total_donations) / math.log1p(EXPERIENCE_SATURATION))
    rw = RELIABILITY_WEIGHTS
    reliability = (rw["response_rate"] * response_rate + rw["completion_rate"] * completion_rate
                   + rw["activity"] * activity + rw["experience"] * experience)

    # ---- availability (current / predicted ability to respond) ------------
    status_score = STATUS_SCORE.get(donor.availability_status, 0.0)
    recent_activity = math.exp(-days_inactive / RECENT_ACTIVITY_DECAY_DAYS)
    avg_min = donor.avg_response_minutes if donor.avg_response_minutes is not None else DEFAULT_RESPONSE_MINUTES
    speed = math.exp(-avg_min / RESPONSE_SPEED_DECAY_MIN)
    aw = AVAILABILITY_WEIGHTS
    availability = aw["status"] * status_score + aw["recent_activity"] * recent_activity + aw["speed"] * speed

    # ---- context -----------------------------------------------------------
    proximity = math.exp(-distance_km / PROXIMITY_SCALE_KM)
    compat = compatibility_score(donor.blood_group, ctx.patient_blood_group)

    return ScoreComponents(
        response_rate=response_rate, completion_rate=completion_rate, activity=activity,
        experience=experience, reliability=reliability, status_score=status_score,
        recent_activity=recent_activity, speed=speed, availability=availability,
        distance_km=distance_km, proximity=proximity, compatibility=compat,
        eligibility_margin=elig.margin_score,
    )


def component_values(c: ScoreComponents) -> dict[str, float]:
    return {
        "reliability": c.reliability,
        "availability": c.availability,
        "proximity": c.proximity,
        "compatibility": c.compatibility,
        "eligibility": c.eligibility_margin,
    }


def get_weights(urgency: str, overrides: Optional[dict[str, float]] = None) -> dict[str, float]:
    weights = dict(SCORE_WEIGHTS[urgency])
    if overrides:
        weights.update(overrides)
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()} if total > 0 else weights
    return weights


def weighted_score(c: ScoreComponents, weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Return (total, per-component contribution)."""
    values = component_values(c)
    contributions = {name: weights.get(name, 0.0) * values[name] for name in COMPONENT_NAMES}
    return sum(contributions.values()), contributions


def explain(c: ScoreComponents, donor: DonorProfile) -> list[str]:
    """Short human-readable reasons shown next to each recommended donor."""
    notes: list[str] = []
    if c.reliability >= 0.70:
        notes.append(f"Strong history: {c.response_rate:.0%} response rate, {c.completion_rate:.0%} follow-through")
    elif c.reliability < 0.45:
        notes.append("Limited or weak response history")
    if c.availability >= 0.70:
        notes.append(f"Likely reachable now (status: {donor.availability_status})")
    elif c.availability < 0.45:
        notes.append("May be slow to respond")
    notes.append(f"{c.distance_km:.1f} km away")
    if c.compatibility == 1.0:
        notes.append("Exact blood-group match")
    elif donor.blood_group == "O-":
        notes.append("Universal donor (O-) - scarce, used as a fallback")
    else:
        notes.append("Compatible (not identical) group")
    return notes
