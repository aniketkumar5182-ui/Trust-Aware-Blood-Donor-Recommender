"""Recommendation pipeline: filter -> score -> rank.

1. Hard filters: blood-group compatibility, search radius, basic eligibility.
2. Soft ranking: weighted trust score (and optionally an ML probability).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .compatibility import is_compatible
from .config import HYBRID_ML_WEIGHT
from .domain import DonorProfile, RequestContext
from .eligibility import check_eligibility
from .geo import haversine_km
from .ml_model import ResponseModel, build_feature_vector
from .scoring import ScoreComponents, compute_components, explain, get_weights, weighted_score

MODES = ("weighted", "ml", "hybrid")


@dataclass
class Candidate:
    donor: DonorProfile
    components: ScoreComponents


@dataclass
class Excluded:
    donor_id: int
    reasons: list[str]


@dataclass
class FilterResult:
    candidates: list[Candidate]
    excluded: list[Excluded]
    stats: dict[str, int]


@dataclass
class Recommendation:
    rank: int
    donor: DonorProfile
    score: float
    weighted_score: float
    ml_probability: Optional[float]
    components: ScoreComponents
    contributions: dict[str, float]
    explanation: list[str]


@dataclass
class RecommendationResult:
    recommendations: list[Recommendation]
    excluded: list[Excluded]
    stats: dict
    mode_requested: str
    mode_used: str
    weights: dict[str, float]
    elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


def filter_candidates(donors: Sequence[DonorProfile], ctx: RequestContext,
                      collect_excluded: bool = False) -> FilterResult:
    stats = {"considered": 0, "compatible": 0, "in_radius": 0, "eligible": 0}
    candidates: list[Candidate] = []
    excluded: list[Excluded] = []
    for d in donors:
        stats["considered"] += 1
        if not is_compatible(d.blood_group, ctx.patient_blood_group):
            continue
        stats["compatible"] += 1
        dist = haversine_km(d.lat, d.lon, ctx.lat, ctx.lon)
        if dist > ctx.max_radius_km:
            if collect_excluded:
                excluded.append(Excluded(d.donor_id, [f"Outside {ctx.max_radius_km:.0f} km radius ({dist:.0f} km)"]))
            continue
        stats["in_radius"] += 1
        elig = check_eligibility(d, ctx.now)
        if not elig.eligible:
            if collect_excluded:
                excluded.append(Excluded(d.donor_id, elig.reasons))
            continue
        stats["eligible"] += 1
        candidates.append(Candidate(d, compute_components(d, ctx, elig, dist)))
    return FilterResult(candidates, excluded, stats)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo, hi = values.min(), values.max()
    return np.zeros_like(values) if hi - lo < 1e-12 else (values - lo) / (hi - lo)


def blend_hybrid(weighted: np.ndarray, ml_prob: np.ndarray, ml_weight: float = HYBRID_ML_WEIGHT) -> np.ndarray:
    """Blend the two signals after min-max normalisation within the candidate set."""
    return (1 - ml_weight) * minmax(weighted) + ml_weight * minmax(ml_prob)


def recommend(
    donors: Sequence[DonorProfile],
    ctx: RequestContext,
    top_k: int = 10,
    mode: str = "weighted",
    model: Optional[ResponseModel] = None,
    include_excluded: bool = False,
    weight_overrides: Optional[dict[str, float]] = None,
) -> RecommendationResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    t0 = time.perf_counter()
    warnings: list[str] = []

    fr = filter_candidates(donors, ctx, collect_excluded=include_excluded)
    weights = get_weights(ctx.urgency, weight_overrides)

    mode_used = mode
    if mode != "weighted" and model is None:
        mode_used = "weighted"
        warnings.append("No trained ML model is available; fell back to the weighted trust score.")

    scored = [weighted_score(c.components, weights) for c in fr.candidates]
    w_scores = np.array([s[0] for s in scored]) if scored else np.array([])

    probs: Optional[np.ndarray] = None
    if mode_used != "weighted" and fr.candidates:
        feats = [build_feature_vector(c.donor, ctx, c.components) for c in fr.candidates]
        probs = model.predict_proba(feats)  # type: ignore[union-attr]

    if mode_used == "weighted" or not fr.candidates:
        final = w_scores
    elif mode_used == "ml":
        final = probs
    else:
        final = blend_hybrid(w_scores, probs)

    order = sorted(range(len(fr.candidates)),
                   key=lambda i: (-final[i], fr.candidates[i].components.distance_km, fr.candidates[i].donor.donor_id))
    recs: list[Recommendation] = []
    for rank, i in enumerate(order[:top_k], start=1):
        cand = fr.candidates[i]
        recs.append(Recommendation(
            rank=rank, donor=cand.donor, score=float(final[i]), weighted_score=float(w_scores[i]),
            ml_probability=float(probs[i]) if probs is not None else None,
            components=cand.components, contributions=scored[i][1],
            explanation=explain(cand.components, cand.donor),
        ))

    elapsed = (time.perf_counter() - t0) * 1000
    return RecommendationResult(
        recommendations=recs, excluded=fr.excluded, stats=fr.stats,
        mode_requested=mode, mode_used=mode_used, weights=weights,
        elapsed_ms=elapsed, warnings=warnings,
    )
