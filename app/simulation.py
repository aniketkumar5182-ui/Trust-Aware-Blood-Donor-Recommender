"""Synthetic world generator + offline evaluation of rankers.

Real donor-response data is sensitive and not publicly available, so the
prototype is trained and evaluated on a *simulated* donor population:

* every donor has a hidden "true reliability" r ~ Beta(3, 2);
* the recommender only sees noisy, history-based proxies of r (response rate,
  follow-through, recency ...) - never r itself;
* the chance that a contacted donor really donates depends on r, their current
  availability, distance and request urgency (see ``success_probability``);
* training rows use *point-in-time* features (history strictly before the event),
  so there is no label leakage.

Results on synthetic data demonstrate the pipeline and the relative value of
the indicators. They are NOT a claim about real-world performance.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional, Sequence

import numpy as np

from .compatibility import is_compatible
from .config import (
    BLOOD_GROUP_PROBS,
    BLOOD_GROUPS,
    DEFAULT_RADIUS_KM,
    URGENCY_LEVELS,
)
from .domain import DonorProfile, RequestContext, utcnow
from .eligibility import check_eligibility
from .geo import haversine_km
from .ml_model import FEATURE_NAMES, ResponseModel, build_feature_vector
from .recommender import blend_hybrid, filter_candidates, recommend
from .scoring import compute_components, get_weights, weighted_score

# (city, lat, lon, population weight) - Delhi NCR
CITIES = [
    ("Greater Noida", 28.4744, 77.5040, 0.22),
    ("Noida", 28.5355, 77.3910, 0.22),
    ("New Delhi", 28.6139, 77.2090, 0.26),
    ("Ghaziabad", 28.6692, 77.4538, 0.10),
    ("Gurugram", 28.4595, 77.0266, 0.12),
    ("Faridabad", 28.4089, 77.3178, 0.08),
]
_CITY_P = np.array([c[3] for c in CITIES])

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Rohan", "Ishaan", "Kabir", "Arjun", "Neha", "Priya", "Anjali",
               "Riya", "Sneha", "Pooja", "Kavya", "Meera", "Rahul", "Amit", "Sandeep", "Vikram", "Simran",
               "Nikhil", "Tanvi", "Karan", "Deepak", "Sana", "Imran", "Gurpreet", "Harsh", "Divya", "Manish"]
LAST_NAMES = ["Sharma", "Verma", "Singh", "Gupta", "Yadav", "Kumar", "Mishra", "Khan", "Joshi", "Patel",
              "Reddy", "Nair", "Das", "Chauhan", "Tiwari", "Pandey", "Mehta", "Bansal", "Kapoor", "Rao"]

AVAIL_TRUTH = {"available": 0.92, "flexible": 0.65, "busy": 0.30, "unavailable": 0.0}
URGENCY_BOOST = {"normal": 1.0, "urgent": 1.05, "critical": 1.10}


# --------------------------------------------------------------------------- truth model
@dataclass
class Outcome:
    responded: bool
    accepted: bool
    donated: bool
    minutes: Optional[float]


def success_probability(r: float, status: str, dist_km: float, urgency: str, noise: float = 1.0) -> float:
    """Hidden ground truth: P(contacted donor responds, accepts and shows up)."""
    avail = min(1.0, AVAIL_TRUTH.get(status, 0.0) * noise)
    p_resp = min(0.98, (0.2 + 0.78 * r) * avail * math.exp(-dist_km / 45.0) * URGENCY_BOOST[urgency])
    return p_resp * (0.55 + 0.40 * r) * (0.60 + 0.38 * r)


def sample_outcome(rng: np.random.Generator, r: float, donor: DonorProfile, dist_km: float, urgency: str) -> Outcome:
    noise = rng.uniform(0.75, 1.25)
    avail = min(1.0, AVAIL_TRUTH.get(donor.availability_status, 0.0) * noise)
    p_resp = min(0.98, (0.2 + 0.78 * r) * avail * math.exp(-dist_km / 45.0) * URGENCY_BOOST[urgency])
    responded = bool(rng.random() < p_resp)
    accepted = responded and bool(rng.random() < 0.55 + 0.40 * r)
    donated = accepted and bool(rng.random() < 0.60 + 0.38 * r)
    minutes = float(rng.exponential(10 + (1 - r) * 110)) if responded else None
    return Outcome(responded, accepted, donated, minutes)


def apply_outcome(d: DonorProfile, out: Outcome, when: datetime) -> None:
    d.total_contacted += 1
    if out.responded:
        d.total_responded += 1
        d.last_active_at = when
        prev = d.avg_response_minutes if d.avg_response_minutes is not None else out.minutes
        d.avg_response_minutes = prev + (out.minutes - prev) / d.total_responded
    if out.accepted:
        d.total_accepted += 1
    if out.donated:
        d.total_donations += 1
        d.last_donation_date = when.date()


# --------------------------------------------------------------------------- generation
def _pick_group(rng: np.random.Generator) -> str:
    return BLOOD_GROUPS[int(rng.choice(len(BLOOD_GROUPS), p=BLOOD_GROUP_PROBS))]


def random_location(rng: np.random.Generator) -> tuple[str, float, float]:
    name, lat, lon, _ = CITIES[int(rng.choice(len(CITIES), p=_CITY_P))]
    return name, lat + rng.normal(0, 0.05), lon + rng.normal(0, 0.05)


def generate_donors(n: int, rng: np.random.Generator, now: datetime, start_id: int = 1):
    """Return (donors, truth) where truth maps donor_id -> hidden reliability."""
    donors: list[DonorProfile] = []
    truth: dict[int, float] = {}
    for i in range(n):
        did = start_id + i
        r = float(rng.beta(3, 2))
        sex = "M" if rng.random() < 0.72 else "F"
        age = int(np.clip(rng.normal(32, 9), 18, 62))
        weight = float(np.clip(rng.normal(70 if sex == "M" else 58, 12 if sex == "M" else 10), 42, 120))
        city, lat, lon = random_location(rng)
        status = ("available", "flexible", "busy", "unavailable")[int(rng.choice(4, p=[0.42, 0.30, 0.20, 0.08]))]

        days_inactive = float(rng.exponential(10 + (1 - r) * 120))
        last_active = now - timedelta(days=days_inactive)

        contacted = int(rng.poisson(5))
        responded = int(rng.binomial(contacted, min(0.98, (0.2 + 0.78 * r) * 0.7)))
        accepted = int(rng.binomial(responded, 0.55 + 0.40 * r))
        donations = int(rng.binomial(accepted, 0.60 + 0.38 * r))
        last_donation = None
        if rng.random() < 0.5 or donations > 0:
            last_donation = (now - timedelta(days=int(rng.integers(10, 400)))).date()
        if last_donation is not None:
            donations = max(donations, 1)
            accepted = max(accepted, donations)
            responded = max(responded, accepted)
            contacted = max(contacted, responded)
        avg_min = None
        if responded > 0:
            avg_min = float(max(2.0, (10 + (1 - r) * 110) * rng.lognormal(0, 0.25)))

        donors.append(DonorProfile(
            donor_id=did, blood_group=_pick_group(rng), sex=sex, age=age, weight_kg=round(weight, 1),
            lat=float(lat), lon=float(lon), availability_status=status, last_donation_date=last_donation,
            last_active_at=last_active, total_contacted=contacted, total_responded=responded,
            total_accepted=accepted, total_donations=donations, avg_response_minutes=avg_min,
            display_name=f"{FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))]} {LAST_NAMES[int(rng.integers(len(LAST_NAMES)))]}",
            city=city,
        ))
        truth[did] = r
    return donors, truth


@dataclass
class World:
    donors: list[DonorProfile]
    truth: dict[int, float]
    now: datetime
    X: np.ndarray = field(default_factory=lambda: np.empty((0, len(FEATURE_NAMES))))
    y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    event_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    n_events: int = 0


def simulate_history(donors, truth, rng, n_events, start, end, per_event=20):
    by_group = {g: [d for d in donors if is_compatible(d.blood_group, g)] for g in BLOOD_GROUPS}
    rows_x, rows_y, rows_e = [], [], []
    span = (end - start).total_seconds()
    for e in range(n_events):
        t = start + timedelta(seconds=span * (e + 0.5) / n_events)
        group = _pick_group(rng)
        _, lat, lon = random_location(rng)
        urgency = URGENCY_LEVELS[int(rng.choice(3, p=[0.5, 0.3, 0.2]))]
        ctx = RequestContext(group, lat, lon, urgency, t, DEFAULT_RADIUS_KM)
        pool = by_group[group]
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(len(pool), per_event * 4), replace=False)
        chosen = []
        for i in idx:
            d = pool[int(i)]
            dist = haversine_km(d.lat, d.lon, lat, lon)
            if dist > ctx.max_radius_km:
                continue
            elig = check_eligibility(d, t)
            if not elig.eligible:
                continue
            chosen.append((d, dist, elig))
            if len(chosen) >= per_event:
                break
        for d, dist, elig in chosen:
            comps = compute_components(d, ctx, elig, dist)
            rows_x.append(build_feature_vector(d, ctx, comps))
            out = sample_outcome(rng, truth[d.donor_id], d, dist, urgency)
            rows_y.append(int(out.donated))
            rows_e.append(e)
            apply_outcome(d, out, t)
    return (np.array(rows_x, dtype=float).reshape(-1, len(FEATURE_NAMES)),
            np.array(rows_y, dtype=int), np.array(rows_e, dtype=int))


def generate_world(n_donors: int = 1500, n_events: int = 4000, seed: int = 42,
                   now: Optional[datetime] = None) -> World:
    """Create donors and (optionally) a year of simulated outreach history."""
    rng = np.random.default_rng(seed)
    end = now or datetime(2026, 5, 1, 12, 0, 0)
    start = end - timedelta(days=365) if n_events > 0 else end
    donors, truth = generate_donors(n_donors, rng, start)
    world = World(donors, truth, end, n_events=n_events)
    if n_events > 0:
        world.X, world.y, world.event_ids = simulate_history(donors, truth, rng, n_events, start, end)
    return world


# --------------------------------------------------------------------------- evaluation
KS = (1, 3, 5, 10)


def ranking_metrics(success_in_order: np.ndarray, total_success: int, ks: Sequence[int] = KS) -> dict[str, float]:
    s = np.asarray(success_in_order, dtype=float)
    out: dict[str, float] = {}
    for k in ks:
        top = s[:k]
        kk = len(top)
        out[f"precision@{k}"] = float(top.sum() / kk) if kk else 0.0
        out[f"recall@{k}"] = float(top.sum() / total_success)
        out[f"hit@{k}"] = float(top.sum() > 0)
        dcg = sum(top[i] / math.log2(i + 2) for i in range(kk))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(total_success, k)))
        out[f"ndcg@{k}"] = float(dcg / idcg) if idcg > 0 else 0.0
    nz = np.flatnonzero(s)
    first = int(nz[0]) + 1 if nz.size else len(s) + 1
    out["mrr"] = 1.0 / first
    out["contacts_to_first_success"] = float(first)
    return out


def _scorers(model: Optional[ResponseModel]) -> dict[str, Callable]:
    """Each scorer maps (candidates, ctx, rng) -> score array (higher = better)."""
    def w_scores(cands, ctx, overrides=None):
        w = get_weights(ctx.urgency, overrides)
        return np.array([weighted_score(c.components, w)[0] for c in cands])

    def ml_probs(cands, ctx):
        return model.predict_proba([build_feature_vector(c.donor, ctx, c.components) for c in cands])

    sc: dict[str, Callable] = {
        "random (blood-group match only)": lambda c, x, rng: rng.random(len(c)),
        "nearest first (distance only)": lambda c, x, rng: np.array([-k.components.distance_km for k in c]),
        "reliability only": lambda c, x, rng: np.array([k.components.reliability for k in c]),
        "trust score w/o reliability": lambda c, x, rng: w_scores(c, x, {"reliability": 0.0}),
        "trust score w/o availability": lambda c, x, rng: w_scores(c, x, {"availability": 0.0}),
        "trust score (weighted)": lambda c, x, rng: w_scores(c, x),
    }
    if model is not None:
        sc["ML only (Random Forest)"] = lambda c, x, rng: ml_probs(c, x)
        sc["hybrid (trust + ML)"] = lambda c, x, rng: blend_hybrid(w_scores(c, x), ml_probs(c, x))
    return sc


def evaluate_rankers(world: World, model: Optional[ResponseModel] = None, n_requests: int = 500,
                     seed: int = 2026, min_candidates: int = 15, timing_requests: int = 100) -> dict:
    """Paired offline evaluation: every ranker orders the same candidates, judged on the same
    sampled ground-truth outcomes (hidden reliability is never shown to the rankers)."""
    rng = np.random.default_rng(seed)
    scorers = _scorers(model)
    per_method: dict[str, list[dict]] = {k: [] for k in scorers}
    skipped_small = skipped_zero = used = 0
    cand_sizes: list[int] = []
    timing_ctxs: list[RequestContext] = []

    for _ in range(n_requests):
        group = _pick_group(rng)
        _, lat, lon = random_location(rng)
        urgency = URGENCY_LEVELS[int(rng.choice(3, p=[0.4, 0.35, 0.25]))]
        ctx = RequestContext(group, lat, lon, urgency, world.now, DEFAULT_RADIUS_KM)
        if len(timing_ctxs) < timing_requests:
            timing_ctxs.append(ctx)
        cands = filter_candidates(world.donors, ctx).candidates
        if len(cands) < min_candidates:
            skipped_small += 1
            continue
        success = np.array([
            sample_outcome(rng, world.truth[c.donor.donor_id], c.donor, c.components.distance_km, urgency).donated
            for c in cands
        ], dtype=bool)
        total = int(success.sum())
        if total == 0:
            skipped_zero += 1
            continue
        used += 1
        cand_sizes.append(len(cands))
        for name, scorer in scorers.items():
            scores = scorer(cands, ctx, rng)
            order = np.argsort(-scores, kind="stable")
            per_method[name].append(ranking_metrics(success[order], total))

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for name, rows in per_method.items():
        summary[name] = {}
        for metric in rows[0].keys() if rows else []:
            vals = np.array([r[metric] for r in rows], dtype=float)
            ci = 1.96 * vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            summary[name][metric] = {"mean": float(vals.mean()), "ci95": float(ci)}

    # retrieval time of the full production pipeline (filter + score + rank)
    times = []
    for ctx in timing_ctxs:
        t0 = time.perf_counter()
        recommend(world.donors, ctx, top_k=10, mode="weighted")
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "n_requests_generated": n_requests,
        "n_requests_evaluated": used,
        "skipped_too_few_candidates": skipped_small,
        "skipped_no_successful_donor": skipped_zero,
        "avg_candidates_per_request": float(np.mean(cand_sizes)) if cand_sizes else 0.0,
        "n_donors": len(world.donors),
        "retrieval_ms_mean": float(np.mean(times)) if times else 0.0,
        "retrieval_ms_p95": float(np.percentile(times, 95)) if times else 0.0,
        "methods": summary,
    }
