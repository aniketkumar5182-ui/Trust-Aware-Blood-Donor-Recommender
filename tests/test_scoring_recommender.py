import math
from datetime import timedelta

import numpy as np
import pytest

from app.config import SCORE_WEIGHTS
from app.domain import RequestContext
from app.ml_model import FEATURE_NAMES, ResponseModel
from app.recommender import filter_candidates, recommend
from app.scoring import compute_components, get_weights, weighted_score

from .conftest import NOW


def test_weights_sum_to_one():
    for urgency, w in SCORE_WEIGHTS.items():
        assert math.isclose(sum(w.values()), 1.0), urgency


def test_urgency_shifts_weight_to_proximity_and_availability():
    n, c = SCORE_WEIGHTS["normal"], SCORE_WEIGHTS["critical"]
    assert c["proximity"] > n["proximity"] and c["availability"] > n["availability"]
    assert c["reliability"] < n["reliability"]


def test_override_weights_are_renormalised():
    w = get_weights("normal", {"reliability": 0.0})
    assert w["reliability"] == 0 and math.isclose(sum(w.values()), 1.0)


def test_contributions_sum_to_total(make_donor, ctx):
    comps = compute_components(make_donor(), ctx)
    total, parts = weighted_score(comps, get_weights("normal"))
    assert math.isclose(total, sum(parts.values()))
    assert 0 <= total <= 1


def test_better_history_scores_higher_reliability(make_donor, ctx):
    good = compute_components(make_donor(total_contacted=20, total_responded=19, total_accepted=15,
                                         total_donations=14), ctx)
    bad = compute_components(make_donor(total_contacted=20, total_responded=3, total_accepted=2,
                                        total_donations=0, last_active_at=NOW - timedelta(days=300)), ctx)
    assert good.reliability > bad.reliability + 0.3


def test_cold_start_donor_is_neutral(make_donor, ctx):
    c = compute_components(make_donor(total_contacted=0, total_responded=0, total_accepted=0, total_donations=0,
                                      avg_response_minutes=None), ctx)
    assert math.isclose(c.response_rate, 0.5) and math.isclose(c.completion_rate, 0.5)


def test_proximity_decreases_with_distance(make_donor, ctx):
    near = compute_components(make_donor(lat=28.47), ctx)
    far = compute_components(make_donor(lat=28.80), ctx)
    assert near.proximity > far.proximity and far.distance_km > near.distance_km


def test_availability_status_matters(make_donor, ctx):
    a = compute_components(make_donor(availability_status="available"), ctx).availability
    b = compute_components(make_donor(availability_status="busy"), ctx).availability
    assert a > b


def test_filters_report_reasons(make_donor, ctx):
    donors = [
        make_donor(donor_id=1),
        make_donor(donor_id=2, blood_group="A+"),                       # incompatible with O+
        make_donor(donor_id=3, lat=30.0),                               # outside radius
        make_donor(donor_id=4, last_donation_date=(NOW - timedelta(days=10)).date()),  # too recent
    ]
    fr = filter_candidates(donors, ctx, collect_excluded=True)
    assert [c.donor.donor_id for c in fr.candidates] == [1]
    assert fr.stats == {"considered": 4, "compatible": 3, "in_radius": 2, "eligible": 1}
    assert {e.donor_id for e in fr.excluded} == {3, 4}


def test_reliable_nearby_donor_outranks_unreliable_far_donor(make_donor, ctx):
    strong = make_donor(donor_id=1, lat=28.48)
    weak = make_donor(donor_id=2, lat=28.60, total_contacted=20, total_responded=2, total_accepted=1,
                      total_donations=0, availability_status="busy", last_active_at=NOW - timedelta(days=200))
    res = recommend([weak, strong], ctx, top_k=5)
    assert [r.donor.donor_id for r in res.recommendations] == [1, 2]
    assert res.recommendations[0].rank == 1 and res.recommendations[0].explanation


def test_top_k_and_determinism(make_donor, ctx):
    donors = [make_donor(donor_id=i, lat=28.47 + i * 0.002) for i in range(1, 30)]
    a = recommend(donors, ctx, top_k=5)
    b = recommend(list(reversed(donors)), ctx, top_k=5)
    assert len(a.recommendations) == 5
    assert [r.donor.donor_id for r in a.recommendations] == [r.donor.donor_id for r in b.recommendations]


def test_ml_mode_falls_back_without_model(make_donor, ctx):
    res = recommend([make_donor()], ctx, mode="hybrid", model=None)
    assert res.mode_used == "weighted" and res.warnings


def test_ml_and_hybrid_modes_with_model(make_donor, ctx):
    rng = np.random.default_rng(0)
    X = rng.random((400, len(FEATURE_NAMES)))
    y = (X[:, 0] + rng.normal(0, 0.2, 400) > 0.6).astype(int)
    model = ResponseModel.fit(X, y, "rf", seed=0)
    donors = [make_donor(donor_id=i, lat=28.47 + i * 0.003) for i in range(1, 12)]
    for mode in ("ml", "hybrid"):
        res = recommend(donors, ctx, top_k=5, mode=mode, model=model)
        assert res.mode_used == mode and all(r.ml_probability is not None for r in res.recommendations)
        scores = [r.score for r in res.recommendations]
        assert scores == sorted(scores, reverse=True)


def test_model_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.random((200, len(FEATURE_NAMES)))
    y = (X[:, 1] > 0.5).astype(int)
    m = ResponseModel.fit(X, y, "rf", seed=1)
    m.save(tmp_path / "m.joblib")
    m2 = ResponseModel.load(tmp_path / "m.joblib")
    assert np.allclose(m.predict_proba(X[:5]), m2.predict_proba(X[:5]))
    assert set(m2.feature_importances()) == set(FEATURE_NAMES)


def test_recommend_rejects_bad_mode(make_donor, ctx):
    with pytest.raises(ValueError):
        recommend([make_donor()], ctx, mode="nonsense")
