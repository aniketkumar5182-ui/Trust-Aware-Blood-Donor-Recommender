import numpy as np

from app.ml_model import FEATURE_NAMES, ResponseModel
from app.simulation import evaluate_rankers, generate_world, ranking_metrics, success_probability


def test_world_is_deterministic_and_well_formed():
    a = generate_world(200, 200, seed=3)
    b = generate_world(200, 200, seed=3)
    assert a.X.shape[1] == len(FEATURE_NAMES) and len(a.X) == len(a.y) == len(a.event_ids) > 500
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)
    assert set(np.unique(a.y)) <= {0, 1}
    for d in a.donors:
        assert d.total_contacted >= d.total_responded >= d.total_accepted >= d.total_donations >= 0


def test_no_history_world_has_no_rows():
    w = generate_world(50, 0, seed=1)
    assert len(w.X) == 0 and len(w.donors) == 50


def test_success_probability_monotonic():
    base = success_probability(0.6, "available", 5, "normal")
    assert success_probability(0.9, "available", 5, "normal") > base      # more reliable
    assert success_probability(0.6, "busy", 5, "normal") < base           # less available
    assert success_probability(0.6, "available", 40, "normal") < base     # farther
    assert success_probability(0.6, "unavailable", 5, "normal") == 0


def test_ranking_metrics_known_values():
    m = ranking_metrics(np.array([1, 0, 1, 0, 0]), total_success=2, ks=(1, 3, 5))
    assert m["precision@1"] == 1.0 and abs(m["precision@3"] - 2 / 3) < 1e-9
    assert m["recall@3"] == 1.0 and m["hit@1"] == 1.0 and m["mrr"] == 1.0
    worst = ranking_metrics(np.array([0, 0, 0, 1]), total_success=1, ks=(3,))
    assert worst["precision@3"] == 0 and worst["mrr"] == 0.25 and worst["contacts_to_first_success"] == 4


def test_trust_score_beats_random_and_nearest():
    world = generate_world(800, 1500, seed=11)
    cut = int(1500 * 0.8)
    tr = world.event_ids < cut
    model = ResponseModel.fit(world.X[tr], world.y[tr], "rf", seed=11)
    res = evaluate_rankers(world, model, n_requests=250, seed=5, timing_requests=10)
    m = res["methods"]
    p5 = lambda name: m[name]["precision@5"]["mean"]  # noqa: E731
    assert res["n_requests_evaluated"] > 100
    assert p5("trust score (weighted)") > p5("random (blood-group match only)") + 0.05
    assert p5("trust score (weighted)") > p5("nearest first (distance only)")
    assert p5("ML only (Random Forest)") > p5("random (blood-group match only)")
