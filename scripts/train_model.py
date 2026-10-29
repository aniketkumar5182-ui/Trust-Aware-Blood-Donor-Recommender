"""Train the donor-response model on a synthetic outreach history.

Usage:
    python -m scripts.train_model                # Random Forest (default)
    python -m scripts.train_model --model xgb    # needs: pip install xgboost
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from app.config import DOCS_DIR, MODEL_PATH
from app.ml_model import FEATURE_NAMES, ResponseModel
from app.simulation import generate_world


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["rf", "xgb"], default="rf")
    ap.add_argument("--donors", type=int, default=1500)
    ap.add_argument("--events", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Simulating {args.donors} donors and {args.events} outreach events (seed={args.seed}) ...")
    world = generate_world(args.donors, args.events, args.seed)
    X, y, ev = world.X, world.y, world.event_ids
    print(f"  -> {len(y):,} labelled rows, success rate {y.mean():.1%} ({time.time() - t0:.0f}s)")

    # Time-ordered split: train on the first 80 % of events, test on the last 20 %.
    cut = int(args.events * 0.8)
    tr, te = ev < cut, ev >= cut
    model = ResponseModel.fit(X[tr], y[tr], args.model, args.seed)
    metrics = model.evaluate(X[te], y[te])
    metrics["train_rows"] = int(tr.sum())
    metrics["test_rows"] = int(te.sum())
    model.metrics = metrics

    print("\nHold-out (last 20 % of time) metrics:")
    for k, v in metrics.items():
        print(f"  {k:>11}: {v:.4f}" if isinstance(v, float) else f"  {k:>11}: {v}")
    print("\nFeature importances:")
    for name, imp in model.feature_importances().items():
        print(f"  {name:<20} {imp:.3f}  {'#' * int(imp * 60)}")

    # Refit on all data for the shipped model, keep the hold-out metrics.
    final = ResponseModel.fit(X, y, args.model, args.seed)
    final.metrics = metrics
    final.save(MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")

    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "model_metrics.json").write_text(json.dumps({
        "model_type": args.model, "features": FEATURE_NAMES, "metrics": metrics,
        "feature_importances": final.feature_importances(),
        "note": "Metrics are on SYNTHETIC data and demonstrate the pipeline only.",
    }, indent=2))


if __name__ == "__main__":
    main()
