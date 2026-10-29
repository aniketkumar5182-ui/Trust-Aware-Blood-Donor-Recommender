"""Response-probability model (Random Forest, optional XGBoost).

The model predicts the probability that a contacted donor will actually come
and donate, from the same interpretable indicators used by the weighted score.
It is only trained on anonymised / synthetic interaction logs.

NOTE: models are stored with joblib (pickle). Only load model files that you
created yourself - never a file from an untrusted source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .config import URGENCY_CODE
from .domain import DonorProfile, RequestContext
from .scoring import ScoreComponents

FEATURE_NAMES = [
    "response_rate", "completion_rate", "activity", "experience",
    "status_score", "recent_activity", "speed",
    "distance_km", "compatibility", "eligibility_margin",
    "urgency_level", "age", "total_donations",
]


def build_feature_vector(donor: DonorProfile, ctx: RequestContext, c: ScoreComponents) -> list[float]:
    return [
        c.response_rate, c.completion_rate, c.activity, c.experience,
        c.status_score, c.recent_activity, c.speed,
        c.distance_km, c.compatibility, c.eligibility_margin,
        float(URGENCY_CODE[ctx.urgency]), float(donor.age), float(donor.total_donations),
    ]


class ResponseModel:
    def __init__(self, estimator, model_type: str = "rf", metrics: Optional[dict] = None,
                 feature_names: Optional[Sequence[str]] = None):
        self.estimator = estimator
        self.model_type = model_type
        self.metrics = metrics or {}
        self.feature_names = list(feature_names or FEATURE_NAMES)

    # ------------------------------------------------------------------ train
    @classmethod
    def fit(cls, X, y, model_type: str = "rf", seed: int = 42) -> "ResponseModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if model_type == "rf":
            est = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=25, max_features="sqrt",
                n_jobs=-1, random_state=seed,
            )
        elif model_type == "xgb":
            try:
                from xgboost import XGBClassifier  # optional dependency
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("xgboost is not installed: pip install xgboost") from exc
            est = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, eval_metric="logloss", random_state=seed,
            )
        else:
            raise ValueError("model_type must be 'rf' or 'xgb'")
        est.fit(X, y)
        return cls(est, model_type)

    # --------------------------------------------------------------- inference
    def predict_proba(self, rows) -> np.ndarray:
        X = np.asarray(rows, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.estimator.predict_proba(X)[:, 1]

    def evaluate(self, X, y) -> dict:
        y = np.asarray(y, dtype=int)
        p = np.clip(self.predict_proba(X), 1e-6, 1 - 1e-6)
        return {
            "n": int(len(y)),
            "base_rate": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p)),
        }

    def feature_importances(self) -> dict[str, float]:
        imp = getattr(self.estimator, "feature_importances_", None)
        if imp is None:
            return {}
        pairs = sorted(zip(self.feature_names, map(float, imp)), key=lambda t: -t[1])
        return dict(pairs)

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "estimator": self.estimator, "model_type": self.model_type,
            "metrics": self.metrics, "feature_names": self.feature_names,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "ResponseModel":
        blob = joblib.load(path)
        return cls(blob["estimator"], blob.get("model_type", "rf"),
                   blob.get("metrics"), blob.get("feature_names"))
