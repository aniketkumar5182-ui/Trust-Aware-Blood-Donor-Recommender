"""Central configuration: paths, scoring weights and eligibility rules.

All medical thresholds below are *demo defaults for an academic prototype*.
They are NOT clinical guidance; final eligibility must always be decided by
authorised healthcare personnel.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

try:  # optional: load variables from a local .env file (never commit it)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
MODEL_PATH = MODEL_DIR / "response_model.joblib"
DOCS_DIR = BASE_DIR / "docs"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'blood_donor.db'}")

# --- Security -------------------------------------------------------------
# Set SECRET_KEY in the environment for any real deployment. If it is missing
# a random key is generated, which simply invalidates tokens on every restart.
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "120"))
PBKDF2_ITERATIONS = int(os.getenv("PBKDF2_ITERATIONS", "600000"))
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# --- Blood groups ---------------------------------------------------------
BLOOD_GROUPS = ("O+", "B+", "A+", "AB+", "O-", "B-", "A-", "AB-")
# Approximate Indian population distribution (used only for synthetic data).
BLOOD_GROUP_PROBS = (0.36, 0.31, 0.21, 0.06, 0.02, 0.015, 0.015, 0.01)

# --- Urgency & scoring ----------------------------------------------------
URGENCY_LEVELS = ("normal", "urgent", "critical")
URGENCY_CODE = {"normal": 0, "urgent": 1, "critical": 2}

# Weights per urgency level (each row sums to 1.0). When a request becomes
# more urgent, proximity and current availability matter more.
SCORE_WEIGHTS = {
    "normal": {"reliability": 0.30, "availability": 0.25, "proximity": 0.20,
               "compatibility": 0.15, "eligibility": 0.10},
    "urgent": {"reliability": 0.28, "availability": 0.30, "proximity": 0.25,
               "compatibility": 0.10, "eligibility": 0.07},
    "critical": {"reliability": 0.25, "availability": 0.33, "proximity": 0.30,
                 "compatibility": 0.07, "eligibility": 0.05},
}

# Reliability sub-weights (sum = 1)
RELIABILITY_WEIGHTS = {"response_rate": 0.35, "completion_rate": 0.30,
                       "activity": 0.20, "experience": 0.15}
# Availability sub-weights (sum = 1)
AVAILABILITY_WEIGHTS = {"status": 0.45, "recent_activity": 0.30, "speed": 0.25}

STATUS_SCORE = {"available": 1.0, "flexible": 0.7, "busy": 0.3, "unavailable": 0.0}
AVAILABILITY_STATUSES = tuple(STATUS_SCORE)

ACTIVITY_DECAY_DAYS = 60.0       # long-term activity (reliability)
RECENT_ACTIVITY_DECAY_DAYS = 14.0  # short-term activity (availability)
RESPONSE_SPEED_DECAY_MIN = 90.0
DEFAULT_RESPONSE_MINUTES = 45.0
EXPERIENCE_SATURATION = 10        # donations after which experience score = 1
PROXIMITY_SCALE_KM = 20.0
DEFAULT_RADIUS_KM = 50.0

# Hybrid mode: share of the ML probability in the final blended score.
HYBRID_ML_WEIGHT = 0.5

# --- Eligibility rules (demo defaults, NOT clinical advice) ---------------
MIN_AGE = 18
MAX_AGE = 65
MIN_WEIGHT_KG = 45.0
MIN_DONATION_INTERVAL_DAYS = {"M": 90, "F": 120, "O": 120}
ELIGIBILITY_MARGIN_DAYS = 60      # days past the minimum interval for margin = 1
