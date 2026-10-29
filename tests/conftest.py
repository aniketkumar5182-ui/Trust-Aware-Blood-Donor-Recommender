"""Test configuration. Environment is set BEFORE the app is imported."""
import os

os.environ["DATABASE_URL"] = "sqlite://"            # in-memory database
os.environ["SECRET_KEY"] = "test-secret-key-only-for-the-test-suite-0123456789"
os.environ["PBKDF2_ITERATIONS"] = "1000"             # fast hashing in tests
os.environ["ADMIN_EMAIL"] = "admin@example.com"
os.environ["ADMIN_PASSWORD"] = "AdminPass123"

from datetime import datetime  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.domain import DonorProfile, RequestContext  # noqa: E402

NOW = datetime(2026, 5, 1, 12, 0, 0)


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def make_donor():
    def _make(**kw):
        base = dict(donor_id=1, blood_group="O+", sex="M", age=30, weight_kg=70.0, lat=28.47, lon=77.50,
                    availability_status="available", last_donation_date=None,
                    last_active_at=NOW, total_contacted=10, total_responded=8, total_accepted=6,
                    total_donations=5, avg_response_minutes=20.0)
        base.update(kw)
        return DonorProfile(**base)
    return _make


@pytest.fixture
def ctx():
    return RequestContext("O+", 28.47, 77.50, "normal", NOW, 50.0)
