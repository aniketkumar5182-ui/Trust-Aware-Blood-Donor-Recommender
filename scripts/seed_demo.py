"""Create demo accounts + synthetic donors for a quick local demonstration.

    python -m scripts.seed_demo

Creates (if missing):
    admin    admin@example.com     / Demo@12345
    hospital hospital@example.com  / Demo@12345
    donor    donor@example.com     / Demo@12345
and 400 clearly-flagged SYNTHETIC donors.

DEMO ONLY - never use these credentials outside your own machine.
"""
from __future__ import annotations

import argparse

from app.auth import hash_password
from app.database import SessionLocal, init_db
from app.db_models import Donor, User
from app.domain import utcnow
from app.simulation import generate_world

DEMO_PASSWORD = "Demo@12345"
ACCOUNTS = [("admin@example.com", "Demo Admin", "admin"),
            ("hospital@example.com", "City General Hospital (demo)", "hospital"),
            ("donor@example.com", "Demo Donor", "donor")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--donors", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    init_db()
    with SessionLocal() as db:
        for email, name, role in ACCOUNTS:
            if not db.query(User).filter(User.email == email).first():
                db.add(User(email=email, name=name, role=role, password_hash=hash_password(DEMO_PASSWORD)))
        existing = db.query(Donor).filter(Donor.is_synthetic.is_(True)).count()
        added = 0
        if existing == 0:
            world = generate_world(args.donors, 0, args.seed, now=utcnow())
            for p in world.donors:
                db.add(Donor(
                    display_name=p.display_name, blood_group=p.blood_group, sex=p.sex, age=p.age,
                    weight_kg=p.weight_kg, lat=p.lat, lon=p.lon, city=p.city, consent_to_contact=True,
                    is_synthetic=True, availability_status=p.availability_status,
                    last_donation_date=p.last_donation_date, last_active_at=p.last_active_at,
                    total_contacted=p.total_contacted, total_responded=p.total_responded,
                    total_accepted=p.total_accepted, total_donations=p.total_donations,
                    avg_response_minutes=p.avg_response_minutes,
                ))
                added += 1
        db.commit()
    print(f"Demo data ready ({added} synthetic donors added, {existing} already present).")
    print(f"Log in with any of {[a[0] for a in ACCOUNTS]} and password '{DEMO_PASSWORD}'.")


if __name__ == "__main__":
    main()
