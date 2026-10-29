from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import require_roles
from ..database import get_db
from ..db_models import BloodRequest, Donor, Notification, User
from ..domain import utcnow
from ..model_registry import get_model
from ..simulation import generate_world

router = APIRouter(prefix="/admin", tags=["admin"])
admin_only = require_roles("admin")


@router.post("/seed-demo")
def seed_demo(n_donors: int = Query(300, ge=10, le=5000), seed: int = 7,
              db: Session = Depends(get_db), _: User = Depends(admin_only)):
    """Insert *synthetic* donors (clearly flagged) so the dashboard can be demonstrated."""
    world = generate_world(n_donors, 0, seed, now=utcnow())
    for p in world.donors:
        db.add(Donor(
            display_name=p.display_name, blood_group=p.blood_group, sex=p.sex, age=p.age,
            weight_kg=p.weight_kg, lat=p.lat, lon=p.lon, city=p.city, phone=None,
            consent_to_contact=True, is_synthetic=True, availability_status=p.availability_status,
            last_donation_date=p.last_donation_date, last_active_at=p.last_active_at,
            total_contacted=p.total_contacted, total_responded=p.total_responded,
            total_accepted=p.total_accepted, total_donations=p.total_donations,
            avg_response_minutes=p.avg_response_minutes,
        ))
    db.commit()
    return {"inserted": len(world.donors), "synthetic": True}


@router.delete("/synthetic-donors")
def delete_synthetic(db: Session = Depends(get_db), _: User = Depends(admin_only)):
    donors = db.query(Donor).filter(Donor.is_synthetic.is_(True)).all()
    for d in donors:
        db.delete(d)
    db.commit()
    return {"deleted": len(donors)}


@router.get("/stats")
def stats(db: Session = Depends(get_db), _: User = Depends(admin_only)):
    by_group = dict(db.query(Donor.blood_group, func.count(Donor.id)).group_by(Donor.blood_group).all())
    return {
        "users": db.query(User).count(),
        "donors": db.query(Donor).count(),
        "synthetic_donors": db.query(Donor).filter(Donor.is_synthetic.is_(True)).count(),
        "requests": db.query(BloodRequest).count(),
        "notifications": db.query(Notification).count(),
        "donors_by_blood_group": by_group,
    }


@router.get("/model")
def model_info(_: User = Depends(admin_only)):
    model = get_model()
    if model is None:
        return {"available": False, "detail": "Run `python -m scripts.train_model` to create a model."}
    return {"available": True, "type": model.model_type, "metrics": model.metrics,
            "feature_importances": model.feature_importances()}
