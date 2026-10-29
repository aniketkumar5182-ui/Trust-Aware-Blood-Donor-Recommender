from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..db_models import BloodRequest, Donor, Notification, User
from ..domain import utcnow
from ..schemas import (
    AvailabilityIn,
    DonorIn,
    DonorNotificationOut,
    DonorPublicOut,
    DonorSelfOut,
)
from ..auth import require_roles
from .common import mask_name

router = APIRouter(prefix="/donors", tags=["donors"])


def _my_donor(db: Session, user: User) -> Donor:
    donor = db.query(Donor).filter(Donor.user_id == user.id).first()
    if donor is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Create your donor profile first")
    return donor


@router.post("/me", response_model=DonorSelfOut, status_code=status.HTTP_201_CREATED)
def create_my_profile(body: DonorIn, db: Session = Depends(get_db), user: User = Depends(require_roles("donor"))):
    if db.query(Donor).filter(Donor.user_id == user.id).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Profile already exists; use PUT /donors/me")
    donor = Donor(user_id=user.id, last_active_at=utcnow(), **body.model_dump())
    db.add(donor)
    db.commit()
    return donor


@router.get("/me", response_model=DonorSelfOut)
def get_my_profile(db: Session = Depends(get_db), user: User = Depends(require_roles("donor"))):
    return _my_donor(db, user)


@router.put("/me", response_model=DonorSelfOut)
def update_my_profile(body: DonorIn, db: Session = Depends(get_db), user: User = Depends(require_roles("donor"))):
    donor = _my_donor(db, user)
    for key, value in body.model_dump().items():
        setattr(donor, key, value)
    donor.last_active_at = utcnow()
    db.commit()
    return donor


@router.put("/me/availability", response_model=DonorSelfOut)
def set_availability(body: AvailabilityIn, db: Session = Depends(get_db), user: User = Depends(require_roles("donor"))):
    donor = _my_donor(db, user)
    donor.availability_status = body.status
    donor.last_active_at = utcnow()
    db.commit()
    return donor


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_my_profile(db: Session = Depends(get_db), user: User = Depends(require_roles("donor"))):
    """Right to erasure: removes the donor profile and its outreach history."""
    donor = _my_donor(db, user)
    db.delete(donor)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me/notifications", response_model=list[DonorNotificationOut])
def my_notifications(db: Session = Depends(get_db), user: User = Depends(require_roles("donor"))):
    donor = _my_donor(db, user)
    rows = (db.query(Notification, BloodRequest)
            .join(BloodRequest, BloodRequest.id == Notification.request_id)
            .filter(Notification.donor_id == donor.id)
            .order_by(Notification.sent_at.desc()).limit(100).all())
    return [
        DonorNotificationOut(
            id=n.id, request_id=n.request_id, donor_id=n.donor_id, rank_at_send=n.rank_at_send,
            score_at_send=n.score_at_send, sent_at=n.sent_at, responded_at=n.responded_at,
            accepted=n.accepted, donated=n.donated, hospital_name=r.hospital_name,
            patient_blood_group=r.patient_blood_group, urgency=r.urgency, city=r.city,
            units_needed=r.units_needed,
        ) for n, r in rows
    ]


@router.get("", response_model=list[DonorPublicOut])
def list_donors(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
                db: Session = Depends(get_db), _: User = Depends(require_roles("admin"))):
    """Admin overview with minimised personal data (no phone, no coordinates)."""
    rows = db.query(Donor).order_by(Donor.id).offset(skip).limit(limit).all()
    return [DonorPublicOut(id=d.id, name=mask_name(d.display_name), blood_group=d.blood_group, city=d.city,
                           availability_status=d.availability_status, total_donations=d.total_donations,
                           is_synthetic=d.is_synthetic) for d in rows]
