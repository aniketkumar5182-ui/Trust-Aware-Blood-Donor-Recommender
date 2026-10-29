from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from .. import notifier
from ..auth import require_roles
from ..compatibility import compatible_donor_groups
from ..database import get_db
from ..db_models import BloodRequest, Donor, Notification, User
from ..domain import RequestContext, utcnow
from ..geo import bounding_box
from ..model_registry import get_model
from ..recommender import Recommendation, recommend
from ..schemas import (
    ComponentBreakdown,
    DonationIn,
    ExcludedOut,
    NotificationOut,
    NotifyIn,
    RecommendationOut,
    RecommendationsResponse,
    RequestIn,
    RequestOut,
    RespondIn,
)
from .common import mask_name

router = APIRouter(tags=["requests"])
staff = require_roles("admin", "hospital")


def _get_request(db: Session, user: User, request_id: int) -> BloodRequest:
    req = db.get(BloodRequest, request_id)
    # a hospital user can only see its own requests (404 instead of 403 to avoid leaking ids)
    if req is None or (user.role != "admin" and req.requester_id != user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found")
    return req


def _context(req: BloodRequest) -> RequestContext:
    return RequestContext(req.patient_blood_group, req.lat, req.lon, req.urgency, utcnow(), req.max_radius_km)


def _candidate_donors(db: Session, req: BloodRequest) -> list[Donor]:
    """SQL prefilter (group, consent, active, bounding box, not yet contacted for this request)."""
    min_lat, max_lat, min_lon, max_lon = bounding_box(req.lat, req.lon, req.max_radius_km)
    contacted = db.query(Notification.donor_id).filter(Notification.request_id == req.id)
    return (db.query(Donor)
            .filter(Donor.blood_group.in_(compatible_donor_groups(req.patient_blood_group)),
                    Donor.consent_to_contact.is_(True), Donor.is_active.is_(True),
                    Donor.lat.between(min_lat, max_lat), Donor.lon.between(min_lon, max_lon),
                    Donor.id.notin_(contacted))
            .all())


def _to_out(r: Recommendation) -> RecommendationOut:
    c = r.components
    return RecommendationOut(
        rank=r.rank, donor_id=r.donor.donor_id, name=mask_name(r.donor.display_name),
        blood_group=r.donor.blood_group, city=r.donor.city, distance_km=round(c.distance_km, 1),
        score=round(r.score, 4), weighted_score=round(r.weighted_score, 4),
        ml_probability=None if r.ml_probability is None else round(r.ml_probability, 4),
        availability_status=r.donor.availability_status,
        component_values=ComponentBreakdown(
            reliability=round(c.reliability, 3), availability=round(c.availability, 3),
            proximity=round(c.proximity, 3), compatibility=round(c.compatibility, 3),
            eligibility=round(c.eligibility_margin, 3)),
        contributions=ComponentBreakdown(**{k: round(v, 4) for k, v in r.contributions.items()}),
        reliability_detail={"response_rate": round(c.response_rate, 3), "completion_rate": round(c.completion_rate, 3),
                            "activity": round(c.activity, 3), "experience": round(c.experience, 3)},
        availability_detail={"status": round(c.status_score, 3), "recent_activity": round(c.recent_activity, 3),
                             "response_speed": round(c.speed, 3)},
        explanation=r.explanation,
    )


# ------------------------------------------------------------------ requests
@router.post("/requests", response_model=RequestOut, status_code=status.HTTP_201_CREATED)
def create_request(body: RequestIn, db: Session = Depends(get_db), user: User = Depends(staff)):
    req = BloodRequest(requester_id=user.id, **body.model_dump())
    db.add(req)
    db.commit()
    return req


@router.get("/requests", response_model=list[RequestOut])
def list_requests(db: Session = Depends(get_db), user: User = Depends(staff)):
    q = db.query(BloodRequest)
    if user.role != "admin":
        q = q.filter(BloodRequest.requester_id == user.id)
    return q.order_by(BloodRequest.created_at.desc()).limit(100).all()


@router.get("/requests/{request_id}", response_model=RequestOut)
def get_request(request_id: int, db: Session = Depends(get_db), user: User = Depends(staff)):
    return _get_request(db, user, request_id)


@router.post("/requests/{request_id}/close", response_model=RequestOut)
def close_request(request_id: int, db: Session = Depends(get_db), user: User = Depends(staff)):
    req = _get_request(db, user, request_id)
    if req.status == "open":
        req.status = "closed"
        db.commit()
    return req


@router.get("/requests/{request_id}/recommendations", response_model=RecommendationsResponse)
def recommendations(
    request_id: int,
    top_k: int = Query(10, ge=1, le=50),
    mode: Literal["auto", "weighted", "ml", "hybrid"] = "auto",
    include_excluded: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(staff),
):
    req = _get_request(db, user, request_id)
    model = get_model()
    effective = ("hybrid" if model else "weighted") if mode == "auto" else mode
    donors = [d.to_profile() for d in _candidate_donors(db, req)]
    result = recommend(donors, _context(req), top_k=top_k, mode=effective, model=model,
                       include_excluded=include_excluded)
    return RecommendationsResponse(
        request_id=req.id, mode_requested=mode, mode_used=result.mode_used,
        weights={k: round(v, 3) for k, v in result.weights.items()}, stats=result.stats,
        elapsed_ms=round(result.elapsed_ms, 2), warnings=result.warnings,
        recommendations=[_to_out(r) for r in result.recommendations],
        excluded=[ExcludedOut(donor_id=e.donor_id, reasons=e.reasons) for e in result.excluded],
    )


@router.get("/requests/{request_id}/notifications", response_model=list[NotificationOut])
def request_notifications(request_id: int, db: Session = Depends(get_db), user: User = Depends(staff)):
    req = _get_request(db, user, request_id)
    return db.query(Notification).filter(Notification.request_id == req.id).order_by(Notification.id).all()


@router.post("/requests/{request_id}/notify", response_model=list[NotificationOut], status_code=status.HTTP_201_CREATED)
def notify_donors(request_id: int, body: NotifyIn, db: Session = Depends(get_db), user: User = Depends(staff)):
    """Contact selected donors. Only donors that pass the compatibility / eligibility / radius
    filters (i.e. appear in the candidate list) can be contacted."""
    req = _get_request(db, user, request_id)
    if req.status != "open":
        raise HTTPException(status.HTTP_409_CONFLICT, "Request is not open")
    rows = {d.id: d for d in _candidate_donors(db, req)}
    result = recommend([d.to_profile() for d in rows.values()], _context(req), top_k=len(rows) or 1,
                       mode="weighted")
    ranked = {r.donor.donor_id: r for r in result.recommendations}
    invalid = [i for i in dict.fromkeys(body.donor_ids) if i not in ranked]
    if invalid:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"Donors not eligible/available for this request: {invalid}")
    created = []
    for donor_id in dict.fromkeys(body.donor_ids):
        rec = ranked[donor_id]
        note = Notification(request_id=req.id, donor_id=donor_id, rank_at_send=rec.rank,
                            score_at_send=round(rec.weighted_score, 4))
        rows[donor_id].total_contacted += 1
        db.add(note)
        created.append(note)
    db.commit()
    for note in created:
        notifier.send(note.donor_id, req.id, req.urgency)
    return created


# --------------------------------------------------- outcomes (feed back into reliability)
@router.post("/notifications/{notification_id}/respond", response_model=NotificationOut)
def respond(notification_id: int, body: RespondIn, db: Session = Depends(get_db),
            user: User = Depends(require_roles("donor"))):
    note = db.get(Notification, notification_id)
    donor = db.query(Donor).filter(Donor.user_id == user.id).first()
    if note is None or donor is None or note.donor_id != donor.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    if note.accepted is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "You have already responded")
    if note.request.status != "open":
        raise HTTPException(status.HTTP_409_CONFLICT, "This request is no longer open")
    now = utcnow()
    minutes = max(0.0, (now - note.sent_at).total_seconds() / 60)
    note.accepted, note.responded_at = body.accepted, now
    donor.total_responded += 1
    prev = donor.avg_response_minutes if donor.avg_response_minutes is not None else minutes
    donor.avg_response_minutes = prev + (minutes - prev) / donor.total_responded
    donor.last_active_at = now
    if body.accepted:
        donor.total_accepted += 1
    db.commit()
    return note


@router.post("/notifications/{notification_id}/donation", response_model=NotificationOut)
def record_donation(notification_id: int, body: DonationIn, db: Session = Depends(get_db),
                    user: User = Depends(staff)):
    """Hospital confirms whether an accepted donor actually donated (or did not show up)."""
    note = db.get(Notification, notification_id)
    if note is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    req = _get_request(db, user, note.request_id)
    if note.accepted is not True:
        raise HTTPException(status.HTTP_409_CONFLICT, "Donor has not accepted this request")
    if note.donated is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Outcome already recorded")
    note.donated = body.completed
    if body.completed:
        donor = note.donor
        donor.total_donations += 1
        donor.last_donation_date = utcnow().date()
        req.units_fulfilled += 1
        if req.units_fulfilled >= req.units_needed:
            req.status = "fulfilled"
    db.commit()
    return note
