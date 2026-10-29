"""Pydantic request / response models."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from .config import BLOOD_GROUPS

BloodGroup = Literal["O+", "B+", "A+", "AB+", "O-", "B-", "A-", "AB-"]
Urgency = Literal["normal", "urgent", "critical"]
AvailabilityStatus = Literal["available", "flexible", "busy", "unavailable"]


# ---------------------------------------------------------------- auth
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(min_length=1, max_length=120)


class CreateUserIn(RegisterIn):
    role: Literal["admin", "hospital", "donor"]


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    name: str
    role: str


# ---------------------------------------------------------------- donors
class DonorIn(BaseModel):
    display_name: str = Field(min_length=1, max_length=120)
    blood_group: BloodGroup
    sex: Literal["M", "F", "O"]
    age: int = Field(ge=16, le=100)
    weight_kg: float = Field(ge=30, le=250)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    city: str = Field(default="", max_length=80)
    phone: Optional[str] = Field(default=None, max_length=32)
    consent_to_contact: bool = False
    last_donation_date: Optional[date] = None

    @field_validator("last_donation_date")
    @classmethod
    def not_future(cls, v):
        if v and v > date.today():
            raise ValueError("last_donation_date cannot be in the future")
        return v


class DonorSelfOut(BaseModel):
    """What a donor sees about themselves."""
    model_config = ConfigDict(from_attributes=True)
    id: int
    display_name: str
    blood_group: str
    sex: str
    age: int
    weight_kg: float
    lat: float
    lon: float
    city: str
    phone: Optional[str]
    consent_to_contact: bool
    availability_status: str
    last_donation_date: Optional[date]
    total_contacted: int
    total_responded: int
    total_accepted: int
    total_donations: int
    avg_response_minutes: Optional[float]


class DonorPublicOut(BaseModel):
    """Privacy-minimised view for staff: no phone, no exact coordinates."""
    id: int
    name: str
    blood_group: str
    city: str
    availability_status: str
    total_donations: int
    is_synthetic: bool


class AvailabilityIn(BaseModel):
    status: AvailabilityStatus


# ---------------------------------------------------------------- requests
class RequestIn(BaseModel):
    hospital_name: str = Field(min_length=1, max_length=160)
    patient_blood_group: BloodGroup
    units_needed: int = Field(default=1, ge=1, le=20)
    urgency: Urgency = "normal"
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    city: str = Field(default="", max_length=80)
    max_radius_km: float = Field(default=50.0, gt=0, le=300)
    notes: Optional[str] = Field(default=None, max_length=500)


class RequestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    hospital_name: str
    patient_blood_group: str
    units_needed: int
    urgency: str
    lat: float
    lon: float
    city: str
    max_radius_km: float
    notes: Optional[str]
    status: str
    units_fulfilled: int
    created_at: datetime


class ComponentBreakdown(BaseModel):
    reliability: float
    availability: float
    proximity: float
    compatibility: float
    eligibility: float


class RecommendationOut(BaseModel):
    rank: int
    donor_id: int
    name: str
    blood_group: str
    city: str
    distance_km: float
    score: float
    weighted_score: float
    ml_probability: Optional[float]
    availability_status: str
    component_values: ComponentBreakdown
    contributions: ComponentBreakdown
    reliability_detail: dict[str, float]
    availability_detail: dict[str, float]
    explanation: list[str]


class ExcludedOut(BaseModel):
    donor_id: int
    reasons: list[str]


class RecommendationsResponse(BaseModel):
    request_id: int
    mode_requested: str
    mode_used: str
    weights: dict[str, float]
    stats: dict[str, int]
    elapsed_ms: float
    warnings: list[str]
    recommendations: list[RecommendationOut]
    excluded: list[ExcludedOut] = []


class NotifyIn(BaseModel):
    donor_ids: list[int] = Field(min_length=1, max_length=50)


class NotificationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    request_id: int
    donor_id: int
    rank_at_send: Optional[int]
    score_at_send: Optional[float]
    sent_at: datetime
    responded_at: Optional[datetime]
    accepted: Optional[bool]
    donated: Optional[bool]


class DonorNotificationOut(NotificationOut):
    hospital_name: str
    patient_blood_group: str
    urgency: str
    city: str
    units_needed: int


class RespondIn(BaseModel):
    accepted: bool


class DonationIn(BaseModel):
    completed: bool
