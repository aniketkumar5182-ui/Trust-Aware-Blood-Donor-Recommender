"""ORM tables."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base
from .domain import DonorProfile, utcnow


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(20))  # admin | hospital | donor
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Donor(Base):
    __tablename__ = "donors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), unique=True, nullable=True)
    display_name: Mapped[str] = mapped_column(String(120))
    blood_group: Mapped[str] = mapped_column(String(3), index=True)
    sex: Mapped[str] = mapped_column(String(1))
    age: Mapped[int] = mapped_column(Integer)
    weight_kg: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float, index=True)
    lon: Mapped[float] = mapped_column(Float, index=True)
    city: Mapped[str] = mapped_column(String(80), default="")
    phone: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # never returned by ranking APIs
    consent_to_contact: Mapped[bool] = mapped_column(Boolean, default=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    availability_status: Mapped[str] = mapped_column(String(16), default="available")
    last_donation_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_active_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    total_contacted: Mapped[int] = mapped_column(Integer, default=0)
    total_responded: Mapped[int] = mapped_column(Integer, default=0)
    total_accepted: Mapped[int] = mapped_column(Integer, default=0)
    total_donations: Mapped[int] = mapped_column(Integer, default=0)
    avg_response_minutes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    notifications: Mapped[list["Notification"]] = relationship(back_populates="donor", cascade="all, delete-orphan")

    def to_profile(self) -> DonorProfile:
        return DonorProfile(
            donor_id=self.id, blood_group=self.blood_group, sex=self.sex, age=self.age,
            weight_kg=self.weight_kg, lat=self.lat, lon=self.lon,
            availability_status=self.availability_status, last_donation_date=self.last_donation_date,
            last_active_at=self.last_active_at, total_contacted=self.total_contacted,
            total_responded=self.total_responded, total_accepted=self.total_accepted,
            total_donations=self.total_donations, avg_response_minutes=self.avg_response_minutes,
            is_active=self.is_active, display_name=self.display_name, city=self.city,
        )


class BloodRequest(Base):
    __tablename__ = "blood_requests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    requester_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    hospital_name: Mapped[str] = mapped_column(String(160))
    patient_blood_group: Mapped[str] = mapped_column(String(3))
    units_needed: Mapped[int] = mapped_column(Integer, default=1)
    urgency: Mapped[str] = mapped_column(String(10), default="normal")
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    city: Mapped[str] = mapped_column(String(80), default="")
    max_radius_km: Mapped[float] = mapped_column(Float, default=50.0)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(12), default="open")  # open | fulfilled | closed
    units_fulfilled: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    notifications: Mapped[list["Notification"]] = relationship(back_populates="request", cascade="all, delete-orphan")


class Notification(Base):
    """One outreach attempt: request -> donor. Its outcome feeds back into reliability."""
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("blood_requests.id"), index=True)
    donor_id: Mapped[int] = mapped_column(ForeignKey("donors.id"), index=True)
    rank_at_send: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    score_at_send: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sent_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    responded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    accepted: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)   # None = no reply yet
    donated: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)    # None = unknown yet

    request: Mapped[BloodRequest] = relationship(back_populates="notifications")
    donor: Mapped[Donor] = relationship(back_populates="notifications")
