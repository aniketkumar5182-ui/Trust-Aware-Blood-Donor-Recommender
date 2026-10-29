"""FastAPI application entry point.

Run:  uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs      Dashboard: http://127.0.0.1:8000/
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .auth import bootstrap_admin
from .database import SessionLocal, init_db
from .model_registry import get_model
from .routes import admin_routes, auth_routes, donor_routes, request_routes

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with SessionLocal() as db:
        bootstrap_admin(db)
    get_model()
    yield


app = FastAPI(
    title="Trust-Aware Emergency Blood Donor Recommendation System",
    description=(
        "Ranks compatible, eligible blood donors by reliability, current availability, proximity "
        "and request urgency, with an explainable score breakdown. Academic prototype - final "
        "medical eligibility decisions belong to authorised healthcare personnel."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


app.include_router(auth_routes.router)
app.include_router(donor_routes.router)
app.include_router(request_routes.router)
app.include_router(admin_routes.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "version": __version__, "ml_model_loaded": get_model() is not None}


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "index.html")
