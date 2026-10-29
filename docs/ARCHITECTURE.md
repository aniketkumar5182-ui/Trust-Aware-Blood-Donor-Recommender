# Architecture

## Components

| Layer | Module(s) | Responsibility |
|---|---|---|
| Presentation | `app/static/index.html` | Single-page dashboard for hospitals, donors and admins (vanilla JS, no build step) |
| API | `app/main.py`, `app/routes/*` | REST endpoints, validation (Pydantic), role-based access |
| Security | `app/auth.py` | PBKDF2 password hashing, JWT bearer tokens, `require_roles` dependency |
| Domain engine | `app/compatibility.py`, `eligibility.py`, `geo.py`, `scoring.py`, `recommender.py` | Pure-Python, database-free ranking logic (easy to unit test) |
| ML | `app/ml_model.py`, `app/model_registry.py` | Train / persist / load the response-probability model |
| Persistence | `app/database.py`, `app/db_models.py` | SQLAlchemy 2 ORM (SQLite by default) |
| Simulation | `app/simulation.py`, `scripts/*` | Synthetic world, model training, offline evaluation |

The domain engine knows nothing about HTTP or SQL: routes convert ORM rows into `DonorProfile` dataclasses, call
`recommend()` and convert the result back to API schemas.

## Request flow

```mermaid
sequenceDiagram
    participant H as Hospital
    participant API as FastAPI
    participant E as Recommender
    participant DB as Database
    participant D as Donor
    H->>API: POST /requests (blood group, urgency, location)
    H->>API: GET /requests/{id}/recommendations
    API->>DB: SQL prefilter (compatible groups, consent, bounding box, not yet contacted)
    API->>E: recommend(donors, context, mode)
    E->>E: filter (compat, radius, eligibility) -> score -> (ML) -> rank
    E-->>API: ranked donors + score breakdown + reasons
    API-->>H: masked names, distance, scores
    H->>API: POST /requests/{id}/notify (selected donors)
    API->>DB: create Notification rows, total_contacted += 1
    D->>API: POST /notifications/{id}/respond (accept / decline)
    API->>DB: responded / accepted counters, average response time, last_active
    H->>API: POST /notifications/{id}/donation (donated / no-show)
    API->>DB: donations counter, last_donation_date, request progress
```

The last three steps form the **feedback loop**: real outcomes continuously refine every donor's reliability indicators.

## Data model

```mermaid
erDiagram
    USERS ||--o| DONORS : "owns profile"
    USERS ||--o{ BLOOD_REQUESTS : "raises"
    BLOOD_REQUESTS ||--o{ NOTIFICATIONS : "contacts"
    DONORS ||--o{ NOTIFICATIONS : "receives"
    USERS {
        int id PK
        string email
        string role
    }
    DONORS {
        int id PK
        int user_id FK
        string blood_group
        string availability_status
        date last_donation_date
        int total_contacted
        int total_responded
        int total_accepted
        int total_donations
        float avg_response_minutes
        bool consent_to_contact
        bool is_synthetic
    }
    BLOOD_REQUESTS {
        int id PK
        int requester_id FK
        string patient_blood_group
        string urgency
        float max_radius_km
        string status
        int units_needed
        int units_fulfilled
    }
    NOTIFICATIONS {
        int id PK
        int request_id FK
        int donor_id FK
        int rank_at_send
        float score_at_send
        datetime responded_at
        bool accepted
        bool donated
    }
```

## Design decisions

* **Transparent first, ML second.** The weighted score is explainable and works with zero training data (cold start);
  ML is optional and falls back gracefully.
* **Filters before scores.** Medical/legal constraints are hard rules, never traded off against convenience.
* **Point-in-time features.** Training rows only use history available *before* each event.
* **Privacy by design.** Masked names, no phone/coordinates in ranking responses, consent flag, right to erasure,
  least-privilege roles (see `PRIVACY_AND_ETHICS.md`).
* **Scalability path.** A SQL bounding-box prefilter keeps the in-memory ranking set small; the engine takes ~4 ms for
  1 500 donors. For very large pools move the prefilter to PostGIS / a geo index.
