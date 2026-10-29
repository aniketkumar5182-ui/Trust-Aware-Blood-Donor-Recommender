# Privacy, security and ethics

Blood-donor data is sensitive personal and health-adjacent information. The prototype applies the following
safeguards; each maps to code you can inspect.

## Data minimisation
* Only what ranking needs is stored: blood group, sex, age, weight, approximate location, donation/response counters.
* Ranking responses expose a **masked name**, city, distance and scores - never phone numbers or coordinates
  (`app/routes/request_routes.py::_to_out`, enforced by `tests/test_api.py`).
* Phone is optional and is only used by a (future) approved notification channel.

## Consent and control
* A donor is contacted only if `consent_to_contact` is true (SQL prefilter).
* Donors can change availability at any time and can **delete their profile and outreach history**
  (`DELETE /donors/me`).

## Access control and security
* Roles: donor / hospital / admin. Staff accounts cannot be self-registered.
* Hospitals only see their own requests; they can only notify donors that passed all filters.
* Salted PBKDF2-HMAC-SHA256 password hashing, constant-time comparison, signed JWTs with expiry.
* Pydantic validation on every input; SQLAlchemy parameterised queries; the dashboard escapes all rendered text.
* Security headers (`nosniff`, `X-Frame-Options`, `no-referrer`).
* **Before any real deployment:** HTTPS, rate limiting, audit logging, database encryption at rest, secrets from a
  vault, a proper consent/retention policy and legal review (e.g. India's DPDP Act 2023).

## Medical safety
* Eligibility rules are demo defaults, **not clinical guidance**. Final eligibility, testing and transfusion decisions
  belong to authorised healthcare personnel.
* The system recommends *whom to contact first*; it never decides who may donate.

## Fairness
* Reliability is learned from past behaviour, which can reflect circumstances (work shifts, distance, caste/class
  barriers to travel). Ranking by it may disadvantage some donors. Mitigations: cold-start prior of 0.5, no protected
  attributes used as features, explainable scores, and a recommendation to audit ranking outcomes across groups before
  any real use.

## Data used in this repository
All donors in the demo database and in the evaluation are **synthetic** (flagged `is_synthetic`). No real personal
data is included.
