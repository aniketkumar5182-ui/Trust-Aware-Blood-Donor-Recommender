# Scoring methodology

All sub-scores lie in **[0, 1]** (higher is better). Constants live in [`app/config.py`](../app/config.py); the
implementation is [`app/scoring.py`](../app/scoring.py).

## 1. Hard filters (applied first)

| Filter | Rule |
|---|---|
| Blood group | ABO/Rh red-cell compatibility (`app/compatibility.py`) |
| Radius | Haversine distance <= request radius (default 50 km) |
| Eligibility | active profile, not self-marked *unavailable*, age 18-65, weight >= 45 kg, minimum interval since last donation (90 days male / 120 days female-other) |
| Consent | donor must have consented to be contacted (enforced by the API) |
| Already contacted | donors already notified for this request are not suggested again |

> The eligibility thresholds are **demo defaults, not clinical guidance**. Final eligibility is decided by authorised
> healthcare personnel.

## 2. Reliability - "does this donor usually come through?"

| Component | Formula | Meaning |
|---|---|---|
| response rate | `(responded + 1) / (contacted + 2)` | how often they reply (accept *or* decline) |
| follow-through | `(donations + 1) / (accepted + 2)` | how often an accepted request ends in a donation |
| activity | `exp(-days_inactive / 60)` | long-term engagement |
| experience | `min(1, ln(1 + donations) / ln(11))` | saturates after 10 donations |

`reliability = 0.35*response_rate + 0.30*follow_through + 0.20*activity + 0.15*experience`

The `+1 / +2` terms are **Laplace smoothing** (a Beta(1,1) prior): a donor with no history scores 0.5 instead of being
punished or over-trusted, and one lucky reply cannot produce a perfect score.

## 3. Availability - "can they respond *now*?"

| Component | Formula |
|---|---|
| status | available 1.0, flexible 0.7, busy 0.3, unavailable 0 (also hard-filtered) |
| recent activity | `exp(-days_inactive / 14)` |
| response speed | `exp(-avg_response_minutes / 90)` (45 min assumed if unknown) |

`availability = 0.45*status + 0.30*recent_activity + 0.25*response_speed`

## 4. Proximity, compatibility, eligibility margin

* `proximity = exp(-distance_km / 20)`
* `compatibility`: identical group 1.0; other compatible group 0.75; **O-** given to a non-O- patient 0.5
  (scarce universal-donor units are used as a fallback when everything else is equal)
* `eligibility margin = min(1, (days_since_donation - minimum_interval) / 60)`, or 1.0 if the donor never donated -
  prefers donors who are comfortably rested over those who just became eligible.

## 5. Composite score and urgency

`score = sum(weight_i * component_i)`, with weights that depend on urgency (each row sums to 1):

| Urgency | reliability | availability | proximity | compatibility | eligibility |
|---|---:|---:|---:|---:|---:|
| normal   | 0.30 | 0.25 | 0.20 | 0.15 | 0.10 |
| urgent   | 0.28 | 0.30 | 0.25 | 0.10 | 0.07 |
| critical | 0.25 | 0.33 | 0.30 | 0.07 | 0.05 |

Rationale: the more urgent the request, the more it matters that a donor is *near* and *reachable right now*.
Every response exposes the per-component contribution (`weight x value`), so a ranking can always be explained.

## 6. ML layer (optional)

* Target: *the contacted donor responded, accepted and donated* (binary).
* Features (13): the reliability / availability sub-scores, distance, compatibility, eligibility margin, urgency
  level, age and number of donations - all computed **point-in-time** (history strictly before the event) to avoid
  label leakage.
* Models: Random Forest (default) or XGBoost (`--model xgb`). Time-ordered 80/20 split for hold-out metrics
  (ROC-AUC, PR-AUC, Brier score, log-loss) - see `docs/model_metrics.json`.
* Modes: `weighted` (no ML), `ml` (rank by predicted probability), `hybrid` (50 / 50 blend of the min-max-normalised
  weighted score and ML probability within the candidate set). The API falls back to `weighted` if no model exists.

## 7. Why a simulation?

Donor-response logs contain sensitive personal data and are not public. We therefore built a generator
(`app/simulation.py`) with a **hidden** per-donor reliability. The recommender only sees noisy proxies of it (response
rate, follow-through, recency, ...) and rankers are compared on identical candidate sets and identical sampled
outcomes. This validates the pipeline and the relative value of each indicator, but it **cannot** replace validation
on real, authorised data.
