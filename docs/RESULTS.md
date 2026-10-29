# Evaluation results

> **Important:** these numbers come from a *simulated* donor population (see `app/simulation.py`). They show that the pipeline works and how much each indicator contributes. They are **not** a claim about real-world performance - that needs real, authorised data.

* Donors in pool: **1,500**
* Emergency requests evaluated: **476** (of 500 generated; 24 skipped for too few candidates, 0 skipped because nobody would have donated)
* Average eligible compatible candidates per request: **67**
* Retrieval time of the full pipeline (filter + score + rank): **5.7 ms** mean, 10.5 ms p95
* Every ranker orders the *same* candidates and is judged on the *same* sampled outcomes (paired comparison). A donor counts as a success if they respond, accept and actually donate.
* Values are mean (+/- 95 % CI).

| Method | P@1 | P@3 | P@5 | P@10 | Recall@10 | NDCG@10 | MRR | Contacts to 1st donor (lower = better) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random (blood-group match only) | 0.187 (+/-0.035) | 0.186 (+/-0.020) | 0.179 (+/-0.016) | 0.175 (+/-0.011) | 0.213 (+/-0.016) | 0.211 (+/-0.014) | 0.380 (+/-0.029) | 5.65 (+/-0.47) |
| nearest first (distance only) | 0.244 (+/-0.039) | 0.277 (+/-0.025) | 0.268 (+/-0.019) | 0.248 (+/-0.013) | 0.297 (+/-0.021) | 0.300 (+/-0.018) | 0.459 (+/-0.030) | 4.22 (+/-0.35) |
| reliability only | 0.431 (+/-0.045) | 0.356 (+/-0.025) | 0.351 (+/-0.019) | 0.337 (+/-0.013) | 0.404 (+/-0.022) | 0.414 (+/-0.018) | 0.610 (+/-0.032) | 2.68 (+/-0.19) |
| trust score w/o reliability | 0.342 (+/-0.043) | 0.293 (+/-0.024) | 0.281 (+/-0.020) | 0.267 (+/-0.014) | 0.315 (+/-0.021) | 0.331 (+/-0.019) | 0.531 (+/-0.032) | 3.69 (+/-0.32) |
| trust score w/o availability | 0.309 (+/-0.042) | 0.284 (+/-0.024) | 0.273 (+/-0.020) | 0.249 (+/-0.013) | 0.299 (+/-0.021) | 0.312 (+/-0.019) | 0.501 (+/-0.032) | 4.05 (+/-0.35) |
| trust score (weighted) | 0.445 (+/-0.045) | 0.359 (+/-0.025) | 0.342 (+/-0.021) | 0.307 (+/-0.014) | 0.358 (+/-0.021) | 0.388 (+/-0.019) | 0.617 (+/-0.032) | 2.91 (+/-0.25) |
| ML only (Random Forest) | 0.527 (+/-0.045) | 0.484 (+/-0.025) | 0.463 (+/-0.020) | 0.406 (+/-0.014) | 0.460 (+/-0.020) | 0.503 (+/-0.017) | 0.707 (+/-0.029) | 2.02 (+/-0.15) |
| hybrid (trust + ML) | 0.492 (+/-0.045) | 0.478 (+/-0.025) | 0.450 (+/-0.021) | 0.392 (+/-0.014) | 0.443 (+/-0.020) | 0.485 (+/-0.017) | 0.686 (+/-0.029) | 2.10 (+/-0.16) |

![Ranking quality](assets/ranking_quality.png)
