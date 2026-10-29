"""Offline evaluation of donor-ranking strategies on the synthetic world.

Usage:
    python -m scripts.train_model     # once, to create the ML model
    python -m scripts.evaluate        # writes docs/RESULTS.md, docs/results.json and a chart
"""
from __future__ import annotations

import argparse
import json

from app.config import DOCS_DIR, MODEL_PATH
from app.ml_model import ResponseModel
from app.simulation import evaluate_rankers, generate_world

HEADLINE = ["precision@1", "precision@3", "precision@5", "precision@10", "recall@10", "ndcg@10", "mrr",
            "contacts_to_first_success"]
LABELS = {"precision@1": "P@1", "precision@3": "P@3", "precision@5": "P@5", "precision@10": "P@10",
          "recall@10": "Recall@10", "ndcg@10": "NDCG@10", "mrr": "MRR",
          "contacts_to_first_success": "Contacts to 1st donor (lower = better)"}


def to_markdown(res: dict) -> str:
    methods = res["methods"]
    lines = [
        "# Evaluation results",
        "",
        "> **Important:** these numbers come from a *simulated* donor population (see `app/simulation.py`). "
        "They show that the pipeline works and how much each indicator contributes. They are **not** a claim "
        "about real-world performance - that needs real, authorised data.",
        "",
        f"* Donors in pool: **{res['n_donors']:,}**",
        f"* Emergency requests evaluated: **{res['n_requests_evaluated']}** "
        f"(of {res['n_requests_generated']} generated; {res['skipped_too_few_candidates']} skipped for too few "
        f"candidates, {res['skipped_no_successful_donor']} skipped because nobody would have donated)",
        f"* Average eligible compatible candidates per request: **{res['avg_candidates_per_request']:.0f}**",
        f"* Retrieval time of the full pipeline (filter + score + rank): "
        f"**{res['retrieval_ms_mean']:.1f} ms** mean, {res['retrieval_ms_p95']:.1f} ms p95",
        "* Every ranker orders the *same* candidates and is judged on the *same* sampled outcomes "
        "(paired comparison). A donor counts as a success if they respond, accept and actually donate.",
        "* Values are mean (+/- 95 % CI).",
        "",
        "| Method | " + " | ".join(LABELS[m] for m in HEADLINE) + " |",
        "|---|" + "|".join("---:" for _ in HEADLINE) + "|",
    ]
    for name, mets in methods.items():
        cells = []
        for m in HEADLINE:
            v = mets[m]
            cells.append(f"{v['mean']:.2f} (+/-{v['ci95']:.2f})" if m == "contacts_to_first_success"
                         else f"{v['mean']:.3f} (+/-{v['ci95']:.3f})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "![Ranking quality](assets/ranking_quality.png)", ""]
    return "\n".join(lines)


def plot(res: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = res["methods"]
    names = list(methods)
    metrics = [("precision@5", "Precision@5"), ("ndcg@10", "NDCG@10"), ("mrr", "MRR")]
    palette = ["#9aa0a6", "#bdc1c6", "#8ab4f8", "#c58af9", "#f6aea9", "#1a73e8", "#e37400", "#188038"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5.2), sharey=False)
    for ax, (key, title) in zip(axes, metrics):
        vals = [methods[n][key]["mean"] for n in names]
        errs = [methods[n][key]["ci95"] for n in names]
        ax.barh(range(len(names)), vals, xerr=errs, color=palette[: len(names)], capsize=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if ax is axes[0] else [""] * len(names), fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(vals):
            ax.text(v + max(errs) * 0.2 + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
    fig.suptitle("Donor ranking quality on simulated emergency requests (higher is better)", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--donors", type=int, default=1500)
    ap.add_argument("--events", type=int, default=4000)
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    world = generate_world(args.donors, args.events, args.seed)
    model = ResponseModel.load(MODEL_PATH) if MODEL_PATH.exists() else None
    if model is None:
        print("No saved model found - evaluating without ML rankers (run scripts.train_model first).")
    res = evaluate_rankers(world, model, n_requests=args.requests)

    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "results.json").write_text(json.dumps(res, indent=2))
    (DOCS_DIR / "RESULTS.md").write_text(to_markdown(res))
    plot(res, DOCS_DIR / "assets" / "ranking_quality.png")
    print(to_markdown(res))
    print(f"Wrote {DOCS_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
