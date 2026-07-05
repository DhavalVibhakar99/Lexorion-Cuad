"""
End-to-end evaluation of the hybrid routing policy.

The hybrid rule mirrors analyze_contract_hybrid (recall-first baseline with
LLM precision-repair on weak flags):
    baseline score >= t + margin -> flag (strong, stays local)
    t <= score < t + margin      -> LLM verdict decides (weak flag triage)
    score < t                    -> clear

Two outputs:
1. hybrid_predictions.parquet — hybrid verdicts on the LLM-evaluated sample
   (the only rows where an LLM verdict exists), so model_comparison.py can
   score BASELINE vs LLM vs HYBRID on identical paragraphs.
2. hybrid_routing_stats.csv — routing volume on the FULL test split (no API
   calls needed: it only depends on baseline scores), i.e. what fraction of
   paragraph x category decisions would ever reach the LLM.

Run: python -m src.evaluation.hybrid_eval
"""

import joblib
import pandas as pd
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
BASELINE_ARTIFACT = CHECKPOINT_DIR / "baseline_tfidf_logreg.joblib"
UNCERTAINTY_MARGIN = 0.15


def evaluate_hybrid(margin: float = UNCERTAINTY_MARGIN) -> pd.DataFrame:
    baseline = pd.read_parquet(PROCESSED_DIR / "baseline_predictions.parquet")
    llm = pd.read_parquet(PROCESSED_DIR / "llm_predictions.parquet")
    thresholds = joblib.load(BASELINE_ARTIFACT)["thresholds"]

    # --- Routing volume on the full test split (baseline scores only) ---
    stats_rows = []
    for category, group in baseline.groupby("risk_category"):
        t = float(thresholds[category])
        strong = (group["y_score"] >= t + margin).sum()
        weak = ((group["y_score"] >= t) & (group["y_score"] < t + margin)).sum()
        total = len(group)
        stats_rows.append(
            {
                "risk_category": category,
                "threshold": t,
                "decisions": total,
                "strong_flags": int(strong),
                "weak_flags_to_llm": int(weak),
                "cleared_locally": int(total - strong - weak),
                "llm_escalation_rate": round(weak / total, 4),
            }
        )
    stats = pd.DataFrame(stats_rows)
    overall_rate = stats["weak_flags_to_llm"].sum() / stats["decisions"].sum()
    stats.to_csv(PROCESSED_DIR / "hybrid_routing_stats.csv", index=False)

    print("=" * 70)
    print("HYBRID ROUTING VOLUME (full test split, no API calls)")
    print("=" * 70)
    print(stats.to_string(index=False))
    print(
        f"\nOverall: {overall_rate:.2%} of paragraph x category decisions "
        f"escalate to the LLM; the rest are decided locally for free."
    )

    # --- Hybrid verdicts on the LLM-evaluated sample ---
    merged = llm.merge(
        baseline[["paragraph_id", "risk_category", "y_score"]].rename(
            columns={"y_score": "baseline_score"}
        ),
        on=["paragraph_id", "risk_category"],
        how="inner",
    )

    def hybrid_verdict(row) -> int:
        t = float(thresholds[row["risk_category"]])
        if row["baseline_score"] >= t + margin:
            return 1
        if row["baseline_score"] >= t:
            return int(row["y_pred"])  # LLM triages the weak flag
        return 0

    merged["baseline_pred"] = merged.apply(
        lambda r: int(r["baseline_score"] >= float(thresholds[r["risk_category"]])),
        axis=1,
    )
    hybrid = merged.copy()
    hybrid["y_pred"] = merged.apply(hybrid_verdict, axis=1)
    hybrid["model"] = "hybrid_recall_first"
    hybrid[
        ["paragraph_id", "risk_category", "y_true", "y_pred", "y_score", "model"]
    ].to_parquet(PROCESSED_DIR / "hybrid_predictions.parquet", index=False)

    # --- Like-for-like mini-benchmark on identical paragraphs ---
    print("\n" + "=" * 70)
    print(f"SAME-SAMPLE COMPARISON (n={len(merged)} LLM-evaluated paragraphs)")
    print("=" * 70)
    for name, preds in [
        ("baseline", merged["baseline_pred"]),
        ("llm", merged["y_pred"]),
        ("hybrid", hybrid["y_pred"]),
    ]:
        print(
            f"  {name:<9} precision={precision_score(merged['y_true'], preds, zero_division=0):.3f} "
            f"recall={recall_score(merged['y_true'], preds, zero_division=0):.3f} "
            f"f1={f1_score(merged['y_true'], preds, zero_division=0):.3f}"
        )
    print(
        "\nNote: this sample is category-balanced (not at deployment prevalence),"
        "\nso precision here is not comparable to full-split numbers. Recall is."
    )
    return hybrid


if __name__ == "__main__":
    evaluate_hybrid()
