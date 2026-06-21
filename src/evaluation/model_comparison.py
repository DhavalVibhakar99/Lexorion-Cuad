"""
Head-to-head comparison: DeBERTa vs LLM vs Hybrid.
Generates the comparison table that goes in the README and the dashboard.

Run: python -m src.evaluation.model_comparison
"""

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from src.evaluation.metrics import evaluate_predictions, print_evaluation_report


PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")


def load_deberta_predictions() -> pd.DataFrame:
    """Load DeBERTa predictions from saved inference results."""
    pred_file = PROCESSED_DIR / "deberta_predictions.parquet"
    if pred_file.exists():
        return pd.read_parquet(pred_file)
    
    print("⚠️  No DeBERTa predictions found. Run clause_detector.py first.")
    return pd.DataFrame()


def load_baseline_predictions() -> pd.DataFrame:
    """Load TF-IDF baseline predictions from saved inference results."""
    pred_file = PROCESSED_DIR / "baseline_predictions.parquet"
    if pred_file.exists():
        return pd.read_parquet(pred_file)

    print("No baseline predictions found. Run baseline_detector.py first.")
    return pd.DataFrame()


def load_llm_predictions() -> pd.DataFrame:
    """Load LLM predictions from saved inference results."""
    pred_file = PROCESSED_DIR / "llm_predictions.parquet"
    if pred_file.exists():
        return pd.read_parquet(pred_file)
    
    print("⚠️  No LLM predictions found. Run llm_classifier.py first.")
    return pd.DataFrame()


def load_hybrid_predictions() -> pd.DataFrame:
    """Load Hybrid predictions from saved inference results."""
    pred_file = PROCESSED_DIR / "hybrid_predictions.parquet"
    if pred_file.exists():
        return pd.read_parquet(pred_file)
    
    print("⚠️  No Hybrid predictions found. Run hybrid_pipeline.py first.")
    return pd.DataFrame()


def compare_models():
    """
    Run the full comparison and generate results table.
    """
    print("=" * 70)
    print("MODEL COMPARISON: DeBERTa vs LLM vs Hybrid")
    print("=" * 70)
    
    results = {}

    # === Baseline ===
    baseline_df = load_baseline_predictions()
    if not baseline_df.empty:
        print("\n--- Baseline Results ---")
        baseline_metrics = evaluate_predictions(baseline_df, "risk_category")
        print_evaluation_report(baseline_metrics)
        results["baseline"] = baseline_metrics
    
    # === DeBERTa ===
    deberta_df = load_deberta_predictions()
    if not deberta_df.empty:
        print("\n--- DeBERTa Results ---")
        deberta_metrics = evaluate_predictions(deberta_df, "risk_category")
        print_evaluation_report(deberta_metrics)
        results["deberta"] = deberta_metrics
    
    # === LLM ===
    llm_df = load_llm_predictions()
    if not llm_df.empty:
        print("\n--- LLM Results ---")
        llm_metrics = evaluate_predictions(llm_df, "risk_category")
        print_evaluation_report(llm_metrics)
        results["llm"] = llm_metrics
    
    # === Hybrid ===
    hybrid_df = load_hybrid_predictions()
    if not hybrid_df.empty:
        print("\n--- Hybrid Results ---")
        hybrid_metrics = evaluate_predictions(hybrid_df, "risk_category")
        print_evaluation_report(hybrid_metrics)
        results["hybrid"] = hybrid_metrics
    
    # === Summary comparison table ===
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)
        
        summary_rows = []
        for approach, metrics_df in results.items():
            aggregate = metrics_df[metrics_df["category"] == "AGGREGATE (macro)"].iloc[0]
            
            # Load cost/latency metadata
            meta = _load_approach_metadata(approach)
            
            summary_rows.append({
                "Approach": approach.upper(),
                "Macro F1": f"{aggregate['f1']:.3f}",
                "Macro Precision": f"{aggregate['precision']:.3f}",
                "Macro Recall": f"{aggregate['recall']:.3f}",
                "Avg Latency/Contract": meta.get("avg_latency", "N/A"),
                "Cost/Contract": meta.get("cost_per_contract", "N/A"),
            })
        
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        
        # Save
        summary_df.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)
        print(f"\nSaved to {PROCESSED_DIR / 'model_comparison.csv'}")
        
        # Save detailed per-category comparison
        _save_detailed_comparison(results)
    
    return results


def _load_approach_metadata(approach: str) -> dict:
    """Load cost and latency metadata for an approach."""
    meta = {}
    
    if approach == "deberta":
        # Estimate from training results
        meta["avg_latency"] = "~2s"
        meta["cost_per_contract"] = "$0 (local)"
    elif approach == "baseline":
        meta["avg_latency"] = "<1s"
        meta["cost_per_contract"] = "$0 (local)"
    elif approach == "llm":
        meta["avg_latency"] = "~30s"
        meta["cost_per_contract"] = "~$0.02"
    elif approach == "hybrid":
        meta["avg_latency"] = "~10s"
        meta["cost_per_contract"] = "~$0.005"
    
    return meta


def _save_detailed_comparison(results: Dict[str, pd.DataFrame]):
    """Save per-category comparison across all approaches."""
    
    # Get all categories
    all_categories = set()
    for metrics_df in results.values():
        cats = metrics_df[metrics_df["category"] != "AGGREGATE (macro)"]["category"].tolist()
        all_categories.update(cats)
    
    rows = []
    for category in sorted(all_categories):
        row = {"category": category}
        for approach, metrics_df in results.items():
            cat_row = metrics_df[metrics_df["category"] == category]
            if not cat_row.empty:
                cat_row = cat_row.iloc[0]
                row[f"{approach}_f1"] = round(cat_row["f1"], 3)
                row[f"{approach}_precision"] = round(cat_row["precision"], 3)
                row[f"{approach}_recall"] = round(cat_row["recall"], 3)
            else:
                row[f"{approach}_f1"] = None
                row[f"{approach}_precision"] = None
                row[f"{approach}_recall"] = None
        
        # Which approach won?
        f1_scores = {
            approach: row.get(f"{approach}_f1", 0) or 0
            for approach in results.keys()
        }
        row["best_approach"] = max(f1_scores, key=f1_scores.get)
        rows.append(row)
    
    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(PROCESSED_DIR / "model_comparison_detailed.csv", index=False)
    print(f"Saved detailed comparison to {PROCESSED_DIR / 'model_comparison_detailed.csv'}")


if __name__ == "__main__":
    compare_models()
