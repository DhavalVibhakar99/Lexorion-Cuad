"""
Error analysis: deep-dive into false negatives and false positives.
This is what interviewers actually care about — not the F1 number, 
but WHY the model failed and what you'd do about it.

Run: python -m src.evaluation.error_analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


PROCESSED_DIR = Path("data/processed")


def analyze_false_negatives(
    predictions_df: pd.DataFrame,
    paragraphs_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Analyze paragraphs where the model missed a clause (FN).
    
    Key questions:
    - Are FNs concentrated in specific categories?
    - Are they longer/shorter than average?
    - Do they use unusual language?
    - Are they at the boundary of two categories?
    """
    fn_mask = (predictions_df["y_true"] == 1) & (predictions_df["y_pred"] == 0)
    fn_df = predictions_df[fn_mask].copy()
    
    if fn_df.empty:
        print("No false negatives found — suspicious, check your eval.")
        return fn_df
    
    print(f"\n{'=' * 60}")
    print(f"FALSE NEGATIVE ANALYSIS ({len(fn_df)} total)")
    print(f"{'=' * 60}")
    
    # By category
    fn_by_category = fn_df["risk_category"].value_counts()
    total_by_category = predictions_df[predictions_df["y_true"] == 1]["risk_category"].value_counts()
    
    print(f"\n--- FN Rate by Category ---")
    for category in fn_by_category.index:
        fn_count = fn_by_category[category]
        total = total_by_category.get(category, 0)
        rate = fn_count / total if total > 0 else 0
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {category:<25} {bar} {rate:.1%} ({fn_count}/{total})")
    
    # By confidence
    if "y_score" in fn_df.columns:
        print(f"\n--- FN Confidence Distribution ---")
        print(f"  Mean confidence: {fn_df['y_score'].mean():.3f}")
        print(f"  Median confidence: {fn_df['y_score'].median():.3f}")
        print(f"  High-confidence FNs (>0.3): {(fn_df['y_score'] > 0.3).sum()}")
        print(f"  Low-confidence FNs (<0.1):  {(fn_df['y_score'] < 0.1).sum()}")
    
    # Sample FNs for manual inspection
    print(f"\n--- Sample False Negatives (inspect these manually) ---")
    sample = fn_df.head(5)
    for _, row in sample.iterrows():
        print(f"\n  Category: {row['risk_category']}")
        if "y_score" in row:
            print(f"  Confidence: {row['y_score']:.3f}")
        if "paragraph_id" in row and paragraphs_df is not None:
            para = paragraphs_df[paragraphs_df["paragraph_id"] == row["paragraph_id"]]
            if not para.empty:
                text = para.iloc[0]["paragraph"][:200]
                print(f"  Text: {text}...")
    
    return fn_df


def analyze_false_positives(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze paragraphs incorrectly flagged as containing a clause (FP)."""
    fp_mask = (predictions_df["y_true"] == 0) & (predictions_df["y_pred"] == 1)
    fp_df = predictions_df[fp_mask].copy()
    
    if fp_df.empty:
        print("No false positives found.")
        return fp_df
    
    print(f"\n{'=' * 60}")
    print(f"FALSE POSITIVE ANALYSIS ({len(fp_df)} total)")
    print(f"{'=' * 60}")
    
    fp_by_category = fp_df["risk_category"].value_counts()
    total_neg = predictions_df[predictions_df["y_true"] == 0]["risk_category"].value_counts()
    
    print(f"\n--- FP Rate by Category ---")
    for category in fp_by_category.index:
        fp_count = fp_by_category[category]
        total = total_neg.get(category, 0)
        rate = fp_count / total if total > 0 else 0
        print(f"  {category:<25} {rate:.1%} ({fp_count}/{total})")
    
    return fp_df


def generate_error_report(approach: str = "llm"):
    """Generate a full error analysis report for a given approach."""
    pred_file = PROCESSED_DIR / f"{approach}_predictions.parquet"
    
    if not pred_file.exists():
        print(f"No predictions found at {pred_file}")
        return
    
    predictions = pd.read_parquet(pred_file)
    
    # Try to load paragraph texts for context
    paragraphs = None
    para_file = PROCESSED_DIR / "paragraphs_test.parquet"
    if para_file.exists():
        paragraphs = pd.read_parquet(para_file)
    
    print(f"\n{'=' * 60}")
    print(f"ERROR ANALYSIS REPORT: {approach.upper()}")
    print(f"{'=' * 60}")
    
    # Overall stats
    total = len(predictions)
    correct = (predictions["y_true"] == predictions["y_pred"]).sum()
    print(f"  Total predictions: {total}")
    print(f"  Correct: {correct} ({correct/total:.1%})")
    
    # FN analysis (most important for legal — missing a risk clause is worse than a false alarm)
    fn_df = analyze_false_negatives(predictions, paragraphs)
    
    # FP analysis
    fp_df = analyze_false_positives(predictions)
    
    # Category-level recommendations
    print(f"\n{'=' * 60}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 60}")
    
    for category in predictions["risk_category"].unique():
        cat_preds = predictions[predictions["risk_category"] == category]
        fn_rate = ((cat_preds["y_true"] == 1) & (cat_preds["y_pred"] == 0)).mean()
        fp_rate = ((cat_preds["y_true"] == 0) & (cat_preds["y_pred"] == 1)).mean()
        
        if fn_rate > 0.3:
            print(f"\n  ⚠️  {category}: High FN rate ({fn_rate:.1%})")
            print(f"      → Consider: more training data, lower threshold, or LLM routing")
        elif fp_rate > 0.2:
            print(f"\n  ⚠️  {category}: High FP rate ({fp_rate:.1%})")
            print(f"      → Consider: higher threshold, negative example mining")
        else:
            print(f"\n  ✅ {category}: Acceptable ({fn_rate:.1%} FN, {fp_rate:.1%} FP)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", default="llm", choices=["deberta", "llm", "hybrid"])
    args = parser.parse_args()
    
    generate_error_report(args.approach)
