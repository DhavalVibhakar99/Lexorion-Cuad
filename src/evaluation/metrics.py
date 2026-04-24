"""
Evaluation metrics for contract clause detection and classification.
Computes per-category and aggregate metrics for model comparison.

Usage:
    from src.evaluation.metrics import evaluate_predictions
    results = evaluate_predictions(y_true, y_pred, y_scores, categories)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
)


def jaccard_similarity(pred: str, truth: str) -> float:
    """
    Compute Jaccard similarity between predicted and ground truth text spans.
    This is the CUAD-standard metric for span overlap.
    """
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    
    pred_tokens = set(pred.lower().split())
    truth_tokens = set(truth.lower().split())
    
    intersection = pred_tokens & truth_tokens
    union = pred_tokens | truth_tokens
    
    return len(intersection) / len(union) if union else 0.0


def compute_aupr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def precision_at_recall(
    y_true: np.ndarray, y_scores: np.ndarray, target_recall: float
) -> float:
    """Compute precision at a given recall threshold (e.g., 80% or 90%)."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    # Find the precision at the target recall level
    valid_idx = recall >= target_recall
    if valid_idx.any():
        return precision[valid_idx].max()
    return 0.0


def evaluate_category(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    pred_texts: Optional[List[str]] = None,
    truth_texts: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate a single clause category.
    
    Args:
        y_true: Binary ground truth (1 = clause present)
        y_pred: Binary predictions
        y_scores: Confidence scores (for AUPR)
        pred_texts: Predicted text spans (for Jaccard)
        truth_texts: Ground truth text spans (for Jaccard)
    
    Returns:
        Dictionary of metric name → value
    """
    results = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "support_positive": int(y_true.sum()),
        "support_negative": int((1 - y_true).sum()),
        "predicted_positive": int(y_pred.sum()),
        "true_positives": int(((y_true == 1) & (y_pred == 1)).sum()),
        "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
        "false_negatives": int(((y_true == 1) & (y_pred == 0)).sum()),
    }
    
    # AUPR if scores available
    if y_scores is not None:
        results["aupr"] = compute_aupr(y_true, y_scores)
        results["precision_at_80_recall"] = precision_at_recall(y_true, y_scores, 0.8)
        results["precision_at_90_recall"] = precision_at_recall(y_true, y_scores, 0.9)
    
    # Jaccard if text spans available
    if pred_texts is not None and truth_texts is not None:
        jaccards = [
            jaccard_similarity(p, t) 
            for p, t in zip(pred_texts, truth_texts)
            if t  # Only compute for positive examples
        ]
        results["mean_jaccard"] = np.mean(jaccards) if jaccards else 0.0
    
    return results


def evaluate_predictions(
    results_df: pd.DataFrame,
    category_column: str = "risk_category",
) -> pd.DataFrame:
    """
    Run full evaluation across all categories.
    
    Args:
        results_df: DataFrame with columns:
            - {category_column}: the category name
            - y_true: ground truth binary
            - y_pred: predicted binary
            - y_score: confidence score (optional)
            - pred_text: predicted span text (optional)
            - truth_text: ground truth span text (optional)
        category_column: column to group by
    
    Returns:
        DataFrame with one row per category + aggregate row
    """
    all_results = []
    
    for category, group in results_df.groupby(category_column):
        y_true = group["y_true"].values
        y_pred = group["y_pred"].values
        y_scores = group["y_score"].values if "y_score" in group.columns else None
        pred_texts = group["pred_text"].tolist() if "pred_text" in group.columns else None
        truth_texts = group["truth_text"].tolist() if "truth_text" in group.columns else None
        
        metrics = evaluate_category(y_true, y_pred, y_scores, pred_texts, truth_texts)
        metrics["category"] = category
        all_results.append(metrics)
    
    # Aggregate (macro average)
    metrics_df = pd.DataFrame(all_results)
    
    aggregate = {
        "category": "AGGREGATE (macro)",
        "precision": metrics_df["precision"].mean(),
        "recall": metrics_df["recall"].mean(),
        "f1": metrics_df["f1"].mean(),
        "support_positive": metrics_df["support_positive"].sum(),
        "predicted_positive": metrics_df["predicted_positive"].sum(),
    }
    
    if "aupr" in metrics_df.columns:
        aggregate["aupr"] = metrics_df["aupr"].mean()
    if "mean_jaccard" in metrics_df.columns:
        aggregate["mean_jaccard"] = metrics_df["mean_jaccard"].mean()
    
    metrics_df = pd.concat([metrics_df, pd.DataFrame([aggregate])], ignore_index=True)
    
    return metrics_df


def print_evaluation_report(metrics_df: pd.DataFrame):
    """Pretty-print the evaluation results."""
    print(f"\n{'=' * 80}")
    print("EVALUATION REPORT")
    print(f"{'=' * 80}")
    
    display_cols = ["category", "precision", "recall", "f1", "support_positive"]
    if "aupr" in metrics_df.columns:
        display_cols.append("aupr")
    if "mean_jaccard" in metrics_df.columns:
        display_cols.append("mean_jaccard")
    
    # Format numbers
    formatted = metrics_df[display_cols].copy()
    for col in formatted.columns:
        if col not in ["category", "support_positive"]:
            formatted[col] = formatted[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    print(formatted.to_string(index=False))
    print(f"{'=' * 80}")
