"""
Fast TF-IDF + logistic regression baseline for clause detection.

This gives the project real, repeatable metrics before expensive DeBERTa or
LLM evaluation. It trains one binary classifier per risk category and writes
predictions in the same format expected by the evaluation pipeline.

Run:
    python -m src.models.baseline_detector
"""

import os
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import yaml

CONFIG_DIR = Path("configs")
PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
BASELINE_MODEL_PATH = CHECKPOINT_DIR / "baseline_tfidf_logreg.joblib"

# Lexorion is a screening tool: missing a risky clause (false negative) is far
# more costly than an extra flag a reviewer dismisses in seconds. Thresholds
# are therefore tuned recall-first, not for max F1.
RECALL_TARGET = float(os.getenv("LEXORION_RECALL_TARGET", "0.90"))


def _label_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("label_")]


def _best_threshold(y_true, y_score) -> tuple[float, float]:
    """Choose a decision threshold on validation data using F1."""
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in [x / 100 for x in range(10, 91, 5)]:
        y_pred = (y_score >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = f1

    return best_threshold, best_f1


def _recall_first_threshold(
    y_true, y_score, recall_target: float = RECALL_TARGET
) -> tuple[float, float, float]:
    """
    Choose the highest threshold whose validation recall still meets the
    target (i.e. maximize precision subject to recall >= target). If no
    threshold reaches the target, fall back to the one with the best recall.

    Returns (threshold, recall_at_threshold, precision_at_threshold).
    """
    candidates = []
    for threshold in [x / 100 for x in range(5, 91, 5)]:
        y_pred = (y_score >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        candidates.append((threshold, recall, precision))

    meeting_target = [c for c in candidates if c[1] >= recall_target]
    if meeting_target:
        # Highest threshold that still meets the recall target.
        return max(meeting_target, key=lambda c: c[0])
    # Target unreachable: take the most recall we can get.
    return max(candidates, key=lambda c: (c[1], c[0]))


def train_baseline() -> pd.DataFrame:
    """Train one baseline classifier per risk category and save test predictions."""
    train_df = pd.read_parquet(PROCESSED_DIR / "paragraphs_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "paragraphs_val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "paragraphs_test.parquet")

    label_cols = _label_columns(train_df)
    if not label_cols:
        raise ValueError("No label_ columns found. Run the chunking pipeline first.")

    all_predictions = []
    summary_rows = []
    trained_models = {}
    thresholds = {}

    print("=" * 70)
    print("TRAINING BASELINE: TF-IDF + Logistic Regression")
    print("=" * 70)

    start_all = time.time()

    for label_col in label_cols:
        category = label_col.replace("label_", "")
        y_train = train_df[label_col].astype(int)
        y_test = test_df[label_col].astype(int)

        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)

        print(f"\n--- {category} ---")
        print(f"  Train positives: {positives}")
        print(f"  Train negatives: {negatives}")

        if positives == 0:
            print("  Skipping: no positive examples in train split.")
            continue

        model = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        max_features=30000,
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        solver="liblinear",
                    ),
                ),
            ]
        )

        start = time.time()
        model.fit(train_df["paragraph"], y_train)
        train_seconds = time.time() - start

        val_y = val_df[label_col].astype(int).to_numpy()
        val_scores = model.predict_proba(val_df["paragraph"])[:, 1]
        threshold, val_recall, val_precision = _recall_first_threshold(
            val_y, val_scores
        )
        trained_models[category] = model
        thresholds[category] = threshold

        y_score = model.predict_proba(test_df["paragraph"])[:, 1]
        y_pred = (y_score >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(
            f"  Threshold: {threshold:.2f} "
            f"(val recall={val_recall:.3f}, val precision={val_precision:.3f}, "
            f"target recall>={RECALL_TARGET})"
        )
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1:        {f1:.3f}")
        print(f"  Train time: {train_seconds:.1f}s")

        summary_rows.append(
            {
                "risk_category": category,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "threshold": threshold,
                "recall_target": RECALL_TARGET,
                "val_recall": val_recall,
                "val_precision": val_precision,
                "train_positive": positives,
                "test_positive": int(y_test.sum()),
                "train_seconds": round(train_seconds, 2),
            }
        )

        category_predictions = pd.DataFrame(
            {
                "paragraph_id": test_df["paragraph_id"],
                "risk_category": category,
                "y_true": y_test.to_numpy(),
                "y_pred": y_pred,
                "y_score": y_score,
                "model": "tfidf_logistic_regression",
            }
        )
        all_predictions.append(category_predictions)

    if not all_predictions:
        raise RuntimeError("No baseline predictions were generated.")

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_path = PROCESSED_DIR / "baseline_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = PROCESSED_DIR / "baseline_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": "tfidf_logistic_regression",
        "models": trained_models,
        "thresholds": thresholds,
        "threshold_policy": f"recall_first>={RECALL_TARGET}",
        "label_columns": label_cols,
        "summary": summary_rows,
    }
    joblib.dump(artifact, BASELINE_MODEL_PATH)

    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved model artifact to {BASELINE_MODEL_PATH}")
    print(f"Total time: {time.time() - start_all:.1f}s")

    return predictions_df


def load_baseline_model(model_path: Path = BASELINE_MODEL_PATH) -> dict:
    """Load the trained baseline model artifact."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"No baseline model artifact found at {model_path}. "
            "Run: python -m src.models.baseline_detector"
        )
    return joblib.load(model_path)


def _category_display_names() -> dict[str, str]:
    try:
        with open(CONFIG_DIR / "category_mapping.yaml") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        return {}

    return {
        key: value.get("display_name", key.replace("_", " ").title())
        for key, value in config.get("risk_categories", {}).items()
    }


def _risk_level(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "moderate"
    if score >= 0.45:
        return "low"
    return "none"


def analyze_contract_with_baseline(
    paragraphs: list[str],
    contract_id: str = "uploaded",
    max_detections: int = 30,
) -> dict:
    """Analyze contract paragraphs using the saved baseline artifact."""
    artifact = load_baseline_model()
    models = artifact["models"]
    thresholds = artifact["thresholds"]
    display_names = _category_display_names()

    start = time.time()
    detections = []
    risk_scores = {category: 0.0 for category in models}

    for idx, paragraph in enumerate(paragraphs):
        paragraph_id = f"{contract_id}_p{idx:04d}"
        for category, model in models.items():
            score = float(model.predict_proba([paragraph])[0, 1])
            threshold = float(thresholds.get(category, 0.5))
            is_present = score >= threshold
            risk_scores[category] = max(
                risk_scores.get(category, 0.0), score if is_present else 0.0
            )

            if not is_present:
                continue

            display_name = display_names.get(
                category, category.replace("_", " ").title()
            )
            detections.append(
                {
                    "paragraph_id": paragraph_id,
                    "risk_category": category,
                    "is_present": True,
                    "confidence": score,
                    "risk_level": _risk_level(score),
                    "summary": (
                        f"Baseline model flagged this paragraph as {display_name} "
                        f"with {score:.0%} confidence."
                    ),
                    "extracted_clause": paragraph,
                    "model_used": artifact.get("model_name", "baseline"),
                    "threshold": threshold,
                }
            )

    detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    flagged_paragraphs = len({item["paragraph_id"] for item in detections})

    return {
        "contract_id": contract_id,
        "total_paragraphs": len(paragraphs),
        "flagged_paragraphs": flagged_paragraphs,
        "risk_scores": {key: round(value, 3) for key, value in risk_scores.items()},
        "processing_time_seconds": round(time.time() - start, 2),
        "total_cost_usd": 0.0,
        "detections": detections[:max_detections],
    }


if __name__ == "__main__":
    train_baseline()
