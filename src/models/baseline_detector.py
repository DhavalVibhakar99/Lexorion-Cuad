"""
Fast TF-IDF + logistic regression baseline for clause detection.

This gives the project real, repeatable metrics before expensive DeBERTa or
LLM evaluation. It trains one binary classifier per risk category and writes
predictions in the same format expected by the evaluation pipeline.

Run:
    python -m src.models.baseline_detector
"""

import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


PROCESSED_DIR = Path("data/processed")


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

        val_scores = model.predict_proba(val_df["paragraph"])[:, 1]
        threshold, val_f1 = _best_threshold(val_df[label_col].astype(int).to_numpy(), val_scores)

        y_score = model.predict_proba(test_df["paragraph"])[:, 1]
        y_pred = (y_score >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"  Threshold: {threshold:.2f} (val F1={val_f1:.3f})")
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
                "val_f1": val_f1,
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

    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Total time: {time.time() - start_all:.1f}s")

    return predictions_df


if __name__ == "__main__":
    train_baseline()
