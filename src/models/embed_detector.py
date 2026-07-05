"""
Transformer-embedding clause detector: frozen MiniLM sentence embeddings +
per-category logistic regression.

This is the CPU-friendly transformer step between the TF-IDF baseline and a
fine-tuned DeBERTa: the encoder is a pretrained transformer used as a frozen
feature extractor, so no GPU fine-tuning is required, but the representation
understands legal phrasing far better than bag-of-words.

Thresholds follow the same recall-first policy as the baseline (screening
tool: false negatives cost more than false positives).

Run:
    python -m src.models.embed_detector
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from src.models.baseline_detector import RECALL_TARGET, _recall_first_threshold

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CACHE_PATH = PROCESSED_DIR / "embeddings_minilm.npz"
EMBED_ARTIFACT_PATH = CHECKPOINT_DIR / "embed_minilm_logreg.joblib"


def _label_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("label_")]


def _load_or_compute_embeddings(splits: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Embed each split's paragraphs, caching to disk (embedding ~25k
    paragraphs takes minutes on CPU; the cache makes reruns instant)."""
    if EMBED_CACHE_PATH.exists():
        cached = np.load(EMBED_CACHE_PATH, allow_pickle=False)
        if all(name in cached for name in splits):
            sizes_match = all(
                cached[name].shape[0] == len(df) for name, df in splits.items()
            )
            if sizes_match:
                print(f"Loaded cached embeddings from {EMBED_CACHE_PATH}")
                return {name: cached[name] for name in splits}

    from sentence_transformers import SentenceTransformer

    print(f"Encoding paragraphs with {EMBED_MODEL_NAME} (first run only)...")
    encoder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = {}
    for name, df in splits.items():
        start = time.time()
        embeddings[name] = encoder.encode(
            df["paragraph"].tolist(),
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        print(f"  {name}: {len(df):,} paragraphs in {time.time() - start:.0f}s")

    np.savez_compressed(EMBED_CACHE_PATH, **embeddings)
    print(f"Cached embeddings to {EMBED_CACHE_PATH}")
    return embeddings


def train_embed_detector() -> pd.DataFrame:
    """Train one logistic head per risk category on MiniLM embeddings."""
    splits = {
        "train": pd.read_parquet(PROCESSED_DIR / "paragraphs_train.parquet"),
        "val": pd.read_parquet(PROCESSED_DIR / "paragraphs_val.parquet"),
        "test": pd.read_parquet(PROCESSED_DIR / "paragraphs_test.parquet"),
    }
    label_cols = _label_columns(splits["train"])
    if not label_cols:
        raise ValueError("No label_ columns found. Run the chunking pipeline first.")

    X = _load_or_compute_embeddings(splits)

    all_predictions = []
    summary_rows = []
    heads = {}
    thresholds = {}

    print("=" * 70)
    print(f"TRAINING EMBED DETECTOR: {EMBED_MODEL_NAME} + LogisticRegression")
    print("=" * 70)

    for label_col in label_cols:
        category = label_col.replace("label_", "")
        y_train = splits["train"][label_col].astype(int)
        y_val = splits["val"][label_col].astype(int).to_numpy()
        y_test = splits["test"][label_col].astype(int)

        if y_train.sum() == 0:
            print(f"--- {category}: skipped (no positives) ---")
            continue

        head = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0, solver="liblinear"
        )
        start = time.time()
        head.fit(X["train"], y_train)
        train_seconds = time.time() - start

        val_scores = head.predict_proba(X["val"])[:, 1]
        threshold, val_recall, val_precision = _recall_first_threshold(
            y_val, val_scores
        )
        heads[category] = head
        thresholds[category] = threshold

        y_score = head.predict_proba(X["test"])[:, 1]
        y_pred = (y_score >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\n--- {category} ---")
        print(
            f"  Threshold: {threshold:.2f} "
            f"(val recall={val_recall:.3f}, val precision={val_precision:.3f})"
        )
        print(f"  Test precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

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
                "train_seconds": round(train_seconds, 2),
            }
        )
        all_predictions.append(
            pd.DataFrame(
                {
                    "paragraph_id": splits["test"]["paragraph_id"],
                    "risk_category": category,
                    "y_true": y_test.to_numpy(),
                    "y_pred": y_pred,
                    "y_score": y_score,
                    "model": "minilm_logistic_regression",
                }
            )
        )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_parquet(PROCESSED_DIR / "embed_predictions.parquet", index=False)
    pd.DataFrame(summary_rows).to_csv(
        PROCESSED_DIR / "embed_training_summary.csv", index=False
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": "minilm_logistic_regression",
            "encoder_name": EMBED_MODEL_NAME,
            "heads": heads,
            "thresholds": thresholds,
            "threshold_policy": f"recall_first>={RECALL_TARGET}",
            "summary": summary_rows,
        },
        EMBED_ARTIFACT_PATH,
    )

    print("\nSaved predictions, summary, and artifact.")
    return predictions_df


if __name__ == "__main__":
    train_embed_detector()
