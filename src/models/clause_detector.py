"""
DeBERTa multi-label clause detector: one encoder, 8 sigmoid heads — "which
risk categories does this paragraph contain?"

Designed for a single free-tier Colab T4 run (~30-45 min): one multi-label
fine-tune instead of 8 separate binary models, 256-token sequences (CUAD
paragraphs are short), fp16, BCE loss with per-class pos_weight for the ~2%
positive rate.

Thresholds follow the project's recall-first policy (precision-max subject to
validation recall >= 0.90 per category), and predictions are written in the
exact format src/evaluation/model_comparison.py expects.

Run (GPU):
    python -m src.models.clause_detector

Smoke test (CPU, tiny model, seconds — validates the whole pipeline):
    python -m src.models.clause_detector --smoke
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.models.baseline_detector import RECALL_TARGET, _recall_first_threshold

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
MODEL_DIR = CHECKPOINT_DIR / "deberta_multilabel"
DEFAULT_MODEL = "microsoft/deberta-v3-base"
SMOKE_MODEL = "hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification"


def _label_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("label_")]


class MultiLabelClauseDataset(Dataset):
    """Paragraphs with an 8-dim float label vector."""

    def __init__(self, df: pd.DataFrame, tokenizer, label_cols: list[str], max_length: int):
        self.texts = df["paragraph"].tolist()
        self.labels = df[label_cols].to_numpy(dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx]),
        }


class PosWeightTrainer(Trainer):
    """BCE-with-logits loss with per-class pos_weight for heavy imbalance."""

    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.pos_weight.to(outputs.logits.device) if self.pos_weight is not None else None
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs.logits, labels, pos_weight=weight
        )
        return (loss, outputs) if return_outputs else loss


def _macro_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def train_multilabel(
    model_name: str = DEFAULT_MODEL,
    max_length: int = 256,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 2,
    max_train_samples: int | None = None,
) -> pd.DataFrame:
    """Train the multi-label detector and write test predictions."""
    train_df = pd.read_parquet(PROCESSED_DIR / "paragraphs_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "paragraphs_val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "paragraphs_test.parquet")

    if max_train_samples:
        train_df = train_df.sample(n=min(max_train_samples, len(train_df)), random_state=42)
        val_df = val_df.head(max(64, max_train_samples // 4))
        test_df = test_df.head(max(64, max_train_samples // 4))

    label_cols = _label_columns(train_df)
    categories = [c.replace("label_", "") for c in label_cols]
    print(f"Training {model_name} on {len(train_df):,} paragraphs, {len(categories)} categories")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_cols),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,  # fresh classification head either way
        # Keep master weights fp32: AMP (fp16=True) scales grads and cannot
        # unscale fp16 params — transformers 5.x otherwise casts the model.
        torch_dtype=torch.float32,
    )

    # pos_weight = negatives/positives per class, capped to avoid instability.
    y_train = train_df[label_cols].to_numpy()
    pos = y_train.sum(axis=0)
    pos_weight = torch.tensor(
        np.clip((len(y_train) - pos) / np.maximum(pos, 1), 1.0, 30.0),
        dtype=torch.float32,
    )
    print("pos_weight:", {c: round(float(w), 1) for c, w in zip(categories, pos_weight)})

    train_ds = MultiLabelClauseDataset(train_df, tokenizer, label_cols, max_length)
    val_ds = MultiLabelClauseDataset(val_df, tokenizer, label_cols, max_length)
    test_ds = MultiLabelClauseDataset(test_df, tokenizer, label_cols, max_length)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "runs"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
    )

    trainer = PosWeightTrainer(
        pos_weight=pos_weight,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_macro_metrics,
    )

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start
    print(f"Training finished in {train_seconds/60:.1f} min")

    # Recall-first thresholds on validation, applied to test.
    val_scores = 1 / (1 + np.exp(-trainer.predict(val_ds).predictions))
    test_scores = 1 / (1 + np.exp(-trainer.predict(test_ds).predictions))
    y_val = val_df[label_cols].to_numpy()
    y_test = test_df[label_cols].to_numpy()

    thresholds, summary_rows, all_predictions = {}, [], []
    for i, category in enumerate(categories):
        threshold, val_recall, val_precision = _recall_first_threshold(
            y_val[:, i], val_scores[:, i]
        )
        thresholds[category] = float(threshold)
        y_pred = (test_scores[:, i] >= threshold).astype(int)
        summary_rows.append(
            {
                "risk_category": category,
                "precision": precision_score(y_test[:, i], y_pred, zero_division=0),
                "recall": recall_score(y_test[:, i], y_pred, zero_division=0),
                "f1": f1_score(y_test[:, i], y_pred, zero_division=0),
                "threshold": threshold,
                "recall_target": RECALL_TARGET,
                "val_recall": val_recall,
                "val_precision": val_precision,
            }
        )
        all_predictions.append(
            pd.DataFrame(
                {
                    "paragraph_id": test_df["paragraph_id"].to_numpy(),
                    "risk_category": category,
                    "y_true": y_test[:, i],
                    "y_pred": y_pred,
                    "y_score": test_scores[:, i],
                    "model": f"deberta_multilabel::{model_name}",
                }
            )
        )

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    print(
        f"\nMacro: precision={summary['precision'].mean():.3f} "
        f"recall={summary['recall'].mean():.3f} f1={summary['f1'].mean():.3f}"
    )

    pd.concat(all_predictions, ignore_index=True).to_parquet(
        PROCESSED_DIR / "deberta_predictions.parquet", index=False
    )
    summary.to_csv(PROCESSED_DIR / "deberta_training_summary.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR / "best_model"))
    tokenizer.save_pretrained(str(MODEL_DIR / "best_model"))
    with open(MODEL_DIR / "results.json", "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "categories": categories,
                "thresholds": thresholds,
                "threshold_policy": f"recall_first>={RECALL_TARGET}",
                "macro_f1": float(summary["f1"].mean()),
                "macro_recall": float(summary["recall"].mean()),
                "macro_precision": float(summary["precision"].mean()),
                "train_seconds": round(train_seconds, 1),
                "max_length": max_length,
            },
            f,
            indent=2,
        )
    print(f"Saved predictions, summary, and model to {MODEL_DIR}")
    return summary


_PREDICT_CACHE: dict = {}


def predict(category: str, texts: list, batch_size: int = 32) -> list:
    """
    Inference helper used by the hybrid pipeline.
    Returns [{"prediction": 0/1, "confidence": float}, ...] for one category.
    """
    if "model" not in _PREDICT_CACHE:
        model_dir = MODEL_DIR / "best_model"
        with open(MODEL_DIR / "results.json") as f:
            meta = json.load(f)
        _PREDICT_CACHE["tokenizer"] = AutoTokenizer.from_pretrained(str(model_dir))
        _PREDICT_CACHE["model"] = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        ).eval()
        _PREDICT_CACHE["meta"] = meta
    meta = _PREDICT_CACHE["meta"]
    if category not in meta["categories"]:
        raise ValueError(f"Unknown category: {category}")
    col = meta["categories"].index(category)
    threshold = meta["thresholds"][category]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _PREDICT_CACHE["model"].to(device)

    results = []
    for i in range(0, len(texts), batch_size):
        encoding = _PREDICT_CACHE["tokenizer"](
            texts[i : i + batch_size],
            max_length=meta.get("max_length", 256),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(**encoding).logits)[:, col]
        results.extend(
            {"prediction": int(p >= threshold), "confidence": float(p)} for p in probs
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU pipeline check: tiny model, 200 samples, 1 epoch.",
    )
    cli = parser.parse_args()

    if cli.smoke:
        train_multilabel(
            model_name=SMOKE_MODEL,
            max_length=64,
            batch_size=16,
            num_epochs=1,
            max_train_samples=200,
        )
    else:
        train_multilabel(
            model_name=cli.model,
            max_length=cli.max_length,
            batch_size=cli.batch_size,
            learning_rate=cli.lr,
            num_epochs=cli.epochs,
        )
