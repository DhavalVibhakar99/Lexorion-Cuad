"""
DeBERTa-v3-base clause detector.
Binary classifier: "Does this paragraph contain a clause from risk category X?"

Trains one model per risk category (or a single multi-label model).
This is Stage 1 of the hybrid pipeline.

Run: python -m src.models.clause_detector --category liability_risk
"""

import argparse
import json
import time
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, precision_score, recall_score


PROCESSED_DIR = Path("data/processed")
CONFIG_DIR = Path("configs")
CHECKPOINT_DIR = Path("checkpoints")


class ClauseDataset(Dataset):
    """PyTorch dataset for paragraph-level clause detection."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, label_column: str, max_length: int = 512):
        self.texts = df["paragraph"].tolist()
        self.labels = df[label_column].tolist()
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
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "accuracy": (preds == labels).mean(),
    }


def compute_class_weights(labels: list) -> torch.Tensor:
    """Compute inverse frequency weights to handle class imbalance."""
    labels_array = np.array(labels)
    n_negative = (labels_array == 0).sum()
    n_positive = (labels_array == 1).sum()
    
    if n_positive == 0:
        return torch.tensor([1.0, 1.0])
    
    weight_negative = len(labels_array) / (2 * n_negative)
    weight_positive = len(labels_array) / (2 * n_positive)
    
    return torch.tensor([weight_negative, weight_positive])


class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss for imbalanced data."""
    
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_detector(
    category: str,
    model_name: str = "microsoft/deberta-v3-base",
    max_seq_length: int = 512,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    use_class_weights: bool = True,
):
    """
    Train a binary clause detector for a specific risk category.
    
    Args:
        category: Risk category key (e.g., 'liability_risk')
    """
    label_column = f"label_{category}"
    output_dir = CHECKPOINT_DIR / f"clause_detector_{category}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print(f"Training Clause Detector: {category}")
    print(f"{'=' * 60}")
    
    # Load data
    train_df = pd.read_parquet(PROCESSED_DIR / "paragraphs_train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "paragraphs_val.parquet")
    
    if label_column not in train_df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {[c for c in train_df.columns if c.startswith('label_')]}")
    
    n_positive_train = train_df[label_column].sum()
    n_negative_train = len(train_df) - n_positive_train
    print(f"  Train: {n_positive_train} positive / {n_negative_train} negative ({n_positive_train/len(train_df):.1%} positive)")
    print(f"  Val:   {val_df[label_column].sum()} positive / {len(val_df) - val_df[label_column].sum()} negative")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Datasets
    train_dataset = ClauseDataset(train_df, tokenizer, label_column, max_seq_length)
    val_dataset = ClauseDataset(val_df, tokenizer, label_column, max_seq_length)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    
    # Class weights
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_df[label_column].tolist())
        print(f"  Class weights: negative={class_weights[0]:.2f}, positive={class_weights[1]:.2f}")
    
    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )
    
    # Trainer
    trainer_cls = WeightedTrainer if use_class_weights else Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }
    if use_class_weights:
        trainer_kwargs["class_weights"] = class_weights
    
    trainer = trainer_cls(**trainer_kwargs)
    
    # Train
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    print(f"\n--- Results for {category} ---")
    print(f"  F1:        {eval_results['eval_f1']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall:    {eval_results['eval_recall']:.4f}")
    print(f"  Train time: {train_time:.0f}s")
    
    # Save model + tokenizer
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    
    # Save results
    results = {
        "category": category,
        "model_name": model_name,
        "eval_f1": eval_results["eval_f1"],
        "eval_precision": eval_results["eval_precision"],
        "eval_recall": eval_results["eval_recall"],
        "train_samples": len(train_df),
        "train_positive": int(n_positive_train),
        "train_time_seconds": train_time,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def predict(
    category: str,
    texts: list,
    model_name: str = "microsoft/deberta-v3-base",
    batch_size: int = 16,
) -> list:
    """
    Run inference with a trained clause detector.
    
    Returns list of dicts: [{"prediction": 0/1, "confidence": float}, ...]
    """
    model_dir = CHECKPOINT_DIR / f"clause_detector_{category}" / "best_model"
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        encoding = tokenizer(
            batch_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        for j in range(len(batch_texts)):
            confidence = probs[j][1].item()  # Probability of positive class
            results.append({
                "prediction": 1 if confidence > 0.5 else 0,
                "confidence": confidence,
            })
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, required=True, help="Risk category to train (e.g., liability_risk)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    
    # Load config for model name
    with open(CONFIG_DIR / "model_config.yaml") as f:
        config = yaml.safe_load(f)
    
    train_detector(
        category=args.category,
        model_name=config["clause_detector"]["model_name"],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
