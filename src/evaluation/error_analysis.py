"""
Error analysis for model predictions.

This report answers the question interviewers usually care about after seeing
an F1 score: where does the model fail, and what would you improve next?

Run:
    python -m src.evaluation.error_analysis --approach baseline
"""

import argparse
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
EXAMPLES_DIR = Path("examples")


def _load_predictions(approach: str) -> pd.DataFrame:
    pred_file = PROCESSED_DIR / f"{approach}_predictions.parquet"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"No predictions found at {pred_file}. "
            f"Run the model first, for example: python -m src.models.baseline_detector"
        )
    return pd.read_parquet(pred_file)


def _load_paragraphs() -> pd.DataFrame:
    para_file = PROCESSED_DIR / "paragraphs_test.parquet"
    if not para_file.exists():
        raise FileNotFoundError(
            f"No test paragraphs found at {para_file}. Run the data pipeline first."
        )
    return pd.read_parquet(para_file)


def _with_paragraph_context(
    errors_df: pd.DataFrame,
    paragraphs_df: pd.DataFrame,
) -> pd.DataFrame:
    context_cols = [
        "paragraph_id",
        "contract_title",
        "paragraph",
        "paragraph_length",
        "risk_categories",
        "cuad_categories",
    ]
    available_cols = [col for col in context_cols if col in paragraphs_df.columns]

    enriched = errors_df.merge(
        paragraphs_df[available_cols],
        on="paragraph_id",
        how="left",
    )
    enriched["text_preview"] = (
        enriched["paragraph"]
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 500)
    )
    enriched["error_type"] = enriched.apply(
        lambda row: (
            "false_negative"
            if row["y_true"] == 1 and row["y_pred"] == 0
            else "false_positive"
        ),
        axis=1,
    )

    sort_cols = ["risk_category", "y_score"]
    ascending = [True, False]
    if "paragraph_length" in enriched.columns:
        sort_cols.append("paragraph_length")
        ascending.append(False)

    return enriched.sort_values(sort_cols, ascending=ascending)


def _category_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for category, group in predictions_df.groupby("risk_category"):
        tp = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum())
        tn = int(((group["y_true"] == 0) & (group["y_pred"] == 0)).sum())
        fp = int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum())
        fn = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())

        actual_positive = tp + fn
        actual_negative = tn + fp
        predicted_positive = tp + fp

        precision = tp / predicted_positive if predicted_positive else 0.0
        recall = tp / actual_positive if actual_positive else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

        rows.append(
            {
                "risk_category": category,
                "support_positive": actual_positive,
                "support_negative": actual_negative,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "false_negative_rate": (
                    round(fn / actual_positive, 3) if actual_positive else 0.0
                ),
                "false_positive_rate": (
                    round(fp / actual_negative, 3) if actual_negative else 0.0
                ),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(["f1", "false_negatives"], ascending=[True, False])


def _recommendation(row: pd.Series) -> str:
    category = row["risk_category"]
    fn_rate = row["false_negative_rate"]
    fp_rate = row["false_positive_rate"]
    f1 = row["f1"]

    if fn_rate >= 0.40:
        return (
            f"{category}: high miss rate. Route uncertain examples to the LLM, "
            "review label mapping, and consider lowering the threshold only if "
            "precision remains acceptable."
        )
    if fp_rate >= 0.08:
        return (
            f"{category}: too many false alarms. Add hard negative examples, "
            "raise the threshold, and inspect repeated legal phrases that confuse "
            "the classifier."
        )
    if f1 < 0.60:
        return (
            f"{category}: weak category. Do manual error review first, then test "
            "whether DeBERTa improves context understanding."
        )
    return (
        f"{category}: usable baseline. Keep as local-screening candidate and "
        "focus LLM budget on weaker categories."
    )


def _sample_errors(errors_df: pd.DataFrame, category: str, limit: int = 3) -> list[str]:
    sample = errors_df[errors_df["risk_category"] == category].head(limit)
    lines = []
    for _, row in sample.iterrows():
        score = row.get("y_score", 0.0)
        preview = row.get("text_preview", "")
        lines.append(f"- score={score:.3f}: {preview}")
    return lines


def _write_markdown_report(
    approach: str,
    summary_df: pd.DataFrame,
    false_negatives: pd.DataFrame,
    false_positives: pd.DataFrame,
) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PROCESSED_DIR / f"{approach}_error_analysis.md"

    macro_precision = summary_df["precision"].mean()
    macro_recall = summary_df["recall"].mean()
    macro_f1 = summary_df["f1"].mean()

    lines = [
        f"# {approach.title()} Error Analysis",
        "",
        "This report explains where the model is failing and what to improve next.",
        "",
        "## Executive Summary",
        "",
        f"- Macro precision: {macro_precision:.3f}",
        f"- Macro recall: {macro_recall:.3f}",
        f"- Macro F1: {macro_f1:.3f}",
        f"- Total false negatives: {len(false_negatives)}",
        f"- Total false positives: {len(false_positives)}",
        "",
        "In contract review, false negatives are especially important because they "
        "represent risks the system missed.",
        "",
        "## Category Summary",
        "",
        "| Category | F1 | Precision | Recall | False Negatives | False Positives | FN Rate | FP Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            "| {risk_category} | {f1:.3f} | {precision:.3f} | {recall:.3f} | "
            "{false_negatives} | {false_positives} | {false_negative_rate:.1%} | "
            "{false_positive_rate:.1%} |".format(**row.to_dict())
        )

    lines.extend(["", "## Priority Fixes", ""])
    for _, row in summary_df.head(5).iterrows():
        lines.append(f"- {_recommendation(row)}")

    lines.extend(["", "## Sample Missed Risks (False Negatives)", ""])
    for category in summary_df.head(4)["risk_category"]:
        lines.append(f"### {category}")
        samples = _sample_errors(false_negatives, category)
        lines.extend(samples or ["- No false negatives found for this category."])
        lines.append("")

    lines.extend(["## Sample False Alarms (False Positives)", ""])
    for category in summary_df.head(4)["risk_category"]:
        lines.append(f"### {category}")
        samples = _sample_errors(false_positives, category)
        lines.extend(samples or ["- No false positives found for this category."])
        lines.append("")

    lines.extend(
        [
            "## Interview Explanation",
            "",
            "The baseline is not only evaluated by F1. I also inspect false negatives "
            "and false positives per risk category. False negatives matter most in a "
            "legal review assistant because a missed risky clause may create hidden "
            "exposure. The weakest categories should be improved through threshold "
            "tuning, hard negative mining, better label mapping, DeBERTa context "
            "modeling, or OpenRouter-powered LLM routing for uncertain cases.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_error_report(approach: str = "baseline") -> dict[str, Path]:
    predictions = _load_predictions(approach)
    paragraphs = _load_paragraphs()

    false_negatives = _with_paragraph_context(
        predictions[(predictions["y_true"] == 1) & (predictions["y_pred"] == 0)],
        paragraphs,
    )
    false_positives = _with_paragraph_context(
        predictions[(predictions["y_true"] == 0) & (predictions["y_pred"] == 1)],
        paragraphs,
    )
    summary = _category_summary(predictions)

    summary_path = PROCESSED_DIR / f"{approach}_error_summary.csv"
    fn_path = PROCESSED_DIR / f"{approach}_false_negatives.csv"
    fp_path = PROCESSED_DIR / f"{approach}_false_positives.csv"

    summary.to_csv(summary_path, index=False)
    false_negatives.to_csv(fn_path, index=False)
    false_positives.to_csv(fp_path, index=False)

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    example_summary_path = EXAMPLES_DIR / f"{approach}_error_summary.csv"
    example_fn_path = EXAMPLES_DIR / f"{approach}_false_negatives_sample.csv"
    example_fp_path = EXAMPLES_DIR / f"{approach}_false_positives_sample.csv"
    summary.to_csv(example_summary_path, index=False)
    false_negatives.head(40).to_csv(example_fn_path, index=False)
    false_positives.head(40).to_csv(example_fp_path, index=False)

    report_path = _write_markdown_report(
        approach=approach,
        summary_df=summary,
        false_negatives=false_negatives,
        false_positives=false_positives,
    )

    print("=" * 70)
    print(f"ERROR ANALYSIS COMPLETE: {approach.upper()}")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {fn_path}")
    print(f"  {fp_path}")
    print(f"  {report_path}")
    print(f"  {example_summary_path}")
    print(f"  {example_fn_path}")
    print(f"  {example_fp_path}")

    return {
        "summary": summary_path,
        "false_negatives": fn_path,
        "false_positives": fp_path,
        "report": report_path,
        "example_summary": example_summary_path,
        "example_false_negatives": example_fn_path,
        "example_false_positives": example_fp_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approach",
        default="baseline",
        choices=["baseline", "deberta", "llm", "hybrid"],
    )
    args = parser.parse_args()

    generate_error_report(args.approach)
