"""
Parse the raw CUAD SQuAD-format data into a clean, analysis-ready format.
Maps the 41 CUAD labels to our 8 risk categories.

Run: python -m src.data_pipeline.parse_cuad
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CONFIG_DIR = Path("configs")


# The 41 CUAD question templates map to these category names
# (extracted from the CUAD question list)
CUAD_QUESTION_TO_CATEGORY = {
    "Highlight the parts (if any) of this contract related to \"Document Name\"": "Document Name",
    "Highlight the parts (if any) of this contract related to \"Parties\"": "Parties",
    "Highlight the parts (if any) of this contract related to \"Agreement Date\"": "Agreement Date",
    "Highlight the parts (if any) of this contract related to \"Effective Date\"": "Effective Date",
    "Highlight the parts (if any) of this contract related to \"Expiration Date\"": "Expiration Date",
    "Highlight the parts (if any) of this contract related to \"Renewal Term\"": "Renewal Term",
    "Highlight the parts (if any) of this contract related to \"Notice Period To Terminate Renewal\"": "Notice Period To Terminate Renewal",
    "Highlight the parts (if any) of this contract related to \"Governing Law\"": "Governing Law",
    "Highlight the parts (if any) of this contract related to \"Most Favored Nation\"": "Most Favored Nation",
    "Highlight the parts (if any) of this contract related to \"Non-Compete\"": "Non-Compete",
    "Highlight the parts (if any) of this contract related to \"Exclusivity\"": "Exclusivity",
    "Highlight the parts (if any) of this contract related to \"No-Solicit Of Customers\"": "No-Solicit Of Customers",
    "Highlight the parts (if any) of this contract related to \"Competitive Restriction Exception\"": "Competitive Restriction Exception",
    "Highlight the parts (if any) of this contract related to \"No-Solicit Of Employees\"": "No-Solicit Of Employees",
    "Highlight the parts (if any) of this contract related to \"Non-Disparagement\"": "Non-Disparagement",
    "Highlight the parts (if any) of this contract related to \"Termination For Convenience\"": "Termination For Convenience",
    "Highlight the parts (if any) of this contract related to \"Rofr/Rofo/Rofn\"": "Rofr/Rofo/Rofn",
    "Highlight the parts (if any) of this contract related to \"Change Of Control\"": "Change Of Control",
    "Highlight the parts (if any) of this contract related to \"Anti-Assignment\"": "Anti-Assignment",
    "Highlight the parts (if any) of this contract related to \"Revenue/Profit Sharing\"": "Revenue/Profit Sharing",
    "Highlight the parts (if any) of this contract related to \"Price Restrictions\"": "Price Restrictions",
    "Highlight the parts (if any) of this contract related to \"Minimum Commitment\"": "Minimum Commitment",
    "Highlight the parts (if any) of this contract related to \"Volume Restriction\"": "Volume Restriction",
    "Highlight the parts (if any) of this contract related to \"Ip Ownership Assignment\"": "Ip Ownership Assignment",
    "Highlight the parts (if any) of this contract related to \"Joint Ip Ownership\"": "Joint Ip Ownership",
    "Highlight the parts (if any) of this contract related to \"License Grant\"": "License Grant",
    "Highlight the parts (if any) of this contract related to \"Non-Transferable License\"": "Non-Transferable License",
    "Highlight the parts (if any) of this contract related to \"Affiliate License-Loss Of Ip\"": "Affiliate License-Loss Of Ip",
    "Highlight the parts (if any) of this contract related to \"Affiliate License-Licensor\"": "Affiliate License-Licensor",
    "Highlight the parts (if any) of this contract related to \"Affiliate License-Licensee\"": "Affiliate License-Licensee",
    "Highlight the parts (if any) of this contract related to \"Unlimited/All-You-Can-Eat-License\"": "Unlimited/All-You-Can-Eat-License",
    "Highlight the parts (if any) of this contract related to \"Irrevocable Or Perpetual License\"": "Irrevocable Or Perpetual License",
    "Highlight the parts (if any) of this contract related to \"Source Code Escrow\"": "Source Code Escrow",
    "Highlight the parts (if any) of this contract related to \"Post-Termination Services\"": "Post-Termination Services",
    "Highlight the parts (if any) of this contract related to \"Audit Rights\"": "Audit Rights",
    "Highlight the parts (if any) of this contract related to \"Uncapped Liability\"": "Uncapped Liability",
    "Highlight the parts (if any) of this contract related to \"Cap On Liability\"": "Cap On Liability",
    "Highlight the parts (if any) of this contract related to \"Liquidated Damages\"": "Liquidated Damages",
    "Highlight the parts (if any) of this contract related to \"Warranty Duration\"": "Warranty Duration",
    "Highlight the parts (if any) of this contract related to \"Insurance\"": "Insurance",
    "Highlight the parts (if any) of this contract related to \"Covenant Not To Sue\"": "Covenant Not To Sue",
    "Highlight the parts (if any) of this contract related to \"Third Party Beneficiary\"": "Third Party Beneficiary",
    "Highlight the parts (if any) of this contract related to \"Limitation Of Liability\"": "Limitation Of Liability",
    "Highlight the parts (if any) of this contract related to \"Effect Of Termination\"": "Effect Of Termination",
    "Highlight the parts (if any) of this contract related to \"Indemnification\"": "Indemnification",
    "Highlight the parts (if any) of this contract related to \"Consent To Assignment\"": "Consent To Assignment",
}


def extract_category_from_question(question: str) -> str:
    """Extract the CUAD category name from the question template."""
    if question in CUAD_QUESTION_TO_CATEGORY:
        return CUAD_QUESTION_TO_CATEGORY[question]
    
    # Fallback: try to parse from the pattern
    import re
    match = re.search(r'"([^"]+)"', question)
    if match:
        return match.group(1)
    return "Unknown"


def load_category_mapping() -> dict:
    """Load the 41→8 risk category mapping from config."""
    with open(CONFIG_DIR / "category_mapping.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Build reverse mapping: cuad_label → risk_category
    label_to_risk = {}
    for risk_key, risk_info in config["risk_categories"].items():
        for cuad_label in risk_info["cuad_labels"]:
            label_to_risk[cuad_label] = {
                "risk_category": risk_key,
                "display_name": risk_info["display_name"],
                "severity_weight": risk_info["severity_weight"],
            }
    
    return label_to_risk, config


def parse_cuad():
    """Parse CUAD into analysis-ready DataFrames."""
    
    print("Loading CUAD dataset...")
    dataset = load_from_disk(str(RAW_DIR / "cuad_hf"))
    
    label_to_risk, config = load_category_mapping()
    
    # === Build clause-level DataFrame ===
    records = []
    
    for example in dataset["train"]:
        cuad_category = extract_category_from_question(example["question"])
        has_answer = len(example["answers"]["text"]) > 0
        
        risk_info = label_to_risk.get(cuad_category, None)
        
        record = {
            "contract_title": example["title"],
            "paragraph": example["context"],
            "cuad_category": cuad_category,
            "has_clause": has_answer,
            "answer_texts": example["answers"]["text"] if has_answer else [],
            "answer_starts": example["answers"]["answer_start"] if has_answer else [],
            "risk_category": risk_info["risk_category"] if risk_info else None,
            "risk_display_name": risk_info["display_name"] if risk_info else None,
            "severity_weight": risk_info["severity_weight"] if risk_info else None,
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # === Summary stats ===
    print(f"\n{'=' * 60}")
    print("PARSED DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total QA pairs:           {len(df):,}")
    print(f"Unique contracts:         {df['contract_title'].nunique()}")
    print(f"Unique CUAD categories:   {df['cuad_category'].nunique()}")
    print(f"Pairs with clauses:       {df['has_clause'].sum():,} ({df['has_clause'].mean():.1%})")
    print(f"Mapped to risk categories:{df['risk_category'].notna().sum():,}")
    
    print(f"\n--- Clause counts per Risk Category ---")
    risk_counts = (
        df[df["has_clause"] & df["risk_category"].notna()]
        .groupby("risk_display_name")
        .size()
        .sort_values(ascending=False)
    )
    for name, count in risk_counts.items():
        print(f"  {name:<30} {count:>5} positive examples")
    
    print(f"\n--- Clause counts per CUAD Category (top 15) ---")
    cuad_counts = (
        df[df["has_clause"]]
        .groupby("cuad_category")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )
    for name, count in cuad_counts.items():
        print(f"  {name:<40} {count:>5}")
    
    # === Save processed data ===
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Full dataset
    df.to_parquet(PROCESSED_DIR / "cuad_parsed.parquet", index=False)
    print(f"\nSaved full dataset to {PROCESSED_DIR / 'cuad_parsed.parquet'}")
    
    # Risk-category-only subset (what we actually train on)
    df_risk = df[df["risk_category"].notna()].copy()
    df_risk.to_parquet(PROCESSED_DIR / "cuad_risk_categories.parquet", index=False)
    print(f"Saved risk subset to {PROCESSED_DIR / 'cuad_risk_categories.parquet'}")
    
    # Positive examples only (for quick reference)
    df_positive = df[df["has_clause"]].copy()
    df_positive.to_parquet(PROCESSED_DIR / "cuad_positive_clauses.parquet", index=False)
    print(f"Saved positive clauses to {PROCESSED_DIR / 'cuad_positive_clauses.parquet'}")
    
    return df


if __name__ == "__main__":
    parse_cuad()
