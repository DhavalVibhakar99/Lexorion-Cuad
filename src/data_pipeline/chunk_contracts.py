"""
Chunk contracts into paragraphs with contract-level train/val/test splitting.

APPROACH: Instead of offset-based span matching (fragile — breaks when text is
cleaned/reformatted), we use TEXT SEARCH: check if each annotation's answer_text
appears inside the paragraph text. This is robust to any text cleaning.

Run: python -m src.data_pipeline.chunk_contracts
"""

import re
import hashlib
import pandas as pd
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split


PROCESSED_DIR = Path("data/processed")
MIN_PARAGRAPH_CHARS = 100
MAX_PARAGRAPH_CHARS = 3000


def split_into_paragraphs(text: str, min_length: int = MIN_PARAGRAPH_CHARS) -> List[str]:
    """Split contract text into paragraphs using multiple delimiter strategies."""
    paragraphs = []

    # Strategy 1: split on double newlines
    chunks = re.split(r"\n\s*\n", text)

    # If that produces only 1 chunk (no double newlines), try single newlines
    # with heuristic: a line starting with a section number or uppercase header
    if len(chunks) <= 1:
        lines = text.split("\n")
        current = ""
        for line in lines:
            # Start new paragraph on section headers or numbered sections
            if re.match(r"^\s*(SECTION|ARTICLE|EXHIBIT|\d+\.\s+[A-Z])", line.strip()):
                if current.strip() and len(current.strip()) >= min_length:
                    paragraphs.append(current.strip())
                current = line
            else:
                current += "\n" + line
        if current.strip() and len(current.strip()) >= min_length:
            paragraphs.append(current.strip())

        # If header-based splitting also failed, fall back to fixed-size chunks
        if not paragraphs:
            words = text.split()
            for i in range(0, len(words), 200):
                chunk = " ".join(words[i:i + 200])
                if len(chunk) >= min_length:
                    paragraphs.append(chunk)
    else:
        paragraphs = [c.strip() for c in chunks if len(c.strip()) >= min_length]

    # Split oversized paragraphs
    final = []
    for para in paragraphs:
        if len(para) <= MAX_PARAGRAPH_CHARS:
            final.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub = ""
            for sent in sentences:
                if len(sub) + len(sent) > MAX_PARAGRAPH_CHARS and sub:
                    if len(sub.strip()) >= min_length:
                        final.append(sub.strip())
                    sub = sent
                else:
                    sub += (" " + sent) if sub else sent
            if len(sub.strip()) >= min_length:
                final.append(sub.strip())

    return final


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n", text)
    return text.strip()


def clean_paragraph(text: str) -> str:
    """Clean a single paragraph."""
    return clean_text(text)


def compute_paragraph_id(contract_title: str, paragraph: str) -> str:
    content = f"{contract_title}::{paragraph[:200]}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def answer_in_paragraph(answer_text: str, paragraph: str) -> bool:
    """Check if an answer span appears inside a paragraph using text search."""
    # Normalize whitespace for matching
    ans_normalized = " ".join(answer_text.lower().split())
    para_normalized = " ".join(paragraph.lower().split())
    return ans_normalized in para_normalized


def create_contract_splits(df, test_size=0.2, val_size=0.1, random_seed=42):
    contracts = df["contract_title"].unique().tolist()
    train_val, test = train_test_split(contracts, test_size=test_size, random_state=random_seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=random_seed)
    split_map = {c: "train" for c in train}
    split_map.update({c: "val" for c in val})
    split_map.update({c: "test" for c in test})
    df["split"] = df["contract_title"].map(split_map)
    return df


def chunk_contracts():
    print("Loading parsed CUAD data...")
    df = pd.read_parquet(PROCESSED_DIR / "cuad_risk_categories.parquet")

    df["contract_title"] = df["contract_title"].astype(str)
    df["paragraph"] = df["paragraph"].astype(str)
    df["risk_category"] = df["risk_category"].astype(str)
    df["cuad_category"] = df["cuad_category"].astype(str)

    n_contracts = df["contract_title"].nunique()
    print(f"Loaded {len(df):,} rows from {n_contracts} contracts")

    # === Step 1: Get unique contract texts and collect all annotations ===
    print("\nCollecting contract texts and annotations...")
    contract_texts = {}  # title -> full text
    contract_annotations = {}  # title -> list of {risk_category, cuad_category, answer_text}

    for _, row in df.iterrows():
        title = row["contract_title"]
        contract_texts[title] = row["paragraph"]  # the context (full contract text)

        if row["has_clause"]:
            answer_texts = row.get("answer_texts", [])
            if hasattr(answer_texts, "tolist"):
                answer_texts = answer_texts.tolist()
            if not isinstance(answer_texts, list) or len(answer_texts) == 0:
                continue

            if title not in contract_annotations:
                contract_annotations[title] = []

            for ans_text in answer_texts:
                if ans_text:
                    contract_annotations[title].append({
                        "risk_category": row["risk_category"],
                        "cuad_category": row["cuad_category"],
                        "answer_text": str(ans_text),
                    })

    total_annotations = sum(len(v) for v in contract_annotations.values())
    print(f"  Unique contracts:       {len(contract_texts)}")
    print(f"  Contracts w/ clauses:   {len(contract_annotations)}")
    print(f"  Total answer spans:     {total_annotations:,}")

    # === Step 2: Split each contract into paragraphs ===
    print("Splitting contracts into paragraphs...")
    risk_categories = df["risk_category"].dropna().unique().tolist()
    paragraph_records = []
    total_matched = 0

    for title, full_text in contract_texts.items():
        paragraphs = split_into_paragraphs(full_text)
        annotations = contract_annotations.get(title, [])

        for para_raw in paragraphs:
            para_clean = clean_paragraph(para_raw)

            # === Step 3: Label using TEXT SEARCH (not offset matching) ===
            matched_categories = set()
            matched_cuad = set()
            matched_answers = []

            for ann in annotations:
                if answer_in_paragraph(ann["answer_text"], para_raw):
                    matched_categories.add(ann["risk_category"])
                    matched_cuad.add(ann["cuad_category"])
                    matched_answers.append(ann["answer_text"])
                    total_matched += 1

            paragraph_records.append({
                "contract_title": title,
                "paragraph_id": compute_paragraph_id(title, para_clean),
                "paragraph": para_clean,
                "paragraph_length": len(para_clean.split()),
                "risk_categories": list(matched_categories),
                "cuad_categories": list(matched_cuad),
                "num_risk_categories": len(matched_categories),
                "has_any_risk": len(matched_categories) > 0,
                "answer_texts": matched_answers,
            })

    df_paragraphs = pd.DataFrame(paragraph_records)
    df_paragraphs = df_paragraphs.drop_duplicates(subset=["paragraph_id"])

    n_positive = df_paragraphs["has_any_risk"].sum()
    n_negative = (~df_paragraphs["has_any_risk"]).sum()

    print(f"\nTotal unique paragraphs:     {len(df_paragraphs):,}")
    print(f"  With risk clauses:         {n_positive:,}")
    print(f"  Without risk clauses:      {n_negative:,}")
    print(f"  Positive rate:             {n_positive / len(df_paragraphs):.1%}")
    print(f"  Avg paragraphs/contract:   {len(df_paragraphs) / n_contracts:.1f}")
    print(f"  Matched answer spans:      {total_matched:,} / {total_annotations:,}")

    if n_positive == 0:
        print("\n⚠️  WARNING: Zero positive paragraphs!")
        # Debug output
        sample_title = list(contract_annotations.keys())[0]
        sample_text = contract_texts[sample_title]
        sample_anns = contract_annotations[sample_title][:3]
        sample_paras = split_into_paragraphs(sample_text)

        print(f"\n  Debug — Contract: {sample_title}")
        print(f"  Full text length: {len(sample_text)} chars")
        print(f"  Split into {len(sample_paras)} paragraphs")
        if sample_paras:
            print(f"  First paragraph (100 chars): {repr(sample_paras[0][:100])}")
        for ann in sample_anns:
            found_in_any = any(answer_in_paragraph(ann["answer_text"], p) for p in sample_paras)
            found_in_full = ann["answer_text"].lower() in sample_text.lower()
            print(f"\n  Annotation: '{ann['answer_text'][:60]}...'")
            print(f"    In full text: {found_in_full}")
            print(f"    In any paragraph: {found_in_any}")
        return df_paragraphs

    # === Step 4: Binary label columns ===
    for cat in risk_categories:
        df_paragraphs[f"label_{cat}"] = df_paragraphs["risk_categories"].apply(
            lambda x, c=cat: 1 if c in x else 0
        )

    # === Step 5: Contract-level splits ===
    df_paragraphs = create_contract_splits(df_paragraphs)

    print(f"\n--- Split Distribution (contract-level) ---")
    for split_name in ["train", "val", "test"]:
        mask = df_paragraphs["split"] == split_name
        n = mask.sum()
        nc = df_paragraphs.loc[mask, "contract_title"].nunique()
        np_ = (mask & df_paragraphs["has_any_risk"]).sum()
        print(f"  {split_name:<6}: {n:>6} paragraphs from {nc:>3} contracts ({np_} positive)")

    print(f"\n--- Per-Category Positive Examples (train split) ---")
    train_df = df_paragraphs[df_paragraphs["split"] == "train"]
    for cat in sorted(risk_categories):
        col = f"label_{cat}"
        if col in train_df.columns:
            count = int(train_df[col].sum())
            flag = " ⚠️  LOW" if count < 50 else ""
            print(f"  {cat:<30} {count:>5} positives{flag}")

    # === Step 6: Save ===
    df_paragraphs.to_parquet(PROCESSED_DIR / "paragraphs_chunked.parquet", index=False)
    for split_name in ["train", "val", "test"]:
        df_paragraphs[df_paragraphs["split"] == split_name].to_parquet(
            PROCESSED_DIR / f"paragraphs_{split_name}.parquet", index=False
        )

    print(f"\nSaved to {PROCESSED_DIR / 'paragraphs_chunked.parquet'}")
    print("Saved train/val/test splits as separate files")
    print(f"\n✅ Data pipeline complete! Ready for modeling.")
    return df_paragraphs


if __name__ == "__main__":
    chunk_contracts()
