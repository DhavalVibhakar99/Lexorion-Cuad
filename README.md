# Lexorion: Contract Risk Intelligence System

A collaborative AI/data science project for automated legal contract risk analysis. Lexorion uses the CUAD contract dataset to detect business-relevant legal risk clauses and turn long contract text into a structured risk profile.

> Current status: data pipeline and dashboard prototype are in place. Model training, final evaluation metrics, and deployment are the next major milestones.

## What This Does

Lexorion is designed to:

1. Accept contract text as input.
2. Split the contract into analyzable paragraphs.
3. Detect clauses related to legal/business risk.
4. Group detailed CUAD clause labels into 8 business-friendly risk categories.
5. Generate confidence scores, plain-English summaries, and a contract-level risk profile.

In plain English: the system reads a contract and highlights the parts that may create liability, IP, termination, revenue, or other business risks.

## What Is CUAD?

CUAD stands for **Contract Understanding Atticus Dataset**. It is a public legal contract dataset where important clauses have been labeled by category.

The original dataset contains 41 detailed legal clause labels. Lexorion maps the most relevant labels into 8 broader risk groups so the output is easier for non-legal business users to understand.

Example:

```text
Limitation of Liability
Cap on Liability
Uncapped Liability
Liquidated Damages
→ Liability Risk
```

## Current Project Status

| Area | Status | Notes |
| --- | --- | --- |
| CUAD download/parsing | Complete | Converts CUAD from QA-style records into structured tables. |
| Risk category mapping | Complete | Maps selected CUAD labels into 8 business-facing risk groups. |
| Paragraph chunking | Complete | Splits contracts into paragraphs and creates paragraph-level labels. |
| Tests | Passing | `22` pipeline/helper tests currently pass. |
| Streamlit dashboard | Prototype | Supports text input and sample risk visualization. |
| DeBERTa training | Planned/In progress | Training/evaluation results still need to be generated. |
| LLM classification | Planned/In progress | Prompting and cache layer exist; final evaluation is pending. |
| Hybrid model comparison | Pending | README will be updated once real metrics exist. |
| Deployment | Pending | Final app deployment has not been completed yet. |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Contract Input  │────▶│  Clause Detector  │────▶│  Risk Classifier │
│  (TXT now)       │     │  (DeBERTa-base)   │     │  (LLM + Rules)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                              ┌────────────────────────────┘
                              ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │  Risk Scorer    │────▶│  Dashboard UI   │
                    │  (Aggregation)  │     │  (Streamlit)    │
                    └─────────────────┘     └─────────────────┘
```

## Risk Categories

| Category | CUAD Labels Mapped | Why It Matters |
|----------|-------------------|----------------|
| **Liability Risk** | Limitation of Liability, Cap on Liability, Uncapped Liability, Liquidated Damages | Financial exposure |
| **IP Risk** | IP Ownership Assignment, License Grant, Non-Compete | Intellectual property loss |
| **Termination Risk** | Termination for Convenience, Effect of Termination | Business continuity |
| **Indemnification** | Indemnification, Insurance | Who pays when things go wrong |
| **Exclusivity** | Exclusivity, No-Solicit of Employees, Non-Compete | Market restrictions |
| **Change of Control** | Change of Control, Anti-Assignment | M&A implications |
| **Revenue Risk** | Minimum Commitment, Revenue/Profit Sharing, Price Restriction | Financial obligations |
| **Renewal & Expiration** | Renewal Term, Expiration Date, Auto-Renewal | Contract lifecycle |

## Tech Stack

- **Python 3.10+**
- **PyTorch + HuggingFace Transformers** — DeBERTa fine-tuning
- **Anthropic/OpenAI API** — LLM reasoning layer
- **Streamlit** — Dashboard UI
- **Pandas + scikit-learn** — Data processing & evaluation

## Project Structure

```
contract-risk-intel/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── model_config.yaml          # Model hyperparameters
│   └── category_mapping.yaml      # CUAD 41 → 8 risk categories
├── data/
│   ├── raw/                       # Original CUAD data (gitignored)
│   ├── processed/                 # Cleaned, chunked, ready for training
│   └── evaluation/                # Hold-out test sets, predictions
├── src/
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── download_cuad.py       # Download & extract CUAD
│   │   ├── parse_cuad.py          # Parse SQuAD JSON → clean format
│   │   ├── chunk_contracts.py     # Intelligent paragraph chunking
│   │   └── category_mapper.py     # Map 41 labels → 8 risk categories
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clause_detector.py     # DeBERTa fine-tuning for clause detection
│   │   ├── llm_classifier.py      # LLM-based clause analysis
│   │   ├── hybrid_pipeline.py     # Combined detector + classifier
│   │   └── risk_scorer.py         # Aggregate clause-level → contract-level risk
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Per-category P/R/F1, Jaccard, AUPR
│   │   ├── error_analysis.py      # False negative deep-dive
│   │   └── model_comparison.py    # Head-to-head: DeBERTa vs LLM vs Hybrid
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py                 # Streamlit dashboard
│   └── utils/
│       ├── __init__.py
│       └── text_processing.py     # Tokenization, cleaning helpers
├── notebooks/
│   ├── 01_eda_exploration.ipynb         # [PERSON B] Data exploration
│   ├── 02_label_distribution.ipynb      # [PERSON B] Category analysis
│   ├── 03_model_experiments.ipynb       # [PERSON A] Training experiments
│   └── 04_results_analysis.ipynb        # [BOTH] Final results
├── tests/
│   ├── test_data_pipeline.py
│   └── test_models.py
└── docs/
    ├── TASK_BOARD.md              # Task tracking (mirror in Trello)
    └── BLOG_DRAFT.md             # Medium post draft
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/sahilshinde-45/Lexorion.git
cd Lexorion
pip install -r requirements.txt

# Download CUAD dataset
python -m src.data_pipeline.download_cuad

# Parse and process
python -m src.data_pipeline.parse_cuad
python -m src.data_pipeline.chunk_contracts

# Run evaluation
python -m src.evaluation.model_comparison

# Launch dashboard
streamlit run src/dashboard/app.py
```

## Results

Final model metrics are not available yet. The next evaluation milestone is to compare DeBERTa, LLM-only, and hybrid predictions on the held-out CUAD test split.

| Approach | Avg F1 | Latency/Contract | Cost/Contract | Best For |
|----------|--------|-----------------|---------------|----------|
| Fine-tuned DeBERTa | Pending | Pending | $0 local inference | Fast, low-cost screening |
| LLM few-shot | Pending | Pending | API usage required | Explainable clause analysis |
| Hybrid Pipeline | Pending | Pending | Lower than LLM-only | Cost-aware risk detection |

## Why A Hybrid Approach?

Contracts are long and legal language can be subtle. Sending every paragraph to a paid LLM can be slow and expensive. Lexorion is designed to use:

- **DeBERTa** for fast local clause detection.
- **LLM classification** for harder cases that need extraction and plain-English explanation.
- **Risk scoring** to aggregate individual clause findings into a contract-level profile.

## Team

This is a collaborative project. Individual responsibilities and final ownership notes should be documented before the project is submitted or published as a portfolio artifact.

## License

MIT
