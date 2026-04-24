# Contract Risk Intelligence System

A hybrid ML/LLM pipeline for automated legal contract risk analysis, trained on CUAD and evaluated against real-world SEC filings.

## What This Does

Upload a contract → System extracts and classifies risk clauses across 8 categories → Returns a risk profile with confidence scores, plain-English summaries, and benchmarking against 500+ contracts.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Contract Input  │────▶│  Clause Detector  │────▶│  Risk Classifier │
│  (PDF/TXT)       │     │  (DeBERTa-base)   │     │  (LLM + Rules)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                              ┌────────────────────────────┘
                              ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │  Risk Scorer    │────▶│  Dashboard UI   │
                    │  (Aggregation)  │     │  (Streamlit)    │
                    └─────────────────┘     └─────────────────┘
```

## Risk Categories (8 focus areas from CUAD's 41)

| Category | CUAD Labels Mapped | Why It Matters |
|----------|-------------------|----------------|
| **Liability Risk** | Limitation of Liability, Cap on Liability, Uncapped Liability | Financial exposure |
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
git clone https://github.com/<your-username>/contract-risk-intel.git
cd contract-risk-intel
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

| Approach | Avg F1 | Latency/Contract | Cost/Contract | Best For |
|----------|--------|-----------------|---------------|----------|
| Fine-tuned DeBERTa | TBD | TBD | $0 (local) | TBD |
| LLM (few-shot) | TBD | TBD | ~$0.02 | TBD |
| Hybrid Pipeline | TBD | TBD | ~$0.005 | TBD |

## Team

- **Dhaval Vibhakar** (Engineering Lead): Data pipeline, model training, dashboard, deployment
- **Sahil Shinde** (Analysis Lead): EDA, evaluation framework, prompt engineering, blog post

## License

MIT
