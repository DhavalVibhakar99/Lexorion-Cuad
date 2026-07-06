---
title: Lexorion API
emoji: ⚖️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Lexorion: Contract Risk Intelligence System

A collaborative AI/data science project for automated legal contract risk analysis. Lexorion uses the CUAD contract dataset to detect business-relevant legal risk clauses and turn long contract text into a structured risk profile.

> Current status: the end-to-end product loop works — contract in (text or PDF), baseline scoring, LLM escalation of uncertain clauses via OpenRouter, and a deployable Streamlit dashboard. DeBERTa fine-tuning remains a planned milestone (requires GPU).

## Try It

Lexorion ships as a React website backed by a FastAPI inference service:

| | What it is | Link |
|---|---|---|
| **Website** | React/TypeScript SPA ([docs/demo/index.html](docs/demo/index.html)): paste a contract or upload a PDF, get live scoring with LLM triage, evidence phrases, and clickable plain-English risk categories | **[sahilshinde-45.github.io/Lexorion/demo](https://sahilshinde-45.github.io/Lexorion/demo/)** (requires the repo to be public with Pages enabled: Settings → Pages → `main`, `/docs`) |
| **API** | FastAPI service ([src/api/main.py](src/api/main.py)) wrapping the hybrid pipeline; holds the OpenRouter key server-side | **Live:** [dhaval99-lexorion-cuad.hf.space](https://dhaval99-lexorion-cuad.hf.space) · [interactive docs](https://dhaval99-lexorion-cuad.hf.space/docs) · [Space](https://huggingface.co/spaces/dhaval99/lexorion_cuad) |
| **Local dev UI** | The original Streamlit dashboard — same pipeline, useful for local exploration | `streamlit run src/dashboard/app.py` |

Run everything locally:

```bash
uvicorn src.api.main:app --port 8000        # backend
open docs/demo/index.html                    # website (defaults to localhost:8000)
```

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
| Streamlit dashboard | Complete | Text, paste, and PDF intake; live hybrid analysis with model badges, routing stats, and cost tracking. |
| Baseline model | Complete | TF-IDF + logistic regression baseline trained across all 8 categories and saved for dashboard inference. |
| Baseline error analysis | Complete | Generates false-positive/false-negative samples and a Markdown report. |
| LLM classification | Complete | Budget-guarded OpenRouter classifier with caching, JSON validation, and guardrails; evaluated on all 8 categories. |
| Hybrid routing | Complete | Baseline scores everything; near-threshold clauses escalate to the LLM, live in the dashboard. |
| Deployment | Ready | Slim `requirements.txt` + secrets template + Streamlit Cloud instructions below. |
| DeBERTa training | Planned | Transformer fine-tuning awaits GPU access; the router is designed to slot it in. |

## Architecture

```
┌──────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
│  Contract Input   │────▶│  Baseline Scorer    │────▶│  Confidence Router    │
│  (TXT / PDF)      │     │  (TF-IDF + LogReg,  │     │  near threshold?      │
└──────────────────┘     │   per category)     │     └──────────┬───────────┘
                         └────────────────────┘        confident │ uncertain
                                                        (free)   ▼
                                                     ┌──────────────────────┐
                                                     │  LLM Second Opinion   │
                                                     │  (OpenRouter, budget- │
                                                     │   capped + cached)    │
                                                     └──────────┬───────────┘
                              ┌───────────────────────────────┘
                              ▼
                    ┌─────────────────┐     ┌──────────────────────┐
                    │  Risk Scorer    │────▶│  FastAPI service      │
                    │  (Aggregation)  │     │  ⇅ React web app      │
                    └─────────────────┘     │  (+ Streamlit local)  │
                                            └──────────────────────┘

Planned: swap the baseline scorer for a fine-tuned DeBERTa once GPU training
is available; the router design stays the same.
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
- **Anthropic/OpenAI/OpenRouter API** — LLM reasoning layer
- **Streamlit** — Dashboard UI
- **Pandas + scikit-learn** — Data processing & evaluation

## Project Structure

```
contract-risk-intel/
├── README.md
├── requirements.txt               # Slim runtime set (dashboard + deploy)
├── requirements-dev.txt           # Full training/eval/dev stack
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
│   └── 01_eda_exploration.ipynb         # Data exploration
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
pip install -r requirements-dev.txt   # full training/eval stack
# (dashboard-only: pip install -r requirements.txt)

# Download CUAD dataset
python -m src.data_pipeline.download_cuad

# Parse and process
python -m src.data_pipeline.parse_cuad
python -m src.data_pipeline.chunk_contracts

# Run evaluation
python -m src.evaluation.model_comparison

# Run baseline error analysis
python -m src.evaluation.error_analysis --approach baseline

# Train and save the baseline dashboard model
python -m src.models.baseline_detector

# Launch dashboard
streamlit run src/dashboard/app.py
```

## Deployment (all free tier)

**1. Backend — Hugging Face Spaces (Docker):**

1. Push to GitHub, including `checkpoints/baseline_tfidf_logreg.joblib`
   (~12 MB, intentionally un-gitignored so the server can score live).
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   with the **Docker** SDK, and connect this repo (or push to the Space's git
   remote). The root [Dockerfile](Dockerfile) serves the API on port 7860.
3. In the Space settings → **Variables and secrets**, add
   `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` (see `.streamlit/secrets.toml.example`
   for values). Without a key the API still works baseline-only.
4. Note your Space URL, e.g. `https://<user>-lexorion.hf.space` — check
   `<url>/health` returns `{"status": "ok"}`.

**2. Frontend — GitHub Pages:**

1. Set the `PROD_API` constant near the top of the script in
   [docs/demo/index.html](docs/demo/index.html) to your Space URL and commit.
   (The API endpoint is deliberately not user-editable on the site; when the
   file is opened locally it talks to `localhost:8000` automatically.)
2. Repo Settings → Pages → deploy from branch → `main`, `/docs` folder.
3. Open `https://<your-github-username>.github.io/Lexorion/demo/`.

**Optional — Streamlit UI on share.streamlit.io:** point an app at
`src/dashboard/app.py`, Python 3.11, secrets from
`.streamlit/secrets.toml.example`. Same pipeline, alternative front-end.

## Results

![Benchmark](docs/assets/benchmark.png)

**Design decision first:** Lexorion is a screening tool, so missing a risky
clause (false negative) costs far more than an extra flag a reviewer dismisses
in seconds. All local models are therefore thresholded **recall-first**
(target recall ≥ 0.90 on validation) rather than for max F1 — the same
operating philosophy as AML transaction monitoring.

**Stage 1 — local screen, full CUAD test split (true prevalence, ~2% positive):**

| Model | Macro Recall | Macro Precision | Notes |
|-------|--------------|-----------------|-------|
| TF-IDF + LogReg | **0.904** | 0.248 | Production screen: catches 9 in 10 risky clauses |
| MiniLM embeddings + LogReg | 0.907 | 0.107 | Frozen transformer — **loses to TF-IDF** (see below) |

**Stage 2 — LLM triage of weak flags (same 61-paragraph eval sample for all
three rows, so the comparison is apples-to-apples):**

| Approach | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Baseline alone | 0.871 | 0.818 | 0.844 |
| LLM alone (gpt-oss-120b:free) | 0.955 | 0.636 | 0.764 |
| **Hybrid (routed)** | **0.929** | **0.788** | **0.852** |

**The headline: the hybrid beats both of its components** — LLM precision
repairs the baseline's weak flags without surrendering its recall — **while
routing only 4.4% of paragraph×category decisions to the LLM** (measured
across all 43,408 decisions on the full test split). The other 95.6% are
decided locally for free.

Honesty notes, because they matter:
- The stage-2 sample is category-balanced, not at deployment prevalence, so
  its precision numbers are not comparable to the full-split table above.
  Recall is prevalence-invariant and is comparable.
- The LLM sample is small (n=61; free-tier rate caps). Rate-limited API errors
  are excluded from scoring, never counted as predictions, and never cached.

**A negative result worth keeping:** we tested frozen MiniLM sentence
embeddings as the "obvious" transformer upgrade over TF-IDF. At matched recall
(~0.90) they *halve* precision (0.107 vs 0.248). Clause detection is heavily
lexical — phrases like "indemnify and hold harmless" are near-perfect signals
that TF-IDF bigrams capture directly, while general-purpose embeddings blur
them into topical similarity. Conclusion: the worthwhile transformer step is
*fine-tuning* (DeBERTa, GPU required — `src/models/clause_detector.py` is
ready for a free Colab T4), not frozen general-purpose embeddings.

Baseline error analysis samples are available in:

- `examples/baseline_error_summary.csv`
- `examples/baseline_false_negatives_sample.csv`
- `examples/baseline_false_positives_sample.csv`

Live dashboard inference uses the local model artifact:

- `checkpoints/baseline_tfidf_logreg.joblib` (generated locally, gitignored)

## Why A Hybrid Approach?

Contracts are long and legal language can be subtle. Sending every paragraph to a paid LLM can be slow and expensive. Lexorion is designed to use:

- **DeBERTa** for fast local clause detection.
- **LLM classification** for harder cases that need extraction and plain-English explanation.
- **Risk scoring** to aggregate individual clause findings into a contract-level profile.

The LLM layer can run through OpenRouter using the OpenAI-compatible API:

```bash
export OPENROUTER_API_KEY="your-key"
python -m src.models.llm_classifier \
  --provider openrouter \
  --model openai/gpt-oss-120b:free \
  --max_samples 4 \
  --max_calls 30 \
  --categories termination_risk revenue_risk exclusivity
```

The LLM evaluator includes guardrails for API-call budgets, category allowlists,
JSON validation, caching, and malformed-response handling.

## Team

- **Dhaval Vibhakar** (Engineering Lead): Data pipeline, model training, dashboard, deployment
- **Sahil Shinde** (Analysis Lead): EDA, evaluation framework, prompt engineering, blog post

## License

MIT
