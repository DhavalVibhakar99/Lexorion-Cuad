# Task Board — Contract Risk Intelligence System

> Mirror this in Trello. Create one board with columns: Backlog → This Week → In Progress → Review → Done

---

## WEEK 1-2: Data Engineering & EDA

### Person A (Engineering Lead)

| Task | Priority | Est. Hours | Status |
|------|----------|-----------|--------|
| Set up GitHub repo, branch protection, .gitignore | P0 | 1h | ⬜ |
| Set up virtual env, install deps, verify GPU access | P0 | 1h | ⬜ |
| Run `download_cuad.py` — verify dataset downloads correctly | P0 | 0.5h | ⬜ |
| Run `parse_cuad.py` — verify parquet files generated | P0 | 0.5h | ⬜ |
| Write `chunk_contracts.py` — split contracts into paragraphs with metadata | P1 | 3h | ⬜ |
| Write `category_mapper.py` — build train/val/test splits per risk category | P1 | 2h | ⬜ |
| Validate data pipeline end-to-end: raw CUAD → model-ready parquet | P0 | 1h | ⬜ |
| Set up `.env` for API keys (Anthropic/OpenAI) | P2 | 0.5h | ⬜ |

### Person B (Analysis Lead)

| Task | Priority | Est. Hours | Status |
|------|----------|-----------|--------|
| Set up Trello board with all tasks from this document | P0 | 1h | ⬜ |
| **Notebook 01**: Distribution of 41 CUAD categories (bar chart) | P0 | 2h | ⬜ |
| **Notebook 01**: Positive vs impossible ratio per category | P0 | 1h | ⬜ |
| **Notebook 01**: Contract length distribution (histogram) | P1 | 1h | ⬜ |
| **Notebook 02**: Clause co-occurrence heatmap across categories | P1 | 2h | ⬜ |
| **Notebook 02**: Risk category distribution after mapping 41→8 | P0 | 1h | ⬜ |
| **Notebook 02**: Label sparsity analysis — flag categories with <50 examples | P1 | 1h | ⬜ |
| Write 1-page data summary for README (key stats, insights) | P2 | 1h | ⬜ |

---

## WEEK 3-4: Model Layer

### Person A

| Task | Priority | Est. Hours | Status |
|------|----------|-----------|--------|
| Write `clause_detector.py` — DeBERTa binary classifier per risk category | P0 | 5h | ⬜ |
| Train DeBERTa on each of the 8 risk categories | P0 | 3h | ⬜ |
| Save model checkpoints + training logs | P1 | 1h | ⬜ |
| Write `hybrid_pipeline.py` — routing logic (easy→DeBERTa, hard→LLM) | P1 | 3h | ⬜ |
| Write `risk_scorer.py` — aggregate clause detections → contract risk score | P1 | 2h | ⬜ |

### Person B

| Task | Priority | Est. Hours | Status |
|------|----------|-----------|--------|
| Write `llm_classifier.py` — prompt templates for each risk category | P0 | 3h | ⬜ |
| Design & test few-shot prompts (3-5 examples per category from CUAD) | P0 | 3h | ⬜ |
| Run LLM inference on test set, save predictions | P0 | 2h | ⬜ |
| Write `error_analysis.py` — analyze false negatives per category | P1 | 2h | ⬜ |
| **Notebook 03**: Document prompt iterations & what worked/didn't | P1 | 2h | ⬜ |

---

## WEEK 5-6: Evaluation & Dashboard

### Person A

| Task | Priority | Est. Hours | Status |
|------|----------|-----------|--------|
| Write `model_comparison.py` — run all 3 approaches, generate comparison table | P0 | 3h | ⬜ |
| Build Streamlit dashboard: file upload + risk heatmap view | P0 | 4h | ⬜ |
| Dashboard: clause drill-down panel (click risk → see extracted clauses) | P1 | 3h | ⬜ |
| Dashboard: model comparison tab (F1 / latency / cost charts) | P1 | 2h | ⬜ |
| Deploy to Streamlit Cloud or HuggingFace Spaces | P1 | 2h | ⬜ |

### Person B

| Task | Priority | Est. Hours | Status |
|------|----------|-----------|--------|
| Run full evaluation across all 3 approaches | P0 | 2h | ⬜ |
| **Notebook 04**: Results visualization — per-category comparison charts | P0 | 3h | ⬜ |
| **Notebook 04**: Cost-latency-accuracy tradeoff analysis | P0 | 2h | ⬜ |
| Write error analysis section — "why did the model fail on X?" | P1 | 2h | ⬜ |
| Dashboard: "contract vs dataset" benchmark view (percentile comparisons) | P2 | 2h | ⬜ |

---

## WEEK 7-8: Polish & Ship

### Both

| Task | Priority | Est. Hours | Status |
|------|----------|-----------|--------|
| Update README with final results table and screenshots | P0 | 2h | ⬜ |
| Record 3-min Loom demo walkthrough | P0 | 2h | ⬜ |
| Write Medium blog post draft (`docs/BLOG_DRAFT.md`) | P1 | 4h | ⬜ |
| Clean up code — docstrings, type hints, remove dead code | P1 | 2h | ⬜ |
| Write tests for data pipeline and metrics | P2 | 2h | ⬜ |
| Publish blog post + share on LinkedIn | P1 | 1h | ⬜ |

---

## Git Workflow

- `main` — always deployable, protected
- `dev` — integration branch
- Feature branches: `feat/data-pipeline`, `feat/deberta-training`, `feat/llm-classifier`, `feat/dashboard`
- PR reviews: Person A reviews B's code, Person B reviews A's code
- Commit messages: `feat:`, `fix:`, `docs:`, `refactor:` prefixes

---

## Weekly Check-ins

- **Sunday night**: quick sync on Trello — what's done, what's blocked
- **Wednesday**: mid-week async update (Slack/WhatsApp message)
- Every PR should have a 1-2 sentence description of what it does

---

## Definition of Done (for each task)

- [ ] Code runs without errors
- [ ] Outputs saved to correct directory
- [ ] At least 1 sentence of docstring
- [ ] Tested manually with sample data
- [ ] PR created and reviewed by partner
