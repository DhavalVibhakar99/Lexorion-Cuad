# Project Log — Lexorion

The original version of this file was an 8-week task board. The project is
shipped; this is the record of what was planned, what was built, and where
reality diverged from the plan.

## Shipped

| Milestone | Status | Where |
|---|---|---|
| CUAD download → parse → chunk pipeline, contract-level splits | ✅ | `src/data_pipeline/`, fully reproducible from public source |
| 41 CUAD labels → 8 business risk categories | ✅ | `configs/category_mapping.yaml` |
| TF-IDF baseline, recall-first thresholds | ✅ | `src/models/baseline_detector.py` — the production screen (recall 0.904) |
| MiniLM embedding detector | ✅ | `src/models/embed_detector.py` — negative result, documented |
| DeBERTa multi-label fine-tune (Colab T4) | ✅ | `src/models/clause_detector.py` + `notebooks/02_deberta_colab.ipynb` — negative result, documented |
| LLM classifier (OpenRouter, guarded, cached) | ✅ | `src/models/llm_classifier.py` |
| Hybrid routing (screen + LLM triage of weak flags) | ✅ | `src/models/hybrid_pipeline.py` — production winner (F1 0.852, 4.4% escalation) |
| Evaluation: 5-way comparison, hybrid eval, error analysis | ✅ | `src/evaluation/` |
| Tests + CI | ✅ | 37 tests, GitHub Actions |
| FastAPI inference service | ✅ | `src/api/main.py`, live on Hugging Face Spaces |
| React/TypeScript web app | ✅ | `docs/demo/index.html`, live on GitHub Pages |
| Streamlit dashboard | ✅ | `src/dashboard/app.py` — kept as local dev UI |
| Blog draft | ✅ | `docs/BLOG_DRAFT.md` |

## Where the plan changed (and why)

- **8 binary DeBERTa models → 1 multi-label model.** The original plan
  (one fine-tune per category) needed ~10 GPU-hours; a free Colab session
  provides ~3. One shared encoder with 8 heads trains in 20 minutes.
- **DeBERTa as the production detector → TF-IDF stays.** The plan assumed
  the transformer would win. Measurement said otherwise (precision 0.107 vs
  0.248 at matched recall), so the architecture follows the data.
- **F1 optimization → recall-first thresholds.** Screening tools should not
  treat a missed clause and a false alarm as equal costs.
- **Streamlit as the product → React + FastAPI.** Streamlit remains for local
  exploration; the public product is a static SPA over an inference API,
  which also keeps the LLM key server-side.

## Deliberately not built

- OCR for scanned PDFs (text-based PDFs only)
- Clause-level span highlighting in the source document (best next UX step)
- Feedback loop / human-in-the-loop corrections
- The 11 unmapped CUAD labels (4 are metadata; 7 are documented out-of-scope
  risks — Governing Law is the first candidate for a 9th category)
