"""
Lexorion inference API — the backend behind the React website.

Wraps the hybrid pipeline (recall-first TF-IDF screen + budgeted LLM triage)
as a small FastAPI service. The OpenRouter key stays server-side; the static
front-end on GitHub Pages only ever sees analysis results.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Deployed on Hugging Face Spaces (Docker SDK) via the repo Dockerfile.
"""

import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data_pipeline.chunk_contracts import split_into_paragraphs
from src.models.baseline_detector import BASELINE_MODEL_PATH
from src.models.hybrid_pipeline import analyze_contract_hybrid
from src.models.llm_classifier import openrouter_available

# Server-side guardrails: callers cannot buy themselves a bigger LLM budget.
MAX_TEXT_CHARS = int(os.getenv("LEXORION_API_MAX_CHARS", "200000"))
MAX_LLM_CALLS = int(os.getenv("LEXORION_API_MAX_LLM_CALLS", "15"))
MAX_PDF_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="Lexorion API",
    description="Contract risk screening: recall-first local model with budgeted LLM triage of weak flags.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # public read-only inference; no cookies or auth involved
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw contract text")
    use_llm: bool = Field(True, description="Escalate weak flags to the LLM")


def _analyze(text: str, use_llm: bool) -> dict:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty contract text.")
    paragraphs = split_into_paragraphs(text[:MAX_TEXT_CHARS])
    if not paragraphs:
        raise HTTPException(
            status_code=400,
            detail="No analyzable paragraphs found — the text may be too short.",
        )
    llm_budget = MAX_LLM_CALLS if (use_llm and openrouter_available()) else 0
    try:
        return analyze_contract_hybrid(
            paragraphs, contract_id="api", llm_max_calls=llm_budget
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Baseline model artifact missing on the server.",
        )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "baseline_ready": BASELINE_MODEL_PATH.exists(),
        "llm_ready": openrouter_available(),
        "llm_budget_per_request": MAX_LLM_CALLS,
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    return _analyze(request.text, request.use_llm)


@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...), use_llm: bool = True) -> dict:
    raw = await file.read()
    if len(raw) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF larger than 10 MB.")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No extractable text — this PDF may be a scan (OCR not supported).",
        )
    profile = _analyze(text, use_llm)
    profile["source_filename"] = file.filename
    profile["pages"] = len(reader.pages)
    return profile
