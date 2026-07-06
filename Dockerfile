# Lexorion inference API — built for Hugging Face Spaces (Docker SDK).
# Spaces expects the app on port 7860 and injects secrets as env vars:
# set OPENROUTER_API_KEY and OPENROUTER_MODEL in the Space settings.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY checkpoints/baseline_tfidf_logreg.joblib checkpoints/
COPY src/ src/

# Writable cache dir for LLM response caching (ephemeral on Spaces restarts)
RUN mkdir -p data/processed/llm_cache && chmod -R 777 data

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
