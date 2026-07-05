"""
Unit tests for the model layer: LLM response validation, cache behavior,
recall-first thresholding, and hybrid routing. No API calls, no data files —
everything external is faked.
"""

import numpy as np
import pytest

from src.models import baseline_detector, hybrid_pipeline, llm_classifier
from src.models.baseline_detector import _recall_first_threshold
from src.models.llm_classifier import LLMClassifier


# === LLM response validation ===


def _make_classifier(monkeypatch, tmp_path):
    """Build an LLMClassifier without touching the network."""
    monkeypatch.setattr(llm_classifier, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    return LLMClassifier(provider="openai", model="fake-model")


VALID_RESPONSE = {
    "classification": "PRESENT",
    "risk_level": "high",
    "extracted_clause": "Liability shall not exceed fees paid.",
    "summary": "Liability cap.",
    "confidence": 0.9,
}


def test_validate_accepts_valid_response(monkeypatch, tmp_path):
    clf = _make_classifier(monkeypatch, tmp_path)
    result = clf._validate_result(dict(VALID_RESPONSE))
    assert result["classification"] == "PRESENT"
    assert result["guardrail_blocked"] is False


@pytest.mark.parametrize("missing", list(VALID_RESPONSE.keys()))
def test_validate_rejects_missing_field(monkeypatch, tmp_path, missing):
    clf = _make_classifier(monkeypatch, tmp_path)
    bad = {k: v for k, v in VALID_RESPONSE.items() if k != missing}
    assert clf._validate_result(bad)["classification"] == "ERROR"


def test_validate_rejects_invalid_enum_values(monkeypatch, tmp_path):
    clf = _make_classifier(monkeypatch, tmp_path)
    assert (
        clf._validate_result({**VALID_RESPONSE, "classification": "MAYBE"})[
            "classification"
        ]
        == "ERROR"
    )
    assert (
        clf._validate_result({**VALID_RESPONSE, "risk_level": "catastrophic"})[
            "classification"
        ]
        == "ERROR"
    )


def test_validate_clamps_confidence_and_clears_absent(monkeypatch, tmp_path):
    clf = _make_classifier(monkeypatch, tmp_path)
    result = clf._validate_result(
        {
            **VALID_RESPONSE,
            "classification": "ABSENT",
            "confidence": 7.5,
        }
    )
    assert result["confidence"] == 1.0
    assert result["risk_level"] == "none"
    assert result["extracted_clause"] == ""


def test_api_errors_are_not_cached(monkeypatch, tmp_path):
    """A transient API failure must never be replayed from cache."""
    clf = _make_classifier(monkeypatch, tmp_path)

    class ExplodingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("rate limited")

    class FakeChat:
        completions = ExplodingCompletions()

    clf.client.chat = FakeChat()
    result = clf.classify_paragraph("Some clause text.", "liability_risk")
    assert result["classification"] == "ERROR"
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_budget_guardrail_blocks_calls(monkeypatch, tmp_path):
    clf = _make_classifier(monkeypatch, tmp_path)
    clf.max_calls = 0
    result = clf.classify_paragraph("Some clause text.", "liability_risk")
    assert result["classification"] == "ERROR"
    assert result["guardrail_blocked"] is True
    assert clf.blocked_calls == 1


# === Recall-first thresholding ===


def test_recall_first_prefers_highest_threshold_meeting_target():
    # Positives score high, negatives low: recall target reachable at 0.5+.
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    threshold, recall, precision = _recall_first_threshold(
        y_true, y_score, recall_target=1.0
    )
    assert threshold == 0.6
    assert recall == 1.0
    assert precision == 1.0


def test_recall_first_falls_back_to_max_recall():
    # One positive is unreachable (score below the lowest threshold), so the
    # target can never be met; we take the most recall available.
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.01, 0.5, 0.4])
    threshold, recall, _ = _recall_first_threshold(y_true, y_score, recall_target=1.0)
    assert recall == 0.5  # the reachable positive


# === Hybrid routing ===


class FakeScoreModel:
    """predict_proba stub returning fixed positive-class scores."""

    def __init__(self, scores):
        self.scores = scores

    def predict_proba(self, paragraphs):
        s = np.array(self.scores[: len(paragraphs)])
        return np.column_stack([1 - s, s])


@pytest.fixture
def fake_artifact(monkeypatch):
    artifact = {
        "model_name": "fake_baseline",
        # Paragraph scores: 0.9 strong flag, 0.55 weak flag, 0.1 clear.
        "models": {"liability_risk": FakeScoreModel([0.9, 0.55, 0.1])},
        "thresholds": {"liability_risk": 0.5},
    }
    monkeypatch.setattr(
        baseline_detector, "load_baseline_model", lambda *a, **k: artifact
    )
    return artifact


PARAGRAPHS = ["strong flag text", "weak flag text", "clean text"]


def _fake_llm_class(classification):
    class FakeLLM:
        model = "fake-llm"

        def __init__(self, *args, **kwargs):
            pass

        def classify_paragraph(self, paragraph, category):
            return {
                "classification": classification,
                "risk_level": "moderate",
                "extracted_clause": paragraph,
                "summary": f"LLM says {classification}",
                "confidence": 0.8,
            }

        def get_cost_estimate(self):
            return {"total_calls": 1, "cache_hits": 0, "estimated_cost_usd": 0.0}

    return FakeLLM


def test_hybrid_llm_clears_weak_flag(monkeypatch, fake_artifact):
    monkeypatch.setattr(hybrid_pipeline, "LLMClassifier", _fake_llm_class("ABSENT"))
    profile = hybrid_pipeline.analyze_contract_hybrid(PARAGRAPHS, contract_id="t")

    assert profile["routing_summary"] == {
        "confident_positive": 1,
        "uncertain": 1,
        "confident_negative": 1,
        "uncertainty_margin": 0.15,
    }
    assert profile["llm_stats"]["cleared_by_llm"] == 1
    assert len(profile["detections"]) == 1
    assert profile["detections"][0]["model_used"] == "baseline"


def test_hybrid_llm_confirms_weak_flag(monkeypatch, fake_artifact):
    monkeypatch.setattr(hybrid_pipeline, "LLMClassifier", _fake_llm_class("PRESENT"))
    profile = hybrid_pipeline.analyze_contract_hybrid(PARAGRAPHS, contract_id="t")

    assert profile["llm_stats"]["confirmed_by_llm"] == 1
    models_used = sorted(d["model_used"] for d in profile["detections"])
    assert models_used == ["baseline", "llm"]


def test_hybrid_keeps_weak_flag_when_llm_unavailable(monkeypatch, fake_artifact):
    def raise_value_error(*args, **kwargs):
        raise ValueError("OPENROUTER_API_KEY is required")

    monkeypatch.setattr(hybrid_pipeline, "LLMClassifier", raise_value_error)
    profile = hybrid_pipeline.analyze_contract_hybrid(PARAGRAPHS, contract_id="t")

    # Recall is never sacrificed: the weak flag survives as a fallback.
    assert profile["llm_stats"]["fallbacks"] == 1
    models_used = sorted(d["model_used"] for d in profile["detections"])
    assert models_used == ["baseline", "baseline_fallback"]
    assert profile["pipeline"] == "baseline_only"
