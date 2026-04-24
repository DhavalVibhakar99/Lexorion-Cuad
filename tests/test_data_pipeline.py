"""
Tests for the data pipeline components.
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_pipeline.chunk_contracts import (
    clean_text,
    split_into_paragraphs,
    compute_paragraph_id,
)
from src.data_pipeline.category_mapper import (
    load_mapping,
    map_cuad_label,
    get_risk_categories,
)
from src.utils.text_processing import (
    count_tokens_approx,
    truncate_to_tokens,
    is_boilerplate,
    normalize_legal_text,
)
from src.evaluation.metrics import (
    jaccard_similarity,
    evaluate_category,
)


# === Text Processing Tests ===

class TestCleanText:
    def test_normalizes_whitespace(self):
        assert "hello world" in clean_text("hello   world")
    
    def test_preserves_paragraph_breaks(self):
        result = clean_text("para one\n\npara two")
        assert "\n\n" in result
    
    def test_collapses_excessive_newlines(self):
        result = clean_text("para one\n\n\n\n\npara two")
        assert "\n\n\n" not in result


class TestSplitParagraphs:
    def test_splits_on_double_newline(self):
        text = "First paragraph here.\n\nSecond paragraph here with enough words to pass the minimum length threshold for inclusion."
        result = split_into_paragraphs(text, min_length=10)
        assert len(result) >= 1
    
    def test_skips_short_fragments(self):
        text = "Hi\n\nThis is a real paragraph with enough content to be meaningful."
        result = split_into_paragraphs(text, min_length=20)
        assert all(len(p) >= 20 for p in result)
    
    def test_handles_empty_text(self):
        assert split_into_paragraphs("") == []


class TestParagraphId:
    def test_deterministic(self):
        id1 = compute_paragraph_id("contract_a", "some text")
        id2 = compute_paragraph_id("contract_a", "some text")
        assert id1 == id2
    
    def test_different_contracts_different_ids(self):
        id1 = compute_paragraph_id("contract_a", "same text")
        id2 = compute_paragraph_id("contract_b", "same text")
        assert id1 != id2


# === Category Mapping Tests ===

class TestCategoryMapping:
    def test_loads_mapping(self):
        label_to_risk, config = load_mapping()
        assert len(label_to_risk) > 0
        assert "risk_categories" in config
    
    def test_all_risk_categories_exist(self):
        categories = get_risk_categories()
        assert len(categories) == 8
        assert "liability_risk" in categories
        assert "ip_risk" in categories
    
    def test_maps_known_label(self):
        result = map_cuad_label("Limitation Of Liability")
        assert result == "liability_risk"
    
    def test_returns_none_for_unknown(self):
        result = map_cuad_label("Totally Made Up Category")
        assert result is None
    
    def test_severity_weights_valid(self):
        _, config = load_mapping()
        for key, info in config["risk_categories"].items():
            assert 0 <= info["severity_weight"] <= 1, f"Invalid weight for {key}"


# === Text Utility Tests ===

class TestTextUtils:
    def test_token_count_reasonable(self):
        text = "This is a ten word sentence with some extra filler"
        count = count_tokens_approx(text)
        assert 10 <= count <= 20
    
    def test_truncation(self):
        long_text = " ".join(["word"] * 1000)
        truncated = truncate_to_tokens(long_text, max_tokens=100)
        assert len(truncated.split()) < 200
    
    def test_boilerplate_detection(self):
        assert is_boilerplate("IN WITNESS WHEREOF, the parties have executed")
        assert is_boilerplate("5")  # Page number
        assert not is_boilerplate(
            "The Vendor shall indemnify and hold harmless the Company "
            "from any claims arising out of breach of this agreement."
        )
    
    def test_legal_normalization(self):
        text = normalize_legal_text('He said "hello" and ***REDACTED***')
        assert "[REDACTED]" in text


# === Evaluation Metrics Tests ===

class TestMetrics:
    def test_jaccard_identical(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0
    
    def test_jaccard_disjoint(self):
        assert jaccard_similarity("hello", "world") == 0.0
    
    def test_jaccard_partial(self):
        score = jaccard_similarity("hello world foo", "hello world bar")
        assert 0 < score < 1
    
    def test_jaccard_empty(self):
        assert jaccard_similarity("", "") == 1.0
        assert jaccard_similarity("hello", "") == 0.0
    
    def test_evaluate_category_basic(self):
        import numpy as np
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        
        results = evaluate_category(y_true, y_pred)
        
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert results["true_positives"] == 2
        assert results["false_positives"] == 1
        assert results["false_negatives"] == 1
