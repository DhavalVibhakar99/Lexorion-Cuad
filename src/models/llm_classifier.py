"""
LLM-based clause analysis using Anthropic Claude API.
This is the "reasoning layer" — given a paragraph flagged as potentially
containing a risk clause, the LLM extracts, classifies, and summarizes it.

Can also be used standalone (without DeBERTa pre-filtering) for comparison.

Run: python -m src.models.llm_classifier --mode evaluate
"""

import os
import json
import time
import yaml
import hashlib
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


CONFIG_DIR = Path("configs")
PROCESSED_DIR = Path("data/processed")
CACHE_DIR = Path("data/processed/llm_cache")


# === Prompt Templates ===

SYSTEM_PROMPT = """You are a legal contract analyst specializing in risk identification.
You analyze contract paragraphs to identify specific risk clauses.
You must be precise — only flag clauses that genuinely match the requested category.
When no relevant clause exists, say so clearly. False positives waste lawyer time."""


def build_classification_prompt(paragraph: str, risk_category: str, category_info: dict) -> str:
    """Build the few-shot prompt for a specific risk category."""
    
    category_descriptions = {
        "liability_risk": {
            "description": "Clauses that limit, cap, or leave uncapped the liability of either party. Includes limitation of liability, liability caps, uncapped liability, and liquidated damages provisions.",
            "examples": [
                {
                    "text": "IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT EXCEED THE TOTAL AMOUNTS PAID BY CUSTOMER IN THE 12 MONTH PERIOD PRECEDING THE CLAIM.",
                    "classification": "PRESENT",
                    "risk_level": "moderate",
                    "summary": "Liability capped at 12 months of fees — standard mutual cap."
                },
                {
                    "text": "The Parties agree to cooperate in good faith to resolve any disputes arising under this Agreement.",
                    "classification": "ABSENT",
                    "risk_level": "none",
                    "summary": "General dispute resolution — no liability clause."
                },
            ]
        },
        "ip_risk": {
            "description": "Clauses affecting intellectual property ownership, licensing, assignment, or non-compete restrictions. Includes IP ownership assignment, license grants, and covenants not to sue.",
            "examples": [
                {
                    "text": "All intellectual property developed by Contractor in the performance of this Agreement shall be the sole property of the Company, including all patents, copyrights, and trade secrets.",
                    "classification": "PRESENT",
                    "risk_level": "high",
                    "summary": "Full IP assignment to Company — Contractor retains no rights."
                },
            ]
        },
        "termination_risk": {
            "description": "Clauses governing how and when the contract can be terminated, including termination for convenience, effects of termination, and post-termination obligations.",
            "examples": [
                {
                    "text": "Either party may terminate this Agreement at any time, for any reason or no reason, upon thirty (30) days' prior written notice.",
                    "classification": "PRESENT",
                    "risk_level": "high",
                    "summary": "Broad termination for convenience — either party, 30 days notice."
                },
            ]
        },
        "indemnification": {
            "description": "Clauses requiring one party to compensate the other for losses, damages, or claims. Includes indemnification obligations and insurance requirements.",
            "examples": [
                {
                    "text": "Vendor shall indemnify, defend, and hold harmless Company from and against any and all claims, damages, losses, and expenses arising from Vendor's breach of this Agreement.",
                    "classification": "PRESENT",
                    "risk_level": "moderate",
                    "summary": "One-way indemnification — Vendor indemnifies Company for breaches."
                },
            ]
        },
        "exclusivity": {
            "description": "Clauses restricting business activities, including exclusivity arrangements, non-solicitation of employees or customers, and competitive restrictions.",
            "examples": []
        },
        "change_of_control": {
            "description": "Clauses triggered by changes in ownership, management, or corporate structure. Includes change of control provisions, anti-assignment clauses, and consent requirements.",
            "examples": []
        },
        "revenue_risk": {
            "description": "Clauses creating financial obligations such as minimum commitments, revenue sharing, price restrictions, most favored nation clauses, and audit rights.",
            "examples": []
        },
        "renewal_expiration": {
            "description": "Clauses governing contract duration, auto-renewal, expiration dates, and notice periods for termination of renewal.",
            "examples": []
        },
    }
    
    cat_info = category_descriptions.get(risk_category, {
        "description": f"Clauses related to {risk_category}",
        "examples": []
    })
    
    # Build examples section
    examples_text = ""
    for i, ex in enumerate(cat_info["examples"], 1):
        examples_text += f"""
Example {i}:
Paragraph: "{ex['text'][:300]}"
Classification: {ex['classification']}
Risk Level: {ex['risk_level']}
Summary: {ex['summary']}
"""
    
    prompt = f"""Analyze the following contract paragraph for **{risk_category.replace('_', ' ').title()}** clauses.

Category Definition: {cat_info['description']}
{examples_text}
---

Now analyze this paragraph:

"{paragraph[:1500]}"

Respond in this exact JSON format (no other text):
{{
    "classification": "PRESENT" or "ABSENT",
    "risk_level": "none" | "low" | "moderate" | "high" | "critical",
    "extracted_clause": "the specific clause text if PRESENT, empty string if ABSENT",
    "summary": "one-sentence plain-English summary of the risk implication",
    "confidence": 0.0 to 1.0
}}"""
    
    return prompt


class LLMClassifier:
    """LLM-based clause classifier with caching and cost tracking."""
    
    def __init__(self, provider: str = "anthropic", model: str = None):
        self.provider = provider
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.total_time = 0
        
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        if provider == "anthropic":
            if Anthropic is None:
                raise ImportError("pip install anthropic")
            self.client = Anthropic()
            self.model = model or "claude-sonnet-4-20250514"
        elif provider == "openai":
            if OpenAI is None:
                raise ImportError("pip install openai")
            self.client = OpenAI()
            self.model = model or "gpt-4o-mini"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _cache_key(self, paragraph: str, category: str) -> str:
        content = f"{self.model}::{category}::{paragraph[:500]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[dict]:
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None
    
    def _save_cache(self, key: str, result: dict):
        cache_file = CACHE_DIR / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(result, f)
    
    def classify_paragraph(self, paragraph: str, risk_category: str) -> dict:
        """
        Classify a single paragraph for a specific risk category.
        Returns parsed JSON response from the LLM.
        """
        # Check cache
        cache_key = self._cache_key(paragraph, risk_category)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        prompt = build_classification_prompt(paragraph, risk_category, {})
        
        start_time = time.time()
        
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_text = response.content[0].text
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens
                
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    max_tokens=512,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_text = response.choices[0].message.content
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            
            self.total_calls += 1
            self.total_time += time.time() - start_time
            
            # Parse JSON response
            # Strip markdown code fences if present
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            
            result = json.loads(cleaned)
            
        except json.JSONDecodeError:
            result = {
                "classification": "ERROR",
                "risk_level": "none",
                "extracted_clause": "",
                "summary": f"Failed to parse LLM response: {raw_text[:200]}",
                "confidence": 0.0,
            }
        except Exception as e:
            result = {
                "classification": "ERROR",
                "risk_level": "none",
                "extracted_clause": "",
                "summary": f"API error: {str(e)}",
                "confidence": 0.0,
            }
        
        # Cache result
        self._save_cache(cache_key, result)
        
        return result
    
    def classify_batch(
        self,
        paragraphs: List[str],
        risk_category: str,
        progress: bool = True,
    ) -> List[dict]:
        """Classify a batch of paragraphs."""
        results = []
        
        for i, paragraph in enumerate(paragraphs):
            if progress and (i + 1) % 10 == 0:
                print(f"  Classified {i + 1}/{len(paragraphs)} paragraphs...")
            
            result = self.classify_paragraph(paragraph, risk_category)
            results.append(result)
        
        return results
    
    def get_cost_estimate(self) -> dict:
        """Estimate API cost based on token usage."""
        # Approximate pricing (check current rates)
        if self.provider == "anthropic":
            input_cost_per_m = 3.0   # Sonnet input
            output_cost_per_m = 15.0  # Sonnet output
        else:
            input_cost_per_m = 0.15   # GPT-4o-mini input
            output_cost_per_m = 0.60  # GPT-4o-mini output
        
        input_cost = (self.total_input_tokens / 1_000_000) * input_cost_per_m
        output_cost = (self.total_output_tokens / 1_000_000) * output_cost_per_m
        
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4),
            "avg_latency_seconds": self.total_time / max(self.total_calls, 1),
        }
    
    def print_cost_report(self):
        """Print a summary of API usage and costs."""
        cost = self.get_cost_estimate()
        print(f"\n--- LLM Usage Report ---")
        print(f"  Total calls:      {cost['total_calls']}")
        print(f"  Input tokens:     {cost['total_input_tokens']:,}")
        print(f"  Output tokens:    {cost['total_output_tokens']:,}")
        print(f"  Estimated cost:   ${cost['estimated_cost_usd']:.4f}")
        print(f"  Avg latency:      {cost['avg_latency_seconds']:.2f}s")


def evaluate_llm_approach(
    risk_categories: List[str] = None,
    max_samples_per_category: int = 100,
    provider: str = "anthropic",
):
    """
    Run LLM classification on the test set and save predictions.
    """
    test_df = pd.read_parquet(PROCESSED_DIR / "paragraphs_test.parquet")
    
    if risk_categories is None:
        risk_categories = [
            col.replace("label_", "")
            for col in test_df.columns
            if col.startswith("label_")
        ]
    
    classifier = LLMClassifier(provider=provider)
    all_predictions = []
    
    for category in risk_categories:
        label_col = f"label_{category}"
        print(f"\n{'=' * 40}")
        print(f"Evaluating LLM on: {category}")
        print(f"{'=' * 40}")
        
        # Sample balanced subset
        positive = test_df[test_df[label_col] == 1]
        negative = test_df[test_df[label_col] == 0]
        
        n_pos = min(len(positive), max_samples_per_category // 2)
        n_neg = min(len(negative), max_samples_per_category // 2)
        
        sample = pd.concat([
            positive.sample(n=n_pos, random_state=42) if n_pos > 0 else positive,
            negative.sample(n=n_neg, random_state=42),
        ])
        
        print(f"  Testing on {len(sample)} paragraphs ({n_pos} pos, {n_neg} neg)")
        
        results = classifier.classify_batch(
            sample["paragraph"].tolist(), category
        )
        
        for idx, (_, row) in enumerate(sample.iterrows()):
            result = results[idx]
            all_predictions.append({
                "paragraph_id": row["paragraph_id"],
                "risk_category": category,
                "y_true": row[label_col],
                "y_pred": 1 if result.get("classification") == "PRESENT" else 0,
                "y_score": result.get("confidence", 0.0),
                "risk_level": result.get("risk_level", "none"),
                "summary": result.get("summary", ""),
                "extracted_clause": result.get("extracted_clause", ""),
                "model": classifier.model,
            })
    
    # Save predictions
    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_parquet(PROCESSED_DIR / "llm_predictions.parquet", index=False)
    print(f"\nSaved predictions to {PROCESSED_DIR / 'llm_predictions.parquet'}")
    
    classifier.print_cost_report()
    
    return pred_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluate", "single"], default="evaluate")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()
    
    if args.mode == "evaluate":
        evaluate_llm_approach(
            provider=args.provider,
            max_samples_per_category=args.max_samples,
        )
