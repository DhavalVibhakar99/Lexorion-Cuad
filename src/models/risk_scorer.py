"""
Risk Scorer: aggregates clause-level detections into a contract-level risk profile.

Turns individual clause flags into a holistic risk assessment:
- Per-category risk scores (0-1)
- Overall contract risk tier (Low / Medium / High / Critical)
- Comparative percentiles against the CUAD dataset

Usage:
    from src.models.risk_scorer import RiskScorer
    scorer = RiskScorer()
    profile = scorer.score_contract(detections)
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


CONFIG_DIR = Path("configs")
PROCESSED_DIR = Path("data/processed")


@dataclass
class RiskAssessment:
    """Contract-level risk assessment."""
    category_scores: Dict[str, float]      # risk_category → score (0-1)
    overall_score: float                    # weighted average
    risk_tier: str                          # Low / Medium / High / Critical
    percentile: Optional[float]            # vs CUAD dataset (0-100)
    flags: List[str]                        # human-readable risk flags
    
    def to_dict(self) -> dict:
        return {
            "category_scores": self.category_scores,
            "overall_score": round(self.overall_score, 3),
            "risk_tier": self.risk_tier,
            "percentile": self.percentile,
            "flags": self.flags,
        }


class RiskScorer:
    """Aggregate clause detections into contract-level risk scores."""
    
    def __init__(self):
        with open(CONFIG_DIR / "category_mapping.yaml") as f:
            self.config = yaml.safe_load(f)
        
        self.severity_weights = {
            key: info["severity_weight"]
            for key, info in self.config["risk_categories"].items()
        }
        
        # Load dataset baseline for percentile comparison
        self.baseline = self._load_baseline()
    
    def _load_baseline(self) -> Optional[pd.DataFrame]:
        """Load CUAD dataset stats for comparative scoring."""
        try:
            df = pd.read_parquet(PROCESSED_DIR / "paragraphs_chunked.parquet")
            
            # Compute per-contract risk profiles across the dataset
            label_cols = [c for c in df.columns if c.startswith("label_")]
            
            contract_profiles = []
            for contract, group in df.groupby("contract_title"):
                profile = {}
                for col in label_cols:
                    cat = col.replace("label_", "")
                    # Score = fraction of paragraphs with this risk type
                    profile[cat] = group[col].mean()
                
                weight_sum = sum(self.severity_weights.get(cat, 0.5) for cat in profile)
                profile["overall"] = sum(
                    score * self.severity_weights.get(cat, 0.5)
                    for cat, score in profile.items()
                ) / max(weight_sum, 1)
                
                contract_profiles.append(profile)
            
            return pd.DataFrame(contract_profiles)
        except FileNotFoundError:
            return None
    
    def score_contract(
        self,
        detections: List[dict],
        total_paragraphs: int = 1,
    ) -> RiskAssessment:
        """
        Score a contract based on clause detections.
        
        Args:
            detections: List of detection dicts with keys:
                - risk_category (str)
                - is_present (bool)
                - confidence (float)
                - risk_level (str)
            total_paragraphs: Total paragraphs in the contract
        """
        # Per-category scores
        category_scores = {}
        flags = []
        
        for category in self.severity_weights:
            cat_detections = [
                d for d in detections
                if d.get("risk_category") == category and d.get("is_present")
            ]
            
            if not cat_detections:
                category_scores[category] = 0.0
                continue
            
            # Score combines detection count, confidence, and severity
            avg_confidence = np.mean([d.get("confidence", 0.5) for d in cat_detections])
            density = len(cat_detections) / max(total_paragraphs, 1)
            
            # Weighted score
            raw_score = avg_confidence * (1 + np.log1p(density))
            category_scores[category] = min(raw_score, 1.0)
            
            # Generate flags
            severity = self.severity_weights[category]
            if category_scores[category] > 0.6 and severity > 0.8:
                display = self.config["risk_categories"][category]["display_name"]
                flags.append(f"High {display} detected ({len(cat_detections)} clauses)")
        
        # Overall score (weighted average)
        total_weight = sum(self.severity_weights.values())
        overall = sum(
            category_scores.get(cat, 0) * weight
            for cat, weight in self.severity_weights.items()
        ) / total_weight
        
        # Risk tier
        if overall >= 0.6:
            tier = "Critical"
        elif overall >= 0.4:
            tier = "High"
        elif overall >= 0.2:
            tier = "Medium"
        else:
            tier = "Low"
        
        # Percentile vs dataset
        percentile = None
        if self.baseline is not None and "overall" in self.baseline.columns:
            percentile = (self.baseline["overall"] <= overall).mean() * 100
            percentile = round(percentile, 1)
        
        return RiskAssessment(
            category_scores=category_scores,
            overall_score=overall,
            risk_tier=tier,
            percentile=percentile,
            flags=flags,
        )


if __name__ == "__main__":
    # Quick test with mock data
    scorer = RiskScorer()
    
    mock_detections = [
        {"risk_category": "liability_risk", "is_present": True, "confidence": 0.85, "risk_level": "high"},
        {"risk_category": "liability_risk", "is_present": True, "confidence": 0.72, "risk_level": "moderate"},
        {"risk_category": "ip_risk", "is_present": True, "confidence": 0.90, "risk_level": "high"},
        {"risk_category": "termination_risk", "is_present": False, "confidence": 0.15, "risk_level": "none"},
    ]
    
    assessment = scorer.score_contract(mock_detections, total_paragraphs=50)
    
    print(f"\nOverall Score: {assessment.overall_score:.3f}")
    print(f"Risk Tier: {assessment.risk_tier}")
    if assessment.percentile is not None:
        print(f"Percentile: {assessment.percentile}th")
    
    print(f"\nCategory Scores:")
    for cat, score in sorted(assessment.category_scores.items(), key=lambda x: -x[1]):
        if score > 0:
            print(f"  {cat:<25} {score:.3f}")
    
    if assessment.flags:
        print(f"\nFlags:")
        for flag in assessment.flags:
            print(f"  ⚠️  {flag}")
