"""
Map CUAD's 41 categories to 8 risk categories.
Utility functions used across the pipeline.

Run: python -m src.data_pipeline.category_mapper  (prints mapping summary)
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CONFIG_DIR = Path("configs")


def load_mapping() -> Tuple[Dict[str, dict], dict]:
    """
    Load category mapping config.
    
    Returns:
        (label_to_risk, full_config)
        label_to_risk: {cuad_label_name: {risk_category, display_name, severity_weight}}
    """
    with open(CONFIG_DIR / "category_mapping.yaml") as f:
        config = yaml.safe_load(f)
    
    label_to_risk = {}
    for risk_key, risk_info in config["risk_categories"].items():
        for cuad_label in risk_info["cuad_labels"]:
            label_to_risk[cuad_label] = {
                "risk_category": risk_key,
                "display_name": risk_info["display_name"],
                "severity_weight": risk_info["severity_weight"],
            }
    
    return label_to_risk, config


def get_risk_categories() -> List[str]:
    """Return list of risk category keys."""
    _, config = load_mapping()
    return list(config["risk_categories"].keys())


def get_category_info(risk_category: str) -> Optional[dict]:
    """Get info for a specific risk category."""
    _, config = load_mapping()
    return config["risk_categories"].get(risk_category)


def map_cuad_label(cuad_label: str) -> Optional[str]:
    """Map a single CUAD label to its risk category key."""
    label_to_risk, _ = load_mapping()
    info = label_to_risk.get(cuad_label)
    return info["risk_category"] if info else None


def print_mapping_summary():
    """Print the full mapping for verification."""
    label_to_risk, config = load_mapping()
    
    print(f"\n{'=' * 60}")
    print("CUAD → Risk Category Mapping")
    print(f"{'=' * 60}")
    
    for risk_key, risk_info in config["risk_categories"].items():
        print(f"\n{risk_info['display_name']} (weight: {risk_info['severity_weight']})")
        for label in risk_info["cuad_labels"]:
            print(f"  ← {label}")
    
    excluded = config.get("excluded_labels", [])
    if excluded:
        print(f"\nExcluded ({len(excluded)} labels):")
        for label in excluded:
            print(f"  ✗ {label}")
    
    # Coverage stats
    mapped_count = len(label_to_risk)
    excluded_count = len(excluded)
    total = mapped_count + excluded_count
    print(f"\nMapped:   {mapped_count}/41")
    print(f"Excluded: {excluded_count}/41")
    print(f"Missing:  {41 - total}/41")


if __name__ == "__main__":
    print_mapping_summary()
