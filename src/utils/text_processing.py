"""
Text processing utilities for contract analysis.
"""

import re
from typing import List, Tuple


def count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3 for subword tokenizers)."""
    return int(len(text.split()) * 1.3)


def truncate_to_tokens(text: str, max_tokens: int = 512) -> str:
    """Truncate text to approximately max_tokens."""
    words = text.split()
    approx_words = int(max_tokens / 1.3)
    if len(words) <= approx_words:
        return text
    return " ".join(words[:approx_words])


def extract_section_headers(text: str) -> List[Tuple[int, str]]:
    """
    Extract section headers from contract text.
    Returns list of (position, header_text) tuples.
    """
    patterns = [
        r"^(\d+\.[\d.]*)\s+([A-Z][A-Z\s]+)",           # "1.1 DEFINITIONS"
        r"^(ARTICLE|SECTION|EXHIBIT)\s+(\w+)",            # "ARTICLE I"
        r"^([A-Z][A-Z\s]{5,})$",                          # "LIMITATION OF LIABILITY"
    ]
    
    headers = []
    for i, line in enumerate(text.split("\n")):
        line = line.strip()
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                headers.append((i, line))
                break
    
    return headers


def normalize_legal_text(text: str) -> str:
    """Normalize common legal text patterns for consistency."""
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Normalize dashes
    text = text.replace("—", " - ").replace("–", " - ")
    
    # Normalize section references
    text = re.sub(r"Section\s+(\d)", r"Section \1", text, flags=re.IGNORECASE)
    text = re.sub(r"Article\s+(\w)", r"Article \1", text, flags=re.IGNORECASE)
    
    # Clean up redacted portions
    text = re.sub(r"\*{3,}", "[REDACTED]", text)
    text = re.sub(r"_{3,}", "[REDACTED]", text)
    
    return text


def is_boilerplate(text: str) -> bool:
    """
    Detect boilerplate paragraphs that rarely contain risk clauses.
    These can be de-prioritized during analysis.
    """
    boilerplate_signals = [
        r"^this agreement is made",
        r"^in witness whereof",
        r"^the undersigned",
        r"^dated as of",
        r"table of contents",
        r"^exhibit\s+[a-z]$",
        r"^schedule\s+\d",
        r"^\d+\s*$",  # Page numbers
    ]
    
    text_lower = text.lower().strip()
    
    for pattern in boilerplate_signals:
        if re.match(pattern, text_lower):
            return True
    
    # Very short text is likely a header or label
    if len(text.split()) < 10:
        return True
    
    return False
