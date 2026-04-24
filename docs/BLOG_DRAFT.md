# We Compared Fine-Tuning vs LLMs vs Hybrid for Legal Contract Review — Here's What We Found

> **TL;DR:** We built a contract risk intelligence system that detects 8 categories of legal risk clauses 
> using three approaches head-to-head. The hybrid pipeline won — but not for the reason you'd expect.

## The Problem

Contract review costs law firms $500-900/hour. A typical M&A due diligence involves reviewing 
hundreds of contracts for risk clauses. We wanted to know: can ML automate the hardest part — 
identifying which clauses actually matter?

## What We Built

[Screenshot of dashboard here]

A pipeline that takes a raw contract and outputs:
- A risk heatmap across 8 categories (Liability, IP, Termination, etc.)
- Extracted clause text with confidence scores
- Plain-English risk summaries

## The Dataset: CUAD

[Stats and charts from EDA here]

Key insight: ...

## Three Approaches, Head-to-Head

### Approach 1: Fine-tuned DeBERTa
- What we did: ...
- What worked: ...
- What didn't: ...

### Approach 2: LLM Few-Shot (Claude)
- What we did: ...
- What worked: ...
- What didn't: ...

### Approach 3: Hybrid Pipeline
- What we did: ...
- The routing logic: ...
- Why it won: ...

## Results

[Comparison table here]

[Per-category chart here]

## The Interesting Failures

[Error analysis section — this is what makes the post worth reading]

## What We'd Do Differently

1. ...
2. ...
3. ...

## Try It Yourself

- **Live demo:** [Streamlit Cloud link]
- **Code:** [GitHub link]
- **Dataset:** [CUAD on HuggingFace]

---

*Built by [Your Name] and [Friend's Name]. We're looking for Data Scientist roles — 
[connect on LinkedIn](#).*
