# We Tested TF-IDF Against Two Transformers for Contract Risk Screening. Bag-of-Words Won Twice.

> **TL;DR:** We built a contract risk screening system on the CUAD legal dataset
> and ran a five-way bake-off: TF-IDF, frozen sentence embeddings, fine-tuned
> DeBERTa, an LLM, and a hybrid router. At matched recall, bag-of-words beat
> both transformers on precision — by 2.3×. The production system ended up
> being TF-IDF + LLM triage of weak flags, it outperforms every individual
> model, and it runs entirely on free-tier cloud.
> [Try it live](https://dhavalvibhakar99.github.io/Lexorion-Cuad/demo/).

## The problem

In July 2024, one bad software update grounded airlines and hospitals
worldwide. The outage was the headline — but the aftermath was a contracts
story: Delta lost roughly $500M, and its recovery is capped by a
limitation-of-liability clause signed years earlier, at single-digit millions.
Thousands of companies had agreed to essentially the same clause. The risk
wasn't hidden. It was sitting in executed contracts, in plain English,
findable by anyone who read them.

Nobody reads every clause of every contract. That's the problem: contract
review is expensive expert time spent mostly on pattern recognition. We wanted
to know how far you could push the pattern-recognition layer with ML —
cheaply — while leaving judgment to humans.

## The setup

**Dataset:** [CUAD](https://www.atticusprojectai.org/cuad) — 408 real
commercial contracts where lawyers labeled 41 clause types. We mapped 30 of
those labels into 8 business-facing risk categories (liability, IP,
termination, indemnification, exclusivity, change of control, revenue,
renewal), split contracts (not paragraphs) into train/val/test to prevent
leakage, and chunked everything into ~35K paragraphs. Positive rate per
category: about 2%.

**The metric decision that shaped everything:** this is a *screening* task.
A missed liability cap costs incomparably more than a false alarm a reviewer
dismisses in seconds — the same asymmetry as AML transaction monitoring. So we
refused to optimize F1. Every model was thresholded **recall-first**: hit
≥90% recall on validation, then report the precision you get at that operating
point. F1 treats a missed clause and a false alarm as equally bad; in this
domain they are not.

## The bake-off

Five approaches, same test split, same recall-first policy for the local
models:

| Approach | Recall | Precision | Notes |
|---|---|---|---|
| **TF-IDF + logistic regression** | **0.904** | **0.248** | 30K sparse features, one classifier per category |
| MiniLM frozen embeddings + LogReg | 0.907 | 0.107 | pretrained sentence-transformer as feature extractor |
| DeBERTa-v3-base, fine-tuned | 0.874 | 0.107 | one multi-label run, 2 epochs on a free Colab T4 |
| LLM few-shot (gpt-oss-120b, free tier) | 0.613* | 0.812* | structured JSON output with validation guardrails |
| **Hybrid: TF-IDF screen + LLM triage** | **0.788*** | **0.933*** | best F1 of all five (0.852) |

\* *LLM rows measured on a category-balanced sample (n=61, free-tier rate
limits), so their precision is not comparable to the full-split rows above —
recall is. Full details and caveats in the
[repo](https://github.com/DhavalVibhakar99/Lexorion-Cuad).*

## Negative result #1: frozen embeddings halve your precision

The "obvious" first upgrade from bag-of-words is sentence embeddings. At
matched recall, MiniLM embeddings scored **0.107 precision vs TF-IDF's
0.248**. Not a small loss — half the precision, double the review noise.

The reason, in hindsight, is the nature of the task. Risk clauses announce
themselves with near-literal signatures: *"indemnify and hold harmless"*,
*"shall not exceed the fees paid"*, *"automatically renews"*. TF-IDF bigrams
latch onto exactly those phrases. General-purpose embeddings blur them into
topical similarity — and a contract is *full* of paragraphs that are topically
"legal obligations" without containing the clause you're hunting.

## Negative result #2: fine-tuned DeBERTa didn't fix it

Fine, frozen embeddings lose — surely *fine-tuning* a transformer wins? We
trained DeBERTa-v3-base as a single multi-label model (8 sigmoid heads, BCE
loss with per-class pos_weight for the 2% positive rate) on a free Colab T4.
Twenty minutes of training, healthy loss curves.

Result: **precision 0.107 at recall 0.874 — identical to the frozen
embeddings.** One diagnostic told the story: the recall-first threshold search
bottomed out at the floor for 7 of 8 categories, meaning the model's scores
barely separated positives from negatives at this training budget.

The honest caveat: 2 epochs is what a free GPU session buys. More epochs and
tuning would likely help, possibly past TF-IDF. But that's precisely the
engineering point — **spending real GPU money to maybe match a free model is a
bad trade** when the precision you need is available elsewhere.

## Where the precision actually came from

The LLM (a free-tier open model via OpenRouter) is the mirror image of the
baseline: conservative, high-precision, lower recall — and far too slow and
rate-limited to read everything.

So the production architecture routes:

```
score ≥ threshold + margin   → strong flag, keep (free)
threshold ≤ score < +margin  → weak flag → LLM confirms or clears
score < threshold            → cleared (free)
```

The recall-first TF-IDF screen over-flags by design; the LLM triages only the
*weak* flags — confirming real risks with a plain-English summary and the
extracted clause, or clearing false alarms. On identical evaluation
paragraphs, the hybrid beat both of its components (F1 0.852 vs 0.844
baseline, 0.764 LLM-only) — while sending **4.4% of the 43,408
paragraph×category decisions** to the LLM. The other 95.6% cost nothing.

Each model is only asked the question it's good at: the baseline decides
what's clean, the LLM adjudicates flags. The LLM never gets the chance to
miss things, so its weakness (recall) never matters.

## What we'd tell you to steal

1. **Pick the operating point before the model.** Recall-first thresholding
   changed our conclusions more than any architecture choice.
2. **Make the cheap baseline strong before reaching for transformers.** Ours
   turned out to be the production model. "Bag-of-words beat a transformer
   twice" is a finding you only get if you actually run the comparison.
3. **Route by uncertainty, not by category.** The LLM budget goes exactly
   where the cheap model is unsure — that's what makes 4.4% escalation enough.
4. **Label your evaluation sets.** Balanced-sample precision and
   true-prevalence precision are different numbers. Mixing them in one table
   flatters whichever model was measured on the easier distribution.
5. **Cache everything, never cache errors.** Our LLM layer caches successful
   verdicts (reruns are free) but never caches rate-limit failures — an early
   bug where transient errors got cached would have silently poisoned every
   future run.

## Try it

- **Live app:** [dhavalvibhakar99.github.io/Lexorion-Cuad/demo](https://dhavalvibhakar99.github.io/Lexorion-Cuad/demo/)
  — paste a contract or upload a PDF; a React front-end on GitHub Pages calls
  a FastAPI inference service on Hugging Face Spaces. Total infrastructure
  cost: $0/month.
- **Code:** [github.com/DhavalVibhakar99/Lexorion-Cuad](https://github.com/DhavalVibhakar99/Lexorion-Cuad)
  — pipeline, evals, tests, CI, and the one-click Colab notebook that
  reproduces the DeBERTa result.
- **Dataset:** [CUAD](https://www.atticusprojectai.org/cuad) by The Atticus
  Project.

---

*Built by Dhaval Vibhakar (engineering: pipeline, models, API, deployment)
and Sahil Shinde (analysis: EDA, evaluation design, prompts). Not legal
advice — it's a screening tool, and the README documents exactly what it
misses.*
