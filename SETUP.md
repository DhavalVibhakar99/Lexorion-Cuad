# Setup Guide — Contract Risk Intelligence System

> Tested on macOS (Apple Silicon & Intel). Both collaborators follow these steps.

---

## Step 1: Install Homebrew (Mac package manager)

Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

It will ask for your Mac login password. You won't see characters as you type — that's normal.

**IMPORTANT:** When it finishes, it prints lines under "Next steps" that look like this:

```bash
echo >> /Users/YOURNAME/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv zsh)"' >> /Users/YOURNAME/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv zsh)"
```

**Copy and run those exact lines** (they'll be different for your username).

Verify:

```bash
brew --version
```

Should print something like `Homebrew 4.x.x`.

---

## Step 2: Install Python 3.11

```bash
brew install python@3.11
```

Verify:

```bash
/opt/homebrew/bin/python3.11 --version
```

Should print `Python 3.11.x`.

> **Why 3.11?** Python 3.9 (the default on many Macs) fails to build spaCy and
> some other ML libraries. 3.11 has pre-built wheels for everything we need.

---

## Step 3: Clone the repo

```bash
cd ~/Documents  # or wherever you keep projects
git clone https://github.com/YOUR-USERNAME/contract-risk-intel.git
cd contract-risk-intel
```

---

## Step 4: Create a virtual environment

```bash
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Your terminal prompt should now start with `(venv)`.

Verify you're on 3.11:

```bash
python --version
```

> **What is a venv?** It gives this project its own isolated set of packages.
> Without it, different projects on your machine fight over package versions.

> **Do I activate every time?** Yes, every new terminal. But VS Code can do it
> automatically — see Step 6.

---

## Step 5: Install dependencies

For the full development environment (training, notebooks, evaluation):

```bash
pip install -r requirements-dev.txt
```

This takes 3-5 minutes (PyTorch is ~2GB). If you see any red errors, share them
in our group chat before trying to fix them yourself.

If you only want to run the dashboard, the slim runtime set is enough
(it's also what Streamlit Cloud installs when deploying):

```bash
pip install -r requirements.txt
```

Verify key packages:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"
python -c "import spacy; print(f'spaCy {spacy.__version__}')"
```

All four should print version numbers with no errors.

---

## Step 6: Configure VS Code (recommended)

1. Open the project folder in VS Code: `code .`
2. Press `Cmd + Shift + P` → type "Python: Select Interpreter"
3. Pick the one that shows `./venv/bin/python` or has `contract-risk-intel` in the path
4. Now every new terminal in VS Code will auto-activate the venv

Install these VS Code extensions (optional but helpful):
- **Python** (Microsoft)
- **Jupyter** (Microsoft)
- **YAML** (Red Hat)

---

## Step 7: Set up API keys

Create a `.env` file in the project root:

```bash
echo "OPENROUTER_API_KEY=your-key-here" > .env
```

Lexorion can use OpenRouter for LLM evaluation. OpenRouter gives access to many
models through one OpenAI-compatible endpoint.

Optional `.env` values:

```bash
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_SITE_URL=https://github.com/DhavalVibhakar99/Lexorion-Cuad
OPENROUTER_APP_NAME=Lexorion Contract Risk Intelligence
```

You can still use direct Anthropic/OpenAI keys later if needed:

```bash
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

> This file is already in `.gitignore` so it won't get pushed to GitHub.

---

## Step 8: Verify the data pipeline

```bash
python -m src.data_pipeline.download_cuad
python -m src.data_pipeline.parse_cuad
python -m src.data_pipeline.chunk_contracts
```

If all three run without errors and you see stats printed in the terminal, you're done.

---

## Step 9: Run tests

```bash
pytest tests/ -v
```

All tests should pass.

---

## Step 10: Run baseline model reports

```bash
python -m src.models.baseline_detector
python -m src.evaluation.error_analysis --approach baseline
python -m src.evaluation.model_comparison
```

This creates local files under `data/processed/` and Git-friendly sample files
under `examples/`. It also creates the dashboard inference artifact:

```text
checkpoints/baseline_tfidf_logreg.joblib
```

That artifact is intentionally gitignored because it is generated locally.

To run a guarded OpenRouter LLM evaluation later:

```bash
python -m src.models.llm_classifier \
  --provider openrouter \
  --model openai/gpt-oss-120b:free \
  --max_samples 4 \
  --max_calls 30 \
  --categories termination_risk revenue_risk exclusivity
```

Guardrails:

- `--max_calls` caps real API calls, even if `--max_samples` is too high.
- `--categories` limits which risk categories can be sent to the LLM.
- By default, Lexorion evaluates only the weak categories unless you pass
  `--all_categories`.
- Malformed LLM JSON is marked as `ERROR` and not trusted.
- Cached responses do not count as new API calls.

---

## Step 11: Launch the dashboard (optional, to see the UI)

```bash
streamlit run src/dashboard/app.py
```

Opens in your browser at `http://localhost:8501`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `command not found: python3.11` | Use full path: `/opt/homebrew/bin/python3.11` |
| `command not found: brew` | You skipped the PATH setup in Step 1. Run the `echo` and `eval` lines again. |
| spaCy fails to build | Make sure you're on Python 3.11, not 3.9. Check with `python --version` inside venv. |
| `ModuleNotFoundError` when running scripts | Make sure `(venv)` shows in your prompt. If not: `source venv/bin/activate` |
| PyTorch is very slow | Expected on CPU. Training happens on Google Colab (free GPU), not locally. |

---

## Google Colab Setup (for DeBERTa training only)

Local machines don't have GPUs. We use Google Colab's free T4 GPU for training.

1. Go to https://colab.research.google.com
2. Upload the notebook or connect to GitHub
3. Runtime → Change runtime type → T4 GPU
4. Mount Google Drive or upload the processed parquet files
5. Run training cells

Detailed Colab notebook will be created when we reach Week 3.
