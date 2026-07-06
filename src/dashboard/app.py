"""
Lexorion product dashboard.
This is the portfolio-facing interface hiring managers see.

Run: streamlit run src/dashboard/app.py
"""

import html as html_lib
import json
import os
import sys
import yaml
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def llm_layer_available() -> bool:
    """Resolve the OpenRouter key from env/.env or Streamlit secrets."""
    if os.getenv("OPENROUTER_API_KEY"):
        return True
    try:
        secrets_key = st.secrets["OPENROUTER_API_KEY"]
    except (KeyError, FileNotFoundError):
        return False
    if not secrets_key:
        return False
    os.environ["OPENROUTER_API_KEY"] = str(secrets_key)
    try:
        os.environ.setdefault("OPENROUTER_MODEL", str(st.secrets["OPENROUTER_MODEL"]))
    except (KeyError, FileNotFoundError):
        pass
    return True

CONFIG_DIR = Path("configs")
PROCESSED_DIR = Path("data/processed")
EXAMPLES_DIR = Path("examples")
CHECKPOINT_DIR = Path("checkpoints")
BASELINE_MODEL_PATH = CHECKPOINT_DIR / "baseline_tfidf_logreg.joblib"


# === Page Config ===
st.set_page_config(
    page_title="Lexorion",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="expanded",
)


_css = (ROOT_DIR / "src" / "dashboard" / "style.css").read_text()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)


# === Sidebar ===
st.sidebar.markdown(
    """
    <div class="lex-sidebar-brand">
        <div class="lex-logo-mark">LX</div>
        <div>
            <div class="lex-sidebar-name">Lexorion</div>
            <div class="lex-sidebar-tagline">Contract risk intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Workspace",
    [
        "Review Console",
        "Model Performance",
        "Error Analysis",
        "Dataset Explorer",
    ],
    label_visibility="collapsed",
)

_llm_ready = llm_layer_available()
st.sidebar.markdown(
    f"""
    <div class="lex-sidebar-panel">
        <div class="lex-sidebar-panel-title">System Status</div>
        <div class="lex-status-line"><span>Baseline</span><strong>Live</strong></div>
        <div class="lex-status-line"><span>LLM Layer</span><strong>{"Live" if _llm_ready else "Key needed"}</strong></div>
        <div class="lex-status-line"><span>Routing</span><strong>Hybrid</strong></div>
    </div>
    <div class="lex-sidebar-panel">
        <div class="lex-sidebar-panel-title">Review Taxonomy</div>
        <div class="lex-status-line"><span>Categories</span><strong>8</strong></div>
        <div class="lex-status-line"><span>Dataset</span><strong>CUAD</strong></div>
        <div class="lex-status-line"><span>Mode</span><strong>Demo</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# === Helper Functions ===


@st.cache_data
def load_category_mapping():
    """Load risk category config."""
    try:
        with open(CONFIG_DIR / "category_mapping.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None


@st.cache_data
def load_dataset_stats():
    """Load parsed dataset for stats."""
    try:
        return pd.read_parquet(PROCESSED_DIR / "paragraphs_chunked.parquet")
    except FileNotFoundError:
        return None


@st.cache_data
def load_comparison_results():
    """Load model comparison results."""
    try:
        return pd.read_csv(PROCESSED_DIR / "model_comparison.csv")
    except FileNotFoundError:
        try:
            return pd.read_csv(EXAMPLES_DIR / "model_comparison.csv")
        except FileNotFoundError:
            return None


@st.cache_data
def load_detailed_comparison():
    """Load per-category comparison."""
    try:
        return pd.read_csv(PROCESSED_DIR / "model_comparison_detailed.csv")
    except FileNotFoundError:
        try:
            return pd.read_csv(EXAMPLES_DIR / "model_comparison_detailed.csv")
        except FileNotFoundError:
            return None


@st.cache_data
def load_error_summary():
    """Load checked-in error analysis summary."""
    try:
        return pd.read_csv(PROCESSED_DIR / "baseline_error_summary.csv")
    except FileNotFoundError:
        try:
            return pd.read_csv(EXAMPLES_DIR / "baseline_error_summary.csv")
        except FileNotFoundError:
            return None


@st.cache_data
def load_error_sample(kind: str):
    """Load checked-in false-positive or false-negative examples."""
    filename = f"baseline_{kind}_sample.csv"
    try:
        return pd.read_csv(EXAMPLES_DIR / filename)
    except FileNotFoundError:
        return None


def baseline_model_available() -> bool:
    """Return whether the saved baseline artifact is available locally."""
    return BASELINE_MODEL_PATH.exists()


@st.cache_data
def load_example_text(filename: str) -> str:
    """Load a small checked-in demo text file."""
    try:
        return (EXAMPLES_DIR / filename).read_text()
    except FileNotFoundError:
        return ""


@st.cache_data
def load_example_json(filename: str) -> dict:
    """Load a small checked-in demo JSON file."""
    try:
        with open(EXAMPLES_DIR / filename) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def render_risk_heatmap(risk_scores: dict):
    """Render a horizontal bar chart of risk scores."""
    categories = list(risk_scores.keys())
    scores = list(risk_scores.values())

    colors = []
    for s in scores:
        if s >= 0.7:
            colors.append("#e5534b")
        elif s >= 0.4:
            colors.append("#d4952a")
        elif s > 0:
            colors.append("#3fb950")
        else:
            colors.append("#2c2920")

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=[c.replace("_", " ").title() for c in categories],
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{s:.2f}" for s in scores],
            textposition="auto",
            textfont=dict(color="#e8e3d8", size=11),
        )
    )

    fig.update_layout(
        title="Risk Exposure by Category",
        xaxis_title="Risk Score",
        yaxis_title="",
        height=400,
        margin=dict(l=8, r=8, t=40, b=8),
        xaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28,26,21,0.6)",
        font=dict(color="#7a7168", family="sans-serif", size=12),
        title_font=dict(color="#e8e3d8", size=14),
    )

    return fig


CATEGORY_STAKES = {
    "liability_risk": "One clause decided the CrowdStrike-Delta fight: a ~$500M loss against a liability cap of single-digit millions.",
    "ip_risk": "Broad IP assignment or license grants can quietly transfer ownership of everything built under the contract.",
    "termination_risk": "Termination-for-convenience lets a counterparty walk away — revenue you planned on can vanish with 30 days' notice.",
    "indemnification": "Determines who pays when things go wrong; one-way indemnification shifts the entire downside to you.",
    "exclusivity": "Non-competes and exclusivity lock you out of markets long after the deal stops making sense.",
    "change_of_control": "When Broadcom bought VMware, renewal rights evaporated and costs jumped ~1,050% for some customers — these clauses decide what an acquisition does to your deals.",
    "revenue_risk": "Minimum commitments and uncapped price escalators compound silently: a 7% uncapped escalator turns $50K/yr into ~$370K over five years.",
    "renewal_expiration": "Auto-renewal with a missed notice window locks you into another full term at the vendor's price.",
}


def category_info() -> dict:
    """Description + severity per category from the mapping config."""
    config = load_category_mapping() or {}
    return {
        key: {
            "display": value.get("display_name", key.replace("_", " ").title()),
            "description": value.get("description", ""),
            "severity": value.get("severity_weight", 0.5),
        }
        for key, value in config.get("risk_categories", {}).items()
    }


def render_verdict(profile: dict):
    """One-sentence narrative verdict, so the reader knows what the numbers say."""
    info = category_info()
    scores = profile.get("risk_scores", {})
    elevated = sorted(
        ((c, s) for c, s in scores.items() if s >= 0.5),
        key=lambda item: -item[1],
    )
    flagged = profile.get("flagged_paragraphs", 0)
    if not elevated:
        text = (
            "No elevated risk categories. "
            f"{flagged} paragraph(s) carry weak flags worth a skim."
            if flagged
            else "No risk clauses detected in this contract."
        )
    else:
        names = ", ".join(info.get(c, {}).get("display", c) for c, _ in elevated[:3])
        text = (
            f"Elevated exposure in {names}. "
            f"{flagged} paragraph(s) queued for review below — start at the top."
        )
    st.markdown(
        f'<div class="lex-verdict">{html_lib.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_pipeline_strip(profile: dict):
    """Show what the pipeline actually did to this contract, stage by stage."""
    routing = profile.get("routing_summary")
    llm_stats = profile.get("llm_stats") or {}
    total = profile.get("total_paragraphs", 0)
    if not routing:
        return
    decisions = (
        routing["confident_positive"]
        + routing["uncertain"]
        + routing["confident_negative"]
    )
    stage2 = (
        f"{llm_stats.get('confirmed_by_llm', 0)} confirmed · "
        f"{llm_stats.get('cleared_by_llm', 0)} cleared as false alarms"
        if llm_stats.get("escalated")
        else (
            "no weak flags to triage"
            if routing["uncertain"] == 0
            else f"{llm_stats.get('fallbacks', 0)} kept without LLM review"
        )
    )
    st.markdown(
        f"""
        <div class="lex-stage-grid">
            <div class="lex-stage">
                <div class="lex-stage-step">1 · Screen (local, free)</div>
                <div class="lex-stage-value">{decisions:,} decisions</div>
                <div class="lex-stage-note">{total} paragraphs × 8 risk categories scored by the
                recall-first baseline; {routing["confident_negative"]:,} cleared instantly</div>
            </div>
            <div class="lex-stage">
                <div class="lex-stage-step">2 · Triage weak flags (LLM)</div>
                <div class="lex-stage-value">{routing["uncertain"]} escalated</div>
                <div class="lex-stage-note">{stage2}</div>
            </div>
            <div class="lex-stage">
                <div class="lex-stage-step">3 · Human review queue</div>
                <div class="lex-stage-value">{profile.get("flagged_paragraphs", 0)} paragraphs</div>
                <div class="lex-stage-note">{routing["confident_positive"]} strong flags kept local ·
                sorted by confidence, highest risk first</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _highlighted_clause(paragraph: str, clause: str) -> str:
    """Paragraph HTML with the extracted clause marked, if it's a substring."""
    para = html_lib.escape(paragraph)
    needle = html_lib.escape(clause.strip())
    if needle and needle in para:
        return para.replace(needle, f"<mark>{needle}</mark>", 1)
    return para


def render_analysis_results(profile: dict):
    """Render a risk profile from either live pipeline output or demo JSON."""
    render_verdict(profile)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Paragraphs Analyzed", profile.get("total_paragraphs", 0))
    col2.metric("Clauses Flagged", profile.get("flagged_paragraphs", 0))
    col3.metric(
        "Review Rate",
        f"{profile.get('flagged_paragraphs', 0) / max(profile.get('total_paragraphs', 1), 1):.1%}",
    )
    col4.metric("Processing Time", f"{profile.get('processing_time_seconds', 0):.1f}s")

    render_pipeline_strip(profile)

    llm_stats = profile.get("llm_stats")
    if llm_stats:
        notes = []
        if llm_stats.get("llm_model"):
            notes.append(f"LLM: {llm_stats['llm_model']}")
        if llm_stats.get("estimated_cost_usd") is not None:
            notes.append(f"est. cost ${llm_stats['estimated_cost_usd']:.4f}")
        if llm_stats.get("cache_hits"):
            notes.append(f"{llm_stats['cache_hits']} served from cache")
        if llm_stats.get("unavailable_reason"):
            notes.append(f"LLM unavailable: {llm_stats['unavailable_reason']}")
        if notes:
            st.caption(" · ".join(notes))

    st.plotly_chart(
        render_risk_heatmap(profile.get("risk_scores", {})),
        width="stretch",
    )

    detections = profile.get("detections", [])
    info = category_info()
    if detections:
        st.subheader("Review Queue")
        st.caption(
            "Each card explains itself: what was found, the exact words that "
            "triggered the model, and why this clause type costs money."
        )
        model_badges = {
            "llm": "LLM verified",
            "baseline": "Baseline (strong signal)",
            "baseline_fallback": "Baseline (unreviewed weak flag)",
            "tfidf_logistic_regression": "Baseline",
        }
        for d in detections:
            category = d.get("risk_category", "")
            cat = info.get(category, {})
            risk_level = d.get("risk_level", "none")
            label = "Priority" if risk_level in ["high", "critical"] else "Review"
            badge = model_badges.get(d.get("model_used", ""), d.get("model_used", ""))
            with st.expander(
                f"{label}: {cat.get('display', category.replace('_', ' ').title())} "
                f"(confidence: {d.get('confidence', 0):.0%} · {badge})"
            ):
                st.markdown(f"**Finding:** {d.get('summary', '')}")

                evidence = d.get("evidence_terms") or []
                if evidence:
                    chips = "".join(
                        f'<span class="lex-chip">{html_lib.escape(t)}</span>'
                        for t in evidence
                    )
                    st.markdown(
                        '<div class="lex-evidence-label">Triggered by these phrases '
                        f"(model's actual features):</div>{chips}",
                        unsafe_allow_html=True,
                    )

                stakes = CATEGORY_STAKES.get(category)
                why = cat.get("description", "")
                if why or stakes:
                    why_text = f"{why.rstrip('.')}." if why else ""
                    st.markdown(
                        f"""<div class="lex-why"><strong>Why this matters:</strong>
                        {html_lib.escape(why_text)}{" " + html_lib.escape(stakes) if stakes else ""}
                        <span class="lex-severity">severity weight {cat.get("severity", 0.5):.2f}</span></div>""",
                        unsafe_allow_html=True,
                    )

                clause = d.get("extracted_clause", "")
                paragraph_text = d.get("paragraph_text", "")
                if paragraph_text and clause and clause.strip() in paragraph_text:
                    st.markdown(
                        '<div class="lex-clause-context">'
                        + _highlighted_clause(paragraph_text, clause)
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                elif clause:
                    st.code(clause, language=None)
    else:
        st.success("No significant risk clauses detected.")

    unknown = profile.get("unknown_risk_candidates", [])
    if unknown:
        st.subheader("Emerging Risk Candidates")
        st.caption(
            "These are risks that may be important but are not part of the current 8-category taxonomy."
        )
        for item in unknown:
            with st.expander(item.get("suggested_category", "Unknown Risk")):
                st.markdown(f"**Reason:** {item.get('reason', '')}")
                if item.get("evidence"):
                    st.code(item["evidence"], language=None)


def render_page_header(title: str, subtitle: str, status: str = "Portfolio build"):
    """Render the product header used across pages."""
    st.markdown(
        f"""
        <div class="lex-header">
            <div class="lex-brand">
                <div class="lex-heading-wrap">
                    <div class="lex-header-mark">LX</div>
                    <div>
                        <div class="lex-overline">Lexorion Command Center</div>
                        <p class="lex-title">{title}</p>
                        <div class="lex-subtitle">{subtitle}</div>
                    </div>
                </div>
                <div class="lex-pill">{status}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_strip(baseline_ready: bool, llm_ready: bool = False):
    """Render product-level operating signals for the review console."""
    baseline_value = "Live scoring" if baseline_ready else "Training needed"
    baseline_note = (
        "Saved TF-IDF model artifact is loaded"
        if baseline_ready
        else "Run baseline training to enable live predictions"
    )
    llm_value = "Live via OpenRouter" if llm_ready else "Key needed"
    llm_note = (
        "Uncertain clauses get an LLM second opinion"
        if llm_ready
        else "Set OPENROUTER_API_KEY to enable escalation"
    )
    st.markdown(
        f"""
        <div class="lex-signal-grid">
            <div class="lex-signal">
                <div class="lex-signal-label">Risk Taxonomy</div>
                <div class="lex-signal-value">8 business categories</div>
                <div class="lex-signal-note">Liability, IP, termination, revenue, and more</div>
            </div>
            <div class="lex-signal">
                <div class="lex-signal-label">Baseline Model</div>
                <div class="lex-signal-value">{baseline_value}</div>
                <div class="lex-signal-note">{baseline_note}</div>
            </div>
            <div class="lex-signal">
                <div class="lex-signal-label">LLM Routing</div>
                <div class="lex-signal-value">{llm_value}</div>
                <div class="lex-signal-note">{llm_note}</div>
            </div>
            <div class="lex-signal">
                <div class="lex-signal-label">Audit Layer</div>
                <div class="lex-signal-value">Error analysis built in</div>
                <div class="lex-signal-note">False negatives and false positives tracked</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# === Pages ===

if page == "Review Console":
    baseline_ready = baseline_model_available()
    render_page_header(
        "Review Console",
        "A contract review workspace that scores clause exposure, queues risky language, and prepares uncertain findings for LLM review.",
        "Live baseline" if baseline_ready else "Demo environment",
    )
    render_signal_strip(baseline_ready, _llm_ready)

    with st.expander("Analysis settings", expanded=False):
        use_llm = st.toggle(
            "LLM triage of weak flags",
            value=_llm_ready,
            disabled=not _llm_ready,
            help=(
                "The baseline is tuned recall-first, so it over-flags by design. "
                "Weak flags (just above the decision threshold) are escalated to "
                "the OpenRouter LLM, which confirms real risks with a plain-"
                "English summary or clears false alarms. Strong flags stay local."
                if _llm_ready
                else "Set OPENROUTER_API_KEY in .env or Streamlit secrets to enable."
            ),
        )
        llm_budget = st.slider(
            "LLM call budget per analysis",
            min_value=5,
            max_value=60,
            value=20,
            step=5,
            disabled=not _llm_ready,
            help="Hard cap on API calls. Leftover uncertain clauses fall back to the baseline verdict.",
        )

    # Input method
    st.markdown(
        '<div class="lex-workspace-label">Document Intake</div>',
        unsafe_allow_html=True,
    )
    input_method = st.segmented_control(
        "Document source",
        ["Demo contract", "Paste text", "Upload file"],
        default="Demo contract",
    )

    contract_text = ""

    if input_method == "Demo contract":
        contract_text = load_example_text("sample_contract.txt")
        st.text_area(
            "Contract text",
            value=contract_text,
            height=300,
            disabled=True,
        )
        st.caption(
            "Demo contract uses the saved baseline model when available; otherwise it falls back to checked-in sample output."
        )
    elif input_method == "Paste text":
        contract_text = st.text_area(
            "Contract text",
            height=300,
            placeholder="Paste your contract text here...",
        )
    else:
        uploaded_file = st.file_uploader("Upload contract", type=["txt", "pdf"])
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                contract_text = uploaded_file.read().decode("utf-8")
            else:
                try:
                    from pypdf import PdfReader

                    reader = PdfReader(uploaded_file)
                    contract_text = "\n\n".join(
                        (page.extract_text() or "") for page in reader.pages
                    ).strip()
                    if contract_text:
                        st.caption(
                            f"Extracted text from {len(reader.pages)} page(s): "
                            f"{uploaded_file.name}"
                        )
                        st.text_area(
                            "Extracted contract text",
                            value=contract_text,
                            height=240,
                            disabled=True,
                        )
                    else:
                        st.warning(
                            "No extractable text found — this PDF may be a scan. "
                            "OCR is not supported yet; paste the text instead."
                        )
                except Exception as exc:
                    st.error(f"Could not read PDF: {exc}")

    if contract_text and st.button("Analyze Contract", type="primary"):
        with st.spinner("Reviewing clauses and scoring risk exposure..."):
            if input_method == "Demo contract" and not baseline_ready:
                profile = load_example_json("demo_risk_profile.json")
                render_analysis_results(profile)
                st.stop()

            try:
                from src.data_pipeline.chunk_contracts import split_into_paragraphs
                from src.models.baseline_detector import analyze_contract_with_baseline
                from src.models.hybrid_pipeline import analyze_contract_hybrid

                paragraphs = split_into_paragraphs(contract_text)

                if not paragraphs:
                    st.error(
                        "No analyzable paragraphs found. Ensure the text is long enough."
                    )
                else:
                    contract_id = (
                        "demo" if input_method == "Demo contract" else "uploaded"
                    )
                    if use_llm:
                        profile = analyze_contract_hybrid(
                            paragraphs,
                            contract_id=contract_id,
                            llm_max_calls=llm_budget,
                        )
                    else:
                        profile = analyze_contract_with_baseline(
                            paragraphs, contract_id=contract_id
                        )
                    render_analysis_results(profile)

            except FileNotFoundError:
                st.error(
                    "Baseline model artifact not found. Run this command first:\n\n"
                    "```bash\n"
                    "python -m src.models.baseline_detector\n"
                    "```"
                )
            except ImportError as e:
                st.error(f"Baseline inference is not ready yet: {e}")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    elif not contract_text:
        st.markdown(
            """
            <div class="lex-stage-grid">
                <div class="lex-stage">
                    <div class="lex-stage-step">1 · Screen (local, free)</div>
                    <div class="lex-stage-value">Every paragraph, 8 categories</div>
                    <div class="lex-stage-note">A recall-first TF-IDF model reads the whole
                    contract — tuned to catch ~9 in 10 risky clauses, over-flagging by design
                    like an AML monitoring screen</div>
                </div>
                <div class="lex-stage">
                    <div class="lex-stage-step">2 · Triage weak flags (LLM)</div>
                    <div class="lex-stage-value">~4% of decisions</div>
                    <div class="lex-stage-note">Flags just above the decision threshold go to an
                    LLM that confirms real risks (with a plain-English summary and the exact
                    clause) or clears false alarms</div>
                </div>
                <div class="lex-stage">
                    <div class="lex-stage-step">3 · Human review queue</div>
                    <div class="lex-stage-value">Ranked findings</div>
                    <div class="lex-stage-note">Each finding shows the trigger phrases, the
                    clause in context, and why that clause type costs money</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("What Lexorion looks for")
        info = category_info()
        glossary = pd.DataFrame(
            [
                {
                    "Risk Category": v["display"],
                    "What it covers": v["description"],
                    "Real-world stakes": CATEGORY_STAKES.get(k, ""),
                }
                for k, v in info.items()
            ]
        )
        st.dataframe(glossary, width="stretch", hide_index=True)
        st.caption(
            "Pick a document source above to run a live analysis — the demo "
            "contract takes about two seconds."
        )


elif page == "Model Performance":
    render_page_header(
        "Model Performance",
        "Recall-first screening benchmark: TF-IDF baseline, MiniLM embeddings, OpenRouter LLM, and the hybrid router on the CUAD test split.",
        "Measured results",
    )

    comparison = load_comparison_results()
    detailed = load_detailed_comparison()

    if comparison is not None:
        # Summary table
        st.subheader("Overall Benchmark")
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.caption(
            "Local models are thresholded recall-first (target ≥0.90): a "
            "screening tool pays more for a missed clause than a false alarm. "
            "Check the Evaluation Set column before comparing rows — "
            "balanced-sample precision (LLM, hybrid) is not comparable to "
            "true-prevalence precision (baseline, embed). Recall always is."
        )

        # Per-category comparison
        if detailed is not None:
            st.subheader("Category-Level F1")

            # Reshape for plotting
            f1_cols = [c for c in detailed.columns if c.endswith("_f1")]
            if f1_cols:
                plot_data = detailed[["category"] + f1_cols].melt(
                    id_vars=["category"],
                    var_name="Approach",
                    value_name="F1 Score",
                )
                plot_data["Approach"] = (
                    plot_data["Approach"].str.replace("_f1", "").str.upper()
                )

                fig = px.bar(
                    plot_data,
                    x="category",
                    y="F1 Score",
                    color="Approach",
                    barmode="group",
                    title="F1 by Risk Category and Approach",
                    height=500,
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width="stretch")

            # Winner table
            if "best_approach" in detailed.columns:
                st.subheader("Best Approach by Category")
                winner_counts = detailed["best_approach"].value_counts()
                col1, col2, col3 = st.columns(3)
                for i, (approach, count) in enumerate(winner_counts.items()):
                    [col1, col2, col3][i % 3].metric(
                        approach.upper(), f"{count} categories"
                    )
    else:
        st.warning(
            "No comparison results found. Run the evaluation pipeline first:\n\n"
            "```bash\n"
            "python -m src.evaluation.model_comparison\n"
            "```"
        )


elif page == "Error Analysis":
    render_page_header(
        "Error Analysis",
        "Inspect false negatives and false positives so model improvements are driven by evidence, not guesswork.",
        "Baseline audit",
    )

    summary = load_error_summary()
    false_negatives = load_error_sample("false_negatives")
    false_positives = load_error_sample("false_positives")

    if summary is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Macro F1", f"{summary['f1'].mean():.3f}")
        col2.metric("Macro Precision", f"{summary['precision'].mean():.3f}")
        col3.metric("Macro Recall", f"{summary['recall'].mean():.3f}")
        col4.metric("Missed Risks", f"{summary['false_negatives'].sum():,}")

        st.subheader("Weakest Categories")
        st.dataframe(
            summary[
                [
                    "risk_category",
                    "f1",
                    "precision",
                    "recall",
                    "false_negatives",
                    "false_positives",
                    "false_negative_rate",
                    "false_positive_rate",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

        plot_data = summary.sort_values("f1")
        fig = px.bar(
            plot_data,
            x="f1",
            y="risk_category",
            orientation="h",
            color="false_negative_rate",
            color_continuous_scale=[[0, "#3fb950"], [0.5, "#d4952a"], [1, "#e5534b"]],
            title="Baseline Weak Spots",
            labels={
                "f1": "F1 Score",
                "risk_category": "Risk Category",
                "false_negative_rate": "Miss Rate",
            },
            height=430,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(28,26,21,0.6)",
            font=dict(color="#7a7168"),
            title_font=dict(color="#e8e3d8"),
        )
        st.plotly_chart(fig, width="stretch")

        tab1, tab2 = st.tabs(["Missed risks", "False alarms"])
        with tab1:
            st.markdown(
                '<div class="lex-section-note">False negatives are risky because the system missed a true clause.</div>',
                unsafe_allow_html=True,
            )
            if false_negatives is not None:
                st.dataframe(
                    false_negatives[
                        ["risk_category", "y_score", "contract_title", "text_preview"]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("False negative sample not generated yet.")

        with tab2:
            st.markdown(
                '<div class="lex-section-note">False positives are review noise: the model flagged a clause that the label does not mark as risky.</div>',
                unsafe_allow_html=True,
            )
            if false_positives is not None:
                st.dataframe(
                    false_positives[
                        ["risk_category", "y_score", "contract_title", "text_preview"]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("False positive sample not generated yet.")
    else:
        st.warning(
            "No baseline error analysis found. Run:\n\n"
            "```bash\n"
            "python -m src.evaluation.error_analysis --approach baseline\n"
            "```"
        )


elif page == "Dataset Explorer":
    render_page_header(
        "Dataset Explorer",
        "Understand the CUAD-derived paragraph dataset behind Lexorion's training and evaluation pipeline.",
        "CUAD foundation",
    )

    df = load_dataset_stats()
    config = load_category_mapping()

    if df is not None:
        # Key stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Paragraphs", f"{len(df):,}")
        col2.metric("Unique Contracts", df["contract_title"].nunique())
        col3.metric("With Risk Clauses", f"{df['has_any_risk'].sum():,}")
        col4.metric("Positive Rate", f"{df['has_any_risk'].mean():.1%}")

        st.markdown("---")

        # Distribution by risk category
        st.subheader("Positive Examples by Risk Category")
        label_cols = [c for c in df.columns if c.startswith("label_")]
        if label_cols:
            cat_counts = {
                col.replace("label_", "").replace("_", " ").title(): df[col].sum()
                for col in label_cols
            }
            fig = px.bar(
                x=list(cat_counts.values()),
                y=list(cat_counts.keys()),
                orientation="h",
                title="Training Examples by Risk Category",
                labels={"x": "Count", "y": "Category"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

        # Split distribution
        st.subheader("Train / Validation / Test Split")
        split_data = (
            df.groupby("split")
            .agg(
                paragraphs=("paragraph_id", "count"),
                contracts=("contract_title", "nunique"),
                positive=("has_any_risk", "sum"),
            )
            .reset_index()
        )
        st.dataframe(split_data, width="stretch", hide_index=True)

        # Paragraph length distribution
        st.subheader("Paragraph Length Distribution")
        fig = px.histogram(
            df,
            x="paragraph_length",
            nbins=50,
            title="Distribution of Paragraph Lengths (words)",
            color="has_any_risk",
            labels={
                "paragraph_length": "Words",
                "has_any_risk": "Contains Risk Clause",
            },
        )
        st.plotly_chart(fig, width="stretch")

    else:
        st.warning(
            "Dataset not processed yet. Run:\n\n"
            "```bash\n"
            "python -m src.data_pipeline.download_cuad\n"
            "python -m src.data_pipeline.parse_cuad\n"
            "python -m src.data_pipeline.chunk_contracts\n"
            "```"
        )


# === Footer ===
st.markdown(
    "<div class='lex-footer'>Lexorion · Contract risk intelligence · CUAD baseline with OpenRouter-ready LLM review</div>",
    unsafe_allow_html=True,
)
