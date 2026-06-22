"""
Lexorion product dashboard.
This is the portfolio-facing interface hiring managers see.

Run: streamlit run src/dashboard/app.py
"""

import json
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

st.sidebar.markdown(
    """
    <div class="lex-sidebar-panel">
        <div class="lex-sidebar-panel-title">System Status</div>
        <div class="lex-status-line"><span>Baseline</span><strong>Live</strong></div>
        <div class="lex-status-line"><span>LLM Layer</span><strong>OpenRouter</strong></div>
        <div class="lex-status-line"><span>Transformer</span><strong>Pending</strong></div>
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


def render_analysis_results(profile: dict):
    """Render a risk profile from either live pipeline output or demo JSON."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Paragraphs Analyzed", profile.get("total_paragraphs", 0))
    col2.metric("Clauses Flagged", profile.get("flagged_paragraphs", 0))
    col3.metric(
        "Review Rate",
        f"{profile.get('flagged_paragraphs', 0) / max(profile.get('total_paragraphs', 1), 1):.1%}",
    )
    col4.metric("Processing Time", f"{profile.get('processing_time_seconds', 0):.1f}s")

    st.plotly_chart(
        render_risk_heatmap(profile.get("risk_scores", {})),
        width="stretch",
    )

    detections = profile.get("detections", [])
    if detections:
        st.subheader("Review Queue")
        for d in detections:
            risk_level = d.get("risk_level", "none")
            label = "Priority" if risk_level in ["high", "critical"] else "Review"
            with st.expander(
                f"{label}: {d.get('risk_category', '').replace('_', ' ').title()} "
                f"(confidence: {d.get('confidence', 0):.0%})"
            ):
                st.markdown(f"**Risk Level:** {risk_level}")
                st.markdown(f"**Summary:** {d.get('summary', '')}")
                st.markdown(f"**Model:** {d.get('model_used', '')}")
                if d.get("extracted_clause"):
                    st.code(d["extracted_clause"], language=None)
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


def render_signal_strip(baseline_ready: bool):
    """Render product-level operating signals for the review console."""
    baseline_value = "Live scoring" if baseline_ready else "Training needed"
    baseline_note = (
        "Saved TF-IDF model artifact is loaded"
        if baseline_ready
        else "Run baseline training to enable live predictions"
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
                <div class="lex-signal-value">OpenRouter ready</div>
                <div class="lex-signal-note">For uncertain or high-value clauses</div>
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
    render_signal_strip(baseline_ready)

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
                st.warning("PDF parsing coming soon. For now, paste the text directly.")

    if contract_text and st.button("Analyze Contract", type="primary"):
        with st.spinner("Reviewing clauses and scoring risk exposure..."):
            if input_method == "Demo contract" and not baseline_ready:
                profile = load_example_json("demo_risk_profile.json")
                render_analysis_results(profile)
                st.stop()

            try:
                from src.data_pipeline.chunk_contracts import split_into_paragraphs
                from src.models.baseline_detector import analyze_contract_with_baseline

                paragraphs = split_into_paragraphs(contract_text)

                if not paragraphs:
                    st.error(
                        "No analyzable paragraphs found. Ensure the text is long enough."
                    )
                else:
                    profile = analyze_contract_with_baseline(
                        paragraphs,
                        contract_id=(
                            "demo" if input_method == "Demo contract" else "uploaded"
                        ),
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
        # Show demo with sample data
        st.info("Paste a contract above, or explore the model and error analysis tabs.")

        # Show sample risk profile
        st.subheader("Sample Exposure Profile")
        sample_scores = {
            "liability_risk": 0.82,
            "ip_risk": 0.65,
            "termination_risk": 0.45,
            "indemnification": 0.71,
            "exclusivity": 0.30,
            "change_of_control": 0.15,
            "revenue_risk": 0.55,
            "renewal_expiration": 0.20,
        }
        st.plotly_chart(render_risk_heatmap(sample_scores), width="stretch")


elif page == "Model Performance":
    render_page_header(
        "Model Performance",
        "Track model quality across the baseline, DeBERTa, LLM-only, and hybrid approaches.",
        "Baseline measured",
    )

    comparison = load_comparison_results()
    detailed = load_detailed_comparison()

    if comparison is not None:
        # Summary table
        st.subheader("Overall Benchmark")
        st.dataframe(comparison, width="stretch", hide_index=True)

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

        # Show placeholder chart
        st.subheader("Expected Output")
        placeholder_data = pd.DataFrame(
            {
                "Category": ["Liability", "IP Risk", "Termination", "Indemnification"]
                * 3,
                "Approach": ["DeBERTa"] * 4 + ["LLM"] * 4 + ["Hybrid"] * 4,
                "F1": [
                    0.72,
                    0.58,
                    0.65,
                    0.70,
                    0.68,
                    0.75,
                    0.71,
                    0.78,
                    0.74,
                    0.73,
                    0.70,
                    0.77,
                ],
            }
        )
        fig = px.bar(
            placeholder_data,
            x="Category",
            y="F1",
            color="Approach",
            barmode="group",
            title="Sample Comparison",
        )
        st.plotly_chart(fig, width="stretch")


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
