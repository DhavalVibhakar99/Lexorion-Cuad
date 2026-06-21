"""
Contract Risk Intelligence Dashboard.
This is the portfolio piece — what hiring managers see.

Run: streamlit run src/dashboard/app.py
"""

import json
import yaml
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path


CONFIG_DIR = Path("configs")
PROCESSED_DIR = Path("data/processed")
EXAMPLES_DIR = Path("examples")


# === Page Config ===
st.set_page_config(
    page_title="Contract Risk Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# === Sidebar ===
st.sidebar.title("⚖️ Contract Risk Intel")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Risk Analysis", "🔬 Model Comparison", "📈 Dataset Explorer"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Built with:**  
    CUAD Dataset · DeBERTa-v3 · Claude API  
    
    [GitHub Repo](#) · [Blog Post](#)
    """
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
        return None


@st.cache_data
def load_detailed_comparison():
    """Load per-category comparison."""
    try:
        return pd.read_csv(PROCESSED_DIR / "model_comparison_detailed.csv")
    except FileNotFoundError:
        return None


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
    
    # Color based on severity
    colors = []
    for s in scores:
        if s >= 0.7:
            colors.append("#ef4444")  # red
        elif s >= 0.4:
            colors.append("#f59e0b")  # amber
        elif s > 0:
            colors.append("#22c55e")  # green
        else:
            colors.append("#94a3b8")  # gray
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=[c.replace("_", " ").title() for c in categories],
        orientation="h",
        marker_color=colors,
        text=[f"{s:.2f}" for s in scores],
        textposition="auto",
    ))
    
    fig.update_layout(
        title="Risk Score by Category",
        xaxis_title="Risk Score (0-1)",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 1]),
    )
    
    return fig


def render_analysis_results(profile: dict):
    """Render a risk profile from either live pipeline output or demo JSON."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Paragraphs Analyzed", profile.get("total_paragraphs", 0))
    col2.metric("Clauses Flagged", profile.get("flagged_paragraphs", 0))
    col3.metric("Processing Time", f"{profile.get('processing_time_seconds', 0):.1f}s")

    st.plotly_chart(
        render_risk_heatmap(profile.get("risk_scores", {})),
        width="stretch",
    )

    detections = profile.get("detections", [])
    if detections:
        st.subheader("Flagged Clauses")
        for d in detections:
            risk_level = d.get("risk_level", "none")
            label = "High" if risk_level in ["high", "critical"] else "Review"
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
        st.subheader("Unknown Risk Candidates")
        st.caption(
            "These are risks that may be important but are not part of the current 8-category taxonomy."
        )
        for item in unknown:
            with st.expander(item.get("suggested_category", "Unknown Risk")):
                st.markdown(f"**Reason:** {item.get('reason', '')}")
                if item.get("evidence"):
                    st.code(item["evidence"], language=None)


# === Pages ===

if page == "📊 Risk Analysis":
    st.title("📊 Contract Risk Analysis")
    st.markdown("Upload a contract or paste text to analyze for legal risk clauses.")
    
    # Input method
    input_method = st.radio(
        "Input method",
        ["Demo contract", "Paste text", "Upload file"],
        horizontal=True,
    )
    
    contract_text = ""
    
    if input_method == "Demo contract":
        contract_text = load_example_text("sample_contract.txt")
        st.text_area(
            "Demo contract text",
            value=contract_text,
            height=300,
            disabled=True,
        )
        st.caption("Demo mode uses checked-in sample outputs, so it works without API keys or trained checkpoints.")
    elif input_method == "Paste text":
        contract_text = st.text_area(
            "Paste contract text here",
            height=300,
            placeholder="Paste your contract text here...",
        )
    else:
        uploaded_file = st.file_uploader("Upload contract (.txt or .pdf)", type=["txt", "pdf"])
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                contract_text = uploaded_file.read().decode("utf-8")
            else:
                st.warning("PDF parsing coming soon. For now, paste the text directly.")
    
    if contract_text and st.button("🔍 Analyze Contract", type="primary"):
        with st.spinner("Analyzing contract..."):
            if input_method == "Demo contract":
                profile = load_example_json("demo_risk_profile.json")
                render_analysis_results(profile)
                st.stop()

            # Import and run pipeline
            try:
                from src.data_pipeline.chunk_contracts import split_into_paragraphs
                from src.models.hybrid_pipeline import HybridPipeline
                
                paragraphs = split_into_paragraphs(contract_text)
                
                if not paragraphs:
                    st.error("No analyzable paragraphs found. Ensure the text is long enough.")
                else:
                    pipeline = HybridPipeline()
                    profile = pipeline.analyze_contract(paragraphs, contract_id="uploaded")
                    render_analysis_results(profile.to_dict())
                    
            except ImportError as e:
                st.error(f"Pipeline not ready yet: {e}. Run the training scripts first.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    
    elif not contract_text:
        # Show demo with sample data
        st.info("👆 Paste a contract above, or explore the other tabs to see model results.")
        
        # Show sample risk profile
        st.subheader("Sample Output")
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


elif page == "🔬 Model Comparison":
    st.title("🔬 Model Comparison")
    st.markdown("Head-to-head evaluation: DeBERTa vs LLM vs Hybrid pipeline.")
    
    comparison = load_comparison_results()
    detailed = load_detailed_comparison()
    
    if comparison is not None:
        # Summary table
        st.subheader("Overall Performance")
        st.dataframe(comparison, width="stretch", hide_index=True)
        
        # Per-category comparison
        if detailed is not None:
            st.subheader("Per-Category F1 Comparison")
            
            # Reshape for plotting
            f1_cols = [c for c in detailed.columns if c.endswith("_f1")]
            if f1_cols:
                plot_data = detailed[["category"] + f1_cols].melt(
                    id_vars=["category"],
                    var_name="Approach",
                    value_name="F1 Score",
                )
                plot_data["Approach"] = plot_data["Approach"].str.replace("_f1", "").str.upper()
                
                fig = px.bar(
                    plot_data,
                    x="category",
                    y="F1 Score",
                    color="Approach",
                    barmode="group",
                    title="F1 Score by Category and Approach",
                    height=500,
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width="stretch")
            
            # Winner table
            if "best_approach" in detailed.columns:
                st.subheader("Best Approach per Category")
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
        st.subheader("Expected Output (placeholder)")
        placeholder_data = pd.DataFrame({
            "Category": ["Liability", "IP Risk", "Termination", "Indemnification"] * 3,
            "Approach": ["DeBERTa"] * 4 + ["LLM"] * 4 + ["Hybrid"] * 4,
            "F1": [0.72, 0.58, 0.65, 0.70, 0.68, 0.75, 0.71, 0.78, 0.74, 0.73, 0.70, 0.77],
        })
        fig = px.bar(
            placeholder_data, x="Category", y="F1", color="Approach",
            barmode="group", title="Sample Comparison (placeholder data)",
        )
        st.plotly_chart(fig, width="stretch")


elif page == "📈 Dataset Explorer":
    st.title("📈 CUAD Dataset Explorer")
    
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
        st.subheader("Positive Examples per Risk Category")
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
                title="Training Examples per Risk Category",
                labels={"x": "Count", "y": "Category"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")
        
        # Split distribution
        st.subheader("Train/Val/Test Split")
        split_data = df.groupby("split").agg(
            paragraphs=("paragraph_id", "count"),
            contracts=("contract_title", "nunique"),
            positive=("has_any_risk", "sum"),
        ).reset_index()
        st.dataframe(split_data, width="stretch", hide_index=True)
        
        # Paragraph length distribution
        st.subheader("Paragraph Length Distribution")
        fig = px.histogram(
            df, x="paragraph_length", nbins=50,
            title="Distribution of Paragraph Lengths (words)",
            color="has_any_risk",
            labels={"paragraph_length": "Words", "has_any_risk": "Contains Risk Clause"},
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
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Contract Risk Intelligence System · Built with CUAD + DeBERTa + Claude"
    "</div>",
    unsafe_allow_html=True,
)
