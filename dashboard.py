"""
dashboard.py
-------------
Streamlit dashboard for the Korean Financial Sentiment Analyser.

Displays:
  1. Sentiment distribution across companies and time
  2. Sentiment score vs post-announcement price returns (scatter)
  3. Per-company sentiment timeline
  4. Live text analyser (paste any Korean text)
  5. Top positive / negative keyword breakdown

Run:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

st.set_page_config(
    page_title="Korean Financial Sentiment Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

:root {
    --bg:      #F0F2FF;
    --white:   #FFFFFF;
    --ink:     #1B1340;
    --purple:  #6C3FC5;
    --blue:    #2563EB;
    --border:  #C7C2E8;
    --muted:   #6B7280;
    --pos:     #0F766E;
    --neg:     #DC2626;
    --neu:     #D97706;
    --pos-lt:  #CCFBF1;
    --neg-lt:  #FEE2E2;
    --neu-lt:  #FEF9C3;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--ink);
}

.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

.page-header {
    background: linear-gradient(135deg, #1B1340 0%, #6C3FC5 50%, #2563EB 100%);
    border-radius: 12px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
}
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    color: #fff;
    margin: 0 0 0.3rem;
}
.page-sub { font-size: 0.85rem; color: rgba(255,255,255,0.75); margin: 0; }

.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.pill-pos { background: var(--pos-lt); color: var(--pos); }
.pill-neg { background: var(--neg-lt); color: var(--neg); }
.pill-neu { background: var(--neu-lt); color: var(--neu); }

.metric-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-val {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: var(--purple);
    line-height: 1;
}
.metric-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
}

.section-head {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--purple);
    border-bottom: 2px solid #EDE8FA;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}

.insight-box {
    background: #DBEAFE;
    border-left: 4px solid var(--blue);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #1E3A8A;
    margin: 0.8rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, var(--purple), var(--blue)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load processed data, generating demo data if not available."""
    processed_path = "data/processed/sentiment_price_correlation.csv"
    demo_path      = "data/raw/dart_transcripts_demo.csv"

    if os.path.exists(processed_path):
        return pd.read_csv(processed_path)

    # Auto-generate demo data if no processed file exists
    st.info("📊 Loading demo data. Run the full pipeline to use real DART transcripts.")

    sys.path.insert(0, os.path.dirname(__file__))
    from dart_scraper import generate_demo_transcripts
    from sentiment_pipeline import KoreanFinancialSentiment
    from price_correlation import demo_correlation

    raw_df = generate_demo_transcripts()
    kfs    = KoreanFinancialSentiment()
    sent_df = kfs.analyse_dataframe(raw_df)
    final   = demo_correlation(sent_df)

    os.makedirs("data/processed", exist_ok=True)
    final.to_csv(processed_path, index=False)
    return final


df = load_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1 class="page-title">📊 Korean Financial Sentiment Analyser</h1>
    <p class="page-sub">
        DART earnings transcripts · KoBERT sentiment · KOSPI price correlation<br>
        Built by <strong>Hazel I.</strong> · English–Korean Bilingual · github.com/hazelkimhyejin
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    companies = ["All"] + sorted(df["company"].unique().tolist())
    sel_company = st.selectbox("Company", companies)
    sel_labels  = st.multiselect(
        "Sentiment",
        ["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"]
    )
    if st.button("Reset Filters"):
        sel_company = "All"
        sel_labels  = ["positive", "negative", "neutral"]

filtered = df.copy()
if sel_company != "All":
    filtered = filtered[filtered["company"] == sel_company]
if sel_labels:
    filtered = filtered[filtered["sentiment_label"].isin(sel_labels)]

# ── KPI row ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Dataset Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

pos_pct = (filtered["sentiment_label"] == "positive").mean() * 100
neg_pct = (filtered["sentiment_label"] == "negative").mean() * 100
avg_score = filtered["sentiment_score"].mean()
avg_post5 = filtered["post_5d_return"].mean() if "post_5d_return" in filtered.columns else 0

with k1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{len(filtered)}</div><div class="metric-label">Transcripts</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{filtered["company"].nunique()}</div><div class="metric-label">Companies</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#0F766E">{pos_pct:.0f}%</div><div class="metric-label">Positive</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#DC2626">{neg_pct:.0f}%</div><div class="metric-label">Negative</div></div>', unsafe_allow_html=True)
with k5:
    color = "#0F766E" if avg_post5 >= 0 else "#DC2626"
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{avg_post5:+.1f}%</div><div class="metric-label">Avg 5D Return</div></div>', unsafe_allow_html=True)

# ── Main charts ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-head">Sentiment by Company</div>', unsafe_allow_html=True)
    if not filtered.empty:
        summary = (
            filtered.groupby(["company", "sentiment_label"])
            .size()
            .reset_index(name="count")
        )
        pivot = summary.pivot(index="company", columns="sentiment_label", values="count").fillna(0)

        import plotly.graph_objects as go
        COLOR_MAP = {"positive": "#0F766E", "negative": "#DC2626", "neutral": "#D97706"}
        fig = go.Figure()
        for label in ["positive", "neutral", "negative"]:
            if label in pivot.columns:
                fig.add_trace(go.Bar(
                    name=label.capitalize(),
                    x=pivot.index,
                    y=pivot[label],
                    marker_color=COLOR_MAP[label],
                ))
        fig.update_layout(
            barmode="stack",
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", size=12),
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="section-head">Sentiment Score vs 5-Day Return</div>', unsafe_allow_html=True)
    if "post_5d_return" in filtered.columns and not filtered.empty:
        valid = filtered.dropna(subset=["sentiment_score", "post_5d_return"])
        COLOR_MAP2 = {"positive": "#0F766E", "negative": "#DC2626", "neutral": "#D97706"}
        fig2 = go.Figure()
        for label, grp in valid.groupby("sentiment_label"):
            fig2.add_trace(go.Scatter(
                x=grp["sentiment_score"],
                y=grp["post_5d_return"],
                mode="markers",
                name=label.capitalize(),
                marker=dict(color=COLOR_MAP2.get(label, "#888"), size=12, opacity=0.8),
                text=grp["company"],
                hovertemplate="<b>%{text}</b><br>Sentiment: %{x:.2f}<br>5D Return: %{y:.2f}%",
            ))
        # Trend line
        if len(valid) >= 3:
            z = np.polyfit(valid["sentiment_score"], valid["post_5d_return"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid["sentiment_score"].min(), valid["sentiment_score"].max(), 50)
            fig2.add_trace(go.Scatter(
                x=x_line, y=p(x_line),
                mode="lines",
                name="Trend",
                line=dict(color="#6C3FC5", dash="dot", width=2),
            ))
        fig2.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", size=12),
            xaxis_title="Sentiment Score",
            yaxis_title="5-Day Return (%)",
            margin=dict(l=0, r=0, t=30, b=0),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Transcript table ────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Transcript Results</div>', unsafe_allow_html=True)

def label_pill(label):
    cls = {"positive": "pill-pos", "negative": "pill-neg", "neutral": "pill-neu"}.get(label, "")
    return f'<span class="pill {cls}">{label}</span>'

if not filtered.empty:
    display_cols = ["company", "report_nm", "rcept_dt", "sentiment_label", "sentiment_score"]
    if "post_5d_return" in filtered.columns:
        display_cols.append("post_5d_return")

    st.dataframe(
        filtered[display_cols].rename(columns={
            "company": "Company",
            "report_nm": "Report",
            "rcept_dt": "Date",
            "sentiment_label": "Sentiment",
            "sentiment_score": "Score",
            "post_5d_return": "5D Return (%)",
        }).sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# ── Live text analyser ─────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Live Korean Text Analyser</div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
    Paste any Korean financial text below — earnings call excerpt, news headline, or analyst note — and get an instant sentiment score.
</div>
""", unsafe_allow_html=True)

sample_texts = {
    "📈 Positive example": "매출액이 전년 대비 15% 증가하였으며 영업이익도 크게 개선되었습니다. HBM 수요가 급증하고 있어 2024년 전망은 매우 긍정적입니다.",
    "📉 Negative example": "업황 악화로 인해 실적이 감소하였으며 재고 조정 압력이 지속되고 있습니다. 불확실성이 높아 보수적인 전망을 유지합니다.",
    "➡ Neutral example":  "검색 광고 매출은 안정적이나 성장세가 둔화되고 있습니다. 신규 사업 투자를 지속하며 시장 상황을 모니터링합니다.",
}

sample_choice = st.selectbox("Load a sample or type your own:", ["Custom"] + list(sample_texts.keys()))
default_text = sample_texts.get(sample_choice, "")

user_text = st.text_area(
    "Korean financial text:",
    value=default_text,
    height=120,
    placeholder="예: 매출액이 전년 대비 증가하였으며...",
)

if st.button("Analyse →") and user_text.strip():
    from sentiment_pipeline import KoreanFinancialSentiment, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS

    kfs = KoreanFinancialSentiment()
    result = kfs.analyse(user_text)

    label = result["label"]
    score = result["score"]
    pos_kw = result["pos_keywords"]
    neg_kw = result["neg_keywords"]

    label_colors = {"positive": "#0F766E", "negative": "#DC2626", "neutral": "#D97706"}
    label_bgs    = {"positive": "#CCFBF1", "negative": "#FEE2E2", "neutral": "#FEF9C3"}
    emoji        = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(label, "⚪")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{label_colors[label]}">{emoji} {label.upper()}</div><div class="metric-label">Sentiment</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{score:.2f}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#0F766E">{pos_kw}</div><div class="metric-label">Positive KW</div></div>', unsafe_allow_html=True)
    with r4:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#DC2626">{neg_kw}</div><div class="metric-label">Negative KW</div></div>', unsafe_allow_html=True)

    found_pos = [w for w in POSITIVE_KEYWORDS if w in user_text]
    found_neg = [w for w in NEGATIVE_KEYWORDS if w in user_text]

    if found_pos or found_neg:
        st.markdown("**Keywords detected:**")
        kw_html = ""
        for w in found_pos:
            kw_html += f'<span class="pill pill-pos" style="margin:2px">{w}</span> '
        for w in found_neg:
            kw_html += f'<span class="pill pill-neg" style="margin:2px">{w}</span> '
        st.markdown(kw_html, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#6B7280">
    <span>Built by <strong>Hazel I.</strong> · DART Open API · KoBERT · KOSPI via yfinance</span>
    <span>
        <a href="https://github.com/hazelkimhyejin" style="color:#6C3FC5;text-decoration:none">GitHub</a> ·
        <a href="https://linkedin.com/in/hazel-ip-jl" style="color:#2563EB;text-decoration:none">LinkedIn</a>
    </span>
</div>
""", unsafe_allow_html=True)
