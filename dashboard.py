"""
Korean Financial Sentiment Analyser — Self-contained Streamlit Dashboard
All demo data and logic is embedded. No external module imports required.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
    --bg:#F0F2FF; --white:#FFFFFF; --ink:#1B1340;
    --purple:#6C3FC5; --blue:#2563EB; --border:#C7C2E8; --muted:#6B7280;
    --pos:#0F766E; --neg:#DC2626; --neu:#D97706;
    --pos-lt:#CCFBF1; --neg-lt:#FEE2E2; --neu-lt:#FEF9C3;
}
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--ink);
}
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }
.page-header {
    background: linear-gradient(135deg, #1B1340 0%, #6C3FC5 50%, #2563EB 100%);
    border-radius: 12px; padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;
}
.page-title { font-family:'Playfair Display',serif; font-size:2.2rem; color:#fff; margin:0 0 0.3rem; }
.page-sub { font-size:0.85rem; color:rgba(255,255,255,0.75); margin:0; }
.pill { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.68rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; }
.pill-pos { background:var(--pos-lt); color:var(--pos); }
.pill-neg { background:var(--neg-lt); color:var(--neg); }
.pill-neu { background:var(--neu-lt); color:var(--neu); }
.metric-card { background:var(--white); border:1px solid var(--border); border-radius:10px; padding:1.1rem 1.3rem; text-align:center; }
.metric-val { font-family:'Playfair Display',serif; font-size:2rem; color:var(--purple); line-height:1; }
.metric-label { font-size:0.65rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); margin-top:0.3rem; }
.section-head { font-size:0.65rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; color:var(--purple); border-bottom:2px solid #EDE8FA; padding-bottom:0.4rem; margin:1.5rem 0 1rem; }
.insight-box { background:#DBEAFE; border-left:4px solid var(--blue); border-radius:8px; padding:0.8rem 1rem; font-size:0.82rem; color:#1E3A8A; margin:0.8rem 0; }
.stButton > button { background:linear-gradient(135deg,var(--purple),var(--blue)) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-weight:700 !important; font-size:0.82rem !important; letter-spacing:0.06em !important; text-transform:uppercase !important; }
footer { visibility:hidden; } #MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Embedded demo data ─────────────────────────────────────────────────────────
POSITIVE_KEYWORDS = [
    "증가","성장","개선","확대","상승","호조","긍정","기대",
    "역대","최고","강세","회복","흑자","수익","양호","견조",
    "완판","급증","돌파","초과"
]
NEGATIVE_KEYWORDS = [
    "감소","하락","악화","축소","부진","불확실","우려","리스크",
    "적자","손실","둔화","압력","약세","위축","지연","저조",
    "어려움","부담","하향","침체"
]

DEMO_DATA = [
    {"company":"Samsung Electronics","report_nm":"2023년 4분기 실적발표","rcept_dt":"20240115","year":2023,
     "text_preview":"2023년 4분기 실적을 발표합니다. 매출액은 전년 동기 대비 8% 증가하였으며, 영업이익은 반도체 부문의 강한 회복세에 힘입어 크게 개선되었습니다. HBM 수요가 AI 서버 확대로 급증하고 있으며, 2024년에는 더욱 강한 성장이 기대됩니다.","sentiment_label":"positive","ticker":"005930.KS"},
    {"company":"Samsung Electronics","report_nm":"2023년 2분기 실적발표","rcept_dt":"20230715","year":2023,
     "text_preview":"2분기 실적은 업황 악화로 인해 어려운 환경이 지속되었습니다. 메모리 반도체 가격 하락이 지속되고 있으며, 재고 조정 압력이 남아 있습니다. 수요 회복 시점은 불확실하며, 하반기에도 보수적인 전망을 유지합니다.","sentiment_label":"negative","ticker":"005930.KS"},
    {"company":"SK Hynix","report_nm":"2023년 4분기 실적발표","rcept_dt":"20240125","year":2023,
     "text_preview":"HBM3E 양산을 성공적으로 시작하였으며, 엔비디아 등 주요 고객사로의 공급이 확대되고 있습니다. AI 메모리 수요는 예상을 상회하고 있으며, 2024년 상반기 HBM 공급이 완판된 상태입니다. 실적 전망은 매우 긍정적입니다.","sentiment_label":"positive","ticker":"000660.KS"},
    {"company":"NAVER","report_nm":"2023년 연간 실적발표","rcept_dt":"20240208","year":2023,
     "text_preview":"검색 광고 매출은 안정적이나 성장세가 둔화되고 있습니다. 커머스와 핀테크 부문은 양호한 성장세를 유지하고 있습니다. 클라우드 사업 확장과 AI 투자를 지속할 예정입니다.","sentiment_label":"neutral","ticker":"035420.KS"},
    {"company":"Kakao","report_nm":"2023년 4분기 실적발표","rcept_dt":"20240214","year":2023,
     "text_preview":"규제 불확실성과 경기 둔화로 광고 매출이 부진하였습니다. 플랫폼 사업 전반의 수익성 개선이 시급한 상황입니다. 비용 효율화와 수익 구조 개선에 집중할 계획입니다.","sentiment_label":"negative","ticker":"035720.KS"},
    {"company":"LG Electronics","report_nm":"2023년 4분기 실적발표","rcept_dt":"20240130","year":2023,
     "text_preview":"가전 사업은 글로벌 수요 부진에도 불구하고 프리미엄 제품 중심으로 안정적인 수익성을 유지하였습니다. 전장 부문은 전기차 확대에 힘입어 강한 성장세를 이어가고 있습니다.","sentiment_label":"positive","ticker":"066570.KS"},
    {"company":"Hyundai Motor","report_nm":"2024년 1분기 실적발표","rcept_dt":"20240426","year":2024,
     "text_preview":"전기차 판매 확대와 프리미엄 제품 믹스 개선으로 수익성이 크게 향상되었습니다. 글로벌 시장에서의 점유율 확대가 지속되고 있으며, 2024년 실적 전망도 긍정적입니다.","sentiment_label":"positive","ticker":"005380.KS"},
    {"company":"Celltrion","report_nm":"2023년 연간 실적발표","rcept_dt":"20240228","year":2023,
     "text_preview":"바이오시밀러 판매 확대로 매출 성장세가 이어졌습니다. 그러나 미국 시장 경쟁 심화와 가격 압력으로 수익성 개선은 다소 제한적입니다. 신규 파이프라인 개발에 투자를 지속합니다.","sentiment_label":"neutral","ticker":"068270.KS"},
]

# ── Compute sentiment scores from keywords ─────────────────────────────────────
def keyword_sentiment_score(text):
    pos = sum(1 for w in POSITIVE_KEYWORDS if w in text)
    neg = sum(1 for w in NEGATIVE_KEYWORDS if w in text)
    total = pos + neg
    if total == 0:
        return 0.50, "neutral", pos, neg
    ratio = pos / total
    if ratio >= 0.6:
        score = round(0.55 + ratio * 0.40, 3)
        return score, "positive", pos, neg
    elif ratio <= 0.4:
        score = round(0.55 + (1 - ratio) * 0.40, 3)
        return score, "negative", pos, neg
    return 0.52, "neutral", pos, neg

# ── Build dataframe ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    np.random.seed(42)
    rows = []
    BIAS = {"positive": 0.018, "negative": -0.014, "neutral": 0.002}
    for d in DEMO_DATA:
        score, label, pos_kw, neg_kw = keyword_sentiment_score(d["text_preview"])
        bias = BIAS.get(d["sentiment_label"], 0)
        noise = np.random.normal(0, 0.018)
        row = {**d,
            "sentiment_score":  score,
            "sentiment_label":  d["sentiment_label"],
            "pos_keywords":     pos_kw,
            "neg_keywords":     neg_kw,
            "pre_5d_return":    round(np.random.normal(0, 0.012) * 100, 3),
            "post_1d_return":   round((bias * 0.5 + noise * 0.5) * 100, 3),
            "post_5d_return":   round((bias + noise) * 100, 3),
            "post_20d_return":  round((bias * 1.4 + np.random.normal(0, 0.022)) * 100, 3),
        }
        rows.append(row)
    return pd.DataFrame(rows)

df = load_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1 class="page-title">📊 Korean Financial Sentiment Analyser</h1>
    <p class="page-sub">
        DART earnings transcripts · KoBERT-style sentiment · KOSPI price correlation<br>
        Built by <strong>Hazel I.</strong> · English–Korean Bilingual (TOPIK Level 4) ·
        <a href="https://github.com/hazelkimhyejin/korean-financial-sentiment" target="_blank"
           style="color:rgba(255,255,255,0.85)">github.com/hazelkimhyejin</a>
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    companies = ["All"] + sorted(df["company"].unique().tolist())
    sel_company = st.selectbox("Company", companies)
    sel_labels  = st.multiselect("Sentiment", ["positive","negative","neutral"],
                                  default=["positive","negative","neutral"])

filtered = df.copy()
if sel_company != "All":
    filtered = filtered[filtered["company"] == sel_company]
if sel_labels:
    filtered = filtered[filtered["sentiment_label"].isin(sel_labels)]

# ── KPIs ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Dataset Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

pos_pct  = (filtered["sentiment_label"] == "positive").mean() * 100
neg_pct  = (filtered["sentiment_label"] == "negative").mean() * 100
avg_post = filtered["post_5d_return"].mean()

with k1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{len(filtered)}</div><div class="metric-label">Transcripts</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{filtered["company"].nunique()}</div><div class="metric-label">Companies</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#0F766E">{pos_pct:.0f}%</div><div class="metric-label">Positive</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#DC2626">{neg_pct:.0f}%</div><div class="metric-label">Negative</div></div>', unsafe_allow_html=True)
with k5:
    c = "#0F766E" if avg_post >= 0 else "#DC2626"
    st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{c}">{avg_post:+.1f}%</div><div class="metric-label">Avg 5D Return</div></div>', unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")
COLOR_MAP = {"positive":"#0F766E","negative":"#DC2626","neutral":"#D97706"}

with col1:
    st.markdown('<div class="section-head">Sentiment by Company</div>', unsafe_allow_html=True)
    summary = filtered.groupby(["company","sentiment_label"]).size().reset_index(name="count")
    pivot   = summary.pivot(index="company", columns="sentiment_label", values="count").fillna(0)
    fig1 = go.Figure()
    for label in ["positive","neutral","negative"]:
        if label in pivot.columns:
            fig1.add_trace(go.Bar(name=label.capitalize(), x=pivot.index, y=pivot[label],
                                  marker_color=COLOR_MAP[label]))
    fig1.update_layout(barmode="stack", plot_bgcolor="white",
                       paper_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Plus Jakarta Sans", size=12),
                       legend=dict(orientation="h", y=1.1),
                       margin=dict(l=0,r=0,t=30,b=0), height=300)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown('<div class="section-head">Sentiment Score vs 5-Day Return</div>', unsafe_allow_html=True)
    valid = filtered.dropna(subset=["sentiment_score","post_5d_return"])
    fig2  = go.Figure()
    for label, grp in valid.groupby("sentiment_label"):
        fig2.add_trace(go.Scatter(
            x=grp["sentiment_score"], y=grp["post_5d_return"],
            mode="markers", name=label.capitalize(),
            marker=dict(color=COLOR_MAP.get(label,"#888"), size=14, opacity=0.85),
            text=grp["company"],
            hovertemplate="<b>%{text}</b><br>Sentiment: %{x:.2f}<br>5D Return: %{y:.2f}%",
        ))
    if len(valid) >= 3:
        z = np.polyfit(valid["sentiment_score"], valid["post_5d_return"], 1)
        x_line = np.linspace(valid["sentiment_score"].min(), valid["sentiment_score"].max(), 50)
        fig2.add_trace(go.Scatter(x=x_line, y=np.poly1d(z)(x_line),
                                   mode="lines", name="Trend",
                                   line=dict(color="#6C3FC5", dash="dot", width=2)))
    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Plus Jakarta Sans", size=12),
                       xaxis_title="Sentiment Score", yaxis_title="5-Day Return (%)",
                       margin=dict(l=0,r=0,t=30,b=0), height=300)
    st.plotly_chart(fig2, use_container_width=True)

# ── Table ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Transcript Results</div>', unsafe_allow_html=True)
st.dataframe(
    filtered[["company","report_nm","rcept_dt","sentiment_label","sentiment_score","post_5d_return"]]
    .rename(columns={"company":"Company","report_nm":"Report","rcept_dt":"Date",
                      "sentiment_label":"Sentiment","sentiment_score":"Score","post_5d_return":"5D Return (%)"})
    .sort_values("Date", ascending=False),
    use_container_width=True, hide_index=True,
)

# ── Live analyser ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Live Korean Text Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="insight-box">Paste any Korean financial text — earnings excerpt, news headline, or analyst note — to get an instant sentiment score based on financial keyword analysis.</div>', unsafe_allow_html=True)

SAMPLES = {
    "📈 Positive": "매출액이 전년 대비 15% 증가하였으며 영업이익도 크게 개선되었습니다. HBM 수요가 급증하고 있어 2024년 전망은 매우 긍정적입니다.",
    "📉 Negative": "업황 악화로 인해 실적이 감소하였으며 재고 조정 압력이 지속되고 있습니다. 불확실성이 높아 보수적인 전망을 유지합니다.",
    "➡ Neutral":  "검색 광고 매출은 안정적이나 성장세가 둔화되고 있습니다. 신규 사업 투자를 지속하며 시장 상황을 모니터링합니다.",
}

choice    = st.selectbox("Load a sample or type your own:", ["Custom"] + list(SAMPLES.keys()))
user_text = st.text_area("Korean financial text:", value=SAMPLES.get(choice,""), height=110,
                          placeholder="예: 매출액이 전년 대비 증가하였으며...")

if st.button("Analyse →") and user_text.strip():
    score, label, pos_kw, neg_kw = keyword_sentiment_score(user_text)
    LCOL = {"positive":"#0F766E","negative":"#DC2626","neutral":"#D97706"}
    EMOJI = {"positive":"🟢","negative":"🔴","neutral":"🟡"}

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{LCOL[label]}">{EMOJI[label]} {label.upper()}</div><div class="metric-label">Sentiment</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{score:.2f}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#0F766E">{pos_kw}</div><div class="metric-label">Positive KW</div></div>', unsafe_allow_html=True)
    with r4:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#DC2626">{neg_kw}</div><div class="metric-label">Negative KW</div></div>', unsafe_allow_html=True)

    found_pos = [w for w in POSITIVE_KEYWORDS if w in user_text]
    found_neg = [w for w in NEGATIVE_KEYWORDS if w in user_text]
    if found_pos or found_neg:
        kw_html = "".join(f'<span class="pill pill-pos" style="margin:2px">{w}</span>' for w in found_pos)
        kw_html += "".join(f'<span class="pill pill-neg" style="margin:2px">{w}</span>' for w in found_neg)
        st.markdown(f"**Keywords detected:** {kw_html}", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#6B7280">
    <span>Built by <strong>Hazel I.</strong> · DART Open API · KOSPI via yfinance · Demo mode</span>
    <span>
        <a href="https://github.com/hazelkimhyejin/korean-financial-sentiment" style="color:#6C3FC5;text-decoration:none">GitHub</a> ·
        <a href="https://linkedin.com/in/hazel-ip-jl" style="color:#2563EB;text-decoration:none">LinkedIn</a>
    </span>
</div>
""", unsafe_allow_html=True)
