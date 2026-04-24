"""
app.py — Intelligent App Testing System
Streamlit-based web application
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
import warnings
import time

warnings.filterwarnings("ignore")

# Local modules
from crawler import crawl_website, generate_issues_from_crawl, validate_url
from analyzer import (
    build_issues_df,
    compute_health_score,
    compute_module_risk,
    train_prediction_model,
    predict_module_risk,
    run_nlp_analysis,
    generate_recommendations,
)

# ─── App Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent App Testing System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "v1.0.0"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff, #7B2FBE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-label {
    color: #888;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.risk-high { background: #3d0000; border-left: 4px solid #ff4444; padding: 0.5rem 1rem; border-radius: 4px; }
.risk-medium { background: #3d2600; border-left: 4px solid #ffaa00; padding: 0.5rem 1rem; border-radius: 4px; }
.risk-low { background: #003d00; border-left: 4px solid #44ff44; padding: 0.5rem 1rem; border-radius: 4px; }
.info-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.85rem;
    color: #8b949e;
}
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #00d4ff;
    border-bottom: 1px solid #2a2a4a;
    padding-bottom: 0.3rem;
    margin: 1rem 0 0.8rem 0;
}
.warning-box {
    background: #2d2000;
    border: 1px solid #ffaa00;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    color: #ffcc44;
    font-size: 0.85rem;
}
.found-tag { background: #003d26; color: #44ffaa; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
.assumed-tag { background: #1a1a2e; color: #8888ff; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ─── Caching ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def cached_crawl(url: str, respect_robots: bool):
    return crawl_website(url, respect_robots=respect_robots)


@st.cache_data(show_spinner=False)
def cached_issues(url: str, respect_robots: bool):
    crawl = cached_crawl(url, respect_robots)
    if crawl.get("error"):
        return crawl, []
    return crawl, generate_issues_from_crawl(crawl)


# ─── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.3rem; color: #00d4ff;'>🔬 AppTester</div>
        <div style='color: #555; font-size: 0.75rem;'>{APP_VERSION}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "📍 Navigate",
        [
            "🌐 Website Overview",
            "📊 Exploratory Analysis",
            "⚠️ Risk Scoring",
            "🤖 Prediction Model",
            "💬 NLP Issue Analysis",
            "✅ Fix Validation & Recs",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**⚙️ Crawl Settings**")
    respect_robots = st.checkbox("Respect robots.txt", value=True)
    st.caption("Uncheck to crawl sites that restrict bots.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color: #555;'>
    <b>Methodology</b><br>
    Risk = (Issues × 0.4) + (Severity × 0.35) + (Broken Links × 0.25)<br><br>
    ML: Random Forest (100 trees)<br>
    NLP: TF-IDF + KMeans<br>
    Max crawl: 50 pages
    </div>
    """, unsafe_allow_html=True)


# ─── Main header ──────────────────────────────────────────────

st.markdown('<div class="main-title">🔬 Intelligent App Testing System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste any URL · Auto-crawl · Full analysis · AI-powered insights</div>', unsafe_allow_html=True)

# ─── URL Input ────────────────────────────────────────────────

col_url, col_btn = st.columns([5, 1])
with col_url:
    input_url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        label_visibility="collapsed",
    )
with col_btn:
    run_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")

st.markdown("---")

# ─── Session state ────────────────────────────────────────────

if "crawl_data" not in st.session_state:
    st.session_state.crawl_data = None
    st.session_state.issues = []
    st.session_state.df = pd.DataFrame()
    st.session_state.risk_df = pd.DataFrame()
    st.session_state.health = {}
    st.session_state.model_data = {}
    st.session_state.nlp_data = {}
    st.session_state.recs = {}
    st.session_state.analyzed_url = ""

# ─── Run Crawl ────────────────────────────────────────────────

if run_btn and input_url.strip():
    valid, clean_url, err = validate_url(input_url.strip())
    if not valid:
        st.error(f"❌ {err}")
    else:
        st.session_state.analyzed_url = clean_url
        progress_bar = st.progress(0, text="Initializing crawler...")
        status_text = st.empty()

        def update_progress(current, total, message):
            pct = min(int((current / max(total, 1)) * 100), 100)
            progress_bar.progress(pct, text=f"🕷️ {message}")

        with st.spinner(""):
            crawl_data = crawl_website(clean_url, respect_robots=respect_robots,
                                        progress_callback=update_progress)

        if crawl_data.get("error"):
            progress_bar.empty()
            err_map = {
                "robots_blocked": "🚫 Crawl blocked by robots.txt. Uncheck 'Respect robots.txt' in sidebar.",
                "timeout": "⏱️ The site timed out. It may be slow or blocking automated requests.",
                "ssl_error": "🔒 SSL certificate error. Try adding http:// instead of https://.",
                "connection_error": "❌ Could not connect to the site. Check the URL and try again.",
            }
            msg = err_map.get(crawl_data["error"], crawl_data.get("message", "Unknown error."))
            st.error(msg)
        else:
            progress_bar.progress(90, text="Generating issue dataset...")
            issues = generate_issues_from_crawl(crawl_data)
            df = build_issues_df(issues)

            progress_bar.progress(95, text="Running analysis pipeline...")
            health = compute_health_score(crawl_data, df)
            risk_df = compute_module_risk(df, crawl_data) if not df.empty else pd.DataFrame()
            model_data = train_prediction_model(df) if not df.empty else {"error": "No data"}
            nlp_data = run_nlp_analysis(df) if not df.empty else {"error": "No data"}
            recs = generate_recommendations(df, crawl_data, risk_df)

            st.session_state.crawl_data = crawl_data
            st.session_state.issues = issues
            st.session_state.df = df
            st.session_state.risk_df = risk_df
            st.session_state.health = health
            st.session_state.model_data = model_data
            st.session_state.nlp_data = nlp_data
            st.session_state.recs = recs

            progress_bar.progress(100, text="✅ Analysis complete!")
            time.sleep(0.5)
            progress_bar.empty()
            st.success(f"✅ Crawled {crawl_data['total_pages']} pages in {crawl_data['crawl_time']}s — {len(issues)} issues found.")

# ─── Guard: no data yet ───────────────────────────────────────

if st.session_state.crawl_data is None:
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color: #444;'>
        <div style='font-size: 3rem;'>🌐</div>
        <div style='font-size: 1.2rem; margin-top: 1rem;'>Enter a website URL above and click <b>Analyze</b></div>
        <div style='font-size: 0.9rem; margin-top: 0.5rem; color: #333;'>
            The system will crawl the site, extract issues, and run full AI-powered analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load from session
crawl_data = st.session_state.crawl_data
issues = st.session_state.issues
df = st.session_state.df
risk_df = st.session_state.risk_df
health = st.session_state.health
model_data = st.session_state.model_data
nlp_data = st.session_state.nlp_data
recs = st.session_state.recs
pages = crawl_data.get("pages", [])


# ══════════════════════════════════════════════════════════════
# PAGE 1 — WEBSITE OVERVIEW
# ══════════════════════════════════════════════════════════════

if page == "🌐 Website Overview":
    st.markdown("## 🌐 Website Overview")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **Crawl Process:** The system uses `requests` + `BeautifulSoup` to visit up to 50 pages 
        starting from the submitted URL. For each page it extracts: title, meta tags, headings, 
        forms, images, links, load time, and HTTP status code. Issues are derived *only* from 
        what was found — nothing is fabricated.

        - 🟢 **Found**: Data directly extracted from crawl  
        - 🔵 **Derived**: Calculated from found data using documented formulas
        """)

    # Health score hero
    score = health.get("score", 0)
    color = health.get("color", "gray")
    grade = health.get("grade", "?")

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("Total Pages", crawl_data.get("total_pages", 0), "🌐"),
        ("Issues Found", len(issues), "⚠️"),
        ("Modules", len(crawl_data.get("modules", [])), "📦"),
        ("Broken Links", len(crawl_data.get("broken_links", [])), "🔗"),
        ("Crawl Time", f"{crawl_data.get('crawl_time', 0)}s", "⏱️"),
    ]
    for col, (label, val, icon) in zip([c1, c2, c3, c4, c5], metrics):
        with col:
            st.metric(f"{icon} {label}", val)

    st.markdown("---")

    col_score, col_breakdown = st.columns([1, 2])

    with col_score:
        st.markdown("### Overall Health Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Grade: {grade}", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35], "color": "#3d0000"},
                    {"range": [35, 65], "color": "#3d2600"},
                    {"range": [65, 100], "color": "#003d00"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        ))
        fig.update_layout(height=280, margin=dict(t=30, b=0, l=20, r=20),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<div class="info-box"><b>Formula (Derived):</b> Weighted composite of Issues (25%), Broken Links (20%), Performance (20%), SEO (15%), Accessibility (10%), Meta Tags (5%), Mobile (5%)</div>', unsafe_allow_html=True)

    with col_breakdown:
        st.markdown("### Score Breakdown by Category")
        breakdown = health.get("breakdown", {})
        if breakdown:
            cats = list(breakdown.keys())
            vals = list(breakdown.values())
            colors_map = ["#44ff44" if v >= 70 else "#ffaa00" if v >= 40 else "#ff4444" for v in vals]
            fig2 = go.Figure(go.Bar(
                x=vals, y=cats, orientation="h",
                marker_color=colors_map,
                text=[f"{v:.0f}" for v in vals],
                textposition="inside",
            ))
            fig2.update_layout(
                xaxis=dict(range=[0, 100], title="Score"),
                yaxis_title="",
                title="Category Scores (0–100)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=280,
                margin=dict(t=40, b=20),
            )
            fig2.update_xaxes(gridcolor="#2a2a4a")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Website structure map
    st.markdown("### 🗺️ Website Structure Map")
    st.caption("🟢 Found: extracted from crawl results")

    if pages:
        module_groups = {}
        for p in pages:
            m = p.get("module", "Unknown")
            if m not in module_groups:
                module_groups[m] = []
            module_groups[m].append(p)

        cols = st.columns(min(len(module_groups), 4))
        for i, (module, mpages) in enumerate(module_groups.items()):
            col = cols[i % len(cols)]
            with col:
                avg_lt = np.mean([p.get("load_time", 0) for p in mpages if p.get("load_time")]) or 0
                errors = sum(1 for p in mpages if p.get("status_code") in (404, 500, 502, 503) or p.get("error"))
                st.markdown(f"""
                <div style='background:#1a1a2e; border:1px solid #2a2a4a; border-radius:8px; padding:0.8rem; margin-bottom:0.5rem;'>
                    <b style='color:#00d4ff;'>{module}</b><br>
                    <span style='color:#888; font-size:0.8rem;'>{len(mpages)} page(s) · {avg_lt:.2f}s avg · {errors} error(s)</span>
                </div>
                """, unsafe_allow_html=True)
                for p in mpages[:3]:
                    sc = p.get("status_code", "?")
                    sc_color = "#44ff44" if sc == 200 else "#ff4444"
                    url_short = p.get("url", "")[-40:]
                    st.markdown(f"<div style='font-size:0.72rem; color:#555; padding-left:0.5rem;'>• <span style='color:{sc_color};'>[{sc}]</span> ...{url_short}</div>", unsafe_allow_html=True)
                if len(mpages) > 3:
                    st.markdown(f"<div style='font-size:0.7rem; color:#444; padding-left:0.5rem;'>+ {len(mpages)-3} more...</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Crawl Summary")
    summary_data = {
        "Metric": ["Base URL", "Pages Crawled", "Modules Detected", "Broken Links", "External Links Checked",
                   "Total Issues Generated", "Crawl Duration", "Robots.txt Respected"],
        "Value": [
            crawl_data.get("url", ""),
            crawl_data.get("total_pages", 0),
            len(crawl_data.get("modules", [])),
            len(crawl_data.get("broken_links", [])),
            crawl_data.get("external_links_checked", 0),
            len(issues),
            f"{crawl_data.get('crawl_time', 0)}s",
            "Yes" if respect_robots else "No",
        ],
        "Source": ["Input", "Found", "Derived", "Found", "Found", "Derived", "Found", "Setting"],
    }
    sum_df = pd.DataFrame(summary_data)
    st.dataframe(sum_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════

elif page == "📊 Exploratory Analysis":
    st.markdown("## 📊 Exploratory Data Analysis")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        All charts are built from the issue dataset generated by the crawler. 
        Each issue is traceable to a specific finding (broken link, missing tag, slow load time, etc.).
        No data is fabricated — sample size is displayed on every chart.
        """)

    if df.empty:
        st.warning("No issues found. The site may be clean or the crawl was limited.")
        st.stop()

    st.caption(f"📊 Analysis based on **{len(df)} issues** across **{df['module'].nunique()} modules** — 🟢 Found from crawl data")

    # Row 1: Issues by module + Severity pie
    col1, col2 = st.columns(2)
    with col1:
        module_counts = df.groupby("module").size().reset_index(name="count").sort_values("count", ascending=False)
        fig = px.bar(module_counts, x="module", y="count",
                     color="count", color_continuous_scale="Blues",
                     title=f"Issues by Module (n={len(df)})",
                     labels={"module": "Module", "count": "Issue Count"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", xaxis_tickangle=-30, showlegend=False)
        fig.update_traces(texttemplate="%{y}", textposition="outside")
        fig.add_annotation(text="Source: crawl findings", x=0, y=-0.25, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=9, color="#555"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sev_counts = df["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig2 = px.pie(sev_counts, names="severity", values="count",
                      color="severity",
                      color_discrete_map={"High": "#ff4444", "Medium": "#ffaa00", "Low": "#44ff44"},
                      title=f"Severity Distribution (n={len(df)})",
                      hole=0.4)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        fig2.add_annotation(text="Source: crawl findings", x=0, y=-0.15, xref="paper", yref="paper",
                            showarrow=False, font=dict(size=9, color="#555"))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Issue type breakdown + Status
    col3, col4 = st.columns(2)
    with col3:
        type_counts = df["issue_type"].value_counts().head(12).reset_index()
        type_counts.columns = ["issue_type", "count"]
        fig3 = px.bar(type_counts, x="count", y="issue_type", orientation="h",
                      color="count", color_continuous_scale="Reds",
                      title="Top Issue Types",
                      labels={"issue_type": "", "count": "Count"})
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", height=350, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        color_map = {"Open": "#ff4444", "Fixed": "#44ff44", "Reopened": "#ffaa00"}
        fig4 = px.bar(status_counts, x="status", y="count",
                      color="status",
                      color_discrete_map=color_map,
                      title="Issue Status Distribution")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Page load times
    st.markdown("### ⏱️ Page Load Times")
    load_data = [
        {"url": p.get("url", "")[-50:], "load_time": p.get("load_time", 0), "module": p.get("module", "?")}
        for p in pages if p.get("load_time") is not None and p.get("load_time", 0) > 0
    ]
    if load_data:
        lt_df = pd.DataFrame(load_data).sort_values("load_time", ascending=False).head(20)
        colors = ["#ff4444" if t > 3 else "#ffaa00" if t > 1.5 else "#44ff44" for t in lt_df["load_time"]]
        fig5 = go.Figure(go.Bar(
            x=lt_df["load_time"], y=lt_df["url"],
            orientation="h",
            marker_color=colors,
            text=[f"{t:.2f}s" for t in lt_df["load_time"]],
            textposition="outside",
        ))
        fig5.add_vline(x=3.0, line_dash="dash", line_color="red", annotation_text="3s threshold")
        fig5.add_vline(x=1.5, line_dash="dot", line_color="orange", annotation_text="1.5s")
        fig5.update_layout(
            title=f"Page Load Times — Top 20 Slowest (n={len(load_data)} total)",
            xaxis_title="Load Time (seconds)", yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=max(250, len(lt_df) * 22),
        )
        st.plotly_chart(fig5, use_container_width=True)
        avg_load = np.mean([d["load_time"] for d in load_data])
        st.caption(f"Average load time: **{avg_load:.2f}s** · Red = >3s (High) · Orange = >1.5s (Medium) · Green = ≤1.5s (Good)")
    else:
        st.info("No load time data available.")

    # Row 4: Most problematic pages
    st.markdown("### 🔥 Most Problematic Pages")
    page_issue_counts = df.groupby("page_url").agg(
        total_issues=("id", "count"),
        high_issues=("is_critical", "sum"),
        open_issues=("is_open", "sum"),
    ).reset_index().sort_values("total_issues", ascending=False).head(10)
    if not page_issue_counts.empty:
        page_issue_counts.columns = ["Page URL", "Total Issues", "High Severity", "Open Issues"]
        st.dataframe(page_issue_counts, use_container_width=True, hide_index=True)

    # Raw data table
    with st.expander("📋 View Raw Issue Dataset"):
        display_df = df[["module", "issue_type", "severity", "status", "occurrences",
                          "fix_time_hours", "description", "source"]].copy()
        display_df.columns = ["Module", "Issue Type", "Severity", "Status", "Occurrences",
                               "Fix Time (h)", "Description", "Data Source"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption("🟢 All issues derived from actual crawl findings · No fabricated data")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — RISK SCORING
# ══════════════════════════════════════════════════════════════

elif page == "⚠️ Risk Scoring":
    st.markdown("## ⚠️ Risk Scoring System")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **Risk Score Formula (Derived — transparent calculation):**
        ```
        Raw Risk = (Issue Count × 0.4) + (Avg Severity Weight × 10 × 0.35) + (Broken Link Rate × 100 × 0.25)
        Final Score = Normalized to 0–100
        ```
        - **Issue Count**: Total issues found in the module (Found)
        - **Severity Weight**: Low=1, Medium=2, High=3 (Found)  
        - **Broken Link Rate**: Broken links in module / pages in module (Found)
        - Modules with fewer than 5 issues are flagged with a sample size warning ⚠️
        """)

    st.markdown("""
    <div class="info-box">
    <b>Formula:</b> Risk Score = <code>(Issue Count × 0.4) + (Severity Weight × 0.35) + (Broken Link Rate × 0.25)</code> → Normalized 0–100
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    if risk_df.empty:
        st.warning("No risk data available. Run a crawl first.")
        st.stop()

    # Color-coded risk table
    st.markdown("### 📊 Module Risk Table")
    st.caption(f"🔵 Derived from crawl data · n={len(df)} issues across {len(risk_df)} modules")

    def highlight_risk(row):
        color_map = {"High": "background-color: #3d0000; color: #ff4444",
                     "Medium": "background-color: #3d2600; color: #ffaa00",
                     "Low": "background-color: #003d00; color: #44ff44"}
        style = color_map.get(row["Risk Level"], "")
        return [style if col == "Risk Level" else "" for col in row.index]

    display_risk = risk_df[["Module", "Issue Count", "Open Issues", "High Severity",
                              "Broken Link Rate", "Risk Score (0-100)", "Risk Level", "Small Sample"]].copy()
    display_risk["Small Sample"] = display_risk["Small Sample"].apply(lambda x: "⚠️ n<5" if x else "✅ OK")
    display_risk["Broken Link Rate"] = display_risk["Broken Link Rate"].apply(lambda x: f"{x:.1%}")
    display_risk.columns = ["Module", "Issues", "Open", "High Sev", "Broken Link Rate",
                             "Risk Score", "Risk Level", "Sample Size"]

    st.dataframe(
        display_risk.style.apply(highlight_risk, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Risk bar chart
    st.markdown("### 📈 Module Risk Scores (0–100)")
    color_list = [
        "#ff4444" if lvl == "High" else "#ffaa00" if lvl == "Medium" else "#44ff44"
        for lvl in risk_df["Risk Level"]
    ]
    fig = go.Figure(go.Bar(
        x=risk_df["Module"],
        y=risk_df["Risk Score (0-100)"],
        marker_color=color_list,
        text=[f"{v:.0f}" for v in risk_df["Risk Score (0-100)"]],
        textposition="outside",
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk (70)")
    fig.add_hline(y=40, line_dash="dot", line_color="orange", annotation_text="Medium Risk (40)")
    fig.update_layout(
        xaxis_title="Module", yaxis_title="Risk Score (0–100)",
        xaxis_tickangle=-30,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        yaxis=dict(range=[0, 110]),
    )
    fig.add_annotation(text="Source: Derived from crawl findings using documented formula",
                       x=0, y=-0.25, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=9, color="#555"))
    st.plotly_chart(fig, use_container_width=True)

    # Risk breakdown radar
    if len(risk_df) >= 3:
        st.markdown("### 🎯 Risk Component Breakdown")
        top5 = risk_df.head(5)
        radar_data = []
        for _, row in top5.iterrows():
            radar_data.append({
                "Module": row["Module"],
                "Issue Count Score": min(row["Issue Count"] * 5, 100),
                "Severity Score": row["Avg Severity Weight"] / 3 * 100,
                "Broken Link Score": min(row["Broken Link Rate"] * 100 * 2, 100),
            })
        rdr_df = pd.DataFrame(radar_data)
        categories = ["Issue Count Score", "Severity Score", "Broken Link Score"]
        fig_radar = go.Figure()
        for _, row in rdr_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[c] for c in categories] + [row[categories[0]]],
                theta=categories + [categories[0]],
                name=row["Module"],
                fill="toself", opacity=0.5,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Risk Components — Top 5 Modules",
            paper_bgcolor="rgba(0,0,0,0)", font_color="white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Small sample warnings
    small = risk_df[risk_df["Small Sample"] == True]
    if not small.empty:
        st.markdown("---")
        st.markdown('<div class="warning-box">⚠️ <b>Sample Size Warning:</b> The following modules have fewer than 5 issues — risk scores may be less reliable: ' + ', '.join(small["Module"].tolist()) + '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION MODEL
# ══════════════════════════════════════════════════════════════

elif page == "🤖 Prediction Model":
    st.markdown("## 🤖 Prediction Model")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **Model:** Random Forest Classifier (100 decision trees, max depth 5, balanced class weights)  
        **Target:** Predict whether an issue is **Critical** (High severity) or **Not Critical**  
        **Features:** Module (encoded), Issue Type (encoded), Occurrences, Fix Time (hours), Severity Weight  
        **Training:** 70% train / 30% test split, stratified, random seed=42 for reproducibility  
        **Confidence:** Probability of predicted class from `predict_proba()` — shown on every prediction  
        **Note:** Model is trained on crawl-derived data. Results reflect site-specific patterns.
        """)

    if model_data.get("error"):
        st.warning(f"⚠️ {model_data['error']}")
        st.stop()

    # Metrics row
    acc = model_data.get("accuracy", 0)
    report = model_data.get("classification_report", {})
    n_test = len(model_data.get("y_test", []))
    conf_scores = model_data.get("confidence_scores", [])
    avg_conf = np.mean(conf_scores) if conf_scores else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy", f"{acc*100:.1f}%")
    c2.metric("Test Set Size", f"n={n_test}")
    c3.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
    c4.metric("Algorithm", "Random Forest")

    st.markdown("---")
    col_cm, col_imp = st.columns(2)

    with col_cm:
        st.markdown("### Confusion Matrix")
        cm = model_data.get("confusion_matrix", [[0, 0], [0, 0]])
        labels = model_data.get("label_names", ["Not Critical", "Critical"])
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    linewidths=0.5)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title("Confusion Matrix — Test Set", fontsize=12)
        fig_cm.patch.set_alpha(0)
        ax.set_facecolor("#0d1117")
        plt.tight_layout()
        st.pyplot(fig_cm)
        st.caption(f"Source: Random Forest predictions on {n_test} test samples")

    with col_imp:
        st.markdown("### Feature Importance")
        importance = model_data.get("feature_importance", {})
        feat_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
        feat_df = feat_df.sort_values("Importance", ascending=True)
        feat_labels = {
            "module_enc": "Module", "type_enc": "Issue Type",
            "occurrences": "Occurrences", "fix_time_hours": "Fix Time",
            "severity_weight": "Severity",
        }
        feat_df["Feature"] = feat_df["Feature"].map(feat_labels)
        fig_imp = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues",
                         title="Feature Importance (Random Forest)",
                         labels={"Feature": "", "Importance": "Gini Importance"})
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="white", showlegend=False, height=300)
        st.plotly_chart(fig_imp, use_container_width=True)

    # Classification report
    st.markdown("### Classification Report")
    report_rows = []
    for label, vals in report.items():
        if isinstance(vals, dict):
            report_rows.append({
                "Class": label,
                "Precision": f"{vals.get('precision', 0):.3f}",
                "Recall": f"{vals.get('recall', 0):.3f}",
                "F1-Score": f"{vals.get('f1-score', 0):.3f}",
                "Support": int(vals.get("support", 0)),
            })
    if report_rows:
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🎯 Live Prediction")
    st.caption("Select a module/issue to get a real-time criticality prediction with confidence score.")

    modules_avail = sorted(df["module"].unique().tolist()) if not df.empty else ["Homepage"]
    types_avail = sorted(df["issue_type"].unique().tolist()) if not df.empty else ["Broken Link"]

    p1, p2 = st.columns(2)
    with p1:
        sel_module = st.selectbox("Module", modules_avail)
        sel_type = st.selectbox("Issue Type", types_avail)
    with p2:
        sel_occ = st.slider("Occurrences", 1, 50, 1)
        sel_fix = st.slider("Estimated Fix Time (hours)", 0.25, 16.0, 2.0, step=0.25)
        sel_sev = st.selectbox("Severity", ["Low", "Medium", "High"])

    if st.button("🔮 Predict Criticality", type="primary"):
        pred_result = predict_module_risk(
            model_data, sel_module, sel_type, sel_occ, sel_fix, sel_sev
        )
        if pred_result.get("error"):
            st.error(pred_result["error"])
        else:
            prediction = pred_result["prediction"]
            confidence = pred_result["confidence"]
            prob_crit = pred_result.get("proba_critical", 0)
            prob_not = pred_result.get("proba_not_critical", 100)

            pred_color = "#ff4444" if prediction == "Critical" else "#44ff44"
            st.markdown(f"""
            <div style='background:#1a1a2e; border: 2px solid {pred_color}; border-radius:10px; padding:1.5rem; margin-top:1rem;'>
                <div style='font-size:1.5rem; font-weight:700; color:{pred_color};'>
                    {'🚨' if prediction == 'Critical' else '✅'} {prediction}
                </div>
                <div style='color:#888; margin-top:0.3rem;'>
                    Confidence: <b style='color:white;'>{confidence}%</b> &nbsp;|&nbsp;
                    P(Critical): <b style='color:#ff4444;'>{prob_crit:.1f}%</b> &nbsp;|&nbsp;
                    P(Not Critical): <b style='color:#44ff44;'>{prob_not:.1f}%</b>
                </div>
                <div style='color:#555; font-size:0.8rem; margin-top:0.5rem;'>
                    Model: Random Forest · Seed: 42 · Based on {len(df)} training samples
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — NLP ISSUE ANALYSIS
# ══════════════════════════════════════════════════════════════

elif page == "💬 NLP Issue Analysis":
    st.markdown("## 💬 NLP Issue Analysis")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **TF-IDF (Term Frequency–Inverse Document Frequency):** Identifies the most statistically 
        significant keywords across all issue descriptions. Words appearing in many descriptions get 
        lower weight; rare but important terms score higher.

        **KMeans Clustering:** Groups similar issues together based on their TF-IDF vectors. 
        Helps identify categories of problems without manual tagging.

        **WordCloud:** Visual representation of keyword frequency — larger = more prominent.

        All text is derived from actual crawl findings — issue descriptions are generated from 
        real data (broken links, missing tags, slow pages, etc.).
        """)

    if nlp_data.get("error"):
        st.warning(f"⚠️ {nlp_data['error']}")
        st.stop()

    col_wc, col_kw = st.columns([1, 1])

    with col_wc:
        st.markdown("### ☁️ WordCloud — Top Issue Keywords")
        word_freq = nlp_data.get("word_freq", {})
        if word_freq:
            try:
                wc = WordCloud(
                    width=600, height=320,
                    background_color="#0d1117",
                    colormap="cool",
                    max_words=50,
                    prefer_horizontal=0.7,
                ).generate_from_frequencies({k: max(v * 100, 1) for k, v in word_freq.items()})
                fig_wc, ax = plt.subplots(figsize=(6, 3.2))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                fig_wc.patch.set_facecolor("#0d1117")
                st.pyplot(fig_wc)
            except Exception as e:
                st.info(f"WordCloud unavailable: {e}")
        st.caption("Source: TF-IDF analysis of issue descriptions (Found + Derived)")

    with col_kw:
        st.markdown("### 📊 Top 20 Keywords by TF-IDF Score")
        top_kws = nlp_data.get("top_keywords", [])
        if top_kws:
            kw_df = pd.DataFrame(top_kws, columns=["Keyword", "TF-IDF Score"])
            fig_kw = px.bar(kw_df, x="TF-IDF Score", y="Keyword", orientation="h",
                            color="TF-IDF Score", color_continuous_scale="Blues",
                            title=f"Top Keywords (n={len(df)} issue descriptions)")
            fig_kw.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 font_color="white", height=380, showlegend=False)
            st.plotly_chart(fig_kw, use_container_width=True)

    # Clusters
    st.markdown("---")
    st.markdown("### 🔵 KMeans Issue Clusters")
    clusters = nlp_data.get("clusters", {})
    n_clusters = nlp_data.get("n_clusters", 0)
    st.caption(f"KMeans with k={n_clusters} clusters · Grouping {len(df)} issues by description similarity")

    if clusters:
        cluster_colors = ["#00d4ff", "#7B2FBE", "#ff4444", "#44ff44", "#ffaa00", "#ff88aa"]
        for cid, cdata in clusters.items():
            color = cluster_colors[cid % len(cluster_colors)]
            with st.expander(f"🔵 {cdata['label']} — {cdata['count']} issue(s)"):
                kw_str = " · ".join([f"`{k}`" for k in cdata["keywords"]])
                st.markdown(f"**Top Keywords:** {kw_str}")

                sev_str = " | ".join([f"{s}: {c}" for s, c in cdata["severities"].items()])
                mod_str = " | ".join([f"{m}: {c}" for m, c in list(cdata["modules"].items())[:4]])
                st.markdown(f"**Severities:** {sev_str}")
                st.markdown(f"**Modules:** {mod_str}")

                if cdata["count"] < 5:
                    st.markdown('<div class="warning-box">⚠️ Sample size warning: fewer than 5 issues in this cluster — patterns may not be representative.</div>', unsafe_allow_html=True)

                st.markdown("**Representative Examples:**")
                for ex in cdata["examples"]:
                    st.markdown(f"- {ex}")

    # Cluster distribution chart
    if clusters:
        cluster_df = pd.DataFrame([
            {"Cluster": cdata["label"], "Count": cdata["count"]}
            for cdata in clusters.values()
        ])
        fig_cl = px.pie(cluster_df, names="Cluster", values="Count",
                        title="Issue Distribution by Cluster", hole=0.3)
        fig_cl.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_cl, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 6 — FIX VALIDATION & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════

elif page == "✅ Fix Validation & Recs":
    st.markdown("## ✅ Fix Validation & Recommendations")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **Recurrence Flags:** Issues are flagged as likely to reappear based on their type and frequency. 
        For example, broken links recur when pages are moved; performance issues recur when new content 
        is added without optimization.

        **Recommendations:** All 8–10 recommendations are derived from actual crawl findings — 
        each cites the specific number of affected pages/issues found.

        **Benchmarks:** Your site's metrics are compared against widely-accepted web quality standards 
        (Google Core Web Vitals, WCAG 2.1, SEO best practices).

        **Priority Tests:** Ranked by risk score from Page 3 — highest risk modules tested first.
        """)

    # Recurrence flags
    recurrence = recs.get("recurrence_flags", [])
    if recurrence:
        st.markdown("### 🔄 Issues Likely to Reappear")
        for flag in recurrence:
            risk_cls = "risk-high" if flag["risk"] == "High" else "risk-medium"
            st.markdown(f"""
            <div class="{risk_cls}" style="margin-bottom:0.5rem;">
                <b>{flag['issue_type']}</b> — Risk: {flag['risk']}<br>
                <span style="font-size:0.85rem; color:#aaa;">{flag['reason']}</span><br>
                <span style="font-size:0.8rem; color:#888;">📅 {flag['frequency']}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")

    # Recommendations
    st.markdown("### 💡 Actionable Recommendations")
    st.caption(f"🟢 All {len(recs.get('recommendations', []))} recommendations derived from actual crawl findings")

    priority_colors = {
        "Critical": "#ff4444",
        "Performance": "#ff8800",
        "Accessibility": "#00aaff",
        "SEO": "#aa44ff",
        "Social/SEO": "#aa44ff",
        "Mobile": "#44ffaa",
        "Security": "#ffaa00",
        "Monitoring": "#44ff88",
    }

    for rec in recs.get("recommendations", []):
        cat = rec.get("category", "General")
        color = priority_colors.get(cat, "#888")
        effort = rec.get("effort", "?")
        impact = rec.get("impact", "?")
        effort_icon = "🔴" if effort == "High" else "🟡" if effort == "Medium" else "🟢"
        impact_icon = "🔴" if impact == "High" else "🟡" if impact == "Medium" else "🟢"
        with st.expander(f"#{rec['priority']} [{cat}] {rec['title']}"):
            st.markdown(f"""
            <div style="background:#1a1a2e; border-left: 3px solid {color}; padding: 1rem; border-radius: 0 8px 8px 0;">
                <b style="color:{color};">{rec['title']}</b><br><br>
                {rec['detail']}<br><br>
                <span style="font-size:0.8rem; color:#888;">
                    {effort_icon} Effort: <b>{effort}</b> &nbsp;|&nbsp; {impact_icon} Impact: <b>{impact}</b>
                </span>
            </div>
            """, unsafe_allow_html=True)

    # Priority test list
    st.markdown("---")
    st.markdown("### 🎯 Testing Priority List (by Risk)")
    priority_tests = recs.get("priority_tests", [])
    if priority_tests:
        for i, test in enumerate(priority_tests, 1):
            lvl_color = "#ff4444" if test["risk_level"] == "High" else "#ffaa00" if test["risk_level"] == "Medium" else "#44ff44"
            st.markdown(f"""
            <div style='background:#1a1a2e; border:1px solid #2a2a4a; border-radius:8px; padding:0.8rem; margin-bottom:0.5rem;'>
                <b style='color:{lvl_color};'>#{i} {test['module']}</b>
                <span style='float:right; color:{lvl_color};'>Risk: {test['risk_score']:.0f}/100 ({test['risk_level']})</span><br>
                <span style='color:#888; font-size:0.85rem;'>📋 {test['test_focus']}</span><br>
                <span style='color:#555; font-size:0.8rem;'>Open issues: {test['open_issues']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No priority tests generated — run crawl first.")

    # Benchmark comparison
    st.markdown("---")
    st.markdown("### 📏 Benchmark Comparison")
    st.caption("Your site vs. web quality benchmarks (Google CWV, WCAG 2.1, SEO best practices)")
    benchmarks = recs.get("benchmarks", {})
    if benchmarks:
        bm_rows = []
        for metric, data in benchmarks.items():
            bm_rows.append({
                "Metric": metric,
                "Your Site": data["yours"],
                "Benchmark": data["benchmark"],
                "Status": data["status"],
            })
        bm_df = pd.DataFrame(bm_rows)
        st.dataframe(bm_df, use_container_width=True, hide_index=True)

    # Download issue report
    st.markdown("---")
    st.markdown("### 📥 Export Issue Data")
    if not df.empty:
        csv = df[["module", "issue_type", "severity", "status", "occurrences",
                   "fix_time_hours", "description", "source", "page_url"]].to_csv(index=False)
        st.download_button(
            "⬇️ Download Issues CSV",
            csv,
            f"issues_{crawl_data.get('base_domain', 'site')}.csv",
            "text/csv",
            type="primary",
        )
        st.caption("CSV contains all issues with full descriptions and source attribution")
