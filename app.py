"""
app.py — Intelligent App Testing System
Streamlit application — 6 structured analysis pages.
Version: 1.0.0
"""
from __future__ import annotations

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

from analyzer import (
    build_issues_df,
    compute_health_score,
    compute_module_risk,
    generate_recommendations,
    predict_module_risk,
    run_nlp_analysis,
    train_prediction_model,
)
from crawler import crawl_website, generate_issues_from_crawl, validate_url

warnings.filterwarnings("ignore")

APP_VERSION = "v1.0.0"
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent App Testing System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header ── */
.app-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #00d4ff, #7B2FBE);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: .15rem;
}
.app-sub { color: #777; font-size: .9rem; margin-bottom: 1.2rem; }

/* ── Section label ── */
.section-label {
    font-size: .7rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: #00d4ff;
    margin-bottom: .2rem;
}
/* ── Section description (the short explanation) ── */
.section-desc {
    background: #0f1923;
    border-left: 3px solid #00d4ff;
    border-radius: 0 6px 6px 0;
    padding: .55rem 1rem;
    font-size: .82rem; color: #9aa8b8;
    margin-bottom: 1.2rem; line-height: 1.55;
}
/* ── Divider with label ── */
.divider-label {
    display: flex; align-items: center; gap: .6rem;
    margin: 1.4rem 0 .9rem;
    font-size: .78rem; font-weight: 600;
    letter-spacing: .1em; text-transform: uppercase; color: #555;
}
.divider-label::before, .divider-label::after {
    content: ''; flex: 1; height: 1px; background: #222;
}
/* ── Metric card ── */
.kpi-row { display: flex; gap: .8rem; margin-bottom: 1rem; flex-wrap: wrap; }
.kpi-card {
    background: #111827; border: 1px solid #1e2a3a;
    border-radius: 10px; padding: .9rem 1.2rem;
    min-width: 130px; flex: 1;
}
.kpi-val { font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #00d4ff; }
.kpi-lbl { font-size: .72rem; color: #6b7280; text-transform: uppercase; letter-spacing: .08em; margin-top: .15rem; }

/* ── Risk pills ── */
.pill-high   { background:#3d0000; color:#ff6b6b; border:1px solid #ff4444; border-radius:20px; padding:2px 10px; font-size:.72rem; font-weight:600; }
.pill-medium { background:#3d2200; color:#ffa94d; border:1px solid #ff8800; border-radius:20px; padding:2px 10px; font-size:.72rem; font-weight:600; }
.pill-low    { background:#003d00; color:#69db7c; border:1px solid #44ff44; border-radius:20px; padding:2px 10px; font-size:.72rem; font-weight:600; }

/* ── Info box ── */
.info-box {
    background: #0d1117; border: 1px solid #1e2a3a;
    border-radius: 8px; padding: .9rem 1rem;
    font-size: .82rem; color: #8b949e; margin: .6rem 0;
}
/* ── Card container ── */
.card {
    background: #111827; border: 1px solid #1e2a3a;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: .6rem;
}
/* ── Warning box ── */
.warn-box {
    background: #2a1a00; border: 1px solid #ff8800;
    border-radius: 6px; padding: .6rem 1rem;
    color: #ffa94d; font-size: .82rem; margin: .5rem 0;
}
/* ── Chart caption ── */
.chart-note { font-size:.72rem; color:#4b5563; margin-top:-.4rem; margin-bottom:.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:.8rem 0 .4rem;'>
        <div style='font-family:Space Mono,monospace;font-size:1.25rem;color:#00d4ff;'>🔬 AppTester</div>
        <div style='color:#374151;font-size:.72rem;margin-top:.1rem;'>{APP_VERSION}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", [
        "🌐 Website Overview",
        "📊 Exploratory Analysis",
        "⚠️ Risk Scoring",
        "🤖 Prediction Model",
        "💬 NLP Issue Analysis",
        "✅ Fix Validation & Recs",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:.75rem;color:#374151;font-weight:600;'>⚙️ CRAWL SETTINGS</div>", unsafe_allow_html=True)
    respect_robots = st.checkbox("Respect robots.txt", value=True)
    st.caption("Uncheck to crawl sites that block bots.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.72rem;color:#374151;line-height:1.7;'>
    <b style='color:#555;'>METHODOLOGY</b><br>
    Risk = (Issues×0.4) + (Severity×0.35) + (BrokenLinks×0.25)<br>
    Normalised 0–100<br><br>
    ML: Random Forest · 100 trees<br>
    NLP: TF-IDF + KMeans clustering<br>
    Crawl limit: 50 pages max
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOP HEADER + URL INPUT
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🔬 Intelligent App Testing System</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Enter any website URL · system crawls it · extracts real issues · delivers AI-powered analysis</div>', unsafe_allow_html=True)

col_u, col_b = st.columns([6, 1])
with col_u:
    input_url = st.text_input("URL", placeholder="https://example.com", label_visibility="collapsed")
with col_b:
    run_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for _k, _v in {
    "crawl_data": None, "issues": [], "df": None,
    "risk_df": None, "health": None, "model_data": None,
    "nlp_data": None, "recs": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────
# RUN CRAWL
# ─────────────────────────────────────────────────────────────
if run_btn and input_url.strip():
    valid, clean_url, err_msg = validate_url(input_url.strip())
    if not valid:
        st.error(f"❌ {err_msg}")
    else:
        prog = st.progress(0, text="Initialising crawler…")

        def _cb(cur, tot, msg):
            prog.progress(min(int(cur / max(tot, 1) * 90), 90), text=f"🕷️ {msg}")

        crawl_data = crawl_website(clean_url, respect_robots=respect_robots, progress_callback=_cb)

        if crawl_data.get("error"):
            prog.empty()
            _err_map = {
                "robots_blocked": "🚫 Blocked by robots.txt — uncheck 'Respect robots.txt' in sidebar.",
                "timeout": "⏱️ Site timed out — it may be slow or blocking requests.",
                "ssl_error": "🔒 SSL error — try using http:// instead of https://.",
                "connection_error": "❌ Cannot connect — verify the URL is correct.",
            }
            st.error(_err_map.get(crawl_data["error"], crawl_data.get("message", "Unknown error.")))
        else:
            prog.progress(91, text="Generating issue dataset…")
            issues  = generate_issues_from_crawl(crawl_data)
            df      = build_issues_df(issues)
            prog.progress(94, text="Computing health score…")
            health  = compute_health_score(crawl_data, df)
            prog.progress(96, text="Computing module risk scores…")
            risk_df = compute_module_risk(df, crawl_data) if not df.empty else pd.DataFrame()
            prog.progress(97, text="Training ML model…")
            mdl     = train_prediction_model(df) if not df.empty else {"error": "No data"}
            prog.progress(98, text="Running NLP analysis…")
            nlp     = run_nlp_analysis(df) if not df.empty else {"error": "No data"}
            prog.progress(99, text="Building recommendations…")
            recs    = generate_recommendations(df, crawl_data, risk_df)

            st.session_state.update({
                "crawl_data": crawl_data, "issues": issues, "df": df,
                "risk_df": risk_df, "health": health, "model_data": mdl,
                "nlp_data": nlp, "recs": recs,
            })
            prog.progress(100, text="✅ Analysis complete!")
            time.sleep(0.5)
            prog.empty()
            st.success(
                f"✅ Crawled **{crawl_data['total_pages']}** pages in "
                f"**{crawl_data['crawl_time']}s** — **{len(issues)}** issues detected."
            )

# ─────────────────────────────────────────────────────────────
# GUARD — no data yet
# ─────────────────────────────────────────────────────────────
if st.session_state["crawl_data"] is None:
    st.markdown("""
    <div style='text-align:center;padding:5rem 2rem;'>
        <div style='font-size:3.5rem;'>🌐</div>
        <div style='font-size:1.15rem;color:#9ca3af;margin-top:1rem;font-weight:600;'>
            Enter a website URL and click Analyze
        </div>
        <div style='font-size:.85rem;color:#4b5563;margin-top:.5rem;'>
            The system will crawl up to 50 pages, extract real issues, and run
            health scoring · risk analysis · ML prediction · NLP clustering
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# UNPACK SESSION
# ─────────────────────────────────────────────────────────────
crawl_data  = st.session_state["crawl_data"]
issues      = st.session_state["issues"]
df          = st.session_state["df"]
risk_df     = st.session_state["risk_df"]
health      = st.session_state["health"]
model_data  = st.session_state["model_data"]
nlp_data    = st.session_state["nlp_data"]
recs        = st.session_state["recs"]
pages       = crawl_data.get("pages", [])

# Shared Plotly dark theme
DARK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")

def section_desc(text: str):
    st.markdown(f'<div class="section-desc">{text}</div>', unsafe_allow_html=True)

def divider(label: str = ""):
    if label:
        st.markdown(f'<div class="divider-label">{label}</div>', unsafe_allow_html=True)
    else:
        st.markdown("<hr style='border-color:#1e2a3a;margin:1.2rem 0;'>", unsafe_allow_html=True)

def chart_note(text: str):
    st.markdown(f'<div class="chart-note">ℹ️ {text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — WEBSITE OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🌐 Website Overview":

    st.markdown("## 🌐 Website Overview")
    section_desc(
        "A top-level snapshot of the crawled site. The <b>health score</b> (0–100) is a weighted "
        "composite of issues found, broken links, page speed, SEO tags, accessibility, and mobile "
        "readiness. The <b>structure map</b> shows every detected module and its page count. "
        "All data is derived from the live crawl — nothing is assumed."
    )

    # ── KPI row ──────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌐 Pages Crawled",    crawl_data.get("total_pages", 0))
    c2.metric("⚠️ Issues Detected",  len(issues))
    c3.metric("📦 Modules Found",    len(crawl_data.get("modules", [])))
    c4.metric("🔗 Broken Links",     len(crawl_data.get("broken_links", [])))
    c5.metric("⏱️ Crawl Duration",   f"{crawl_data.get('crawl_time', 0)}s")

    divider("HEALTH SCORE & CATEGORY BREAKDOWN")

    col_gauge, col_bar = st.columns([1, 2])

    with col_gauge:
        score = health.get("score", 0)
        color = health.get("color", "gray")
        grade = health.get("grade", "?")

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Grade: {grade}", "font": {"size": 17, "color": "white"}},
            number={"font": {"size": 42, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "#555"}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#111",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  35],  "color": "#1a0000"},
                    {"range": [35, 65],  "color": "#1a1000"},
                    {"range": [65, 100], "color": "#001a00"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": score},
            },
        ))
        fig_g.update_layout(height=240, margin=dict(t=30, b=0, l=10, r=10), **DARK)
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(
            '<div class="info-box">'
            '<b>How the score is calculated:</b><br>'
            'Issues (25%) + Broken Links (20%) + Performance (20%) + SEO (15%) '
            '+ Accessibility (10%) + Meta Tags (5%) + Mobile (5%)'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_bar:
        breakdown = health.get("breakdown", {})
        if breakdown:
            cats = list(breakdown.keys())
            vals = list(breakdown.values())
            bar_colors = ["#44ff44" if v >= 70 else "#ffaa00" if v >= 40 else "#ff4444" for v in vals]
            fig_br = go.Figure(go.Bar(
                x=vals, y=cats, orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.0f}/100" for v in vals],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Score: %{x:.0f}<extra></extra>",
            ))
            fig_br.add_vline(x=70, line_dash="dot", line_color="#444", annotation_text="Good (70)", annotation_font_color="#555")
            fig_br.update_layout(
                title="Score by Category — green ≥70 · orange 40–70 · red <40",
                xaxis=dict(range=[0, 110], title="Score (0–100)"),
                yaxis_title="",
                height=260,
                margin=dict(t=40, b=10),
                **DARK,
            )
            fig_br.update_xaxes(gridcolor="#1e2a3a")
            st.plotly_chart(fig_br, use_container_width=True)
        chart_note("Each category is scored independently then combined into the overall health score.")

    divider("WEBSITE MODULE STRUCTURE")
    section_desc(
        "Modules are identified from URL path patterns (e.g. /login → Authentication, /cart → Cart). "
        "Each card shows page count, average load time, and error count for that module."
    )

    module_groups: dict = {}
    for p in pages:
        module_groups.setdefault(p.get("module", "Unknown"), []).append(p)

    n_cols = min(len(module_groups), 4)
    if n_cols > 0:
        cols = st.columns(n_cols)
        for i, (mod, mpages) in enumerate(module_groups.items()):
            lts = [p["load_time"] for p in mpages if p.get("load_time", 0) > 0]
            avg_lt = float(np.mean(lts)) if lts else 0.0
            errs   = sum(1 for p in mpages if p.get("status_code") in (404, 500, 502, 503) or p.get("error"))
            health_col = "#ff4444" if errs > 0 else "#44ff44"

            with cols[i % n_cols]:
                st.markdown(f"""
                <div class="card">
                  <div style='font-size:.9rem;font-weight:700;color:#00d4ff;margin-bottom:.3rem;'>{mod}</div>
                  <div style='font-size:.75rem;color:#6b7280;'>
                    📄 {len(mpages)} page(s) &nbsp;·&nbsp; ⏱️ {avg_lt:.2f}s avg<br>
                    <span style='color:{health_col};'>● {errs} error(s)</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                for p in mpages[:2]:
                    sc     = p.get("status_code", "?")
                    sc_col = "#44ff44" if sc == 200 else "#ff4444"
                    short  = ("…" + p.get("url","")[-36:]) if len(p.get("url","")) > 36 else p.get("url","")
                    st.markdown(
                        f"<div style='font-size:.68rem;color:#374151;padding:.1rem .4rem;'>"
                        f"<span style='color:{sc_col};'>[{sc}]</span> {short}</div>",
                        unsafe_allow_html=True,
                    )
                if len(mpages) > 2:
                    st.markdown(f"<div style='font-size:.68rem;color:#374151;padding:.1rem .4rem;'>+ {len(mpages)-2} more…</div>", unsafe_allow_html=True)

    divider("CRAWL SUMMARY TABLE")
    section_desc("Raw numbers from the crawl — what the system found and checked.")
    st.dataframe(pd.DataFrame({
        "Metric": [
            "Site URL", "Pages Crawled", "Modules Detected", "Total Issues",
            "Broken / Error Links", "External Links Checked", "Crawl Duration", "robots.txt Respected",
        ],
        "Value": [
            crawl_data.get("url",""),
            crawl_data.get("total_pages", 0),
            len(crawl_data.get("modules", [])),
            len(issues),
            len(crawl_data.get("broken_links", [])),
            crawl_data.get("external_links_checked", 0),
            f"{crawl_data.get('crawl_time', 0)}s",
            "Yes" if respect_robots else "No",
        ],
        "What it means": [
            "The URL that was crawled",
            "Number of unique pages visited (max 50)",
            "Feature areas identified from URL structure",
            "Total problems found across all pages",
            "Links returning 4xx / 5xx / unreachable",
            "Off-site links checked for validity",
            "Wall-clock time to complete the crawl",
            "Whether robots.txt rules were followed",
        ],
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":

    st.markdown("## 📊 Exploratory Data Analysis")
    section_desc(
        "Visual breakdown of every issue found during the crawl. Charts are ordered from "
        "<b>broadest view → specific detail → page-level drill-down</b>. "
        "Use these charts to understand <i>where</i> problems concentrate, <i>how severe</i> they are, "
        "<i>what types</i> recur most, and <i>which pages</i> are slowest."
    )

    if df is None or df.empty:
        st.warning("No issues found — the site may be very clean or the crawl returned limited data.")
        st.stop()

    # Summary strip
    open_count  = int(df["is_open"].sum())
    high_count  = int((df["severity"] == "High").sum())
    fixed_count = int((df["status"] == "Fixed").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Issues",    len(df))
    c2.metric("Open / Active",   open_count)
    c3.metric("High Severity",   high_count)
    c4.metric("Fixed",           fixed_count)

    # ── Chart 1 + 2: Distribution overview ────────────────────
    divider("DISTRIBUTION OVERVIEW")
    section_desc(
        "<b>Left:</b> How many issues belong to each site module — reveals which areas have the most problems.<br>"
        "<b>Right:</b> Proportion of Low / Medium / High severity across all issues."
    )

    c_left, c_right = st.columns(2)
    with c_left:
        mc = (df.groupby("module").size()
                .reset_index(name="count")
                .sort_values("count", ascending=False))
        fig1 = px.bar(
            mc, x="module", y="count",
            color="count", color_continuous_scale="Blues",
            title="Issues per Module",
            labels={"module": "Module", "count": "Issue Count"},
        )
        fig1.update_layout(**DARK, xaxis_tickangle=-30, showlegend=False,
                           margin=dict(t=40, b=60))
        fig1.update_traces(texttemplate="%{y}", textposition="outside")
        fig1.update_xaxes(gridcolor="#1e2a3a")
        fig1.update_yaxes(gridcolor="#1e2a3a")
        st.plotly_chart(fig1, use_container_width=True)
        chart_note(f"Tallest bar = most issue-dense module. Total: {len(df)} issues.")

    with c_right:
        svc = df["severity"].value_counts().reset_index()
        svc.columns = ["severity", "count"]
        fig2 = px.pie(
            svc, names="severity", values="count", hole=0.45,
            color="severity",
            color_discrete_map={"High": "#ff4444", "Medium": "#ffaa00", "Low": "#44ff44"},
            title="Severity Distribution",
        )
        fig2.update_traces(textinfo="percent+label", textfont_size=13)
        fig2.update_layout(**DARK, margin=dict(t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        chart_note("Outer ring = share of each severity. High severity issues require immediate attention.")

    # ── Chart 3 + 4: Type & Status ────────────────────────────
    divider("ISSUE TYPE & STATUS BREAKDOWN")
    section_desc(
        "<b>Left:</b> Which specific issue types occur most — helps prioritise what to fix first.<br>"
        "<b>Right:</b> Current resolution status — Open (unresolved), Fixed, Reopened (regressed)."
    )

    c3_l, c3_r = st.columns(2)
    with c3_l:
        tc = df["issue_type"].value_counts().head(12).reset_index()
        tc.columns = ["issue_type", "count"]
        fig3 = px.bar(
            tc, x="count", y="issue_type", orientation="h",
            color="count", color_continuous_scale="Oranges",
            title="Top 12 Issue Types (by frequency)",
            labels={"issue_type": "", "count": "Occurrences"},
        )
        fig3.update_layout(**DARK, height=370, showlegend=False, margin=dict(t=40, b=10))
        fig3.update_traces(texttemplate="%{x}", textposition="outside")
        fig3.update_xaxes(gridcolor="#1e2a3a")
        st.plotly_chart(fig3, use_container_width=True)
        chart_note("Longest bar = most common problem type across all pages.")

    with c3_r:
        stc = df["status"].value_counts().reset_index()
        stc.columns = ["status", "count"]
        fig4 = px.bar(
            stc, x="status", y="count", color="status",
            color_discrete_map={"Open": "#ff4444", "Fixed": "#44ff44", "Reopened": "#ffaa00"},
            title="Issue Status (Open / Fixed / Reopened)",
            labels={"status": "Status", "count": "Count"},
            text_auto=True,
        )
        fig4.update_layout(**DARK, showlegend=False, margin=dict(t=40, b=10))
        fig4.update_yaxes(gridcolor="#1e2a3a")
        st.plotly_chart(fig4, use_container_width=True)
        chart_note("'Reopened' = previously fixed but regressed. High reopen rate signals systemic issues.")

    # ── Chart 5: Module × Severity heatmap ────────────────────
    divider("MODULE × SEVERITY HEATMAP")
    section_desc(
        "Cross-tabulation of modules vs severity levels. Darker red cells = more high-severity issues "
        "in that module. Quickly shows which modules carry the most critical risk."
    )
    pivot = df.pivot_table(index="module", columns="severity", values="id", aggfunc="count", fill_value=0)
    for sev in ["Low", "Medium", "High"]:
        if sev not in pivot.columns:
            pivot[sev] = 0
    pivot = pivot[["Low", "Medium", "High"]]
    fig_hm = px.imshow(
        pivot.values,
        x=["Low", "Medium", "High"],
        y=pivot.index.tolist(),
        color_continuous_scale=[[0, "#0d1117"], [0.5, "#3d1a00"], [1.0, "#ff2200"]],
        title="Issue Count by Module & Severity",
        labels=dict(x="Severity", y="Module", color="Count"),
        text_auto=True,
        aspect="auto",
    )
    fig_hm.update_layout(**DARK, margin=dict(t=50, b=10), height=max(250, len(pivot)*38))
    st.plotly_chart(fig_hm, use_container_width=True)
    chart_note("Each cell = number of issues of that severity in that module. Red = high risk concentration.")

    # ── Chart 6: Page load times ──────────────────────────────
    divider("PAGE LOAD TIMES")
    section_desc(
        "Load time in seconds for each crawled page. "
        "<b>Red</b> = >3s (critical — users leave), "
        "<b>Orange</b> = 1.5–3s (acceptable but improvable), "
        "<b>Green</b> = ≤1.5s (good). "
        "Google's Core Web Vitals target is under 2.5s."
    )
    lt_data = [
        {"Page": ("…" + p.get("url","")[-45:]), "Load Time (s)": p.get("load_time",0), "Module": p.get("module","?")}
        for p in pages if p.get("load_time", 0) and p["load_time"] > 0
    ]
    if lt_data:
        lt_df = pd.DataFrame(lt_data).sort_values("Load Time (s)", ascending=False).head(25)
        lt_colors = ["#ff4444" if t > 3 else "#ffaa00" if t > 1.5 else "#44ff44" for t in lt_df["Load Time (s)"]]
        fig5 = go.Figure(go.Bar(
            x=lt_df["Load Time (s)"], y=lt_df["Page"], orientation="h",
            marker_color=lt_colors,
            text=[f"{t:.2f}s" for t in lt_df["Load Time (s)"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Load: %{x:.2f}s<extra></extra>",
        ))
        fig5.add_vline(x=3.0,  line_dash="dash", line_color="#ff4444", annotation_text="3s — critical", annotation_font_color="#ff4444")
        fig5.add_vline(x=1.5,  line_dash="dot",  line_color="#ffaa00", annotation_text="1.5s — target", annotation_font_color="#ffaa00")
        avg_lt = float(np.mean([d["Load Time (s)"] for d in lt_data]))
        fig5.add_vline(x=avg_lt, line_dash="dot", line_color="#aaa", annotation_text=f"avg {avg_lt:.2f}s", annotation_font_color="#aaa")
        fig5.update_layout(
            title="Page Load Times — slowest 25 pages",
            xaxis_title="Seconds", yaxis_title="",
            height=max(280, len(lt_df) * 24),
            margin=dict(t=50, b=10), **DARK,
        )
        fig5.update_xaxes(gridcolor="#1e2a3a")
        st.plotly_chart(fig5, use_container_width=True)
        chart_note(f"Showing slowest 25 of {len(lt_data)} pages with load data. Average: {avg_lt:.2f}s.")
    else:
        st.info("No load time data available from this crawl.")

    # ── Table: Most problematic pages ─────────────────────────
    divider("MOST PROBLEMATIC PAGES")
    section_desc("Pages ranked by total issue count. Focus your manual testing effort on pages near the top.")
    pic = (
        df.groupby("page_url")
          .agg(total_issues=("id","count"), high=("is_critical","sum"), open_=("is_open","sum"))
          .reset_index()
          .sort_values("total_issues", ascending=False)
          .head(10)
    )
    pic.columns = ["Page URL", "Total Issues", "High Severity", "Open Issues"]
    st.dataframe(pic, use_container_width=True, hide_index=True)
    chart_note("Sorted by total issues descending. High Severity = issues that need immediate attention.")

    # ── Raw data ──────────────────────────────────────────────
    with st.expander("📋 View Full Issue Dataset"):
        disp = df[["module","issue_type","severity","status","occurrences","fix_time_hours","description","source"]].copy()
        disp.columns = ["Module","Issue Type","Severity","Status","Occurrences","Fix Time (h)","Description","Data Source"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption("'Data Source' shows which crawl signal generated each issue (e.g. crawl_seo, crawl_forms).")


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — RISK SCORING
# ══════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Scoring":

    st.markdown("## ⚠️ Risk Scoring System")
    section_desc(
        "Each site module is given a <b>Risk Score (0–100)</b> combining three signals: "
        "how many issues it has, how severe they are, and how many broken links it contains. "
        "The score is then normalised so the most dangerous module always appears at 100. "
        "Use this to decide <i>which parts of the site to test first</i>."
    )

    if risk_df is None or (hasattr(risk_df, "empty") and risk_df.empty):
        st.warning("No risk data available — run a crawl first.")
        st.stop()

    # Formula box
    st.markdown("""
    <div class="info-box">
    <b>Risk Formula (fully transparent):</b><br>
    <code>Raw Risk = (Issue Count × 0.4) + (Avg Severity Weight × 10 × 0.35) + (Broken Link Rate × 100 × 0.25)</code><br>
    <code>Final Score = Normalised to 0–100 &nbsp;·&nbsp; Severity weights: Low=1, Medium=2, High=3</code><br><br>
    <span style='color:#555;'>Weightings: Issue volume (40%) matters most, severity (35%) second, broken links (25%) third.</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart 1: Risk score bar chart ─────────────────────────
    divider("MODULE RISK SCORES")
    section_desc(
        "Modules sorted highest → lowest risk. Red bars are high-risk (≥70), orange are medium (40–70), "
        "green are low (<40). The dashed lines mark the thresholds."
    )

    r_colors = ["#ff4444" if l=="High" else "#ffaa00" if l=="Medium" else "#44ff44"
                for l in risk_df["Risk Level"]]
    fig_r = go.Figure(go.Bar(
        x=risk_df["Module"],
        y=risk_df["Risk Score (0-100)"],
        marker_color=r_colors,
        text=[f"{v:.0f}" for v in risk_df["Risk Score (0-100)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Risk Score: %{y:.0f}/100<extra></extra>",
    ))
    fig_r.add_hline(y=70, line_dash="dash", line_color="#ff4444",   line_width=1, annotation_text="High Risk ≥70",    annotation_font_color="#ff4444")
    fig_r.add_hline(y=40, line_dash="dot",  line_color="#ffaa00",   line_width=1, annotation_text="Medium Risk ≥40",  annotation_font_color="#ffaa00")
    fig_r.update_layout(
        title="Risk Score per Module (0–100) — higher = test first",
        xaxis_title="Module", yaxis_title="Risk Score",
        xaxis_tickangle=-30, yaxis=dict(range=[0, 118]),
        margin=dict(t=50, b=60), **DARK,
    )
    fig_r.update_yaxes(gridcolor="#1e2a3a")
    st.plotly_chart(fig_r, use_container_width=True)
    chart_note("Score is normalised — the highest-risk module is always 100; others are relative to it.")

    # ── Chart 2: Risk components stacked ──────────────────────
    divider("RISK COMPONENTS — WHAT DRIVES EACH SCORE")
    section_desc(
        "Stacked bar showing the three contributing signals for each module. "
        "Helps you understand <i>why</i> a module is high risk — "
        "is it issue volume, severity, or broken links?"
    )
    comp_df = risk_df.copy()
    comp_df["Issue Component"]       = (comp_df["Issue Count"] * 0.4).round(1)
    comp_df["Severity Component"]    = (comp_df["Avg Severity Weight"] * 10 * 0.35).round(1)
    comp_df["Broken Link Component"] = (comp_df["Broken Link Rate"] * 100 * 0.25).round(1)
    fig_stack = go.Figure()
    for col_name, color in [
        ("Issue Component",       "#00d4ff"),
        ("Severity Component",    "#7B2FBE"),
        ("Broken Link Component", "#ff4444"),
    ]:
        fig_stack.add_trace(go.Bar(
            name=col_name.replace(" Component",""),
            x=comp_df["Module"],
            y=comp_df[col_name],
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{col_name}: %{{y:.1f}}<extra></extra>",
        ))
    fig_stack.update_layout(
        barmode="stack",
        title="Raw Risk Components per Module",
        xaxis_title="Module", yaxis_title="Component Score",
        xaxis_tickangle=-30, margin=dict(t=50, b=60),
        legend=dict(orientation="h", y=1.1, x=0),
        **DARK,
    )
    fig_stack.update_yaxes(gridcolor="#1e2a3a")
    st.plotly_chart(fig_stack, use_container_width=True)
    chart_note("Blue = issue count contribution · Purple = severity contribution · Red = broken link contribution.")

    # ── Table ─────────────────────────────────────────────────
    divider("DETAILED RISK TABLE")
    section_desc("Full numbers behind every risk score. Modules flagged ⚠️ have fewer than 5 issues — scores are less statistically reliable.")

    def _row_style(row):
        c = {"High":   "background-color:#2a0000;color:#ff6b6b",
             "Medium": "background-color:#2a1500;color:#ffa94d",
             "Low":    "background-color:#002a00;color:#69db7c"}.get(row["Risk Level"], "")
        return [c if col == "Risk Level" else "" for col in row.index]

    disp_r = risk_df[["Module","Issue Count","Open Issues","High Severity",
                       "Broken Link Rate","Risk Score (0-100)","Risk Level","Small Sample"]].copy()
    disp_r["Small Sample"]       = disp_r["Small Sample"].apply(lambda x: "⚠️ n<5" if x else "✅ OK")
    disp_r["Broken Link Rate"]   = disp_r["Broken Link Rate"].apply(lambda x: f"{x:.1%}")
    disp_r["Risk Score (0-100)"] = disp_r["Risk Score (0-100)"].apply(lambda x: f"{x:.1f}")
    st.dataframe(disp_r.style.apply(_row_style, axis=1), use_container_width=True, hide_index=True)

    small = risk_df[risk_df["Small Sample"] == True]
    if not small.empty:
        st.markdown(
            '<div class="warn-box">⚠️ <b>Small sample warning:</b> '
            + ", ".join(small["Module"].tolist()) +
            " — fewer than 5 issues; interpret their scores with caution.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — PREDICTION MODEL
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Prediction Model":

    st.markdown("## 🤖 Prediction Model")
    section_desc(
        "A <b>Random Forest classifier</b> is trained on the crawl-derived issue dataset to predict "
        "whether any given issue is likely to be <b>Critical</b> (High severity) or not. "
        "Charts show how accurately the model performs and which features drive its decisions. "
        "The <b>Live Prediction</b> panel at the bottom lets you test any hypothetical issue."
    )

    if not model_data or model_data.get("error"):
        st.warning(f"⚠️ {model_data.get('error','Model unavailable') if model_data else 'Run a crawl first.'}")
        st.stop()

    acc       = model_data.get("accuracy", 0)
    n_test    = len(model_data.get("y_test", []))
    confs     = model_data.get("confidence_scores", [])
    avg_conf  = float(np.mean(confs)) if confs else 0.0

    # ── KPIs ─────────────────────────────────────────────────
    divider("MODEL PERFORMANCE SUMMARY")
    section_desc(
        "Key numbers that describe how well the model performs on unseen data (the 30% test split). "
        "Accuracy = % of test issues correctly classified. Avg Confidence = how certain the model is on average."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy",   f"{acc*100:.1f}%",  help="% of test issues correctly labelled")
    c2.metric("Test Set Size",    f"n = {n_test}",    help="Issues held back for testing (30% split)")
    c3.metric("Avg Confidence",   f"{avg_conf*100:.1f}%", help="Average predict_proba() score on test set")
    c4.metric("Algorithm",        "Random Forest",   help="100 decision trees, max depth 5, balanced classes")

    # ── Confusion matrix + Feature importance ─────────────────
    divider("CONFUSION MATRIX & FEATURE IMPORTANCE")
    section_desc(
        "<b>Left — Confusion Matrix:</b> Rows = actual label, Columns = predicted label. "
        "Diagonal cells (top-left, bottom-right) = correct predictions. Off-diagonal = errors.<br>"
        "<b>Right — Feature Importance:</b> Which input features the model relies on most when deciding. "
        "Higher bar = stronger influence on the prediction."
    )

    col_cm, col_fi = st.columns(2)
    with col_cm:
        cm     = model_data.get("confusion_matrix", [[0,0],[0,0]])
        labels = model_data.get("label_names", ["Not Critical","Critical"])
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
            linewidths=.5, annot_kws={"size": 14},
        )
        ax.set_xlabel("Predicted Label", fontsize=10, color="white")
        ax.set_ylabel("Actual Label",    fontsize=10, color="white")
        ax.set_title(f"Confusion Matrix  (n={n_test} test samples)", color="white", fontsize=11)
        ax.tick_params(colors="white")
        fig_cm.patch.set_alpha(0); ax.set_facecolor("#0d1117")
        plt.tight_layout()
        st.pyplot(fig_cm); plt.close(fig_cm)
        chart_note("Large numbers on the diagonal = model is classifying correctly.")

    with col_fi:
        imp = model_data.get("feature_importance", {})
        feat_labels = {
            "module_enc":      "Module",
            "type_enc":        "Issue Type",
            "occurrences":     "Occurrences",
            "fix_time_hours":  "Fix Time",
            "severity_weight": "Severity",
        }
        fi_df = pd.DataFrame(
            [(feat_labels.get(k,k), round(v*100,1)) for k,v in imp.items()],
            columns=["Feature","Importance (%)"]
        ).sort_values("Importance (%)")
        fig_fi = px.bar(
            fi_df, x="Importance (%)", y="Feature", orientation="h",
            color="Importance (%)", color_continuous_scale="Blues",
            title="Feature Importance — what drives the prediction",
            labels={"Feature":"","Importance (%)":"Importance (%)"},
            text_auto=True,
        )
        fig_fi.update_layout(**DARK, height=310, showlegend=False, margin=dict(t=50,b=10))
        fig_fi.update_xaxes(gridcolor="#1e2a3a")
        st.plotly_chart(fig_fi, use_container_width=True)
        chart_note("Importance sums to 100%. Higher % = feature contributes more to each decision.")

    # ── Classification report ─────────────────────────────────
    divider("CLASSIFICATION REPORT")
    section_desc(
        "Per-class performance metrics. "
        "<b>Precision</b> = of issues predicted Critical, how many actually were. "
        "<b>Recall</b> = of all actual Critical issues, how many were caught. "
        "<b>F1</b> = harmonic mean of Precision and Recall (best single metric). "
        "<b>Support</b> = number of test samples in that class."
    )
    report = model_data.get("classification_report", {})
    rows = [
        {"Class": k,
         "Precision": f"{v.get('precision',0):.3f}",
         "Recall":    f"{v.get('recall',0):.3f}",
         "F1-Score":  f"{v.get('f1-score',0):.3f}",
         "Support":   int(v.get("support",0))}
        for k,v in report.items() if isinstance(v, dict)
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Live prediction ───────────────────────────────────────
    divider("LIVE PREDICTION TOOL")
    section_desc(
        "Select any module and issue parameters below. The trained model will output "
        "a <b>Critical / Not Critical</b> verdict with a confidence percentage. "
        "Use this to quickly assess hypothetical issues before raising them formally."
    )

    mods = sorted(df["module"].unique().tolist()) if df is not None and not df.empty else ["Homepage"]
    typs = sorted(df["issue_type"].unique().tolist()) if df is not None and not df.empty else ["Broken Link"]

    p1, p2 = st.columns(2)
    with p1:
        sel_mod  = st.selectbox("Module",       mods,  help="Which site module does this issue belong to?")
        sel_type = st.selectbox("Issue Type",   typs,  help="What category of issue is it?")
        sel_sev  = st.selectbox("Severity",     ["Low","Medium","High"], help="How severe is the issue?")
    with p2:
        sel_occ  = st.slider("Occurrences",         1, 50,   1,    help="How many times does this issue appear?")
        sel_fix  = st.slider("Estimated Fix Time (hours)", 0.25, 16.0, 2.0, step=0.25,
                              help="How long would it take to fix?")

    if st.button("🔮 Predict Criticality", type="primary"):
        res = predict_module_risk(model_data, sel_mod, sel_type, sel_occ, sel_fix, sel_sev)
        if res.get("error"):
            st.error(res["error"])
        else:
            pred = res["prediction"]
            conf = res["confidence"]
            pc   = res.get("proba_critical", 0)
            pnc  = res.get("proba_not_critical", 100)
            pcol = "#ff4444" if pred == "Critical" else "#44ff44"
            icon = "🚨" if pred == "Critical" else "✅"
            st.markdown(f"""
            <div class="card" style="border:2px solid {pcol}; margin-top:.8rem;">
              <div style="font-size:1.4rem;font-weight:700;color:{pcol};">{icon} Prediction: {pred}</div>
              <div style="color:#9ca3af;margin-top:.4rem;font-size:.9rem;">
                Overall Confidence: <b style="color:white;">{conf}%</b>
                &nbsp;·&nbsp;
                P(Critical): <b style="color:#ff6b6b;">{pc:.1f}%</b>
                &nbsp;·&nbsp;
                P(Not Critical): <b style="color:#69db7c;">{pnc:.1f}%</b>
              </div>
              <div style="color:#374151;font-size:.75rem;margin-top:.4rem;">
                Random Forest · seed=42 · trained on {len(df)} issues
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — NLP ISSUE ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "💬 NLP Issue Analysis":

    st.markdown("## 💬 NLP Issue Analysis")
    section_desc(
        "Natural-language analysis of every issue description generated by the crawl. "
        "<b>TF-IDF</b> finds the most statistically distinctive words. "
        "<b>WordCloud</b> visualises word prominence at a glance. "
        "<b>KMeans clustering</b> groups similar issues together automatically — "
        "useful for spotting theme patterns without reading every issue individually."
    )

    if not nlp_data or nlp_data.get("error"):
        st.warning(f"⚠️ {nlp_data.get('error','Not enough data for NLP') if nlp_data else 'Run a crawl first.'}")
        st.stop()

    # ── Chart 1: WordCloud ────────────────────────────────────
    divider("WORDCLOUD — MOST PROMINENT TERMS")
    section_desc(
        "Larger words appear more frequently (by TF-IDF weight) across issue descriptions. "
        "A dominant word like 'missing' or 'broken' tells you the main type of problems on this site."
    )
    col_wc, col_kw = st.columns([1, 1])
    with col_wc:
        wf = nlp_data.get("word_freq", {})
        if wf:
            try:
                wc_img = WordCloud(
                    width=620, height=340,
                    background_color="#0d1117",
                    colormap="cool", max_words=60,
                    prefer_horizontal=0.75,
                ).generate_from_frequencies({k: max(v*100, 1) for k,v in wf.items()})
                fig_wc, ax = plt.subplots(figsize=(6.2, 3.4))
                ax.imshow(wc_img, interpolation="bilinear"); ax.axis("off")
                fig_wc.patch.set_facecolor("#0d1117")
                fig_wc.tight_layout(pad=0)
                st.pyplot(fig_wc); plt.close(fig_wc)
            except Exception as e:
                st.info(f"WordCloud could not render: {e}")
        chart_note("Word size = TF-IDF prominence. Words common to all docs are suppressed.")

    # ── Chart 2: Top keywords bar ─────────────────────────────
    with col_kw:
        section_desc(
            "Exact TF-IDF scores for the top 20 keywords. "
            "Higher score = word is more distinctive to this site's issues."
        )
        kws = nlp_data.get("top_keywords", [])
        if kws:
            kw_df = pd.DataFrame(kws, columns=["Keyword","TF-IDF Score"])
            fig_kw = px.bar(
                kw_df, x="TF-IDF Score", y="Keyword", orientation="h",
                color="TF-IDF Score", color_continuous_scale="Blues",
                title="Top 20 Keywords by TF-IDF Score",
                text_auto=".3f",
            )
            fig_kw.update_layout(**DARK, height=400, showlegend=False, margin=dict(t=50,b=10))
            fig_kw.update_xaxes(gridcolor="#1e2a3a")
            st.plotly_chart(fig_kw, use_container_width=True)
            chart_note(f"Computed across {len(df)} issue descriptions using TF-IDF (1- and 2-word phrases).")

    # ── Chart 3: Cluster overview pie ─────────────────────────
    divider("ISSUE CLUSTERS — KMeans GROUPING")
    section_desc(
        "Issues are grouped into clusters of similar descriptions using KMeans. "
        "<b>Left pie</b> shows how issues are distributed across clusters. "
        "<b>Cards below</b> give a keyword summary and examples for each cluster. "
        f"k = {nlp_data.get('n_clusters', '?')} clusters were automatically chosen based on dataset size."
    )

    clusters = nlp_data.get("clusters", {})
    n_cl     = nlp_data.get("n_clusters", 0)

    cl_left, cl_right = st.columns([1, 2])
    with cl_left:
        if clusters:
            cl_df = pd.DataFrame([
                {"Cluster": cd["label"].split(":")[0], "Count": cd["count"]}
                for cd in clusters.values()
            ])
            fig_pie = px.pie(
                cl_df, names="Cluster", values="Count",
                title="Issue Distribution by Cluster",
                hole=0.4,
            )
            fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
            fig_pie.update_layout(**DARK, margin=dict(t=50,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            chart_note(f"k={n_cl} KMeans clusters on TF-IDF vectors. Each slice = one issue theme.")

    with cl_right:
        CLRS = ["#00d4ff","#7B2FBE","#ff6b6b","#69db7c","#ffa94d","#f783ac"]
        for cid, cd in clusters.items():
            col = CLRS[cid % len(CLRS)]
            sev_str = " · ".join(f"<b>{s}</b>: {c}" for s,c in cd["severities"].items())
            mod_str = " · ".join(f"{m}: {c}" for m,c in list(cd["modules"].items())[:3])
            kw_str  = " &nbsp; ".join(f"<code>{k}</code>" for k in cd["keywords"][:5])
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {col}; margin-bottom:.5rem;">
              <div style="font-size:.85rem;font-weight:700;color:{col};">{cd['label']} &nbsp;
                <span style="font-size:.72rem;color:#4b5563;font-weight:400;">({cd['count']} issues)</span>
              </div>
              <div style="font-size:.78rem;color:#6b7280;margin-top:.3rem;">
                🔑 Keywords: {kw_str}<br>
                ⚠️ Severities: {sev_str}<br>
                📦 Modules: {mod_str}
              </div>
            </div>
            """, unsafe_allow_html=True)
            if cd["count"] < 5:
                st.markdown('<div class="warn-box">⚠️ Fewer than 5 issues in this cluster — patterns may not be representative.</div>', unsafe_allow_html=True)

    # ── Cluster detail expanders ───────────────────────────────
    divider("CLUSTER DETAIL — REPRESENTATIVE EXAMPLES")
    section_desc("Expand each cluster to read real example issue descriptions that belong to that group.")
    for cid, cd in clusters.items():
        with st.expander(f"{'🔵🟣🔴🟢🟡🩷'[cid % 6]} {cd['label']} — {cd['count']} issue(s)"):
            for ex in cd["examples"]:
                st.markdown(f"- {ex}")
            st.caption("These are real descriptions generated from crawl findings.")


# ══════════════════════════════════════════════════════════════
#  PAGE 6 — FIX VALIDATION & RECS
# ══════════════════════════════════════════════════════════════
elif page == "✅ Fix Validation & Recs":

    st.markdown("## ✅ Fix Validation & Recommendations")
    section_desc(
        "Everything you need to act on the analysis. "
        "<b>Recurrence flags</b> highlight issues that will reappear unless root causes are fixed. "
        "<b>Recommendations</b> are specific, actionable steps derived from actual crawl findings. "
        "<b>Priority tests</b> tell you which modules to test first. "
        "<b>Benchmarks</b> compare your site against industry standards."
    )

    if not recs:
        st.warning("Run a crawl first.")
        st.stop()

    # ── Section 1: Recurrence flags ───────────────────────────
    rec_flags = recs.get("recurrence_flags", [])
    if rec_flags:
        divider("ISSUES LIKELY TO REAPPEAR")
        section_desc(
            "These issue types have a high probability of coming back after being fixed, "
            "because they stem from process gaps rather than one-time mistakes. "
            "They need preventive measures, not just one-time patches."
        )
        for flag in rec_flags:
            cls  = "risk-high" if flag["risk"]=="High" else "risk-medium"
            pill = f'<span class="pill-high">High Risk</span>' if flag["risk"]=="High" else f'<span class="pill-medium">Medium Risk</span>'
            st.markdown(f"""
            <div class="{cls}" style="margin-bottom:.5rem;">
              <div style="font-size:.9rem;font-weight:700;">{flag['issue_type']} &nbsp; {pill}</div>
              <div style="font-size:.8rem;color:#9ca3af;margin-top:.2rem;">📌 {flag['reason']}</div>
              <div style="font-size:.75rem;color:#6b7280;margin-top:.1rem;">📅 {flag['frequency']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Section 2: Recommendations ────────────────────────────
    divider("ACTIONABLE RECOMMENDATIONS")
    section_desc(
        f"All {len(recs.get('recommendations',[]))} recommendations below are derived from actual crawl data — "
        "each cites the real number of affected pages or issues found. "
        "Sorted by priority (1 = fix first). Effort = how hard to implement. Impact = benefit gained."
    )

    CAT_COLORS = {
        "Critical":"#ff4444","Performance":"#ff8800","Accessibility":"#00aaff",
        "SEO":"#aa44ff","Social/SEO":"#aa44ff","Mobile":"#44ffaa",
        "Security":"#ffaa00","Monitoring":"#44ff88","Info":"#888",
    }
    EFFORT_ICON = {"High":"🔴 High","Medium":"🟡 Medium","Low":"🟢 Low"}

    for rec in recs.get("recommendations", []):
        cat   = rec.get("category","General")
        color = CAT_COLORS.get(cat,"#888")
        ei    = EFFORT_ICON.get(rec.get("effort",""), rec.get("effort",""))
        ii    = EFFORT_ICON.get(rec.get("impact",""),  rec.get("impact",""))
        with st.expander(f"#{rec['priority']}  [{cat}]  {rec['title']}"):
            st.markdown(f"""
            <div style="background:#0f1923;border-left:3px solid {color};
                        padding:.9rem 1.1rem;border-radius:0 8px 8px 0;">
              <div style="font-size:.85rem;color:#9ca3af;line-height:1.6;">{rec['detail']}</div>
              <div style="font-size:.78rem;color:#4b5563;margin-top:.6rem;">
                Effort: <b style="color:#d1d5db;">{ei}</b>
                &nbsp;·&nbsp;
                Impact: <b style="color:#d1d5db;">{ii}</b>
                &nbsp;·&nbsp;
                Category: <b style="color:{color};">{cat}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Section 3: Priority test list ─────────────────────────
    divider("TESTING PRIORITY LIST")
    section_desc(
        "Modules ranked by their Risk Score from Page 3. "
        "Test the top modules first — they have the highest concentration of issues and severity. "
        "Each entry includes a suggested test focus so testers know exactly what to check."
    )
    pt = recs.get("priority_tests", [])
    if pt:
        for i, t in enumerate(pt, 1):
            lc   = "#ff4444" if t["risk_level"]=="High" else "#ffaa00" if t["risk_level"]=="Medium" else "#44ff44"
            pill = (f'<span class="pill-high">{t["risk_level"]}</span>' if t["risk_level"]=="High"
                    else f'<span class="pill-medium">{t["risk_level"]}</span>' if t["risk_level"]=="Medium"
                    else f'<span class="pill-low">{t["risk_level"]}</span>')
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {lc}; margin-bottom:.5rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:.9rem;font-weight:700;color:{lc};">#{i} &nbsp; {t['module']}</span>
                <span style="font-size:.8rem;">
                  {pill} &nbsp; Risk Score: <b style="color:{lc};">{t['risk_score']:.0f}/100</b>
                </span>
              </div>
              <div style="font-size:.8rem;color:#6b7280;margin-top:.3rem;">
                📋 Test focus: {t['test_focus']}<br>
                <span style="font-size:.72rem;color:#374151;">Open issues: {t['open_issues']}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No priority test data available.")

    # ── Section 4: Benchmark comparison ───────────────────────
    divider("BENCHMARK COMPARISON")
    section_desc(
        "Your site's key metrics compared against accepted industry standards: "
        "Google Core Web Vitals, WCAG 2.1 accessibility guidelines, and SEO best practices. "
        "✅ = meets standard · ⚠️ = needs improvement · ❌ = below standard."
    )
    bm = recs.get("benchmarks", {})
    if bm:
        bm_df = pd.DataFrame([
            {"Metric": k, "Your Site": v["yours"], "Industry Standard": v["benchmark"], "Status": v["status"]}
            for k,v in bm.items()
        ])
        st.dataframe(bm_df, use_container_width=True, hide_index=True)
        chart_note("Standards: Google Core Web Vitals (<2s load), WCAG 2.1 (100% alt text), SEO (100% meta tags).")

    # ── Section 5: Export ─────────────────────────────────────
    divider("EXPORT DATA")
    if df is not None and not df.empty:
        section_desc("Download the full issue dataset as CSV for use in your issue tracker or spreadsheet.")
        csv = df[["module","issue_type","severity","status","occurrences",
                  "fix_time_hours","description","source","page_url"]].to_csv(index=False)
        st.download_button(
            "⬇️  Download Issues CSV",
            csv,
            file_name=f"issues_{crawl_data.get('base_domain','site')}.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption(f"Contains {len(df)} issues · columns: Module, Issue Type, Severity, Status, Occurrences, Fix Time, Description, Source, Page URL")
