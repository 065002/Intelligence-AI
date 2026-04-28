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

.app-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #00d4ff, #7B2FBE);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: .15rem;
}
.app-sub { color: #6b7280; font-size: .88rem; margin-bottom: 1.2rem; }

/* One-line plain-English description below each section header */
.section-desc {
    background: #0f1923;
    border-left: 3px solid #00d4ff;
    border-radius: 0 6px 6px 0;
    padding: .5rem 1rem;
    font-size: .82rem; color: #9aa8b8;
    margin-bottom: 1rem; line-height: 1.5;
}

/* Section divider with uppercase label */
.divider-label {
    display: flex; align-items: center; gap: .6rem;
    margin: 1.6rem 0 .8rem;
    font-size: .72rem; font-weight: 700;
    letter-spacing: .12em; text-transform: uppercase; color: #4b5563;
}
.divider-label::before, .divider-label::after {
    content: ''; flex: 1; height: 1px; background: #1e2a3a;
}

/* Small grey note below a chart */
.chart-note { font-size: .72rem; color: #4b5563; margin: -.3rem 0 .9rem; }

/* Category explanation table rows */
.cat-row {
    display: flex; align-items: flex-start; gap: .8rem;
    padding: .55rem .8rem; border-bottom: 1px solid #1e2a3a;
    font-size: .8rem;
}
.cat-row:last-child { border-bottom: none; }
.cat-icon { font-size: 1rem; min-width: 1.4rem; }
.cat-name { font-weight: 700; color: #e5e7eb; min-width: 110px; }
.cat-score { font-family: 'Space Mono', monospace; font-weight: 700; min-width: 52px; text-align: right; }
.cat-desc { color: #6b7280; flex: 1; }
.cat-how  { color: #374151; font-size: .72rem; margin-top: .18rem; }

/* Info box */
.info-box {
    background: #0d1117; border: 1px solid #1e2a3a;
    border-radius: 8px; padding: .85rem 1rem;
    font-size: .82rem; color: #8b949e; margin: .5rem 0;
}

/* Card */
.card {
    background: #111827; border: 1px solid #1e2a3a;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: .6rem;
}

/* Warning box */
.warn-box {
    background: #2a1a00; border: 1px solid #ff8800;
    border-radius: 6px; padding: .55rem 1rem;
    color: #ffa94d; font-size: .8rem; margin: .4rem 0;
}

/* Risk pills */
.pill-high   { background:#3d0000; color:#ff6b6b; border:1px solid #ff4444; border-radius:20px; padding:1px 10px; font-size:.72rem; font-weight:700; }
.pill-medium { background:#3d2200; color:#ffa94d; border:1px solid #ff8800; border-radius:20px; padding:1px 10px; font-size:.72rem; font-weight:700; }
.pill-low    { background:#003d00; color:#69db7c; border:1px solid #44ff44; border-radius:20px; padding:1px 10px; font-size:.72rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
DARK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")

def section_desc(text: str):
    st.markdown(f'<div class="section-desc">{text}</div>', unsafe_allow_html=True)

def divider(label: str = ""):
    if label:
        st.markdown(f'<div class="divider-label">{label}</div>', unsafe_allow_html=True)
    else:
        st.markdown("<hr style='border-color:#1e2a3a;margin:1.2rem 0;'>", unsafe_allow_html=True)

def chart_note(text: str):
    st.markdown(f'<div class="chart-note">ℹ️ &nbsp;{text}</div>', unsafe_allow_html=True)

def pill(level: str) -> str:
    cls = {"High":"pill-high","Medium":"pill-medium","Low":"pill-low"}.get(level,"pill-low")
    return f'<span class="{cls}">{level}</span>'


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:.8rem 0 .4rem;'>
        <div style='font-family:Space Mono,monospace;font-size:1.2rem;color:#00d4ff;'>🔬 AppTester</div>
        <div style='color:#374151;font-size:.7rem;margin-top:.1rem;'>{APP_VERSION}</div>
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
    st.markdown("<div style='font-size:.73rem;color:#4b5563;font-weight:700;letter-spacing:.08em;'>⚙️ CRAWL SETTINGS</div>", unsafe_allow_html=True)
    respect_robots = st.checkbox("Respect robots.txt", value=True)
    st.caption("Uncheck to crawl sites that restrict automated access.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.71rem;color:#374151;line-height:1.75;'>
    <b style='color:#4b5563;letter-spacing:.06em;'>METHODOLOGY</b><br>
    Health = weighted average of 7 categories<br>
    Risk = (Issues×0.4)+(Severity×0.35)+(BrokenLinks×0.25)<br>
    ML: Random Forest · 100 trees · seed 42<br>
    NLP: TF-IDF + KMeans clustering<br>
    Crawl: up to 50 pages · 10s timeout/page
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER + URL INPUT
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🔬 Intelligent App Testing System</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Paste any website URL — the system crawls it, extracts real issues from the HTML, and delivers a full analysis report.</div>', unsafe_allow_html=True)

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
                "robots_blocked": "🚫 Blocked by robots.txt. Uncheck 'Respect robots.txt' in the sidebar to override.",
                "timeout":        "⏱️ The site timed out — it may be slow or blocking automated requests.",
                "ssl_error":      "🔒 SSL certificate error — try using http:// instead of https://.",
                "connection_error":"❌ Cannot connect — check that the URL is correct and the site is reachable.",
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
            prog.progress(97, text="Training prediction model…")
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
            time.sleep(0.4)
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
        <div style='font-size:1.1rem;color:#9ca3af;margin-top:1rem;font-weight:600;'>
            Enter a website URL above and click Analyze
        </div>
        <div style='font-size:.83rem;color:#4b5563;margin-top:.5rem;line-height:1.6;'>
            The system crawls up to 50 pages · identifies issues from raw HTML ·
            scores each site category · trains a prediction model · clusters issues by topic
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


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — WEBSITE OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🌐 Website Overview":

    st.markdown("## 🌐 Website Overview")

    # ── KPI strip ─────────────────────────────────────────────
    # Each card shows a single number from the crawl + a sub-label explaining what it means.
    high_count_kpi  = int((df["severity"] == "High").sum()) if df is not None and not df.empty else 0
    open_count_kpi  = int(df["is_open"].sum()) if df is not None and not df.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "🌐 Pages Crawled",
        crawl_data.get("total_pages", 0),
        help="Number of unique pages the crawler actually visited (max 50).",
    )
    c2.metric(
        "⚠️ Issues Found",
        len(issues),
        delta=f"{high_count_kpi} critical",
        delta_color="inverse",
        help=f"Total issues detected. {high_count_kpi} are High-severity (need immediate attention). {open_count_kpi} are still Open.",
    )
    c3.metric(
        "📦 Site Sections",
        len(crawl_data.get("modules", [])),
        help="Distinct functional areas detected from URL patterns — e.g. /login → Authentication, /cart → Cart.",
    )
    c4.metric(
        "🔗 Broken Links",
        len(crawl_data.get("broken_links", [])),
        delta="should be 0" if len(crawl_data.get("broken_links", [])) > 0 else "✓ none found",
        delta_color="inverse" if len(crawl_data.get("broken_links", [])) > 0 else "normal",
        help="Links that returned HTTP 404/5xx or could not be reached at all.",
    )
    c5.metric(
        "⏱️ Crawl Time",
        f"{crawl_data.get('crawl_time', 0)}s",
        help="Wall-clock time to complete the crawl. Longer times may indicate slow server response.",
    )

    # ── Why the issue count and health grade can both be high ─
    low_count_kpi  = int((df["severity"] == "Low").sum())  if df is not None and not df.empty else 0
    med_count_kpi  = int((df["severity"] == "Medium").sum()) if df is not None and not df.empty else 0
    broken_count   = len(crawl_data.get("broken_links", []))
    total_pages    = max(crawl_data.get("total_pages", 1), 1)

    # Build an explanation that resolves the apparent contradiction
    score_now = health.get("score", 0)
    grade_now = health.get("grade", "?")
    issue_note_parts = []
    if high_count_kpi == 0:
        issue_note_parts.append(f"none of the {len(issues)} issues are High-severity")
    else:
        issue_note_parts.append(
            f"only {high_count_kpi} of the {len(issues)} issues are High-severity "
            f"({high_count_kpi/total_pages:.1f} per page) — the Issues score rewards low high-severity density"
        )
    if broken_count > 0:
        issue_note_parts.append(
            f"{broken_count} broken link(s) are the biggest drag on your grade "
            f"(Broken Links score = {health.get('breakdown',{}).get('Broken Links',0):.0f}/100)"
        )
    explanation = "; ".join(issue_note_parts)

    st.markdown(f"""
    <div style="background:#0f1923;border-left:3px solid #ffaa00;border-radius:0 8px 8px 0;
                padding:.65rem 1.1rem;margin:.4rem 0 .2rem;font-size:.82rem;color:#9aa8b8;line-height:1.6;">
      <b style="color:#ffaa00;">❓ Why does the site have {len(issues)} issues but still score {score_now:.0f}/100 (Grade {grade_now})?</b><br>
      The health score only uses <b>High-severity</b> issues to score the Issues category —
      Low and Medium issues ({low_count_kpi} Low + {med_count_kpi} Medium here) don't reduce that score.
      In this crawl, {explanation}.
      Scroll down to the category table to see exactly how each score is calculated.
    </div>
    """, unsafe_allow_html=True)

    # ── Health score gauge ────────────────────────────────────
    divider("OVERALL HEALTH SCORE")

    col_gauge, col_cats = st.columns([1, 2])

    with col_gauge:
        score = health.get("score", 0)
        color = health.get("color", "gray")
        grade = health.get("grade", "?")

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Grade: {grade}", "font": {"size": 16, "color": "white"}},
            number={"font": {"size": 44, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "#555"}},
                "bar": {"color": color, "thickness": 0.26},
                "bgcolor": "#111", "borderwidth": 0,
                "steps": [
                    {"range": [0,  35],  "color": "#1a0000"},
                    {"range": [35, 65],  "color": "#1a1000"},
                    {"range": [65, 100], "color": "#001a00"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": score},
            },
        ))
        fig_g.update_layout(height=230, margin=dict(t=30, b=0, l=10, r=10), **DARK)
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown(f"""
        <div class="info-box" style="font-size:.77rem;line-height:1.7;">
          <b>What this score means</b><br>
          This site scored <b style="color:{color};">{score:.0f}/100</b> — Grade <b>{grade}</b>.<br><br>
          🟢 <b>A (80–100)</b> — Healthy. Minor polish only.<br>
          🟡 <b>B (65–79)</b> — Good overall, some gaps to close.<br>
          🟠 <b>C (50–64)</b> — Problems noticeable to real users.<br>
          🔴 <b>D/F (&lt;50)</b> — Fix critical issues before next release.
        </div>
        """, unsafe_allow_html=True)

    # ── Category bar chart ────────────────────────────────────
    with col_cats:
        breakdown = health.get("breakdown", {})
        if breakdown:
            # Sort by weight (importance): Issues first, Mobile last
            weight_order = ["Issues", "Broken Links", "Performance", "SEO", "Accessibility", "Meta Tags", "Mobile"]
            ordered = {k: breakdown[k] for k in weight_order if k in breakdown}
            # Reverse so highest-weight is at top of horizontal bar chart
            cats_ord = list(reversed(list(ordered.keys())))
            vals_ord = [min(ordered[c], 100) for c in cats_ord]   # cap at 100 — formula can exceed for fast pages
            bar_clrs = ["#44ff44" if v >= 70 else "#ffaa00" if v >= 40 else "#ff4444" for v in vals_ord]

            fig_br = go.Figure(go.Bar(
                x=vals_ord, y=cats_ord, orientation="h",
                marker_color=bar_clrs,
                text=[f"{v:.0f}" for v in vals_ord],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="<b>%{y}</b><br>Score: %{x:.0f} / 100<extra></extra>",
            ))
            fig_br.add_vline(x=70, line_dash="dot", line_color="#444",
                             annotation_text="70 = Good threshold",
                             annotation_font_color="#888",
                             annotation_position="top right")
            fig_br.update_layout(
                title="Category Scores — sorted by importance (top = highest weight)",
                xaxis=dict(range=[0, 118], title="Score (0–100)", gridcolor="#1e2a3a"),
                yaxis_title="",
                height=260,
                margin=dict(t=45, b=10),
                **DARK,
            )
            st.plotly_chart(fig_br, use_container_width=True)
            chart_note(
                "Top categories (Issues, Broken Links, Performance) carry the most weight in the final score. "
                "Red bars are the biggest drags on your grade — fix those first."
            )

    # ── Category explanation table ────────────────────────────
    divider("HOW EACH CATEGORY SCORE IS CALCULATED")
    section_desc(
        "Every number comes from inspecting raw HTML — no external API. "
        "The table shows exactly what the crawler checked, the formula used to turn that into a 0–100 score, "
        "and a worked example so you can verify the maths yourself."
    )

    CATEGORY_META = [
        # Ordered by weight (highest first) so the table matches the bar chart above
        {
            "icon": "⚠️", "category": "Issues",       "weight": "25%",
            "what_is_checked": "How many High-severity issues exist per page crawled",
            "formula": "100 − (high_issue_count ÷ pages × 50)",
            "worked_example": "10 high issues, 26 pages → 100 − (10÷26×50) = 81",
            "good_score": "≥ 70 means fewer than ~0.6 high issues per page",
        },
        {
            "icon": "🔗", "category": "Broken Links",  "weight": "20%",
            "what_is_checked": "Links returning HTTP 404 / 5xx or that could not connect",
            "formula": "100 − (broken_count ÷ pages × 200)",
            "worked_example": "10 broken links, 26 pages → 100 − (10÷26×200) = 23 (red)",
            "good_score": "0 broken links = 100. Even 1 per 10 pages drops score to 80",
        },
        {
            "icon": "⚡", "category": "Performance",   "weight": "20%",
            "what_is_checked": "Average HTTP response time across all pages (timed by crawler stopwatch)",
            "formula": "100 − (avg_load_seconds − 1.0) × 15",
            "worked_example": "Avg 1.5s → score 92 · Avg 3s → score 70 · Avg 5s → score 40",
            "good_score": "Pages loading under 2.5s score above 76 (Google Core Web Vitals target)",
        },
        {
            "icon": "🔍", "category": "SEO",           "weight": "15%",
            "what_is_checked": "Pages that have a non-empty <title> tag",
            "formula": "pages_with_title ÷ total_pages × 100",
            "worked_example": "21 of 26 pages have a title → 21÷26×100 = 81",
            "good_score": "100 = every page has a title. Missing titles lose clicks in search results",
        },
        {
            "icon": "♿", "category": "Accessibility", "weight": "10%",
            "what_is_checked": "<img> tags with a non-empty alt attribute (WCAG 2.1 rule 1.1.1)",
            "formula": "images_with_alt ÷ total_images × 100",
            "worked_example": "40 of 50 images have alt text → 40÷50×100 = 80",
            "good_score": "100 = all images described. Screen readers need this to work",
        },
        {
            "icon": "🏷️", "category": "Meta Tags",    "weight": "5%",
            "what_is_checked": "Pages with a <meta name='description'> tag (the text shown in Google search results)",
            "formula": "pages_with_meta_desc ÷ total_pages × 100",
            "worked_example": "13 of 26 pages have a meta description → 50",
            "good_score": "100 = every page controls its search snippet. Missing = Google writes its own",
        },
        {
            "icon": "📱", "category": "Mobile",        "weight": "5%",
            "what_is_checked": "Pages with a <meta name='viewport'> tag (tells mobile browsers how to scale the page)",
            "formula": "pages_with_viewport ÷ total_pages × 100",
            "worked_example": "All 26 pages have viewport tag → 100",
            "good_score": "100 = all pages are mobile-ready. Missing = page appears zoomed-out on phones",
        },
    ]

    breakdown_now = health.get("breakdown", {})
    if breakdown_now:
        # Compute real numbers from the crawl for live "this site" column
        total_p   = max(crawl_data.get("total_pages", 1), 1)
        n_high    = int((df["severity"] == "High").sum()) if df is not None and not df.empty else 0
        n_broken  = len(crawl_data.get("broken_links", []))
        n_title   = sum(1 for p in pages if p.get("has_title"))
        n_meta    = sum(1 for p in pages if p.get("has_meta_description"))
        n_vp      = sum(1 for p in pages if p.get("has_viewport"))
        t_imgs    = sum(p.get("img_total", 0) for p in pages)
        a_imgs    = sum(p.get("img_total", 0) - p.get("img_missing_alt", 0) for p in pages)
        load_ts   = [p["load_time"] for p in pages if p.get("load_time", 0) > 0]
        avg_lt    = float(np.mean(load_ts)) if load_ts else 0.0

        live_calc = {
            "Issues":       f"{n_high} high-sev issues ÷ {total_p} pages → score {min(breakdown_now.get('Issues',0),100):.0f}",
            "Broken Links": f"{n_broken} broken ÷ {total_p} pages → score {min(breakdown_now.get('Broken Links',0),100):.0f}",
            "Performance":  f"avg load {avg_lt:.2f}s → score {min(breakdown_now.get('Performance',0),100):.0f}",
            "SEO":          f"{n_title} of {total_p} pages have <title> → score {min(breakdown_now.get('SEO',0),100):.0f}",
            "Accessibility":f"{a_imgs} of {t_imgs} images have alt text → score {min(breakdown_now.get('Accessibility',0),100):.0f}",
            "Meta Tags":    f"{n_meta} of {total_p} pages have meta desc → score {min(breakdown_now.get('Meta Tags',0),100):.0f}",
            "Mobile":       f"{n_vp} of {total_p} pages have viewport tag → score {min(breakdown_now.get('Mobile',0),100):.0f}",
        }

        table_rows = []
        for meta in CATEGORY_META:
            cat = meta["category"]
            val = min(breakdown_now.get(cat, 0), 100)
            status = "✅ Good" if val >= 70 else ("⚠️ Needs work" if val >= 40 else "❌ Poor")
            table_rows.append({
                " ":                    meta["icon"],
                "Category":             cat,
                "Weight":               meta["weight"],
                "Your Score":           f"{val:.0f} / 100",
                "Status":               status,
                "What is checked":      meta["what_is_checked"],
                "This site":            live_calc.get(cat, ""),
                "Formula":              meta["formula"],
                "Good score means":     meta["good_score"],
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        chart_note(
            "Final health score = (Issues×0.25) + (Broken Links×0.20) + (Performance×0.20) + "
            "(SEO×0.15) + (Accessibility×0.10) + (Meta Tags×0.05) + (Mobile×0.05). "
            "All values computed locally from raw HTML — no paid API."
        )


    # ── Module structure ──────────────────────────────────────
    divider("DETECTED SITE MODULES")
    section_desc(
        "URL path patterns identify site sections: /login → Authentication, /cart → Cart, /blog → Blog/Content. "
        "Each card shows page count, average load time, and error count for that section."
    )

    module_groups: dict = {}
    for p in pages:
        module_groups.setdefault(p.get("module", "Unknown"), []).append(p)

    n_cols = min(len(module_groups), 4)
    if n_cols > 0:
        cols = st.columns(n_cols)
        for i, (mod, mpages) in enumerate(module_groups.items()):
            lts    = [p["load_time"] for p in mpages if p.get("load_time", 0) > 0]
            avg_lt = float(np.mean(lts)) if lts else 0.0
            errs   = sum(1 for p in mpages if p.get("status_code") in (404, 500, 502, 503) or p.get("error"))
            e_clr  = "#ff4444" if errs > 0 else "#44ff44"
            with cols[i % n_cols]:
                st.markdown(f"""
                <div class="card">
                  <div style='font-size:.88rem;font-weight:700;color:#00d4ff;margin-bottom:.25rem;'>{mod}</div>
                  <div style='font-size:.74rem;color:#6b7280;line-height:1.6;'>
                    📄 {len(mpages)} page(s)<br>
                    ⏱️ {avg_lt:.2f}s avg load<br>
                    <span style='color:{e_clr};'>● {errs} error(s)</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                for p in mpages[:2]:
                    sc    = p.get("status_code", "?")
                    s_clr = "#44ff44" if sc == 200 else "#ff4444"
                    short = ("…" + p.get("url","")[-34:]) if len(p.get("url","")) > 34 else p.get("url","")
                    st.markdown(
                        f"<div style='font-size:.67rem;color:#374151;padding:.1rem .3rem;'>"
                        f"<span style='color:{s_clr};'>[{sc}]</span> {short}</div>",
                        unsafe_allow_html=True,
                    )
                if len(mpages) > 2:
                    st.markdown(f"<div style='font-size:.67rem;color:#374151;padding:.05rem .3rem;'>+ {len(mpages)-2} more…</div>", unsafe_allow_html=True)

    # ── Crawl summary table ───────────────────────────────────
    divider("CRAWL SUMMARY")
    section_desc("Raw numbers from the crawl — what was visited, what was checked, and how long it took.")
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
            "The starting URL the crawler used",
            "Unique pages visited — capped at 50 to stay within Streamlit Cloud limits",
            "Feature areas detected from URL patterns (e.g. /blog, /checkout)",
            "Total problems found and recorded across all crawled pages",
            "Links whose server returned 4xx / 5xx or could not be reached",
            "Off-site links sampled and checked — up to 20 per crawl",
            "Total wall-clock time to complete the crawl",
            "If Yes, pages blocked by robots.txt were skipped",
        ],
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":

    st.markdown("## 📊 Exploratory Data Analysis")
    section_desc(
        "Six charts progressing from broad (module-level totals) to specific (individual page load times). "
        "Start at the top to understand the overall picture, then drill down to find individual problem pages."
    )

    if df is None or df.empty:
        st.warning("No issues were found — the site may be very clean, or the crawl returned very few pages.")
        st.stop()

    # ── Summary strip ─────────────────────────────────────────
    open_count  = int(df["is_open"].sum())
    high_count  = int((df["severity"] == "High").sum())
    fixed_count = int((df["status"] == "Fixed").sum())
    reopen_count= int((df["status"] == "Reopened").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Issues",   len(df),        help="All issues detected across every crawled page")
    c2.metric("Open / Active",  open_count,     help="Issues that are unresolved or have regressed")
    c3.metric("High Severity",  high_count,     help="Issues that can break functionality or fail audits")
    c4.metric("Fixed",          fixed_count,    help="Issues previously open but now resolved")

    # ── Chart 1 + 2 ───────────────────────────────────────────
    divider("ISSUE VOLUME & SEVERITY DISTRIBUTION")

    c_l, c_r = st.columns(2)
    with c_l:
        mc = (df.groupby("module").size()
                .reset_index(name="count")
                .sort_values("count", ascending=False))
        fig1 = px.bar(
            mc, x="module", y="count",
            color="count", color_continuous_scale="Blues",
            title="Issue Count per Module",
            labels={"module": "Module", "count": "Number of Issues"},
        )
        fig1.update_layout(**DARK, xaxis_tickangle=-30, showlegend=False, margin=dict(t=40, b=60))
        fig1.update_traces(texttemplate="%{y}", textposition="outside")
        fig1.update_xaxes(gridcolor="#1e2a3a")
        fig1.update_yaxes(gridcolor="#1e2a3a")
        st.plotly_chart(fig1, use_container_width=True)
        chart_note("Modules with the tallest bars should be tested and fixed first. Sorted highest → lowest.")

    with c_r:
        svc = df["severity"].value_counts().reset_index()
        svc.columns = ["severity", "count"]
        fig2 = px.pie(
            svc, names="severity", values="count", hole=0.45,
            color="severity",
            color_discrete_map={"High":"#ff4444","Medium":"#ffaa00","Low":"#44ff44"},
            title="Severity Split Across All Issues",
        )
        fig2.update_traces(textinfo="percent+label", textfont_size=13)
        fig2.update_layout(**DARK, margin=dict(t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        chart_note(
            f"High = breaks functionality or fails audits · "
            f"Medium = degrades experience · Low = best-practice gap. "
            f"This site has {high_count} High-severity issues."
        )

    # ── Chart 3 + 4 ───────────────────────────────────────────
    divider("ISSUE TYPES & RESOLUTION STATUS")

    c3_l, c3_r = st.columns(2)
    with c3_l:
        tc = df["issue_type"].value_counts().head(12).reset_index()
        tc.columns = ["issue_type", "count"]
        fig3 = px.bar(
            tc, x="count", y="issue_type", orientation="h",
            color="count", color_continuous_scale="Oranges",
            title="Top 12 Issue Types by Frequency",
            labels={"issue_type": "", "count": "Times Found"},
        )
        fig3.update_layout(**DARK, height=370, showlegend=False, margin=dict(t=40, b=10))
        fig3.update_traces(texttemplate="%{x}", textposition="outside")
        fig3.update_xaxes(gridcolor="#1e2a3a")
        st.plotly_chart(fig3, use_container_width=True)
        chart_note("The most frequent issue type tells you the biggest systemic gap on this site.")

    with c3_r:
        stc = df["status"].value_counts().reset_index()
        stc.columns = ["status", "count"]
        fig4 = px.bar(
            stc, x="status", y="count", color="status",
            color_discrete_map={"Open":"#ff4444","Fixed":"#44ff44","Reopened":"#ffaa00"},
            title="Issue Resolution Status",
            labels={"status": "Status", "count": "Count"},
            text_auto=True,
        )
        fig4.update_layout(**DARK, showlegend=False, margin=dict(t=40, b=10))
        fig4.update_yaxes(gridcolor="#1e2a3a")
        st.plotly_chart(fig4, use_container_width=True)
        chart_note(
            f"Open = still needs fixing · Fixed = resolved · Reopened = regressed after a fix. "
            f"There are {reopen_count} reopened issue(s) on this site."
        )

    # ── Chart 5: Heatmap ──────────────────────────────────────
    divider("SEVERITY × MODULE HEATMAP")
    section_desc("Each cell = number of issues at that severity level in that module. Darker red = more critical issues concentrated there.")

    pivot = df.pivot_table(index="module", columns="severity", values="id", aggfunc="count", fill_value=0)
    for sev in ["Low","Medium","High"]:
        if sev not in pivot.columns:
            pivot[sev] = 0
    pivot = pivot[["Low","Medium","High"]]
    fig_hm = px.imshow(
        pivot.values,
        x=["Low","Medium","High"],
        y=pivot.index.tolist(),
        color_continuous_scale=[[0,"#0d1117"],[0.5,"#3d1a00"],[1.0,"#ff2200"]],
        title="Issue Count by Module × Severity",
        labels=dict(x="Severity Level", y="Module", color="Count"),
        text_auto=True,
        aspect="auto",
    )
    fig_hm.update_layout(**DARK, margin=dict(t=50,b=10), height=max(240, len(pivot)*38))
    st.plotly_chart(fig_hm, use_container_width=True)
    chart_note("Red cells in the 'High' column = modules with the most critical issues. Focus testing effort there first.")

    # ── Chart 6: Load times ───────────────────────────────────
    divider("WHICH PAGES ARE SLOWEST TO LOAD?")
    section_desc(
        "Measured by timing each HTTP response during the crawl. "
        "Red = over 3s (poor) · Orange = 1.5–3s (acceptable) · Green = under 1.5s (good). "
        "Google Core Web Vitals target: under 2.5s."
    )

    lt_data = [
        {"Page": ("…" + p.get("url","")[-45:]),
         "Load Time (s)": p.get("load_time",0),
         "Module": p.get("module","?")}
        for p in pages if p.get("load_time",0) and p["load_time"] > 0
    ]
    if lt_data:
        lt_df    = pd.DataFrame(lt_data).sort_values("Load Time (s)", ascending=False).head(25)
        lt_clrs  = ["#ff4444" if t > 3 else "#ffaa00" if t > 1.5 else "#44ff44" for t in lt_df["Load Time (s)"]]
        avg_lt   = float(np.mean([d["Load Time (s)"] for d in lt_data]))
        fig5 = go.Figure(go.Bar(
            x=lt_df["Load Time (s)"], y=lt_df["Page"], orientation="h",
            marker_color=lt_clrs,
            text=[f"{t:.2f}s" for t in lt_df["Load Time (s)"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Load time: %{x:.2f}s<extra></extra>",
        ))
        fig5.add_vline(x=3.0,  line_dash="dash", line_color="#ff4444",
                       annotation_text="3s — poor (red)",   annotation_font_color="#ff4444")
        fig5.add_vline(x=1.5,  line_dash="dot",  line_color="#ffaa00",
                       annotation_text="1.5s — acceptable", annotation_font_color="#ffaa00")
        fig5.add_vline(x=avg_lt, line_dash="dot", line_color="#9ca3af",
                       annotation_text=f"site avg {avg_lt:.2f}s", annotation_font_color="#9ca3af")
        fig5.update_layout(
            title="Page Load Times — slowest 25 pages (sorted slowest first)",
            xaxis_title="Load Time (seconds)", yaxis_title="",
            height=max(280, len(lt_df)*24),
            margin=dict(t=50, b=10), **DARK,
        )
        fig5.update_xaxes(gridcolor="#1e2a3a")
        st.plotly_chart(fig5, use_container_width=True)
        slow3 = sum(1 for d in lt_data if d["Load Time (s)"] > 3)
        chart_note(
            f"Site average: {avg_lt:.2f}s across {len(lt_data)} pages. "
            f"{slow3} page(s) exceed the 3s critical threshold."
        )
    else:
        st.info("No load time data was captured during this crawl.")

    # ── Table: worst pages ─────────────────────────────────────
    divider("WHICH INDIVIDUAL PAGES HAVE THE MOST ISSUES?")
    section_desc(
        "Pages with the highest total issue count — open these first during manual testing."
    )
    pic = (
        df.groupby("page_url")
          .agg(total_issues=("id","count"), high=("is_critical","sum"), open_=("is_open","sum"))
          .reset_index()
          .sort_values("total_issues", ascending=False)
          .head(10)
    )
    pic.columns = ["Page URL","Total Issues","High Severity","Open Issues"]
    st.dataframe(pic, use_container_width=True, hide_index=True)
    chart_note("Sorted by Total Issues descending. 'High Severity' = issues that should be fixed before the next release.")

    with st.expander("📋 View full raw issue dataset"):
        disp = df[["module","issue_type","severity","status","occurrences","fix_time_hours","description","source"]].copy()
        disp.columns = ["Module","Issue Type","Severity","Status","Occurrences","Est. Fix Time (h)","Description","Data Source"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption("'Data Source' shows which crawl signal generated each issue — e.g. crawl_seo, crawl_forms, crawl_accessibility.")


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — RISK SCORING
# ══════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Scoring":

    st.markdown("## ⚠️ Risk Scoring System")
    section_desc(
        "Each module gets a 0–100 Risk Score combining issue count (40%), severity (35%), and broken link rate (25%). "
        "The riskiest module is always 100; all others are scaled relative to it. Test high-scoring modules first."
    )

    if risk_df is None or (hasattr(risk_df,"empty") and risk_df.empty):
        st.warning("No risk data available — run a crawl first.")
        st.stop()

    # Formula box
    st.markdown("""
    <div class="info-box">
    <b>How the score is calculated (fully transparent — no black box):</b><br><br>
    <code>Raw Risk = (Issue Count × 0.4) + (Avg Severity Weight × 10 × 0.35) + (Broken Link Rate × 100 × 0.25)</code><br><br>
    <b>Why these weights?</b><br>
    · <b>Issue Count (40%)</b> — volume of problems is the strongest signal of how much work a module needs<br>
    · <b>Severity (35%)</b> — a few critical issues matter more than many low-level ones; weight: Low=1, Medium=2, High=3<br>
    · <b>Broken Links (25%)</b> — broken links directly harm users and SEO but are less common than general issues<br><br>
    <code>Final Score = (Raw Risk − min) ÷ (max − min) × 100</code>&nbsp; — normalised so the worst module always shows 100.
    </div>
    """, unsafe_allow_html=True)

    # ── Chart 1: Risk bar ─────────────────────────────────────
    divider("RISK SCORE PER MODULE")

    r_clrs = ["#ff4444" if l=="High" else "#ffaa00" if l=="Medium" else "#44ff44"
              for l in risk_df["Risk Level"]]
    fig_r = go.Figure(go.Bar(
        x=risk_df["Module"],
        y=risk_df["Risk Score (0-100)"],
        marker_color=r_clrs,
        text=[f"{v:.0f}" for v in risk_df["Risk Score (0-100)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Risk Score: %{y:.0f} / 100<extra></extra>",
    ))
    fig_r.add_hline(y=70, line_dash="dash", line_color="#ff4444", line_width=1,
                    annotation_text="≥70 High Risk", annotation_font_color="#ff4444")
    fig_r.add_hline(y=40, line_dash="dot",  line_color="#ffaa00", line_width=1,
                    annotation_text="≥40 Medium Risk", annotation_font_color="#ffaa00")
    fig_r.update_layout(
        title="Module Risk Score (0–100)",
        xaxis_title="Module", yaxis_title="Risk Score",
        xaxis_tickangle=-30, yaxis=dict(range=[0,118]),
        margin=dict(t=50, b=60), **DARK,
    )
    fig_r.update_yaxes(gridcolor="#1e2a3a")
    st.plotly_chart(fig_r, use_container_width=True)
    chart_note(
        "Score is relative — the riskiest module always scores 100; others are scaled against it. "
        "Red bars need immediate attention. Green bars are lower priority."
    )

    # ── Chart 2: Stacked components ───────────────────────────
    divider("WHAT IS DRIVING EACH MODULE'S RISK SCORE?")
    section_desc(
        "Blue = issue volume · Purple = severity · Red = broken links. "
        "A tall blue bar means many issues. A tall purple bar means the issues are serious. "
        "This tells you the root cause of each module's risk score."
    )

    comp_df = risk_df.copy()
    comp_df["Issue Volume"]  = (comp_df["Issue Count"] * 0.4).round(2)
    comp_df["Severity"]      = (comp_df["Avg Severity Weight"] * 10 * 0.35).round(2)
    comp_df["Broken Links"]  = (comp_df["Broken Link Rate"] * 100 * 0.25).round(2)

    fig_stk = go.Figure()
    for col_name, clr, tip in [
        ("Issue Volume",  "#00d4ff", "issue count × 0.4"),
        ("Severity",      "#7B2FBE", "avg severity weight × 10 × 0.35"),
        ("Broken Links",  "#ff4444", "broken link rate × 100 × 0.25"),
    ]:
        fig_stk.add_trace(go.Bar(
            name=col_name,
            x=comp_df["Module"],
            y=comp_df[col_name],
            marker_color=clr,
            hovertemplate=f"<b>%{{x}}</b><br>{col_name} ({tip}): %{{y:.2f}}<extra></extra>",
        ))
    fig_stk.update_layout(
        barmode="stack",
        title="Raw Risk — Component Breakdown per Module",
        xaxis_title="Module", yaxis_title="Contribution to Raw Score",
        xaxis_tickangle=-30,
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(t=60, b=60), **DARK,
    )
    fig_stk.update_yaxes(gridcolor="#1e2a3a")
    st.plotly_chart(fig_stk, use_container_width=True)
    chart_note("Blue = volume of issues · Purple = severity of those issues · Red = broken link contribution. Taller blue = more issues; taller purple = more serious issues.")

    # ── Detailed table ────────────────────────────────────────
    divider("FULL RISK TABLE — NUMBERS BEHIND EVERY SCORE")
    section_desc(
        "Issue Count = all issues found · Open = unresolved · High Severity = critical only · "
        "Broken Link Rate = broken links ÷ pages in module · Risk Score = final 0–100 value · "
        "Sample Size ⚠️ = fewer than 5 issues, score is directional only."
    )

    def _row_style(row):
        c = {
            "High":   "background-color:#2a0000;color:#ff6b6b",
            "Medium": "background-color:#2a1500;color:#ffa94d",
            "Low":    "background-color:#002a00;color:#69db7c",
        }.get(row["Risk Level"], "")
        return [c if col=="Risk Level" else "" for col in row.index]

    disp_r = risk_df[["Module","Issue Count","Open Issues","High Severity",
                       "Broken Link Rate","Risk Score (0-100)","Risk Level","Small Sample"]].copy()
    disp_r["Small Sample"]       = disp_r["Small Sample"].apply(lambda x: "⚠️ n<5" if x else "✅ OK")
    disp_r["Broken Link Rate"]   = disp_r["Broken Link Rate"].apply(lambda x: f"{x:.1%}")
    disp_r["Risk Score (0-100)"] = disp_r["Risk Score (0-100)"].apply(lambda x: f"{x:.1f}")
    st.dataframe(disp_r.style.apply(_row_style, axis=1), use_container_width=True, hide_index=True)

    small = risk_df[risk_df["Small Sample"] == True]
    if not small.empty:
        st.markdown(
            '<div class="warn-box">⚠️ <b>Small sample:</b> '
            + ", ".join(small["Module"].tolist())
            + " have fewer than 5 issues — their scores are directional only, not statistically reliable.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — PREDICTION MODEL
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Prediction Model":

    st.markdown("## 🤖 Prediction Model")
    section_desc(
        "A Random Forest (100 decision trees) trained on this site's issue data predicts whether any issue is Critical. "
        "Use the Live Prediction tool at the bottom to test a hypothetical issue instantly."
    )

    if not model_data or model_data.get("error"):
        st.warning(f"⚠️ {model_data.get('error','Model unavailable') if model_data else 'Run a crawl first.'}")
        st.stop()

    acc      = model_data.get("accuracy", 0)
    n_test   = len(model_data.get("y_test", []))
    confs    = model_data.get("confidence_scores", [])
    avg_conf = float(np.mean(confs)) if confs else 0.0

    # ── Model summary ─────────────────────────────────────────
    divider("MODEL PERFORMANCE AT A GLANCE")
    section_desc(
        "Performance on the 30% test split — issues the model never saw during training. "
        "Accuracy = % correctly labelled. Confidence = average probability score from predict_proba()."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",       f"{acc*100:.1f}%",       help="% of test issues correctly classified as Critical or Not Critical")
    c2.metric("Test Set Size",  f"n = {n_test}",         help="30% of total issues held back for unbiased evaluation")
    c3.metric("Avg Confidence", f"{avg_conf*100:.1f}%",  help="Average predict_proba() score on test samples — how certain the model is")
    c4.metric("Algorithm",      "Random Forest",         help="100 decision trees · max depth 5 · balanced class weights · seed 42")

    # ── Confusion matrix + Feature importance ─────────────────
    divider("CONFUSION MATRIX & FEATURE IMPORTANCE")

    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown("**Confusion Matrix** — how many predictions were correct vs wrong")
        st.caption("Rows = actual label · Columns = what the model predicted. Numbers on the diagonal = correct predictions.")
        cm     = model_data.get("confusion_matrix", [[0,0],[0,0]])
        labels = model_data.get("label_names", ["Not Critical","Critical"])
        fig_cm, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
            linewidths=.5, annot_kws={"size":14},
        )
        ax.set_xlabel("Predicted",  fontsize=10, color="white")
        ax.set_ylabel("Actual",     fontsize=10, color="white")
        ax.set_title(f"n = {n_test} test issues", color="#6b7280", fontsize=10)
        ax.tick_params(colors="white")
        fig_cm.patch.set_alpha(0); ax.set_facecolor("#0d1117")
        plt.tight_layout()
        st.pyplot(fig_cm); plt.close(fig_cm)
        chart_note("Large off-diagonal numbers = the model made mistakes there. Small → the model is accurate.")

    with col_fi:
        st.markdown("**Feature Importance** — which inputs drive the model's decisions")
        st.caption("Higher % = that feature influences the prediction more. Measured by Gini impurity reduction across all trees.")
        imp = model_data.get("feature_importance", {})
        feat_labels = {
            "module_enc":     "Module",
            "type_enc":       "Issue Type",
            "occurrences":    "Occurrences",
            "fix_time_hours": "Fix Time",
            "severity_weight":"Severity",
        }
        fi_df = pd.DataFrame(
            [(feat_labels.get(k,k), round(v*100,1)) for k,v in imp.items()],
            columns=["Feature","Importance (%)"]
        ).sort_values("Importance (%)")
        fig_fi = px.bar(
            fi_df, x="Importance (%)", y="Feature", orientation="h",
            color="Importance (%)", color_continuous_scale="Blues",
            title="Feature Importance (% of decision weight)",
            text_auto=".1f",
        )
        fig_fi.update_layout(**DARK, height=310, showlegend=False, margin=dict(t=50,b=10))
        fig_fi.update_xaxes(gridcolor="#1e2a3a")
        st.plotly_chart(fig_fi, use_container_width=True)
        chart_note("All bars sum to 100%. If 'Severity' dominates, the model leans heavily on how critical an issue is labelled.")

    # ── Classification report ─────────────────────────────────
    divider("CLASSIFICATION REPORT — PER-CLASS PERFORMANCE")
    section_desc(
        "Precision = of all predicted Critical, how many actually were. "
        "Recall = of all actual Critical issues, how many were caught. "
        "F1 = single metric combining both — above 0.70 is good. Support = test samples in that class."
    )
    report = model_data.get("classification_report", {})
    rows = [
        {"Class":     k,
         "Precision": f"{v.get('precision',0):.3f}",
         "Recall":    f"{v.get('recall',0):.3f}",
         "F1-Score":  f"{v.get('f1-score',0):.3f}",
         "Support":   int(v.get("support",0))}
        for k,v in report.items() if isinstance(v, dict)
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    chart_note("F1-Score above 0.70 is considered good for imbalanced datasets. 'weighted avg' accounts for class size.")

    # ── Live prediction ───────────────────────────────────────
    divider("LIVE PREDICTION TOOL")
    section_desc(
        "Fill in any issue parameters and click Predict. "
        "The model returns a Critical / Not Critical verdict plus the probability from all 100 trees."
    )

    mods = sorted(df["module"].unique().tolist()) if df is not None and not df.empty else ["Homepage"]
    typs = sorted(df["issue_type"].unique().tolist()) if df is not None and not df.empty else ["Broken Link"]

    p1, p2 = st.columns(2)
    with p1:
        sel_mod  = st.selectbox("Module",      mods, help="Which part of the site does this issue belong to?")
        sel_type = st.selectbox("Issue Type",  typs, help="Category of the issue (from the crawl-detected types)")
        sel_sev  = st.selectbox("Severity",    ["Low","Medium","High"], help="How severe do you consider this issue?")
    with p2:
        sel_occ  = st.slider("Occurrences",          1, 50,   1,    help="How many pages does this issue appear on?")
        sel_fix  = st.slider("Est. Fix Time (hours)", 0.25, 16.0, 2.0, step=0.25, help="How long would it realistically take to fix?")

    if st.button("🔮 Predict Criticality", type="primary"):
        res = predict_module_risk(model_data, sel_mod, sel_type, sel_occ, sel_fix, sel_sev)
        if res.get("error"):
            st.error(res["error"])
        else:
            pred = res["prediction"]
            conf = res["confidence"]
            pc   = res.get("proba_critical",     0)
            pnc  = res.get("proba_not_critical", 100)
            pcol = "#ff4444" if pred=="Critical" else "#44ff44"
            icon = "🚨" if pred=="Critical" else "✅"
            st.markdown(f"""
            <div class="card" style="border:2px solid {pcol}; margin-top:.8rem;">
              <div style="font-size:1.35rem;font-weight:700;color:{pcol};">{icon} {pred}</div>
              <div style="color:#9ca3af;margin-top:.4rem;font-size:.88rem;">
                Confidence: <b style="color:white;">{conf}%</b>
                &nbsp;·&nbsp;
                P(Critical): <b style="color:#ff6b6b;">{pc:.1f}%</b>
                &nbsp;·&nbsp;
                P(Not Critical): <b style="color:#69db7c;">{pnc:.1f}%</b>
              </div>
              <div style="color:#374151;font-size:.74rem;margin-top:.35rem;">
                Random Forest · 100 trees · seed 42 · trained on {len(df)} issues from this site's crawl
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — NLP ISSUE ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "💬 NLP Issue Analysis":

    st.markdown("## 💬 NLP Issue Analysis")
    section_desc(
        "Analyses the text of every issue description. "
        "TF-IDF surfaces the most distinctive words. KMeans groups issues by topic — "
        "so you can spot recurring problem themes without reading each issue individually."
    )

    if not nlp_data or nlp_data.get("error"):
        st.warning(f"⚠️ {nlp_data.get('error','Not enough issues for NLP') if nlp_data else 'Run a crawl first.'}")
        st.stop()

    # ── WordCloud + Top keywords ───────────────────────────────
    divider("WHAT WORDS DEFINE THIS SITE'S ISSUES?")

    col_wc, col_kw = st.columns([1,1])
    with col_wc:
        st.markdown("**WordCloud** — visual prominence of top terms")
        st.caption("Larger word = higher TF-IDF score. Words common to all documents are suppressed so only distinctive terms appear large.")
        wf = nlp_data.get("word_freq", {})
        if wf:
            try:
                wc_img = WordCloud(
                    width=620, height=340,
                    background_color="#0d1117",
                    colormap="cool", max_words=60,
                    prefer_horizontal=0.75,
                ).generate_from_frequencies({k: max(v*100,1) for k,v in wf.items()})
                fig_wc, ax = plt.subplots(figsize=(6.2,3.4))
                ax.imshow(wc_img, interpolation="bilinear"); ax.axis("off")
                fig_wc.patch.set_facecolor("#0d1117")
                fig_wc.tight_layout(pad=0)
                st.pyplot(fig_wc); plt.close(fig_wc)
            except Exception as e:
                st.info(f"WordCloud could not render: {e}")

    with col_kw:
        st.markdown("**Top 20 Keywords** — exact TF-IDF scores")
        st.caption("TF-IDF score = how frequently a word appears in this site's issues, offset by how common it is across all issues. Higher = more distinctive.")
        kws = nlp_data.get("top_keywords", [])
        if kws:
            kw_df = pd.DataFrame(kws, columns=["Keyword","TF-IDF Score"])
            fig_kw = px.bar(
                kw_df, x="TF-IDF Score", y="Keyword", orientation="h",
                color="TF-IDF Score", color_continuous_scale="Blues",
                title=f"Top 20 Keywords (from {len(df)} issue descriptions)",
                text_auto=".3f",
            )
            fig_kw.update_layout(**DARK, height=400, showlegend=False, margin=dict(t=50,b=10))
            fig_kw.update_xaxes(gridcolor="#1e2a3a")
            st.plotly_chart(fig_kw, use_container_width=True)

    # ── Clusters ──────────────────────────────────────────────
    divider("HOW ARE ISSUES GROUPED BY TOPIC?")

    clusters = nlp_data.get("clusters", {})
    n_cl     = nlp_data.get("n_clusters", 0)
    section_desc(
        f"KMeans (k={n_cl}) grouped all {len(df)} issues by description similarity. "
        "Each cluster = a distinct problem theme. The pie shows how issues are distributed across clusters."
    )

    cl_l, cl_r = st.columns([1,2])
    with cl_l:
        if clusters:
            cl_df = pd.DataFrame([
                {"Cluster": f"Cluster {cid+1}", "Count": cd["count"]}
                for cid, cd in clusters.items()
            ])
            fig_pie = px.pie(
                cl_df, names="Cluster", values="Count",
                title="Share of Issues per Cluster",
                hole=0.4,
            )
            fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
            fig_pie.update_layout(**DARK, margin=dict(t=50,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            chart_note("Each slice = one issue theme. A dominant slice means one problem type is very prevalent.")

    with cl_r:
        CLRS = ["#00d4ff","#7B2FBE","#ff6b6b","#69db7c","#ffa94d","#f783ac"]
        for cid, cd in clusters.items():
            clr     = CLRS[cid % len(CLRS)]
            kw_str  = " &nbsp;".join(f"<code>{k}</code>" for k in cd["keywords"][:5])
            sev_str = " · ".join(f"<b>{s}</b>: {c}" for s,c in cd["severities"].items())
            mod_str = " · ".join(f"{m}: {c}" for m,c in list(cd["modules"].items())[:3])
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {clr}; margin-bottom:.5rem;">
              <div style="font-size:.85rem;font-weight:700;color:{clr};">
                {cd['label']}
                <span style="font-size:.72rem;color:#4b5563;font-weight:400;"> · {cd['count']} issues</span>
              </div>
              <div style="font-size:.78rem;color:#6b7280;margin-top:.3rem;line-height:1.6;">
                🔑 Top terms: {kw_str}<br>
                ⚠️ {sev_str}<br>
                📦 {mod_str}
              </div>
            </div>
            """, unsafe_allow_html=True)
            if cd["count"] < 5:
                st.markdown('<div class="warn-box">⚠️ Fewer than 5 issues — this cluster may not represent a real pattern.</div>', unsafe_allow_html=True)

    divider("EXAMPLE ISSUES FROM EACH CLUSTER")
    section_desc("Expand any cluster to read the exact issue descriptions the algorithm grouped together.")
    icons = "🔵🟣🔴🟢🟡🩷"
    for cid, cd in clusters.items():
        with st.expander(f"{icons[cid % len(icons)]} {cd['label']} — {cd['count']} issue(s)"):
            for ex in cd["examples"]:
                st.markdown(f"- {ex}")
            st.caption("These descriptions come directly from crawl findings — no text was fabricated.")


# ══════════════════════════════════════════════════════════════
#  PAGE 6 — FIX VALIDATION & RECS
# ══════════════════════════════════════════════════════════════
elif page == "✅ Fix Validation & Recs":

    st.markdown("## ✅ Fix Validation & Recommendations")
    section_desc(
        "Actionable output: which issues will reappear, what to fix and in what order, "
        "which modules to test first, and how this site compares to published standards."
    )

    if not recs:
        st.warning("Run a crawl first.")
        st.stop()

    # ── Recurrence flags ──────────────────────────────────────
    rec_flags = recs.get("recurrence_flags", [])
    if rec_flags:
        divider("ISSUES THAT WILL COME BACK UNLESS THE ROOT CAUSE IS FIXED")
        section_desc(
            "These issues come back because the underlying process is missing, not because someone forgot to fix them. "
            "Each one needs a preventive measure, not just a one-time patch."
        )
        for flag in rec_flags:
            cls = "risk-high" if flag["risk"]=="High" else "risk-medium"
            st.markdown(f"""
            <div class="{cls}" style="margin-bottom:.5rem;">
              <div style="font-size:.88rem;font-weight:700;">{flag['issue_type']} &nbsp; {pill(flag['risk'])}</div>
              <div style="font-size:.8rem;color:#9ca3af;margin-top:.2rem;">📌 Why it recurs: {flag['reason']}</div>
              <div style="font-size:.75rem;color:#6b7280;margin-top:.1rem;">📅 Prevention cadence: {flag['frequency']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────
    divider("WHAT TO FIX — SORTED BY PRIORITY")
    section_desc(
        f"All {len(recs.get('recommendations',[]))} items below cite real numbers from the crawl. "
        "Priority 1 = fix first. Effort = how hard to implement. Impact = benefit when done."
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
                        padding:.85rem 1.1rem;border-radius:0 8px 8px 0;">
              <div style="font-size:.83rem;color:#9ca3af;line-height:1.65;">{rec['detail']}</div>
              <div style="font-size:.76rem;color:#4b5563;margin-top:.55rem;">
                Effort: <b style="color:#d1d5db;">{ei}</b>
                &nbsp;·&nbsp;
                Impact: <b style="color:#d1d5db;">{ii}</b>
                &nbsp;·&nbsp;
                Category: <b style="color:{color};">{cat}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Testing priority ──────────────────────────────────────
    divider("WHICH MODULES TO TEST FIRST")
    section_desc(
        "Ordered by Risk Score. Each card specifies exactly what a tester should verify in that module."
    )
    pt = recs.get("priority_tests", [])
    if pt:
        for i, t in enumerate(pt, 1):
            lc = "#ff4444" if t["risk_level"]=="High" else "#ffaa00" if t["risk_level"]=="Medium" else "#44ff44"
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {lc}; margin-bottom:.5rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:.9rem;font-weight:700;color:{lc};">#{i} &nbsp; {t['module']}</span>
                <span style="font-size:.8rem;">
                  {pill(t['risk_level'])} &nbsp;
                  Risk Score: <b style="color:{lc};">{t['risk_score']:.0f} / 100</b>
                </span>
              </div>
              <div style="font-size:.8rem;color:#6b7280;margin-top:.3rem;">
                📋 What to test: {t['test_focus']}<br>
                <span style="font-size:.72rem;color:#374151;">Open issues in this module: {t['open_issues']}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No priority test data — the risk scoring page may have insufficient data.")

    # ── Benchmark comparison ───────────────────────────────────
    divider("HOW DOES THIS SITE COMPARE TO INDUSTRY STANDARDS?")
    section_desc(
        "Measured values vs published benchmarks: Google Core Web Vitals (load time), "
        "WCAG 2.1 SC 1.1.1 (alt text), SEO best practice (meta descriptions), and mobile standards (viewport tag)."
    )
    bm = recs.get("benchmarks", {})
    if bm:
        bm_df = pd.DataFrame([
            {"Metric": k,
             "This Site": v["yours"],
             "Industry Standard": v["benchmark"],
             "Result": v["status"]}
            for k,v in bm.items()
        ])
        st.dataframe(bm_df, use_container_width=True, hide_index=True)
        chart_note("✅ meets standard · ⚠️ needs improvement · ❌ below standard. All values measured from live crawl data.")

    # ── Export ────────────────────────────────────────────────
    divider("EXPORT")
    if df is not None and not df.empty:
        section_desc("Full issue dataset as CSV — ready to import into Jira, Linear, Notion, or any spreadsheet.")
        csv = df[["module","issue_type","severity","status","occurrences",
                  "fix_time_hours","description","source","page_url"]].to_csv(index=False)
        st.download_button(
            "⬇️  Download Issues CSV",
            csv,
            file_name=f"issues_{crawl_data.get('base_domain','site')}.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption(f"{len(df)} rows · columns: Module, Issue Type, Severity, Status, Occurrences, Est. Fix Time, Description, Data Source, Page URL")
