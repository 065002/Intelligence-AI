"""
app.py — Intelligent App Testing System
Streamlit application — all 6 analysis pages.
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

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent App Testing System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main-title{font-family:'Space Mono',monospace;font-size:2.1rem;font-weight:700;
  background:linear-gradient(135deg,#00d4ff,#7B2FBE);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;margin-bottom:.2rem;}
.subtitle{color:#888;font-size:.95rem;margin-bottom:1.4rem;}
.info-box{background:#0d1117;border:1px solid #30363d;border-radius:8px;
  padding:1rem;font-size:.85rem;color:#8b949e;}
.risk-high{background:#3d0000;border-left:4px solid #ff4444;padding:.5rem 1rem;border-radius:4px;margin-bottom:.4rem;}
.risk-medium{background:#3d2600;border-left:4px solid #ffaa00;padding:.5rem 1rem;border-radius:4px;margin-bottom:.4rem;}
.risk-low{background:#003d00;border-left:4px solid #44ff44;padding:.5rem 1rem;border-radius:4px;margin-bottom:.4rem;}
.warning-box{background:#2d2000;border:1px solid #ffaa00;border-radius:6px;
  padding:.6rem 1rem;color:#ffcc44;font-size:.85rem;margin:.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:1rem 0;'>
      <div style='font-family:Space Mono,monospace;font-size:1.3rem;color:#00d4ff;'>🔬 AppTester</div>
      <div style='color:#555;font-size:.75rem;'>{APP_VERSION}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("📍 Navigate", [
        "🌐 Website Overview",
        "📊 Exploratory Analysis",
        "⚠️ Risk Scoring",
        "🤖 Prediction Model",
        "💬 NLP Issue Analysis",
        "✅ Fix Validation & Recs",
    ], label_visibility="collapsed")
    st.markdown("---")
    respect_robots = st.checkbox("Respect robots.txt", value=True)
    st.caption("Uncheck to crawl sites that restrict bots.")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.75rem;color:#555;'>
    <b>Risk formula:</b><br>
    (Issues×0.4)+(Severity×0.35)+(BrokenLinks×0.25)<br>
    Normalised 0–100<br><br>
    <b>ML:</b> Random Forest (100 trees)<br>
    <b>NLP:</b> TF-IDF + KMeans<br>
    <b>Max crawl:</b> 50 pages
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 Intelligent App Testing System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste any URL · Auto-crawl · Full AI-powered analysis · Actionable insights</div>', unsafe_allow_html=True)

# ── URL input ─────────────────────────────────────────────────
col_u, col_b = st.columns([5, 1])
with col_u:
    input_url = st.text_input("URL", placeholder="https://example.com", label_visibility="collapsed")
with col_b:
    run_btn = st.button("🚀 Analyze", use_container_width=True, type="primary")
st.markdown("---")

# ── Session state ─────────────────────────────────────────────
for key in ["crawl_data", "issues", "df", "risk_df", "health",
            "model_data", "nlp_data", "recs", "analyzed_url"]:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ("issues",) else []

# ── Run crawl ─────────────────────────────────────────────────
if run_btn and input_url.strip():
    valid, clean_url, err = validate_url(input_url.strip())
    if not valid:
        st.error(f"❌ {err}")
    else:
        st.session_state["analyzed_url"] = clean_url
        prog = st.progress(0, text="Initialising crawler…")

        def cb(cur, tot, msg):
            prog.progress(min(int(cur / max(tot, 1) * 95), 95), text=f"🕷️ {msg}")

        crawl_data = crawl_website(clean_url, respect_robots=respect_robots, progress_callback=cb)

        if crawl_data.get("error"):
            prog.empty()
            err_map = {
                "robots_blocked": "🚫 Blocked by robots.txt — uncheck in sidebar to override.",
                "timeout": "⏱️ Site timed out.",
                "ssl_error": "🔒 SSL error — try http:// instead.",
                "connection_error": "❌ Cannot connect — check the URL.",
            }
            st.error(err_map.get(crawl_data["error"], crawl_data.get("message", "Unknown error.")))
        else:
            prog.progress(95, text="Generating issue dataset…")
            issues = generate_issues_from_crawl(crawl_data)
            df = build_issues_df(issues)
            prog.progress(97, text="Running analysis pipeline…")
            health   = compute_health_score(crawl_data, df)
            risk_df  = compute_module_risk(df, crawl_data) if not df.empty else pd.DataFrame()
            mdl      = train_prediction_model(df) if not df.empty else {"error": "No data"}
            nlp      = run_nlp_analysis(df) if not df.empty else {"error": "No data"}
            recs     = generate_recommendations(df, crawl_data, risk_df)

            st.session_state.update({
                "crawl_data": crawl_data, "issues": issues, "df": df,
                "risk_df": risk_df, "health": health, "model_data": mdl,
                "nlp_data": nlp, "recs": recs,
            })
            prog.progress(100, text="✅ Done!")
            time.sleep(0.4)
            prog.empty()
            st.success(
                f"✅ Crawled **{crawl_data['total_pages']}** pages in "
                f"**{crawl_data['crawl_time']}s** — **{len(issues)}** issues found."
            )

# ── Guard ─────────────────────────────────────────────────────
if st.session_state["crawl_data"] is None:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;color:#444;'>
      <div style='font-size:3rem;'>🌐</div>
      <div style='font-size:1.2rem;margin-top:1rem;'>
        Enter a website URL above and click <b>Analyze</b>
      </div>
      <div style='font-size:.9rem;margin-top:.5rem;color:#333;'>
        The system crawls the site, extracts real issues, and runs full AI-powered analysis.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Unpack session
crawl_data  = st.session_state["crawl_data"]
issues      = st.session_state["issues"]
df          = st.session_state["df"]
risk_df     = st.session_state["risk_df"]
health      = st.session_state["health"]
model_data  = st.session_state["model_data"]
nlp_data    = st.session_state["nlp_data"]
recs        = st.session_state["recs"]
pages       = crawl_data.get("pages", [])

DARK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — WEBSITE OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🌐 Website Overview":
    st.markdown("## 🌐 Website Overview")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        The crawler visits up to 50 pages starting from the submitted URL using
        `requests` + `BeautifulSoup`. For each page it records: title, meta tags,
        headings, forms, images, links, load time, and HTTP status.
        Issues are derived **only** from actual findings — nothing is fabricated.
        """)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌐 Pages",        crawl_data.get("total_pages", 0))
    c2.metric("⚠️ Issues",       len(issues))
    c3.metric("📦 Modules",      len(crawl_data.get("modules", [])))
    c4.metric("🔗 Broken Links", len(crawl_data.get("broken_links", [])))
    c5.metric("⏱️ Crawl Time",   f"{crawl_data.get('crawl_time', 0)}s")

    st.markdown("---")
    col_g, col_b2 = st.columns([1, 2])

    with col_g:
        score = health.get("score", 0)
        color = health.get("color", "gray")
        grade = health.get("grade", "?")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Grade: {grade}", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35],  "color": "#3d0000"},
                    {"range": [35, 65], "color": "#3d2600"},
                    {"range": [65, 100],"color": "#003d00"},
                ],
            },
        ))
        fig.update_layout(height=270, margin=dict(t=30, b=0, l=20, r=20), **DARK)
        st.markdown("### Overall Health Score")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="info-box"><b>Formula (Derived):</b> Weighted composite — '
            'Issues (25%), Broken Links (20%), Performance (20%), SEO (15%), '
            'Accessibility (10%), Meta Tags (5%), Mobile (5%)</div>',
            unsafe_allow_html=True,
        )

    with col_b2:
        breakdown = health.get("breakdown", {})
        if breakdown:
            vals = list(breakdown.values())
            cats = list(breakdown.keys())
            colors = ["#44ff44" if v >= 70 else "#ffaa00" if v >= 40 else "#ff4444" for v in vals]
            fig2 = go.Figure(go.Bar(
                x=vals, y=cats, orientation="h",
                marker_color=colors,
                text=[f"{v:.0f}" for v in vals], textposition="inside",
            ))
            fig2.update_layout(xaxis=dict(range=[0, 100]), title="Category Scores",
                               height=270, margin=dict(t=40, b=20), **DARK)
            st.markdown("### Score Breakdown")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗺️ Website Structure Map")
    module_groups = {}
    for p in pages:
        m = p.get("module", "Unknown")
        module_groups.setdefault(m, []).append(p)

    cols = st.columns(min(len(module_groups), 4))
    for i, (mod, mpages) in enumerate(module_groups.items()):
        col = cols[i % len(cols)]
        lts = [p["load_time"] for p in mpages if p.get("load_time", 0) > 0]
        avg_lt = float(np.mean(lts)) if lts else 0.0
        errs = sum(1 for p in mpages if p.get("status_code") in (404, 500, 502, 503) or p.get("error"))
        with col:
            st.markdown(f"""
            <div style='background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;
                        padding:.8rem;margin-bottom:.5rem;'>
              <b style='color:#00d4ff;'>{mod}</b><br>
              <span style='color:#888;font-size:.8rem;'>
                {len(mpages)} page(s) · {avg_lt:.2f}s avg · {errs} error(s)
              </span>
            </div>
            """, unsafe_allow_html=True)
            for p in mpages[:3]:
                sc = p.get("status_code", "?")
                sc_col = "#44ff44" if sc == 200 else "#ff4444"
                short = p.get("url", "")[-40:]
                st.markdown(
                    f"<div style='font-size:.72rem;color:#555;padding-left:.5rem;'>"
                    f"• <span style='color:{sc_col};'>[{sc}]</span> …{short}</div>",
                    unsafe_allow_html=True,
                )
            if len(mpages) > 3:
                st.markdown(
                    f"<div style='font-size:.7rem;color:#444;padding-left:.5rem;'>"
                    f"+ {len(mpages)-3} more…</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("### 📋 Crawl Summary")
    st.dataframe(pd.DataFrame({
        "Metric": ["Base URL", "Pages Crawled", "Modules Detected", "Broken Links",
                   "External Links Checked", "Total Issues", "Crawl Time", "robots.txt Respected"],
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
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.markdown("## 📊 Exploratory Data Analysis")
    with st.expander("ℹ️ How this works"):
        st.markdown("All charts are built from issues generated by the crawler. "
                    "Every issue is traceable to a specific crawl finding.")

    if df.empty:
        st.warning("No issues found — the site may be clean or the crawl was limited.")
        st.stop()

    st.caption(f"📊 **{len(df)} issues** across **{df['module'].nunique()} modules**")

    c1, c2 = st.columns(2)
    with c1:
        mc = df.groupby("module").size().reset_index(name="count").sort_values("count", ascending=False)
        fig = px.bar(mc, x="module", y="count", color="count", color_continuous_scale="Blues",
                     title=f"Issues by Module (n={len(df)})",
                     labels={"module": "Module", "count": "Issues"})
        fig.update_layout(**DARK, xaxis_tickangle=-30, showlegend=False)
        fig.update_traces(texttemplate="%{y}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sc = df["severity"].value_counts().reset_index()
        sc.columns = ["severity", "count"]
        fig2 = px.pie(sc, names="severity", values="count", hole=0.4,
                      color="severity",
                      color_discrete_map={"High":"#ff4444","Medium":"#ffaa00","Low":"#44ff44"},
                      title=f"Severity Distribution (n={len(df)})")
        fig2.update_layout(**DARK)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        tc = df["issue_type"].value_counts().head(12).reset_index()
        tc.columns = ["issue_type", "count"]
        fig3 = px.bar(tc, x="count", y="issue_type", orientation="h",
                      color="count", color_continuous_scale="Reds",
                      title="Top Issue Types", labels={"issue_type":"","count":"Count"})
        fig3.update_layout(**DARK, height=350, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        stc = df["status"].value_counts().reset_index()
        stc.columns = ["status", "count"]
        fig4 = px.bar(stc, x="status", y="count", color="status",
                      color_discrete_map={"Open":"#ff4444","Fixed":"#44ff44","Reopened":"#ffaa00"},
                      title="Status Distribution")
        fig4.update_layout(**DARK, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### ⏱️ Page Load Times")
    lt_data = [{"url": p.get("url","")[-50:], "load_time": p.get("load_time",0),
                "module": p.get("module","?")}
               for p in pages if p.get("load_time",0) and p["load_time"] > 0]
    if lt_data:
        lt_df = pd.DataFrame(lt_data).sort_values("load_time", ascending=False).head(20)
        colors = ["#ff4444" if t > 3 else "#ffaa00" if t > 1.5 else "#44ff44"
                  for t in lt_df["load_time"]]
        fig5 = go.Figure(go.Bar(
            x=lt_df["load_time"], y=lt_df["url"], orientation="h",
            marker_color=colors,
            text=[f"{t:.2f}s" for t in lt_df["load_time"]], textposition="outside",
        ))
        fig5.add_vline(x=3.0, line_dash="dash", line_color="red", annotation_text="3s threshold")
        fig5.add_vline(x=1.5, line_dash="dot", line_color="orange", annotation_text="1.5s")
        fig5.update_layout(title="Slowest Pages (top 20)", xaxis_title="Seconds",
                           height=max(250, len(lt_df)*22), **DARK)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### 🔥 Most Problematic Pages")
    pic = (df.groupby("page_url")
             .agg(total_issues=("id","count"), high=("is_critical","sum"), open_=("is_open","sum"))
             .reset_index()
             .sort_values("total_issues", ascending=False)
             .head(10))
    pic.columns = ["Page URL","Total Issues","High Severity","Open Issues"]
    st.dataframe(pic, use_container_width=True, hide_index=True)

    with st.expander("📋 Raw Issue Dataset"):
        disp = df[["module","issue_type","severity","status","occurrences",
                   "fix_time_hours","description","source"]].copy()
        disp.columns = ["Module","Issue Type","Severity","Status","Occurrences",
                        "Fix Time (h)","Description","Source"]
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — RISK SCORING
# ══════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Scoring":
    st.markdown("## ⚠️ Risk Scoring System")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        ```
        Raw Risk = (Issue Count × 0.4)
                 + (Avg Severity Weight × 10 × 0.35)
                 + (Broken Link Rate × 100 × 0.25)
        Final    = Normalised to 0–100
        ```
        Severity weights: Low=1, Medium=2, High=3.
        Modules with < 5 issues are flagged ⚠️.
        """)

    if risk_df is None or (hasattr(risk_df, "empty") and risk_df.empty):
        st.warning("No risk data — run a crawl first.")
        st.stop()

    st.markdown("""
    <div class="info-box">
    <b>Formula:</b>
    Risk = <code>(Issues×0.4) + (Severity×0.35) + (BrokenLinks×0.25)</code> → Normalised 0–100
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    def row_style(row):
        c = {"High":"background-color:#3d0000;color:#ff4444",
             "Medium":"background-color:#3d2600;color:#ffaa00",
             "Low":"background-color:#003d00;color:#44ff44"}.get(row["Risk Level"],"")
        return [c if col=="Risk Level" else "" for col in row.index]

    disp_r = risk_df[["Module","Issue Count","Open Issues","High Severity",
                       "Broken Link Rate","Risk Score (0-100)","Risk Level","Small Sample"]].copy()
    disp_r["Small Sample"] = disp_r["Small Sample"].apply(lambda x: "⚠️ n<5" if x else "✅ OK")
    disp_r["Broken Link Rate"] = disp_r["Broken Link Rate"].apply(lambda x: f"{x:.1%}")
    st.dataframe(disp_r.style.apply(row_style, axis=1), use_container_width=True, hide_index=True)

    colors = ["#ff4444" if l=="High" else "#ffaa00" if l=="Medium" else "#44ff44"
              for l in risk_df["Risk Level"]]
    fig = go.Figure(go.Bar(
        x=risk_df["Module"], y=risk_df["Risk Score (0-100)"],
        marker_color=colors,
        text=[f"{v:.0f}" for v in risk_df["Risk Score (0-100)"]],
        textposition="outside",
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="High (70)")
    fig.add_hline(y=40, line_dash="dot",  line_color="orange",annotation_text="Medium (40)")
    fig.update_layout(title="Module Risk Scores (0–100)",
                      xaxis_title="Module", yaxis_title="Risk Score",
                      xaxis_tickangle=-30, yaxis=dict(range=[0,115]),
                      **DARK)
    st.plotly_chart(fig, use_container_width=True)

    small = risk_df[risk_df["Small Sample"] == True]
    if not small.empty:
        st.markdown(
            '<div class="warning-box">⚠️ <b>Small sample warning</b> — fewer than 5 issues: '
            + ", ".join(small["Module"].tolist()) + "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# PAGE 4 — ML PREDICTION MODEL
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Prediction Model":
    st.markdown("## 🤖 Prediction Model")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **Algorithm:** Random Forest (100 trees, max depth 5, balanced class weights)  
        **Target:** Critical (High severity) vs Not Critical  
        **Features:** Module, Issue Type, Occurrences, Fix Time, Severity Weight  
        **Split:** 70/30 train/test, stratified, seed=42  
        **Confidence:** from `predict_proba()` — shown on every prediction
        """)

    if not model_data or model_data.get("error"):
        st.warning(f"⚠️ {model_data.get('error','No model') if model_data else 'No model'}")
        st.stop()

    acc = model_data.get("accuracy", 0)
    n_test = len(model_data.get("y_test", []))
    confs = model_data.get("confidence_scores", [])
    avg_conf = float(np.mean(confs)) if confs else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",      f"{acc*100:.1f}%")
    c2.metric("Test Set",      f"n={n_test}")
    c3.metric("Avg Confidence",f"{avg_conf*100:.1f}%")
    c4.metric("Algorithm",     "Random Forest")
    st.markdown("---")

    col_cm, col_imp = st.columns(2)
    with col_cm:
        st.markdown("### Confusion Matrix")
        cm = model_data.get("confusion_matrix", [[0,0],[0,0]])
        labels = model_data.get("label_names", ["Not Critical","Critical"])
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {n_test} test samples")
        fig_cm.patch.set_alpha(0); ax.set_facecolor("#0d1117")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_imp:
        st.markdown("### Feature Importance")
        imp = model_data.get("feature_importance", {})
        feat_labels = {"module_enc":"Module","type_enc":"Issue Type",
                       "occurrences":"Occurrences","fix_time_hours":"Fix Time",
                       "severity_weight":"Severity"}
        fi_df = pd.DataFrame(
            [(feat_labels.get(k, k), v) for k, v in imp.items()],
            columns=["Feature","Importance"]
        ).sort_values("Importance")
        fig_i = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                       color="Importance", color_continuous_scale="Blues",
                       title="Feature Importance (Gini)")
        fig_i.update_layout(**DARK, height=300, showlegend=False)
        st.plotly_chart(fig_i, use_container_width=True)

    st.markdown("### Classification Report")
    report = model_data.get("classification_report", {})
    rows = [{"Class": k, "Precision": f"{v.get('precision',0):.3f}",
              "Recall": f"{v.get('recall',0):.3f}",
              "F1": f"{v.get('f1-score',0):.3f}",
              "Support": int(v.get("support",0))}
             for k, v in report.items() if isinstance(v, dict)]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🎯 Live Prediction")
    mods = sorted(df["module"].unique().tolist()) if not df.empty else ["Homepage"]
    typs = sorted(df["issue_type"].unique().tolist()) if not df.empty else ["Broken Link"]

    p1, p2 = st.columns(2)
    with p1:
        sel_mod  = st.selectbox("Module",      mods)
        sel_type = st.selectbox("Issue Type",  typs)
    with p2:
        sel_occ  = st.slider("Occurrences", 1, 50, 1)
        sel_fix  = st.slider("Fix Time (hours)", 0.25, 16.0, 2.0, step=0.25)
        sel_sev  = st.selectbox("Severity", ["Low","Medium","High"])

    if st.button("🔮 Predict Criticality", type="primary"):
        res = predict_module_risk(model_data, sel_mod, sel_type, sel_occ, sel_fix, sel_sev)
        if res.get("error"):
            st.error(res["error"])
        else:
            pred = res["prediction"]
            conf = res["confidence"]
            pc   = res.get("proba_critical", 0)
            pnc  = res.get("proba_not_critical", 100)
            col  = "#ff4444" if pred == "Critical" else "#44ff44"
            icon = "🚨" if pred == "Critical" else "✅"
            st.markdown(f"""
            <div style='background:#1a1a2e;border:2px solid {col};
                        border-radius:10px;padding:1.5rem;margin-top:1rem;'>
              <div style='font-size:1.5rem;font-weight:700;color:{col};'>{icon} {pred}</div>
              <div style='color:#888;margin-top:.3rem;'>
                Confidence: <b style='color:white;'>{conf}%</b> &nbsp;|&nbsp;
                P(Critical): <b style='color:#ff4444;'>{pc:.1f}%</b> &nbsp;|&nbsp;
                P(Not Critical): <b style='color:#44ff44;'>{pnc:.1f}%</b>
              </div>
              <div style='color:#555;font-size:.8rem;margin-top:.5rem;'>
                Random Forest · seed=42 · trained on {len(df)} samples
              </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — NLP
# ══════════════════════════════════════════════════════════════
elif page == "💬 NLP Issue Analysis":
    st.markdown("## 💬 NLP Issue Analysis")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **TF-IDF** identifies statistically significant keywords across all issue descriptions.
        **KMeans** clusters similar issues together.
        **WordCloud** shows keyword prominence visually.
        All text comes from real crawl-derived issue descriptions.
        """)

    if not nlp_data or nlp_data.get("error"):
        st.warning(f"⚠️ {nlp_data.get('error','Not enough data') if nlp_data else 'No data'}")
        st.stop()

    col_wc, col_kw = st.columns(2)
    with col_wc:
        st.markdown("### ☁️ WordCloud")
        wf = nlp_data.get("word_freq", {})
        if wf:
            try:
                wc = WordCloud(
                    width=600, height=320,
                    background_color="#0d1117",
                    colormap="cool", max_words=50,
                ).generate_from_frequencies({k: max(v*100, 1) for k, v in wf.items()})
                fig_wc, ax = plt.subplots(figsize=(6, 3.2))
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                fig_wc.patch.set_facecolor("#0d1117")
                st.pyplot(fig_wc)
                plt.close(fig_wc)
            except Exception as e:
                st.info(f"WordCloud unavailable: {e}")
        st.caption("Source: TF-IDF on crawl-derived issue descriptions")

    with col_kw:
        st.markdown("### 📊 Top Keywords (TF-IDF)")
        kws = nlp_data.get("top_keywords", [])
        if kws:
            kw_df = pd.DataFrame(kws, columns=["Keyword","Score"])
            fig_k = px.bar(kw_df, x="Score", y="Keyword", orientation="h",
                           color="Score", color_continuous_scale="Blues",
                           title=f"Top Keywords (n={len(df)} descriptions)")
            fig_k.update_layout(**DARK, height=380, showlegend=False)
            st.plotly_chart(fig_k, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔵 KMeans Issue Clusters")
    clusters = nlp_data.get("clusters", {})
    n_cl = nlp_data.get("n_clusters", 0)
    st.caption(f"k={n_cl} clusters · {len(df)} issues")

    CLRS = ["#00d4ff","#7B2FBE","#ff4444","#44ff44","#ffaa00","#ff88aa"]
    for cid, cd in clusters.items():
        with st.expander(f"🔵 {cd['label']} — {cd['count']} issue(s)"):
            st.markdown("**Keywords:** " + " · ".join(f"`{k}`" for k in cd["keywords"]))
            st.markdown("**Severities:** " + " | ".join(f"{s}: {c}" for s, c in cd["severities"].items()))
            st.markdown("**Modules:** " + " | ".join(f"{m}: {c}" for m, c in list(cd["modules"].items())[:4]))
            if cd["count"] < 5:
                st.markdown('<div class="warning-box">⚠️ Fewer than 5 issues — patterns may not be representative.</div>', unsafe_allow_html=True)
            st.markdown("**Representative Examples:**")
            for ex in cd["examples"]:
                st.markdown(f"- {ex}")

    if clusters:
        cl_df = pd.DataFrame([{"Cluster": cd["label"], "Count": cd["count"]} for cd in clusters.values()])
        fig_cl = px.pie(cl_df, names="Cluster", values="Count", title="Issues by Cluster", hole=0.3)
        fig_cl.update_layout(**DARK)
        st.plotly_chart(fig_cl, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 6 — FIX VALIDATION & RECS
# ══════════════════════════════════════════════════════════════
elif page == "✅ Fix Validation & Recs":
    st.markdown("## ✅ Fix Validation & Recommendations")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        **Recurrence Flags** — issues likely to reappear based on type and frequency.  
        **Recommendations** — derived exclusively from actual crawl findings; each cites real numbers.  
        **Benchmarks** — compared against Google Core Web Vitals, WCAG 2.1, SEO best practices.  
        **Priority Tests** — ranked by risk score from Page 3.
        """)

    if not recs:
        st.warning("Run a crawl first.")
        st.stop()

    # Recurrence
    rec_flags = recs.get("recurrence_flags", [])
    if rec_flags:
        st.markdown("### 🔄 Issues Likely to Reappear")
        for flag in rec_flags:
            cls = "risk-high" if flag["risk"]=="High" else "risk-medium"
            st.markdown(f"""
            <div class="{cls}">
              <b>{flag['issue_type']}</b> — Risk: {flag['risk']}<br>
              <span style='font-size:.85rem;color:#aaa;'>{flag['reason']}</span><br>
              <span style='font-size:.8rem;color:#888;'>📅 {flag['frequency']}</span>
            </div>
            """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("### 💡 Actionable Recommendations")
    st.caption(f"All {len(recs.get('recommendations', []))} recs derived from actual crawl findings")
    CAT_COLORS = {
        "Critical":"#ff4444","Performance":"#ff8800","Accessibility":"#00aaff",
        "SEO":"#aa44ff","Social/SEO":"#aa44ff","Mobile":"#44ffaa",
        "Security":"#ffaa00","Monitoring":"#44ff88","Info":"#888",
    }
    EFFORT_ICON = {"High":"🔴","Medium":"🟡","Low":"🟢"}
    for rec in recs.get("recommendations", []):
        cat   = rec.get("category","General")
        color = CAT_COLORS.get(cat,"#888")
        ei    = EFFORT_ICON.get(rec.get("effort",""),"")
        ii    = EFFORT_ICON.get(rec.get("impact",""),"")
        with st.expander(f"#{rec['priority']} [{cat}] {rec['title']}"):
            st.markdown(f"""
            <div style='background:#1a1a2e;border-left:3px solid {color};
                        padding:1rem;border-radius:0 8px 8px 0;'>
              <b style='color:{color};'>{rec['title']}</b><br><br>
              {rec['detail']}<br><br>
              <span style='font-size:.8rem;color:#888;'>
                {ei} Effort: <b>{rec.get('effort','?')}</b> &nbsp;|&nbsp;
                {ii} Impact: <b>{rec.get('impact','?')}</b>
              </span>
            </div>
            """, unsafe_allow_html=True)

    # Priority test list
    st.markdown("---")
    st.markdown("### 🎯 Testing Priority List (by Risk Score)")
    pt = recs.get("priority_tests", [])
    if pt:
        for i, t in enumerate(pt, 1):
            lc = "#ff4444" if t["risk_level"]=="High" else "#ffaa00" if t["risk_level"]=="Medium" else "#44ff44"
            st.markdown(f"""
            <div style='background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;
                        padding:.8rem;margin-bottom:.5rem;'>
              <b style='color:{lc};'>#{i} {t['module']}</b>
              <span style='float:right;color:{lc};'>
                Risk: {t['risk_score']:.0f}/100 ({t['risk_level']})
              </span><br>
              <span style='color:#888;font-size:.85rem;'>📋 {t['test_focus']}</span><br>
              <span style='color:#555;font-size:.8rem;'>Open issues: {t['open_issues']}</span>
            </div>
            """, unsafe_allow_html=True)

    # Benchmarks
    st.markdown("---")
    st.markdown("### 📏 Benchmark Comparison")
    bm = recs.get("benchmarks", {})
    if bm:
        st.dataframe(
            pd.DataFrame([{"Metric":k,"Your Site":v["yours"],
                           "Benchmark":v["benchmark"],"Status":v["status"]}
                          for k, v in bm.items()]),
            use_container_width=True, hide_index=True,
        )

    # Export
    st.markdown("---")
    if not df.empty:
        csv = df[["module","issue_type","severity","status","occurrences",
                  "fix_time_hours","description","source","page_url"]].to_csv(index=False)
        st.download_button(
            "⬇️ Download Issues CSV", csv,
            f"issues_{crawl_data.get('base_domain','site')}.csv",
            "text/csv", type="primary",
        )
