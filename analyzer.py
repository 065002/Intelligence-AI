"""
analyzer.py
Intelligent App Testing System v1.0.0
Risk scoring, ML prediction model, NLP pipeline, recommendations.
"""
from __future__ import annotations

import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SEV_WEIGHT = {"Low": 1, "Medium": 2, "High": 3}


# ─── 1. DataFrame builder ─────────────────────────────────────

def build_issues_df(issues):
    if not issues:
        return pd.DataFrame()
    df = pd.DataFrame(issues)
    for col in ["module", "issue_type", "severity", "status",
                "occurrences", "fix_time_hours", "description"]:
        if col not in df.columns:
            df[col] = "Unknown"
    df["severity_weight"] = df["severity"].map(SEV_WEIGHT).fillna(1).astype(float)
    df["is_critical"] = (df["severity"] == "High").astype(int)
    df["is_recurring"] = (df["status"] == "Reopened").astype(int)
    df["is_open"] = df["status"].isin(["Open", "Reopened"]).astype(int)
    return df


# ─── 2. Health Score ──────────────────────────────────────────

def compute_health_score(crawl_data, df):
    pages = crawl_data.get("pages", [])
    if not pages:
        return {"score": 0, "grade": "F", "color": "red", "breakdown": {}}

    n = len(pages)

    broken = len(crawl_data.get("broken_links", []))
    broken_score = max(0.0, 100.0 - (broken / max(n, 1)) * 200)

    high_issues = int(df["is_critical"].sum()) if not df.empty else 0
    issue_score = max(0.0, 100.0 - (high_issues / max(n, 1)) * 50)

    pages_with_title = sum(1 for p in pages if p.get("has_title"))
    seo_score = (pages_with_title / n) * 100

    pages_with_meta = sum(1 for p in pages if p.get("has_meta_description"))
    meta_score = (pages_with_meta / n) * 100

    load_times = [p["load_time"] for p in pages
                  if p.get("load_time") and p["load_time"] > 0]
    avg_load = float(np.mean(load_times)) if load_times else 0.0
    perf_score = max(0.0, 100.0 - (avg_load - 1.0) * 15)

    pages_with_vp = sum(1 for p in pages if p.get("has_viewport"))
    mobile_score = (pages_with_vp / n) * 100

    total_imgs = sum(p.get("img_total", 0) for p in pages)
    missing_alts = sum(p.get("img_missing_alt", 0) for p in pages)
    access_score = (
        100.0 if total_imgs == 0
        else max(0.0, (1 - missing_alts / total_imgs) * 100)
    )

    weights = {
        "Issues":        (issue_score,  0.25),
        "Broken Links":  (broken_score, 0.20),
        "Performance":   (perf_score,   0.20),
        "SEO":           (seo_score,    0.15),
        "Accessibility": (access_score, 0.10),
        "Meta Tags":     (meta_score,   0.05),
        "Mobile":        (mobile_score, 0.05),
    }
    score = round(min(100.0, max(0.0, sum(s * w for s, w in weights.values()))), 1)

    if score >= 80:
        grade, color = "A", "green"
    elif score >= 65:
        grade, color = "B", "orange"
    elif score >= 50:
        grade, color = "C", "#FF8C00"
    elif score >= 35:
        grade, color = "D", "red"
    else:
        grade, color = "F", "darkred"

    return {
        "score": score,
        "grade": grade,
        "color": color,
        "breakdown": {k: round(v[0], 1) for k, v in weights.items()},
    }


# ─── 3. Module Risk Scoring ───────────────────────────────────

def compute_module_risk(df, crawl_data):
    """
    Risk Score = (Issue Count × 0.4) + (Avg Severity Weight × 10 × 0.35)
                 + (Broken Link Rate × 100 × 0.25)
    Normalised to 0–100.
    """
    if df.empty:
        return pd.DataFrame()

    pages = crawl_data.get("pages", [])
    module_page_count = Counter(p.get("module", "Unknown") for p in pages)

    from crawler import identify_module
    module_broken = Counter()
    for bl in crawl_data.get("broken_links", []):
        m = identify_module(bl.get("url", ""), crawl_data.get("url", ""))
        module_broken[m] += 1

    records = []
    for module, grp in df.groupby("module"):
        n_issues = len(grp)
        avg_sev = float(grp["severity_weight"].mean())
        n_pages = max(module_page_count.get(module, 1), 1)
        broken_rate = module_broken.get(module, 0) / n_pages
        records.append({
            "Module": module,
            "Issue Count": n_issues,
            "Open Issues": int(grp["is_open"].sum()),
            "High Severity": int((grp["severity"] == "High").sum()),
            "Avg Severity Weight": round(avg_sev, 2),
            "Broken Link Rate": round(broken_rate, 3),
            "Raw Risk Score": (n_issues * 0.4) + (avg_sev * 10 * 0.35) + (broken_rate * 100 * 0.25),
            "Small Sample": n_issues < 5,
        })

    risk_df = pd.DataFrame(records)
    if risk_df.empty:
        return risk_df

    mn = risk_df["Raw Risk Score"].min()
    mx = risk_df["Raw Risk Score"].max()
    if mx > mn:
        risk_df["Risk Score (0-100)"] = (
            (risk_df["Raw Risk Score"] - mn) / (mx - mn) * 100
        ).round(1)
    else:
        risk_df["Risk Score (0-100)"] = 50.0

    risk_df["Risk Level"] = risk_df["Risk Score (0-100)"].apply(
        lambda s: "High" if s >= 70 else ("Medium" if s >= 40 else "Low")
    )
    return risk_df.sort_values("Risk Score (0-100)", ascending=False).reset_index(drop=True)


# ─── 4. ML Prediction Model ───────────────────────────────────

def train_prediction_model(df):
    if df.empty or len(df) < 8:
        return {"error": "Not enough data for ML model (need ≥ 8 issues)."}

    df = df.copy()
    le_module = LabelEncoder()
    le_type = LabelEncoder()
    df["module_enc"] = le_module.fit_transform(df["module"].astype(str))
    df["type_enc"] = le_type.fit_transform(df["issue_type"].astype(str))

    features = ["module_enc", "type_enc", "occurrences", "fix_time_hours", "severity_weight"]
    for f in features:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)

    X = df[features].values
    y = df["is_critical"].values

    # Guarantee at least two classes
    if len(set(y)) < 2:
        y_aug = y.copy()
        y_aug[0] = 1 - y_aug[0]
        X = np.vstack([X, X])
        y = np.hstack([y, y_aug])

    strat = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=strat
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        random_state=RANDOM_SEED, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    return {
        "model": clf,
        "le_module": le_module,
        "le_type": le_type,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "feature_importance": dict(zip(features, clf.feature_importances_.tolist())),
        "features": features,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "confidence_scores": y_proba.max(axis=1).tolist(),
        "label_names": ["Not Critical", "Critical"],
        "error": None,
    }


def predict_module_risk(model_data, module, issue_type, occurrences, fix_time, severity):
    if model_data.get("error"):
        return {"error": model_data["error"]}
    clf = model_data["model"]
    le_module = model_data["le_module"]
    le_type = model_data["le_type"]
    try:
        m_enc = le_module.transform([module])[0]
    except ValueError:
        m_enc = 0
    try:
        t_enc = le_type.transform([issue_type])[0]
    except ValueError:
        t_enc = 0
    sev_w = SEV_WEIGHT.get(severity, 1)
    X = np.array([[m_enc, t_enc, occurrences, fix_time, sev_w]])
    pred = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0].tolist()
    return {
        "prediction": "Critical" if pred == 1 else "Not Critical",
        "confidence": round(float(max(proba)) * 100, 1),
        "proba_critical": round(float(proba[1]) * 100, 1) if len(proba) > 1 else 0.0,
        "proba_not_critical": round(float(proba[0]) * 100, 1),
    }


# ─── 5. NLP Analysis ──────────────────────────────────────────

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "not","with","this","that","has","have","are","was","were","be","been",
    "being","do","does","did","will","would","could","should","may","might",
    "shall","can","which","from","by","as","its","their","there","than","also",
    "no","all","more","into","url","page","http","https","www","html","css",
    "js","tag","tags","returns","return","has","lack","lacks","impacts",
}


def _preprocess(text):
    text = text.lower()
    text = re.sub(r"https?\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words) if words else "issue"


def run_nlp_analysis(df):
    if df.empty or len(df) < 3:
        return {"error": "Need ≥ 3 issues for NLP analysis."}

    docs = [_preprocess(str(d)) for d in df["description"].fillna("")]

    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=1)
    try:
        tfidf = vectorizer.fit_transform(docs)
    except Exception as exc:
        return {"error": f"TF-IDF failed: {exc}"}

    names = vectorizer.get_feature_names_out()
    sums = np.asarray(tfidf.sum(axis=0)).flatten()
    top_idx = sums.argsort()[::-1][:20]
    top_keywords = [(names[i], round(float(sums[i]), 3)) for i in top_idx]

    n_clusters = min(max(2, len(df) // 3), 6)
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    try:
        labels = km.fit_predict(tfidf)
    except Exception:
        labels = [0] * len(docs)

    df = df.copy()
    df["cluster"] = labels

    clusters = {}
    for cid in sorted(set(labels)):
        sub = df[df["cluster"] == cid]
        idx = [i for i, l in enumerate(labels) if l == cid]
        if idx:
            sub_tfidf = tfidf[idx]
            sc = np.asarray(sub_tfidf.sum(axis=0)).flatten()
            kw_idx = sc.argsort()[::-1][:5]
            kws = [names[j] for j in kw_idx if sc[j] > 0]
        else:
            kws = []
        clusters[int(cid)] = {
            "label": f"Cluster {cid + 1}: {kws[0] if kws else 'issues'}",
            "keywords": kws,
            "count": len(sub),
            "examples": sub["description"].head(3).tolist(),
            "modules": sub["module"].value_counts().to_dict(),
            "severities": sub["severity"].value_counts().to_dict(),
        }

    return {
        "top_keywords": top_keywords,
        "clusters": clusters,
        "n_clusters": n_clusters,
        "word_freq": {k: v for k, v in top_keywords},
        "df_with_clusters": df,
        "error": None,
    }


# ─── 6. Recommendations ──────────────────────────────────────

def generate_recommendations(df, crawl_data, risk_df):
    pages = crawl_data.get("pages", [])
    n = max(len(pages), 1)
    recs = []
    recurrence_flags = []

    if df.empty:
        return {
            "recommendations": [{"priority": 1, "category": "Info",
                                  "title": "No issues detected",
                                  "detail": "The site appears healthy.",
                                  "effort": "Low", "impact": "Low"}],
            "priority_tests": [],
            "recurrence_flags": [],
            "benchmarks": {},
        }

    ic = df["issue_type"].value_counts().to_dict()

    broken = len(crawl_data.get("broken_links", []))
    if broken > 0:
        recs.append({"priority": 1, "category": "Critical",
                     "title": f"Fix {broken} Broken Link(s)",
                     "detail": ("Broken links hurt UX and SEO. Use monthly link-checker scans. "
                                "Add 301 redirects for moved pages and a custom 404 page."),
                     "effort": "Low", "impact": "High"})
        recurrence_flags.append({"issue_type": "Broken Link (404)", "risk": "High",
                                  "reason": "Links break whenever pages move without redirects.",
                                  "frequency": "Check monthly"})

    forms_no_val = ic.get("Form Without Client-Side Validation", 0)
    if forms_no_val > 0:
        recs.append({"priority": 2, "category": "Critical",
                     "title": f"Add Validation to {forms_no_val} Form(s)",
                     "detail": ("Add HTML5 required/pattern/type=email attributes. "
                                "Implement server-side validation. Prevents invalid data and XSS risk."),
                     "effort": "Medium", "impact": "High"})

    slow = ic.get("Slow Page Load", 0)
    if slow > 0:
        lts = [p["load_time"] for p in pages if p.get("load_time", 0) > 3]
        recs.append({"priority": 3, "category": "Performance",
                     "title": f"Optimise {slow} Slow Page(s) — max {max(lts):.1f}s",
                     "detail": ("Enable gzip/Brotli, use WebP images, add a CDN, "
                                "defer non-critical JS, enable browser caching."),
                     "effort": "High", "impact": "High"})
        recurrence_flags.append({"issue_type": "Slow Page Load", "risk": "Medium",
                                  "reason": "Performance degrades as content grows.",
                                  "frequency": "Re-test after every major release"})

    missing_alts = sum(p.get("img_missing_alt", 0) for p in pages)
    if missing_alts > 0:
        recs.append({"priority": 4, "category": "Accessibility",
                     "title": f"Add Alt Text to {missing_alts} Image(s) — WCAG 2.1",
                     "detail": "Add descriptive alt attributes; use alt='' for decorative images.",
                     "effort": "Low", "impact": "Medium"})

    no_title = ic.get("Missing Page Title", 0)
    no_meta = ic.get("Missing Meta Description", 0)
    if no_title + no_meta > 0:
        recs.append({"priority": 5, "category": "SEO",
                     "title": f"Fix {no_title} Missing Titles & {no_meta} Meta Descriptions",
                     "detail": ("Every page needs a unique title (50–60 chars) and "
                                "meta description (150–160 chars)."),
                     "effort": "Low", "impact": "Medium"})

    no_vp = ic.get("Missing Viewport Meta Tag", 0)
    if no_vp > 0:
        recs.append({"priority": 6, "category": "Mobile",
                     "title": f"Add Viewport Meta to {no_vp} Page(s)",
                     "detail": "Add <meta name='viewport' content='width=device-width, initial-scale=1'>.",
                     "effort": "Low", "impact": "Medium"})

    srv_err = ic.get("Server Error", 0)
    if srv_err > 0:
        recs.append({"priority": 1, "category": "Critical",
                     "title": f"Resolve {srv_err} Server Error(s) (5xx)",
                     "detail": "Check server logs; set up uptime monitoring and alerting.",
                     "effort": "High", "impact": "High"})

    no_canon = ic.get("Missing Canonical Tag", 0)
    if no_canon > 3:
        recs.append({"priority": 7, "category": "SEO",
                     "title": f"Add Canonical Tags to {no_canon} Pages",
                     "detail": "Prevents duplicate-content SEO penalties on paginated/filtered URLs.",
                     "effort": "Low", "impact": "Low"})

    # Always include general high-value recs
    recs += [
        {"priority": 8, "category": "Security",
         "title": "Enforce HTTPS + HSTS Headers",
         "detail": ("Ensure all pages use HTTPS. Add Strict-Transport-Security header. "
                    "Check for mixed-content warnings."),
         "effort": "Medium", "impact": "High"},
        {"priority": 9, "category": "Monitoring",
         "title": "Set Up Continuous Automated Testing",
         "detail": ("Run Lighthouse CI on every deploy; schedule weekly broken-link checks "
                    "and Pa11y accessibility scans."),
         "effort": "High", "impact": "High"},
    ]

    recs.sort(key=lambda r: r["priority"])

    # Priority tests from risk table
    priority_tests = []
    if not risk_df.empty:
        for _, row in risk_df.head(5).iterrows():
            priority_tests.append({
                "module": row["Module"],
                "risk_score": row["Risk Score (0-100)"],
                "risk_level": row["Risk Level"],
                "open_issues": row["Open Issues"],
                "test_focus": _module_focus(row["Module"]),
            })

    # Benchmarks
    avg_load = float(np.mean([p["load_time"] for p in pages
                               if p.get("load_time", 0) > 0])) if pages else 0.0
    benchmarks = {
        "Average Load Time": {
            "yours": f"{avg_load:.2f}s",
            "benchmark": "< 2.0s",
            "status": ("✅ Good" if avg_load < 2 else
                       ("⚠️ Needs Work" if avg_load < 4 else "❌ Poor")),
        },
        "Broken Link Rate": {
            "yours": f"{broken}/{n}",
            "benchmark": "0",
            "status": "✅ Good" if broken == 0 else "❌ Poor",
        },
        "Pages With Meta Description": {
            "yours": f"{sum(1 for p in pages if p.get('has_meta_description'))}/{n}",
            "benchmark": "100%",
            "status": ("✅ Good"
                       if all(p.get("has_meta_description") for p in pages)
                       else "⚠️ Needs Work"),
        },
        "Images With Alt Text": {
            "yours": (
                f"{sum(p.get('img_total',0)-p.get('img_missing_alt',0) for p in pages)}"
                f"/{sum(p.get('img_total',0) for p in pages)}"
            ),
            "benchmark": "100%",
            "status": "✅ Good" if missing_alts == 0 else "⚠️ Needs Work",
        },
        "Pages With Viewport Meta": {
            "yours": f"{sum(1 for p in pages if p.get('has_viewport'))}/{n}",
            "benchmark": "100%",
            "status": ("✅ Good"
                       if all(p.get("has_viewport") for p in pages)
                       else "❌ Poor"),
        },
    }

    return {
        "recommendations": recs,
        "priority_tests": priority_tests,
        "recurrence_flags": recurrence_flags,
        "benchmarks": benchmarks,
    }


def _module_focus(module):
    mapping = {
        "Authentication": "Login/logout flows, session management, password reset, brute force",
        "Checkout": "Payment flow, cart persistence, discount codes, order confirmation",
        "Cart": "Add/remove items, quantity updates, total accuracy, session persistence",
        "Dashboard": "Data accuracy, chart rendering, real-time updates, permissions",
        "User Profile": "Edit profile, avatar upload, password change, privacy settings",
        "Search": "Empty results, special chars, pagination, relevance ranking",
        "Product Pages": "Images, pricing, stock status, add-to-cart functionality",
        "Admin Panel": "Access control, CRUD, audit logs, bulk actions",
        "Blog/Content": "Rich text, media embedding, comments, pagination",
        "Help/Support": "Search accuracy, ticket submission, response tracking",
    }
    for key, focus in mapping.items():
        if key.lower() in module.lower():
            return focus
    return "Functional testing, link validation, form submission, content accuracy"
