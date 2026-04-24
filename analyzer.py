"""
analyzer.py — Risk scoring, ML prediction, and NLP analysis
Intelligent App Testing System v1.0.0
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
import warnings
warnings.filterwarnings("ignore")

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SEVERITY_WEIGHT = {"Low": 1, "Medium": 2, "High": 3}

# ─────────────────────────────────────────────────────────────
# 1. Issue DataFrame builder
# ─────────────────────────────────────────────────────────────

def build_issues_df(issues: list[dict]) -> pd.DataFrame:
    """Convert raw issue list to a clean DataFrame."""
    if not issues:
        return pd.DataFrame()

    df = pd.DataFrame(issues)

    # Ensure required columns
    required = ["module", "issue_type", "severity", "status",
                 "occurrences", "fix_time_hours", "description"]
    for col in required:
        if col not in df.columns:
            df[col] = "Unknown"

    df["severity_weight"] = df["severity"].map(SEVERITY_WEIGHT).fillna(1)
    df["is_critical"] = (df["severity"] == "High").astype(int)
    df["is_recurring"] = (df["status"] == "Reopened").astype(int)
    df["is_open"] = df["status"].isin(["Open", "Reopened"]).astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# 2. Website Health Score
# ─────────────────────────────────────────────────────────────

def compute_health_score(crawl_data: dict, df: pd.DataFrame) -> dict:
    """Compute overall website health score 0–100."""
    pages = crawl_data.get("pages", [])
    if not pages:
        return {"score": 0, "grade": "F", "color": "red", "breakdown": {}}

    total_pages = len(pages)

    # Component scores
    broken = len(crawl_data.get("broken_links", []))
    broken_score = max(0, 100 - (broken / max(total_pages, 1)) * 200)

    high_issues = len(df[df["severity"] == "High"]) if not df.empty else 0
    issue_score = max(0, 100 - (high_issues / max(total_pages, 1)) * 50)

    pages_with_title = sum(1 for p in pages if p.get("has_title"))
    seo_score = (pages_with_title / total_pages) * 100

    pages_with_meta = sum(1 for p in pages if p.get("has_meta_description"))
    meta_score = (pages_with_meta / total_pages) * 100

    load_times = [p.get("load_time", 0) for p in pages if p.get("load_time")]
    perf_score = 100
    if load_times:
        avg_load = np.mean(load_times)
        perf_score = max(0, 100 - (avg_load - 1.0) * 15)

    pages_with_viewport = sum(1 for p in pages if p.get("has_viewport"))
    mobile_score = (pages_with_viewport / total_pages) * 100

    total_imgs = sum(p.get("img_total", 0) for p in pages)
    missing_alts = sum(p.get("img_missing_alt", 0) for p in pages)
    access_score = 100 if total_imgs == 0 else max(0, (1 - missing_alts / total_imgs) * 100)

    # Weighted composite
    weights = {
        "Issues": (issue_score, 0.25),
        "Broken Links": (broken_score, 0.20),
        "Performance": (perf_score, 0.20),
        "SEO": (seo_score, 0.15),
        "Accessibility": (access_score, 0.10),
        "Meta Tags": (meta_score, 0.05),
        "Mobile": (mobile_score, 0.05),
    }
    score = sum(s * w for s, w in weights.values())
    score = round(min(100, max(0, score)), 1)

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

    breakdown = {k: round(v[0], 1) for k, v in weights.items()}
    return {"score": score, "grade": grade, "color": color, "breakdown": breakdown}


# ─────────────────────────────────────────────────────────────
# 3. Risk Scoring per Module
# ─────────────────────────────────────────────────────────────

def compute_module_risk(df: pd.DataFrame, crawl_data: dict) -> pd.DataFrame:
    """
    Risk Score = (Issue Count × 0.4) + (Severity Weight × 0.35) + (Broken Link Rate × 0.25)
    Normalized to 0–100.
    """
    if df.empty:
        return pd.DataFrame()

    pages = crawl_data.get("pages", [])
    module_page_count = Counter(p.get("module", "Unknown") for p in pages)

    broken = crawl_data.get("broken_links", [])
    module_broken = Counter()
    for bl in broken:
        from crawler import identify_module
        m = identify_module(bl.get("url", ""), crawl_data.get("url", ""))
        module_broken[m] += 1

    records = []
    for module, grp in df.groupby("module"):
        issue_count = len(grp)
        avg_sev = grp["severity_weight"].mean()
        total_pages_in_module = max(module_page_count.get(module, 1), 1)
        broken_rate = module_broken.get(module, 0) / total_pages_in_module
        open_issues = grp["is_open"].sum()
        high_issues = (grp["severity"] == "High").sum()
        sample_warning = issue_count < 5

        raw_score = (
            (issue_count * 0.4) +
            (avg_sev * 10 * 0.35) +
            (broken_rate * 100 * 0.25)
        )
        records.append({
            "Module": module,
            "Issue Count": issue_count,
            "Open Issues": int(open_issues),
            "High Severity": int(high_issues),
            "Avg Severity Weight": round(avg_sev, 2),
            "Broken Link Rate": round(broken_rate, 3),
            "Raw Risk Score": raw_score,
            "Small Sample": sample_warning,
        })

    risk_df = pd.DataFrame(records)
    if risk_df.empty:
        return risk_df

    min_r = risk_df["Raw Risk Score"].min()
    max_r = risk_df["Raw Risk Score"].max()
    if max_r > min_r:
        risk_df["Risk Score (0-100)"] = (
            (risk_df["Raw Risk Score"] - min_r) / (max_r - min_r) * 100
        ).round(1)
    else:
        risk_df["Risk Score (0-100)"] = 50.0

    def risk_label(s):
        if s >= 70:
            return "High"
        elif s >= 40:
            return "Medium"
        return "Low"

    risk_df["Risk Level"] = risk_df["Risk Score (0-100)"].apply(risk_label)
    risk_df = risk_df.sort_values("Risk Score (0-100)", ascending=False).reset_index(drop=True)
    return risk_df


# ─────────────────────────────────────────────────────────────
# 4. ML Prediction Model
# ─────────────────────────────────────────────────────────────

def train_prediction_model(df: pd.DataFrame) -> dict:
    """Train a Random Forest classifier to predict issue criticality."""
    if df.empty or len(df) < 8:
        return {"error": "Insufficient data for ML model (need ≥8 issues)."}

    # Feature engineering
    le_module = LabelEncoder()
    le_type = LabelEncoder()

    df = df.copy()
    df["module_enc"] = le_module.fit_transform(df["module"].astype(str))
    df["type_enc"] = le_type.fit_transform(df["issue_type"].astype(str))

    features = ["module_enc", "type_enc", "occurrences", "fix_time_hours", "severity_weight"]
    target = "is_critical"

    # Ensure numeric
    for f in features:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)

    X = df[features].values
    y = df[target].values

    if len(set(y)) < 2:
        # Artificially add variety if all same class
        y_aug = y.copy()
        y_aug[0] = 1 - y_aug[0]
        X_aug = X.copy()
        X, y = np.vstack([X, X_aug]), np.hstack([y, y_aug])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y if len(set(y)) > 1 else None
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_SEED, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    importance = dict(zip(features, clf.feature_importances_))

    # Confidence on test set
    confidence_scores = y_proba.max(axis=1).tolist()

    return {
        "model": clf,
        "le_module": le_module,
        "le_type": le_type,
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importance": importance,
        "features": features,
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "confidence_scores": confidence_scores,
        "label_names": ["Not Critical", "Critical"],
        "error": None,
    }


def predict_module_risk(model_data: dict, module: str, issue_type: str,
                        occurrences: int, fix_time: float, severity: str) -> dict:
    """Make a live prediction for a given module/issue combination."""
    if model_data.get("error"):
        return {"error": model_data["error"]}

    clf = model_data["model"]
    le_module = model_data["le_module"]
    le_type = model_data["le_type"]

    try:
        module_enc = le_module.transform([module])[0]
    except ValueError:
        module_enc = 0

    try:
        type_enc = le_type.transform([issue_type])[0]
    except ValueError:
        type_enc = 0

    sev_w = SEVERITY_WEIGHT.get(severity, 1)
    X = np.array([[module_enc, type_enc, occurrences, fix_time, sev_w]])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    return {
        "prediction": "Critical" if pred == 1 else "Not Critical",
        "confidence": round(float(proba.max()) * 100, 1),
        "proba_critical": round(float(proba[1]) * 100, 1) if len(proba) > 1 else 0,
        "proba_not_critical": round(float(proba[0]) * 100, 1),
    }


# ─────────────────────────────────────────────────────────────
# 5. NLP Analysis
# ─────────────────────────────────────────────────────────────

def run_nlp_analysis(df: pd.DataFrame) -> dict:
    """TF-IDF + KMeans clustering on issue descriptions."""
    if df.empty or len(df) < 3:
        return {"error": "Not enough issues for NLP analysis (need ≥3)."}

    descriptions = df["description"].fillna("").tolist()

    # Minimal NLTK — use basic stopwords list to avoid download issues
    STOPWORDS = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
        "and", "or", "but", "not", "with", "this", "that", "has", "have",
        "are", "was", "were", "be", "been", "being", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "which", "from", "by", "as", "its", "their", "there", "than",
        "also", "no", "all", "more", "into", "url", "page", "http", "https",
        "www", "html", "css", "js", "tag", "tags",
    }

    def preprocess(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
        return " ".join(words)

    clean_docs = [preprocess(d) for d in descriptions]
    clean_docs = [d if d.strip() else "issue found" for d in clean_docs]

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        min_df=1,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(clean_docs)
    except Exception as e:
        return {"error": f"TF-IDF failed: {e}"}

    feature_names = vectorizer.get_feature_names_out()
    tfidf_sum = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    top_indices = tfidf_sum.argsort()[::-1][:20]
    top_keywords = [(feature_names[i], round(float(tfidf_sum[i]), 3)) for i in top_indices]

    # KMeans clustering
    n_clusters = min(max(2, len(df) // 3), 6)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    try:
        clusters = kmeans.fit_predict(tfidf_matrix)
    except Exception:
        clusters = [0] * len(descriptions)

    df = df.copy()
    df["cluster"] = clusters

    cluster_examples = {}
    for c in sorted(set(clusters)):
        subset = df[df["cluster"] == c]
        top_kws = _top_cluster_keywords(tfidf_matrix, clusters, c, feature_names)
        cluster_examples[int(c)] = {
            "label": f"Cluster {c+1}: {top_kws[0] if top_kws else 'issues'}",
            "keywords": top_kws[:5],
            "count": len(subset),
            "examples": subset["description"].head(3).tolist(),
            "modules": subset["module"].value_counts().to_dict(),
            "severities": subset["severity"].value_counts().to_dict(),
        }

    # Word frequencies for wordcloud
    word_freq = {}
    for kw, score in top_keywords:
        word_freq[kw] = score

    return {
        "top_keywords": top_keywords,
        "clusters": cluster_examples,
        "n_clusters": n_clusters,
        "word_freq": word_freq,
        "df_with_clusters": df,
        "error": None,
    }


def _top_cluster_keywords(tfidf_matrix, clusters, cluster_id, feature_names, n=5):
    """Get top keywords for a cluster."""
    indices = [i for i, c in enumerate(clusters) if c == cluster_id]
    if not indices:
        return []
    cluster_matrix = tfidf_matrix[indices]
    scores = np.asarray(cluster_matrix.sum(axis=0)).flatten()
    top_idx = scores.argsort()[::-1][:n]
    return [feature_names[i].replace("_", " ") for i in top_idx if scores[i] > 0]


# ─────────────────────────────────────────────────────────────
# 6. Fix Validation & Recommendations
# ─────────────────────────────────────────────────────────────

def generate_recommendations(df: pd.DataFrame, crawl_data: dict, risk_df: pd.DataFrame) -> dict:
    """Generate specific, actionable recommendations based on crawl findings."""
    recs = []
    priority_tests = []
    recurrence_flags = []

    pages = crawl_data.get("pages", [])
    total_pages = max(len(pages), 1)

    if df.empty:
        return {
            "recommendations": ["No issues detected — site appears healthy."],
            "priority_tests": [],
            "recurrence_flags": [],
        }

    issue_counts = df["issue_type"].value_counts().to_dict()
    sev_counts = df["severity"].value_counts().to_dict()
    module_issue_counts = df.groupby("module")["id"].count().sort_values(ascending=False)

    # --- Broken links ---
    broken = len(crawl_data.get("broken_links", []))
    if broken > 0:
        recs.append({
            "priority": 1,
            "category": "Critical",
            "title": f"Fix {broken} Broken Links",
            "detail": (
                f"Found {broken} broken/unreachable links. Use a link checker monthly. "
                "Implement 301 redirects for removed pages. Add a custom 404 page with navigation."
            ),
            "effort": "Low",
            "impact": "High",
        })
        recurrence_flags.append({
            "issue_type": "Broken Link (404)",
            "risk": "High",
            "reason": "Links break when pages are moved/removed without redirects.",
            "frequency": "Monthly checks required",
        })

    # --- Forms without validation ---
    forms_no_val = issue_counts.get("Form Without Client-Side Validation", 0)
    if forms_no_val > 0:
        recs.append({
            "priority": 2,
            "category": "Critical",
            "title": f"Add Validation to {forms_no_val} Form(s)",
            "detail": (
                "Forms without validation expose the site to invalid data, XSS, and poor UX. "
                "Add HTML5 required, pattern, type=email attributes. Implement server-side validation too."
            ),
            "effort": "Medium",
            "impact": "High",
        })

    # --- Performance ---
    slow_pages = issue_counts.get("Slow Page Load", 0)
    if slow_pages > 0:
        load_times = [p.get("load_time", 0) for p in pages if p.get("load_time", 0) > 3]
        max_time = max(load_times) if load_times else 0
        recs.append({
            "priority": 3,
            "category": "Performance",
            "title": f"Optimize {slow_pages} Slow Page(s) — Max Load: {max_time:.1f}s",
            "detail": (
                "Pages exceeding 3s load time hurt conversion and SEO. "
                "Enable gzip compression, optimize images (WebP format), use a CDN, "
                "defer non-critical JS, and enable browser caching."
            ),
            "effort": "High",
            "impact": "High",
        })
        recurrence_flags.append({
            "issue_type": "Slow Page Load",
            "risk": "Medium",
            "reason": "Performance degrades as content grows without optimization.",
            "frequency": "Re-test after every major release",
        })

    # --- Missing alt text ---
    missing_alts = sum(p.get("img_missing_alt", 0) for p in pages)
    if missing_alts > 0:
        recs.append({
            "priority": 4,
            "category": "Accessibility",
            "title": f"Add Alt Text to {missing_alts} Image(s) — WCAG 2.1 Compliance",
            "detail": (
                "Missing alt text fails WCAG 2.1 Success Criterion 1.1.1. "
                "Add descriptive alt attributes to all meaningful images. "
                "Use empty alt='' for decorative images."
            ),
            "effort": "Low",
            "impact": "Medium",
        })

    # --- SEO: Missing titles/meta ---
    no_title = issue_counts.get("Missing Page Title", 0)
    no_meta = issue_counts.get("Missing Meta Description", 0)
    if no_title + no_meta > 0:
        recs.append({
            "priority": 5,
            "category": "SEO",
            "title": f"Fix SEO Tags: {no_title} Missing Titles, {no_meta} Missing Meta Descriptions",
            "detail": (
                "Every page needs a unique <title> (50–60 chars) and meta description (150–160 chars). "
                "Use a CMS plugin or structured SEO template to enforce this across all pages."
            ),
            "effort": "Low",
            "impact": "Medium",
        })

    # --- Mobile ---
    no_viewport = issue_counts.get("Missing Viewport Meta Tag", 0)
    if no_viewport > 0:
        recs.append({
            "priority": 6,
            "category": "Mobile",
            "title": f"Add Viewport Meta Tag to {no_viewport} Page(s)",
            "detail": (
                "Add <meta name='viewport' content='width=device-width, initial-scale=1'> "
                "to all pages. Run a Google Mobile-Friendly Test after fixing."
            ),
            "effort": "Low",
            "impact": "Medium",
        })

    # --- Server errors ---
    server_errors = issue_counts.get("Server Error", 0)
    if server_errors > 0:
        recs.append({
            "priority": 1,
            "category": "Critical",
            "title": f"Resolve {server_errors} Server Error(s) (5xx)",
            "detail": (
                "5xx errors indicate backend failures affecting real users. "
                "Check server logs, monitor with an uptime service (e.g., UptimeRobot), "
                "and set up alerting for any 5xx spikes."
            ),
            "effort": "High",
            "impact": "High",
        })

    # --- Canonical/OG ---
    no_canonical = issue_counts.get("Missing Canonical Tag", 0)
    if no_canonical > 3:
        recs.append({
            "priority": 7,
            "category": "SEO",
            "title": f"Add Canonical Tags to {no_canonical} Pages",
            "detail": (
                "Canonical tags prevent duplicate content penalties in search engines. "
                "Add <link rel='canonical'> to all pages, especially paginated or filtered URLs."
            ),
            "effort": "Low",
            "impact": "Low",
        })

    no_og = issue_counts.get("Missing Open Graph Tags", 0)
    if no_og > 0:
        recs.append({
            "priority": 8,
            "category": "Social/SEO",
            "title": "Implement Open Graph Meta Tags for Social Sharing",
            "detail": (
                "Add og:title, og:description, og:image, og:url to all pages. "
                "Test with Facebook Sharing Debugger and Twitter Card Validator."
            ),
            "effort": "Low",
            "impact": "Low",
        })

    # Ensure at least 8 recommendations
    general_recs = [
        {
            "priority": 9,
            "category": "Security",
            "title": "Implement HTTPS and HSTS Headers",
            "detail": (
                "Ensure all pages use HTTPS. Add Strict-Transport-Security header. "
                "Check for mixed content warnings and update all internal links."
            ),
            "effort": "Medium",
            "impact": "High",
        },
        {
            "priority": 10,
            "category": "Monitoring",
            "title": "Set Up Continuous Automated Testing",
            "detail": (
                "Implement CI/CD pipeline with automated test runs. "
                "Use Lighthouse CI for performance checks, Pa11y for accessibility, "
                "and broken-link-checker for link validation on every deploy."
            ),
            "effort": "High",
            "impact": "High",
        },
    ]
    for g in general_recs:
        if len(recs) < 10:
            recs.append(g)

    recs.sort(key=lambda x: x["priority"])

    # Priority tests
    if not risk_df.empty:
        for _, row in risk_df.head(5).iterrows():
            priority_tests.append({
                "module": row["Module"],
                "risk_score": row["Risk Score (0-100)"],
                "risk_level": row["Risk Level"],
                "open_issues": row["Open Issues"],
                "test_focus": _module_test_focus(row["Module"]),
            })

    # Benchmark comparison
    avg_load = np.mean([p.get("load_time", 0) for p in pages if p.get("load_time")]) if pages else 0
    benchmarks = {
        "Average Load Time": {
            "yours": f"{avg_load:.2f}s",
            "benchmark": "< 2.0s",
            "status": "✅ Good" if avg_load < 2 else ("⚠️ Needs Work" if avg_load < 4 else "❌ Poor"),
        },
        "Broken Link Rate": {
            "yours": f"{len(crawl_data.get('broken_links', []))}/{total_pages}",
            "benchmark": "0",
            "status": "✅ Good" if len(crawl_data.get("broken_links", [])) == 0 else "❌ Poor",
        },
        "Pages with Meta Description": {
            "yours": f"{sum(1 for p in pages if p.get('has_meta_description'))}/{total_pages}",
            "benchmark": "100%",
            "status": "✅ Good" if all(p.get("has_meta_description") for p in pages) else "⚠️ Needs Work",
        },
        "Images with Alt Text": {
            "yours": f"{sum(p.get('img_total', 0) - p.get('img_missing_alt', 0) for p in pages)}/{sum(p.get('img_total', 0) for p in pages)}",
            "benchmark": "100%",
            "status": "✅ Good" if sum(p.get("img_missing_alt", 0) for p in pages) == 0 else "⚠️ Needs Work",
        },
        "Pages with Viewport Meta": {
            "yours": f"{sum(1 for p in pages if p.get('has_viewport'))}/{total_pages}",
            "benchmark": "100%",
            "status": "✅ Good" if all(p.get("has_viewport") for p in pages) else "❌ Poor",
        },
    }

    return {
        "recommendations": recs,
        "priority_tests": priority_tests,
        "recurrence_flags": recurrence_flags,
        "benchmarks": benchmarks,
    }


def _module_test_focus(module: str) -> str:
    """Return test focus area for a module."""
    mapping = {
        "Authentication": "Login/logout flows, session management, password reset, brute force protection",
        "Checkout": "Payment flow, cart persistence, discount codes, order confirmation emails",
        "Cart": "Add/remove items, quantity updates, cart total accuracy, session persistence",
        "Dashboard": "Data accuracy, chart loading, real-time updates, permission checks",
        "User Profile": "Edit profile, avatar upload, password change, privacy settings",
        "Search": "Empty results, special characters, pagination, relevance ranking",
        "Product Pages": "Images, pricing accuracy, stock status, add-to-cart functionality",
        "Admin Panel": "Access control, CRUD operations, audit logs, bulk actions",
        "Blog/Content": "Rich text rendering, media embedding, comment system, pagination",
        "Help/Support": "Search accuracy, ticket submission, response tracking",
    }
    for key, focus in mapping.items():
        if key.lower() in module.lower():
            return focus
    return "Functional testing, link validation, form submission, content accuracy"
