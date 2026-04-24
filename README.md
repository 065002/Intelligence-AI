# 🔬 Intelligent App Testing System

> Paste any URL → Auto-crawl → Full AI-powered analysis → Actionable risk report

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 🚀 What It Does

The Intelligent App Testing System accepts any live website URL, automatically crawls it using `requests` + `BeautifulSoup`, extracts real structural and content data, and generates a complete AI-powered testing and risk analysis report — **no file uploads, no manual data entry**.

Every insight is traceable back to actual crawl findings. Nothing is fabricated.

---

## 📋 Features

| Page | What You Get |
|------|-------------|
| 🌐 **Website Overview** | Health score, crawl summary, structure map, module breakdown |
| 📊 **Exploratory Analysis** | Issue charts, severity distribution, load time comparison, top problem pages |
| ⚠️ **Risk Scoring** | Per-module risk scores (0–100), color-coded table, radar chart |
| 🤖 **Prediction Model** | Random Forest classifier, confusion matrix, live prediction with confidence |
| 💬 **NLP Analysis** | TF-IDF keywords, KMeans clustering, WordCloud |
| ✅ **Fix Validation** | Recurrence flags, 8–10 actionable recommendations, benchmark comparison |

---

## 🏗️ Architecture

```
intelligent-app-tester/
├── app.py              # Streamlit UI — all 6 pages
├── crawler.py          # Website crawling, data extraction, issue generation
├── analyzer.py         # Risk scoring, ML model, NLP pipeline
├── requirements.txt    # Dependencies (Streamlit Cloud compatible)
└── README.md           # This file
```

### How It Works

```
URL Input
    ↓
crawler.py: crawl_website()
    ├── Fetch pages (requests + BeautifulSoup)
    ├── Extract: title, meta, headings, forms, images, links, load time
    ├── Check HTTP status codes
    ├── Identify modules from URL structure
    └── generate_issues_from_crawl() → structured issue dataset
         ↓
analyzer.py:
    ├── build_issues_df()       → clean DataFrame
    ├── compute_health_score()  → 0–100 weighted score
    ├── compute_module_risk()   → per-module risk table
    ├── train_prediction_model() → Random Forest classifier
    ├── run_nlp_analysis()      → TF-IDF + KMeans
    └── generate_recommendations() → actionable insights
         ↓
app.py: Streamlit UI → 6 interactive pages
```

---

## ⚙️ Methodology

### Health Score
```
Score = (Issues × 25%) + (Broken Links × 20%) + (Performance × 20%) + 
        (SEO × 15%) + (Accessibility × 10%) + (Meta Tags × 5%) + (Mobile × 5%)
```
Normalized to 0–100. Grade: A (≥80), B (≥65), C (≥50), D (≥35), F (<35).

### Risk Score per Module
```
Raw Risk = (Issue Count × 0.4) + (Avg Severity Weight × 10 × 0.35) + (Broken Link Rate × 100 × 0.25)
Final = Normalized to 0–100
```
Severity weights: Low=1, Medium=2, High=3.

### ML Model
- **Algorithm:** Random Forest Classifier (100 trees, max depth 5, balanced class weights)
- **Features:** Module (encoded), Issue Type (encoded), Occurrences, Fix Time, Severity Weight
- **Target:** Critical (High severity) vs Not Critical
- **Split:** 70% train / 30% test, stratified, seed=42
- **Output:** Prediction + confidence score from `predict_proba()`

### NLP Pipeline
- **TF-IDF:** `TfidfVectorizer(max_features=100, ngram_range=(1,2))` on issue descriptions
- **Clustering:** KMeans (k = min(max(2, n_issues//3), 6))
- **WordCloud:** Frequency-weighted from TF-IDF scores

---

## 💻 Local Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/intelligent-app-tester.git
cd intelligent-app-tester

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 🌐 Deploy to Streamlit Cloud

### One-Click Deploy

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

### Step-by-Step

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/intelligent-app-tester.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repository: `YOUR_USERNAME/intelligent-app-tester`
   - Set **Main file path**: `app.py`
   - Click **"Deploy!"**

3. **Wait ~2 minutes** for the build to complete

4. **Your app is live** at `https://YOUR_USERNAME-intelligent-app-tester-app-XXXX.streamlit.app`

### Streamlit Cloud Requirements
- All dependencies in `requirements.txt` (no conda, no OS-level packages)
- Python 3.9+ (set in Advanced Settings if needed)
- Max 1GB RAM on free tier — crawl limit set to 50 pages to stay within limits

---

## 🔒 Privacy & Ethics

- Adds proper `User-Agent` header to identify as a bot
- Respects `robots.txt` by default (toggle in sidebar)
- No crawl data is stored — all analysis is in-memory per session
- Crawl limit: 50 pages maximum
- Timeout: 10 seconds per page

---

## 🧪 Testing the App

Try these URLs to test different scenarios:

| URL | Expected Result |
|-----|----------------|
| `https://example.com` | Simple 1-page site, minimal issues |
| `https://httpbin.org` | API site with multiple endpoints |
| `https://books.toscrape.com` | Full e-commerce with many pages and modules |
| `https://quotes.toscrape.com` | Content site with pagination |

---

## 📊 Issue Types Detected

| Issue Type | Severity | Source |
|-----------|----------|--------|
| Broken Link (404) | High | HTTP status code |
| Server Error (5xx) | High | HTTP status code |
| Page Timeout | High | Request timeout |
| Form Without Validation | High | HTML form analysis |
| Slow Page Load (>3s) | High/Medium | Response time measurement |
| Missing Page Title | Medium | `<title>` tag check |
| Missing H1 Tag | Medium | Heading structure analysis |
| Missing Viewport Meta | Medium | Mobile meta tag check |
| Missing Image Alt Text | Low/Medium | `<img alt="">` attribute check |
| Missing Meta Description | Low | Meta tag extraction |
| Multiple H1 Tags | Low | Heading count |
| Missing Canonical Tag | Low | `<link rel="canonical">` check |
| Missing Open Graph Tags | Low | OG meta tag check |

---

## 🔧 Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Max pages | 50 | Maximum pages to crawl |
| Timeout | 10s | Per-page timeout |
| Random seed | 42 | For reproducible ML results |
| Respect robots.txt | Yes | Toggle in sidebar |
| Cache TTL | 1 hour | Same URL won't be re-crawled |

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.35.0 | Web UI framework |
| requests | 2.31.0 | HTTP crawling |
| beautifulsoup4 | 4.12.3 | HTML parsing |
| pandas | 2.1.4 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.4.2 | ML model + NLP |
| plotly | 5.20.0 | Interactive charts |
| matplotlib | 3.8.2 | Static charts |
| seaborn | 0.13.2 | Confusion matrix heatmap |
| wordcloud | 1.9.3 | Word frequency visualization |
| lxml | 5.1.0 | Fast HTML parsing backend |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 FAQ

**Q: Why does the crawl return 0 pages?**  
A: The site may be blocking automated requests. Try unchecking "Respect robots.txt" or the site may require JavaScript rendering (this tool uses static HTML parsing only).

**Q: Can I crawl sites that require login?**  
A: No — the crawler doesn't support authentication. It only crawls publicly accessible pages.

**Q: Why is the ML model accuracy variable?**  
A: The model is trained on crawl-derived data. Small sites (< 20 issues) have limited training samples, which reduces accuracy. A warning is shown when sample size is low.

**Q: Will this work on JavaScript-heavy SPAs?**  
A: Partially. The crawler fetches raw HTML — client-side rendered content won't be visible. Server-side rendered pages work best.

**Q: Is there a rate limit?**  
A: The crawler adds a natural delay through sequential requests. It won't overwhelm sites, but for very large sites, the 50-page limit ensures reasonable crawl times.
