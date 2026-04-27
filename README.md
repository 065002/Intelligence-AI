# 🔬 Intelligent App Testing System

> Paste any URL → Auto-crawl → Full AI-powered analysis → Actionable risk report

---

## 🚀 One-Click Deploy to Streamlit Cloud

1. **Push all files to GitHub** (root of your repo):
   ```
   app.py
   crawler.py
   analyzer.py
   requirements.txt
   README.md
   ```

2. Go to **[share.streamlit.io](https://share.streamlit.io)** → New app  
3. Select your repo · Branch: `main` · Main file: `app.py`  
4. Click **Deploy** — done in ~2 minutes

---

## 💻 Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/intelligent-app-tester.git
cd intelligent-app-tester
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`

---

## 📋 Features

| Page | What You Get |
|------|-------------|
| 🌐 **Website Overview** | Health score (0-100), crawl summary, structure map |
| 📊 **Exploratory Analysis** | Issue charts, severity distribution, load times |
| ⚠️ **Risk Scoring** | Per-module risk scores with colour-coded table |
| 🤖 **Prediction Model** | Random Forest classifier + live predictions |
| 💬 **NLP Analysis** | TF-IDF keywords, KMeans clusters, WordCloud |
| ✅ **Fix Validation** | Recurrence flags, recommendations, benchmarks |

---

## ⚙️ Risk Formula

```
Risk = (Issue Count × 0.4) + (Avg Severity Weight × 10 × 0.35) + (Broken Link Rate × 100 × 0.25)
Final = Normalised to 0–100
```

---

## 📦 Dependencies

All unpinned so pip picks the latest wheel compatible with your Python version (tested on Python 3.9–3.14):

```
streamlit, requests, beautifulsoup4, pandas, numpy,
matplotlib, seaborn, scikit-learn, plotly, wordcloud, urllib3, lxml
```
