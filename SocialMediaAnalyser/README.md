# 📊 Social Media Engagement Analyser
### Full Python Data Science Project | Streamlit + Plotly | Age Category Analysis

---

## 🚀 Quick Start in VS Code

```bash
# Step 1 — Open project folder
# File → Open Folder → select "SocialMediaAnalyser"

# Step 2 — Create virtual environment
python -m venv venv

# Step 3 — Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Step 4 — Install dependencies
pip install -r requirements.txt

# Step 5 — Run dashboard
streamlit run app.py
```
Opens at → **http://localhost:8501**

---

## 📁 Project Structure

```
SocialMediaAnalyser/
│
├── app.py                      ← 🚀 Main Streamlit dashboard (run this)
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── assets/
│   └── style.css               ← Custom CSS (White/Purple theme)
│
├── data/
│   ├── __init__.py
│   ├── generate_data.py        ← 5000-record synthetic dataset generator
│   └── preprocess.py           ← Cleaning, aggregations, KPI summary
│
├── models/
│   ├── __init__.py
│   ├── predictor.py            ← Random Forest engagement predictor
│   └── clusterer.py            ← K-Means persona clustering (5 personas)
│
└── utils/
    ├── __init__.py
    └── charts.py               ← 20 Plotly interactive charts
```

---

## 🖥️ Dashboard Tabs

| Tab | Content |
|---|---|
| **📊 Overview** | Hero KPIs, engagement bar, donut, heatmap, stacked bar, content grouped bar, monthly area trend, bubble matrix |
| **🔬 Deep Analysis** | Box plot, violin, radar, scatter, hourly heatmap, correlation matrix, funnel, top-20 posts table, summary stats |
| **🌍 Demographics** | Gender heatmap, region grouped bar, gender × age table, region × age table |
| **🤖 ML Prediction** | RF predictor form, feature importance chart, model metrics, smart optimisation tips |
| **👥 Persona Clusters** | 5 persona cards, PCA scatter, persona × age heatmap, summary table, pivot table |
| **🔴 Live Feed** | Adjustable real-time simulation, live chart updates, latest posts table, CSV export |

---

## 📊 Data Fields

| Field | Description |
|---|---|
| Age_Group | 13-17, 18-24, 25-34, 35-44, 45-54, 55+ |
| Platform | Instagram, Facebook, Twitter/X, TikTok, YouTube, LinkedIn |
| Content_Type | Video, Image, Story, Reel, Article, Live |
| Gender | Male, Female, Non-binary |
| Region | North America, Europe, Asia, South America, Middle East, Africa |
| Engagement_Rate | (Likes+Comments+Shares+Saves) / Views × 100 |
| Virality_Score | Shares / Views × 100 |
| Click_Rate | Clicks / Impressions × 100 |
| Reach_Rate | Reach / Followers × 100 |

---

## 🤖 ML Models

### Random Forest Predictor
- **Input**: Platform, Age Group, Content Type, Gender, Region, Day, Hour, Followers, Quarter
- **Output**: Predicted Engagement Rate (%)
- **Metrics**: R², RMSE, MAE, Cross-Validation R²

### K-Means Clustering (5 Personas)
- 🔥 Viral Creators
- 💬 Active Commenters
- 👁️ Passive Scrollers
- 📤 Share Champions
- ⭐ Niche Enthusiasts
- **Evaluated with**: Silhouette Score + PCA visualisation

---

## 🛠 Tech Stack

| Library | Version | Use |
|---|---|---|
| Streamlit | ≥1.32 | Dashboard framework |
| Plotly | ≥5.18 | 20 interactive charts |
| Pandas | ≥2.0 | Data wrangling |
| NumPy | ≥1.24 | Numerical processing |
| Scikit-learn | ≥1.3 | ML models |

---

## 🎨 Design System
- **Framework**: Streamlit + custom CSS
- **Theme**: Light Modern White/Purple
- **Fonts**: Syne (headings) · DM Sans (body)
- **Colors**: `#7C3AED` Purple · `#10B981` Green · `#F59E0B` Amber · `#3B82F6` Blue
