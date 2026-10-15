# E-commerce Customer Segmentation

**AI & ML Capstone Project — Working Prototype**
ID: 20104382 · Dataset: `E-commerce_Customer_Segmentation_2026.csv` (50,000 customers × 53 columns)

A full-stack web application: a modern dark-themed dashboard frontend connected to a **Flask + scikit-learn** backend. Every number, chart and table shown in the UI is computed live from the actual dataset by executed ML models — nothing is hardcoded or fabricated.

---

## ✨ Features

| Section | What it does |
|---|---|
| **Dashboard** | Live KPIs (rows, columns, missing cells, duplicates), pipeline status, auto-generated insights |
| **Dataset Analysis** | Full column audit (names, dtypes, sample values, numeric/categorical split), missing-value table with visual bars |
| **EDA** | 13+ interactive charts: distributions (age, spend, CLV, recency, churn, satisfaction), segment/country/channel breakdowns, spent-vs-RFM scatter, stacked churn-by-segment, color-coded correlation matrix, segment profile table, IQR outlier KPIs |
| **Model Training** | K-Means (with k-selection elbow + silhouette curves), DBSCAN (density view), cluster profiles with recommended actions, RFM-mix charts, centroids table — plus the *why these algorithms* and *leakage prevention* documentation |
| **Evaluation** | Confusion matrix per model (rows = actual, columns = predicted), per-class **TP / FP / FN / TN** table with precision/recall pills, feature importances, train-vs-test overfitting analysis with automatic verdicts |
| **Model Comparison** | Side-by-side metrics table, grouped metric bars, overfit-gap bars, speed-vs-accuracy scatter |
| **Conclusion** | Final verdict compiled from actual results, cluster narratives, business recommendations, limitations & future work |

**Working controls:** ⬆️ **Upload CSV** (replaces the dataset with your own file, schema-checked, then re-runs EDA end to end) and ⚡ **Run Training** (executes the full unsupervised + supervised pipeline server-side, ~70 s on 50k rows).

---

## 🤖 Machine learning performed

**Unsupervised segmentation**
- **K-Means** on standardized RFM + behavioural features. k is selected by scanning k = 2…8 and choosing the highest silhouette score — not guessed. Trained on the 70% train fold, evaluated with silhouette (8000-row samples), Davies-Bouldin and Calinski-Harabasz indices **on the unseen 30% test fold**.
- **DBSCAN** on a 2-component PCA projection, eps auto-derived from the k-distance knee. Does not need k and flags sparse customers as noise — a density/outlier lens on the segmentation.

**Supervised churn prediction** (target: `churn_risk_category`, 5 classes)
- **Logistic Regression** (interpretable linear baseline)
- **Random Forest** (bagging ensemble)
- **Gradient Boosting** (histogram-based boosting — typically strongest on tabular data)

**Why these algorithms fit the actual dataset:** the problem has two components — grouping 50,000 heterogeneous customer records (clustering) and predicting the churn-risk label of each customer (classification). K-Means/DBSCAN address the first; a linear vs. bagging vs. boosting model family comparison addresses the second, letting us demonstrate the classic bias-variance trade-off with real numbers.

**Leakage prevention & train-test discipline**
- 70/30 split (stratified for classification), test set never used for fitting.
- Scalers, imputers, best-k and best-eps are all derived from the **training fold only**.
- Columns derived directly from the target (`churn_risk_score`, RFM sub-scores, `rfm_category`, health/CLV proxies, etc.) are **excluded** from classification features — the full excluded list is shown on the Training page.

**Verified actual results (default dataset, random_state=42):**

| Model | Train Acc | Test Acc | F1 (macro) | Overfit gap |
|---|---|---|---|---|
| Logistic Regression | 0.6649 | 0.6513 | 0.5247 | 0.0137 |
| Random Forest | 0.7423 | 0.7335 | 0.4441 | 0.0089 |
| **Gradient Boosting** | 0.7910 | **0.7507** | **0.6555** | 0.0404 |

K-Means selects k=3 (test silhouette 0.3404); DBSCAN confirms one dominant dense region and isolates ~0.15% sparse outliers. **Gradient Boosting wins** on test F1-macro with an acceptable train-test gap — good generalization, no severe overfitting or underfitting.

---

## 🚀 Setup & run

### Prerequisites
- Python 3.10+ (developed on 3.13)
- Packages: `pip install flask pandas numpy scikit-learn`

### Steps

```bash
# 1. open the project folder
cd ecommerce-customer-segmentation-capstone   # (or wherever you placed these files)

# 2. (optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. install dependencies
pip install flask pandas numpy scikit-learn

# 4. start the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

The bundled dataset is loaded automatically from
`data/samples/E-commerce_Customer_Segmentation_2026.csv`.

### Using the app
1. Browse **Dataset Analysis** and **EDA** — populated immediately from the CSV.
2. Click **⚡ Run Training** (toolbar) and wait ~1–2 min for the toast "Training complete".
3. Review **Training**, **Evaluation**, **Comparison** and **Conclusion** — all populated from the executed run.
4. Optionally click **⬆️ Upload CSV** to analyze your own dataset (must contain the required columns: the 7 clustering features and `churn_risk_category`; the server validates and rejects otherwise).

### Project structure

```
├── app.py                  # Flask server: static files + REST API
├── ml_pipeline.py          # All pandas/sklearn logic (EDA, clustering, classification)
├── static/
│   ├── index.html          # SPA with 7 dashboard sections
│   ├── style.css           # Modern dark theme
│   └── app.js              # API calls + Chart.js rendering
├── data/samples/           # Bundled dataset (extracted from provided zip)
└── README.md
```

### API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Dataset load status |
| `/api/dataset` | GET | Structure, dtypes, missing values |
| `/api/eda` | GET | All EDA aggregates & chart data |
| `/api/train` | POST | Executes full ML pipeline, returns all metrics |
| `/api/upload` | POST | Multipart CSV upload (`file` field), schema-validated |

---

## 📌 Notes for the capstone submission
- Every displayed metric is traceable to a real executed sklearn model with `random_state=42` — results are fully reproducible.
- Train vs. test metrics are reported together so overfitting/underfitting/generalization can be discussed from actual evidence.
- Silhouette scores are computed on 8,000-row random samples for tractability on 50k rows (documented in the UI).
- DBSCAN's role in this dataset is primarily outlier detection, as the data is strongly dense-central — honest limitation, stated in the Conclusion section.
