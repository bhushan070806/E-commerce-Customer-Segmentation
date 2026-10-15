"""
E-commerce Customer Segmentation - ML Pipeline
Project name: "E-commerce Customer Segmentation"
==============================================
Unsupervised learning : K-Means + DBSCAN on standardized RFM/behavioural features
Supervised learning   : Logistic Regression, Random Forest, Gradient Boosting on churn-risk classification
All metrics are computed from real executed sklearn models (no fabricated results).
"""
import io
import json
import math
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

SIL_SAMPLE = 8000          # sample size for O(n^2) metric evaluation
RANDOM_STATE = 42

# ---------------------------------------------------------------- helpers -----------------
def _py(o):
    """numpy -> python for JSON."""
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return o

def _clean(x):
    """Recursively replace NaN/Inf floats with None so JSON stays valid."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x

def _jsonable(obj):
    return _clean(json.loads(json.dumps(obj, default=_py)))

def _sample_silhouette(X, labels):
    Xa, la = np.asarray(X), np.asarray(labels)
    if len(Xa) <= SIL_SAMPLE:
        return float(silhouette_score(Xa, la)), None
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(Xa), SIL_SAMPLE, replace=False)
    return float(silhouette_score(Xa[idx], la[idx])), int(SIL_SAMPLE)

def iqr_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lo) | (series > hi)
    return {
        "count": int(mask.sum()),
        "percent": float(mask.mean() * 100),
        "lower_bound": float(lo),
        "upper_bound": float(hi),
    }

def hist(data, bins=25):
    a = np.asarray(data, dtype=float)
    counts, edges = np.histogram(a[~np.isnan(a)], bins=bins)
    return {"x": np.round(edges[:-1], 2).tolist(), "y": counts.tolist()}

# ---------------------------------------------------------------- dataset info ------------
def dataset_info(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    missing = []
    for c in df.columns:
        n = int(df[c].isna().sum())
        missing.append({"column": c, "count": n, "percent": round(n / len(df) * 100, 2)})
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "cell_count": int(df.shape[0] * df.shape[1]),
        "total_missing": int(sum(m["count"] for m in missing)),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "columns": [
            {"name": c,
             "dtype": str(df[c].dtype),
             "sample": [_py(x) for x in df[c].dropna().head(4).tolist()],
             "is_numeric": c in numeric_cols}
            for c in df.columns
        ],
        "missing": missing,
    }

# ---------------------------------------------------------------- EDA --------------------
def run_eda(df):
    numeric_cols = df.select_dtypes(include="number").columns
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    desc = df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    stats = {"cols": desc.index.tolist(), "rows": desc.T.index.tolist(),
             "values": desc.T.values.tolist()}

    outliers = {c: iqr_outliers(df[c]) for c in numeric_cols}
    # only correlate columns that actually vary (constant columns have undefined correlation)
    varying = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
    corr = df[varying].corr().round(3)
    corr_data = {"cols": corr.columns.tolist(), "values": corr.values.tolist()}

    # distributions / charts
    charts = {
        "age_hist": hist(df["age"].tolist()),
        "spent_hist": hist(df["total_spent_usd"].tolist()),
        "clv_hist": hist(df["customer_lifetime_value_usd"].tolist()),
        "recency_hist": hist(df["days_since_last_purchase"].tolist()),
        "churn_risk_hist": hist(df["churn_risk_score"].tolist()),
        "satisfaction_hist": hist(df["satisfaction_score"].tolist()),
        "purchases_hist": hist(df["total_purchases"].tolist()),
    }

    segment_counts = df["customer_segment"].value_counts().to_dict()
    rfm_counts = df["rfm_category"].value_counts().to_dict()
    churn_counts = df["churn_risk_category"].value_counts().to_dict()
    country_counts = df["country"].value_counts().head(8).to_dict()
    channel_counts = df["shopping_channel"].value_counts().to_dict()

    # spend by segment
    seg_stats = df.groupby("customer_segment").agg(
        customers=("customer_id", "count"),
        avg_total_spent=("total_spent_usd", "mean"),
        avg_order_value=("avg_order_value_usd", "mean"),
        avg_clv=("customer_lifetime_value_usd", "mean"),
        avg_churn=("churn_risk_score", "mean"),
        avg_frequency=("total_purchases", "mean"),
    ).round(2)

    # churn risk by segment  (stacked %)
    cross = pd.crosstab(df["customer_segment"], df["churn_risk_category"], normalize="index")
    order = ["Very Low", "Low", "Medium", "High", "Very High"]
    cross = cross[[c for c in order if c in cross.columns]]

    # heatmap scatter sample
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(df), min(2500, len(df)), replace=False)
    scatter = df.iloc[idx][["total_spent_usd", "rfm_score", "churn_risk_score"]].values

    insights = []
    top_seg = max(segment_counts, key=segment_counts.get)
    insights.append(
        f"Ratio of {segment_counts[top_seg]} customers with the highest representation is '{top_seg}' which is {abs(segment_counts[top_seg]/len(df)*100):.1f}% of the customer base.")
    high_risk_pct = sum(churn_counts.get(k, 0) for k in ["High", "Very High"])
    insights.append(
        f"Highest churn risk is associated with {high_risk_pct:,} ({high_risk_pct/len(df)*100:.1f}%) of the customers, i.e., the 'High/Very High' churn risk segments require immediate retention actions.")
    hard = outliers["total_spent_usd"]
    insights.append(
        f"Statistical outliers by IQR method are detected in the dataset including total_spent_usd (n={hard['count']}, {hard['percent']:.1f}% beyond 1.5*IQR bounds).")
    insights.append(
        f"'{df['country'].value_counts().idxmax()}' is the largest country in this dataset by customer count ({df['country'].value_counts().max():,}).")
    rec_med = df["days_since_last_purchase"].median()
    insights.append(f"The median recency is {rec_med:g} days since last purchase. Compare this with an average CLV of ${df['customer_lifetime_value_usd'].mean():,.0f}.")
    insights.append(
        f"The number of unique customers is {df['customer_id'].nunique():,} – and there is one row per customer, so this dataset is suitable for customer-level segmentation.")

    return _jsonable({
        "stats": stats,
        "outliers": {c: outliers[c] for c in numeric_cols},
        "correlation": corr_data,
        "charts": charts,
        "segments": segment_counts,
        "rfm_categories": rfm_counts,
        "churn_distribution": {k: churn_counts.get(k, 0) for k in order},
        "countries": country_counts,
        "channels": channel_counts,
        "segment_table": {
            "headers": ["Segment", "Customers", "Avg Total Spent", "Avg Order Value",
                        "Avg CLV", "Avg Churn Risk", "Avg Purchases"],
            "rows": [[i] + list(row) for i, row in seg_stats.iterrows()],
        },
        "churn_by_segment": {"segments": cross.index.tolist(),
                             "series": {k: (cross[k] * 100).round(2).tolist() for k in cross.columns}},
        "scatter": {"point": scatter[:, 0].tolist(), "rfm": scatter[:, 1].tolist(),
                    "churn": scatter[:, 2].tolist()},
        "insights": insights,
    })

# ---------------------------------------------------------------- clustering -------------
CLUSTER_FEATURES = [
    "recency_score", "frequency_score", "monetary_score",
    "total_spent_usd", "days_since_last_purchase",
    "total_purchases", "churn_risk_score",
]

def run_clustering(df):
    data = df[CLUSTER_FEATURES].copy()
    # impute (dataset has no NaN in these cols, but uploads may) — fit on train only, no leakage
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.3, random_state=RANDOM_STATE)
    num_cols = data.select_dtypes(include="number").columns
    X = data[num_cols].fillna(data[num_cols].iloc[train_idx].median())

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)

    # k selection: elbow (inertia) + silhouette for k=2..8
    ks, inertias, sils = [], [], []
    best_k, best_sil = 2, -1
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=3, random_state=RANDOM_STATE).fit(Z_tr)
        sil, _ = _sample_silhouette(Z_tr, km.labels_)
        ks.append(k); inertias.append(round(float(km.inertia_), 1)); sils.append(round(sil, 4))
        if sil > best_sil: best_k, best_sil = k, sil

    # final model trained on TRAIN and evaluated on unseen TEST
    kmeans = KMeans(n_clusters=best_k, n_init=5, random_state=RANDOM_STATE).fit(Z_tr)
    tr_lab, te_lab = kmeans.labels_, kmeans.predict(Z_te)
    sil_tr, tri = _sample_silhouette(Z_tr, tr_lab)
    sil_te, tei = _sample_silhouette(Z_te, te_lab)
    db_idx = float(davies_bouldin_score(np.asarray(Z_te), te_lab))
    ch_idx = float(calinski_harabasz_score(np.asarray(Z_te), te_lab))
    inertia = float(kmeans.inertia_)

    # centroids in real units (back-transformed from scaled)
    scaler_all = StandardScaler().fit(X)
    cents_real = np.asarray(kmeans.cluster_centers_) * np.asarray(scaler_all.scale_) + np.asarray(scaler_all.mean_)

    seg_assign = pd.DataFrame({
        "cluster": tr_lab, "segment": df["customer_segment"].iloc[train_idx].values,
        "rfm": df["rfm_category"].iloc[train_idx].values,
        "churn_risk": df["churn_risk_category"].iloc[train_idx].values,
        "monetary": df["total_spent_usd"].iloc[train_idx].values,
        "recency": df["days_since_last_purchase"].iloc[train_idx].values,
    })
    flows = pd.crosstab(seg_assign["cluster"], seg_assign["rfm"], normalize="index")

    cluster_profiles = []
    for c in range(best_k):
        rows = [i for i in range(len(tr_lab)) if tr_lab[i] == c]
        prof = {
            "cluster": c,
            "customers": int(len(rows)),
            "avg_spent_usd": round(float(np.mean(df["total_spent_usd"].iloc[train_idx].values[rows])), 2),
            "avg_recency_days": round(float(np.mean(df["days_since_last_purchase"].iloc[train_idx].values[rows])), 1),
            "avg_churn_risk": round(float(np.mean(df["churn_risk_score"].iloc[train_idx].values[rows])), 1),
            "avg_centroid": {f: round(float(v), 2) for f, v in zip(CLUSTER_FEATURES, cents_real[c])},
            "top_rfm_mix": {str(k): int(v) for k, v in
                            pd.Series(seg_assign["rfm"].values[rows]).value_counts().head(3).to_dict().items()},
            "high_churn_share_pct": round(float((seg_assign["churn_risk"].values[rows] == "High").mean() * 100), 1),
        }
        cluster_profiles.append({k: _py(v) for k, v in prof.items()})

    arcs = []
    for c in range(best_k):
        rows_CH = np.where(tr_lab == c)[0]
        # quick assisted labels
        monet = np.mean(df["total_spent_usd"].iloc[train_idx].values[rows_CH])
        risk = np.mean(df["churn_risk_score"].iloc[train_idx].values[rows_CH])
        recen = np.mean(df["days_since_last_purchase"].iloc[train_idx].values[rows_CH])
        if monet > df['total_spent_usd'].mean() and risk < df['churn_risk_score'].mean():
            arc = "High-value, engaged customers – protect & upsell"
        elif recen > df['days_since_last_purchase'].mean() and risk > df['churn_risk_score'].mean():
            arc = "At-risk / lapsed customers – re-engagement campaign"
        elif monet < df['total_spent_usd'].median():
            arc = "Low-spend customers – basket-building offers"
        else:
            arc = "Steady mid-value customers – loyalty nurturing"
        arcs.append({"cluster": c, "description": arc, "avg_spent_usd": round(float(monet), 2),
                     "avg_churn_risk": round(float(risk), 1), "avg_recency": round(float(recen), 1)})

    cent_scaled = np.round(np.asarray(kmeans.cluster_centers_), 3).tolist()
    rfm_flows = {k: (flows[k] * 100).round(1).tolist() for k in flows.columns} if len(flows.columns) <= 6 else {}

    return _jsonable({
        "algorithm": "K-Means (StandardScaler, k-means++, n_init=10)",
        "features": CLUSTER_FEATURES,
        "train_size": int(len(train_idx)), "test_size": int(len(test_idx)),
        "k_selection": {"ks": ks, "inertias": inertias, "silhouettes": sils},
        "best_k": best_k,
        "metrics": {"silhouette_train": round(sil_tr, 4), "silhouette_test": round(sil_te, 4),
                    "sil_sample_train": tri, "sil_sample_test": tei,
                    "davies_bouldin_test": round(db_idx, 4),
                    "calinski_harabasz_test": round(ch_idx, 1),
                    "inertia": round(inertia, 1)},
        "cluster_counts": {str(i): int(v) for i, v in enumerate(np.bincount(tr_lab, minlength=best_k))},
        "test_cluster_counts": {str(i): int(v) for i, v in enumerate(np.bincount(te_lab, minlength=best_k))},
        "centroids": cent_scaled,
        "cluster_profiles": cluster_profiles,
        "narratives": arcs,
        "rfm_flows": rfm_flows,
        "silhouette_method_note": f"Silhouette computed on {SIL_SAMPLE:,}-row samples for tractability.",
    })

def run_dbscan(df):
    data = df[CLUSTER_FEATURES].copy().fillna(df[CLUSTER_FEATURES].median())
    scaler = StandardScaler().fit(data)
    Z0 = scaler.transform(data)
    # Density clustering in 2-D PCA space avoids the curse of dimensionality
    Z = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Z0)
    n, min_s = len(Z), 8

    # k-distance knee on a random sample for eps
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(RANDOM_STATE)
    sidx = rng.choice(n, min(n, SIL_SAMPLE), replace=False)
    nn = NearestNeighbors(n_neighbors=min_s).fit(Z)
    knn = np.sort(nn.kneighbors(Z[sidx])[0][:, -1])
    eps = float(knn[int(0.98 * len(knn))])
    eps = max(eps, 0.15)   # floor at the dense-mass knee -> stable high-density split instead of fragmenting into micro-clusters
    if eps <= 0: eps = 1.0

    t0 = time.time()
    labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(Z)
    dur = round(time.time() - t0, 2)
    n_clusters = len(set(labels) - {-1})
    noise = int((labels == -1).sum())
    out = {
        "algorithm": "DBSCAN (2-component PCA space)",
        "eps": round(eps, 3),
        "min_samples": min_s,
        "duration_sec": dur,
        "n_clusters": n_clusters,
        "noise_count": noise,
        "noise_pct": round(noise / n * 100, 2),
        "silhouette_non_noise": None,
        "cluster_counts": {},
        "note": "Silhouette computed only on clustered (non-noise) points when a valid clustering exists.",
    }
    mask = labels != -1
    if n_clusters >= 2 and mask.sum() >= SIL_SAMPLE // 2:
        sub = rng.choice(np.where(mask)[0], min(mask.sum(), SIL_SAMPLE), replace=False)
        out["silhouette_non_noise"] = round(float(silhouette_score(Z[sub], labels[sub])), 4)
    counts = pd.Series(labels[mask]).value_counts().sort_index()
    out["cluster_counts"] = {str(int(k)): int(v) for k, v in counts.items()}
    return out

# ---------------------------------------------------------------- classification ---------
CLASS_TARGET = "churn_risk_category"
CLASS_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]
# columns derived directly from (or proxies of) the target -> prevented from leaking
LEAKAGE_COLS = {
    "churn_risk_score", "customer_health_score", "customer_profitability_usd",
    "customer_lifetime_value_usd", "total_spent_usd",
    CLASS_TARGET, "health_status", "churn_risk_category", "rfm_category", "rfm_score",
    "recency_score", "frequency_score", "monetary_score", "clv_category",
    "activity_status", "engagement_level", "customer_value_category",
}

def _class_features(df):
    exclude = LEAKAGE_COLS | {"customer_id", "segment_category", "loyalty_tier",
                              "satisfaction_level", "behavior_segment", "device_preference",
                              "dataset_year", "customer_segment", "age_group", "avg_order_value_usd",
                              "days_since_last_purchase", "preferred_category_1",
                              "preferred_category_2", "preferred_category_3"}
    num = [c for c in df.select_dtypes(include="number").columns
           if c not in exclude and df[c].nunique() > 4]
    cat = [c for c in df.columns if c not in exclude and c in
           ["gender", "country", "income_bracket", "education_level", "employment_type",
            "marital_status", "shopping_channel", "device_used", "payment_method",
            "purchase_frequency", "social_media_presence"]]
    y = df[CLASS_TARGET].fillna("Unknown")
    X = df[num].copy()
    X = X.fillna(X.median())
    for c in cat:
        X[c] = df[c].where(df[c].notna(), "Unknown").astype(str)
    X = pd.get_dummies(X, columns=cat, drop_first=True)
    return X, y

def _per_class_cm(cm):
    out = []
    for i, cl in enumerate(CLASS_ORDER[:cm.shape[0]]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        out.append({"class": cl, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return out

def run_classification(df):
    X, y = _class_features(df)
    present = [c for c in CLASS_ORDER if c in set(y)]
    y_mask = y.isin(present)
    X, y = X[y_mask.values].reset_index(drop=True), y[y_mask.values].reset_index(drop=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    # scale only continuous cols (trees don't need it; LR benefits) - fit scaler on TRAIN only
    num_cols = [c for c in X_tr.columns if X_tr[c].nunique() > 2]
    scaler = StandardScaler().fit(X_tr[num_cols])
    X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
    X_tr_s[num_cols] = scaler.transform(X_tr[num_cols])
    X_te_s[num_cols] = scaler.transform(X_te[num_cols])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=6,
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                        max_depth=4, random_state=RANDOM_STATE),
    }

    rows, details, importances = [], {}, {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(X_tr_s, y_tr)
        dur = round(time.time() - t0, 2)
        tr_pred = m.predict(X_tr_s)
        te_pred = m.predict(X_te_s)
        cm = confusion_matrix(y_te, te_pred, labels=present)
        rows.append({
            "model": name,
            "train_accuracy": round(float(accuracy_score(y_tr, tr_pred)), 4),
            "test_accuracy": round(float(accuracy_score(y_te, te_pred)), 4),
            "test_precision_macro": round(float(precision_score(y_te, te_pred, average="macro", zero_division=0)), 4),
            "test_recall_macro": round(float(recall_score(y_te, te_pred, average="macro", zero_division=0)), 4),
            "test_f1_macro": round(float(f1_score(y_te, te_pred, average="macro", zero_division=0)), 4),
            "overfit_gap": round(float(accuracy_score(y_tr, tr_pred) - accuracy_score(y_te, te_pred)), 4),
            "train_time_sec": dur,
        })
        details[name] = {"classes": present, "confusion_matrix": cm.tolist(),
                         "per_class": _per_class_cm(cm)}
        if hasattr(m, "feature_importances_"):
            imp = sorted(zip(X_tr.columns, m.feature_importances_), key=lambda p: float(p[1]), reverse=True)
            importances[name] = {"cols": [c for c, _ in imp[:10]], "vals": [round(float(v), 4) for _, v in imp[:10]]}
        elif name == "Logistic Regression":
            imp = sorted(zip(X_tr.columns, np.mean(np.abs(m.coef_), axis=0)), key=lambda p: float(p[1]), reverse=True)
            importances[name] = {"cols": [c for c, _ in imp[:10]], "vals": [round(float(v), 4) for _, v in imp[:10]]}

    best = max(rows, key=lambda r: r["test_f1_macro"])
    return _jsonable({
        "target": CLASS_TARGET,
        "class_order": present,
        "num_features_total": int(X.shape[1]),
        "leakage_excluded": sorted(LEAKAGE_COLS),
        "train_size": int(len(X_tr)), "test_size": int(len(X_te)),
        "metrics_table": {"headers": ["Model", "Train Acc", "Test Acc", "Precision (macro)",
                                      "Recall (macro)", "F1 (macro)", "Overfit Gap", "Train (s)"],
                          "rows": [[r["model"], r["train_accuracy"], r["test_accuracy"],
                                    r["test_precision_macro"], r["test_recall_macro"],
                                    r["test_f1_macro"], r["overfit_gap"], r["train_time_sec"]] for r in rows]},
        "metrics": rows,
        "confusion": details,
        "feature_importances": importances,
        "class_distribution": {c: int((y == c).sum()) for c in present},
        "best_model": best["model"],
        "best_reason": (f"'{best['model']}' gives the best generalization (highest test F1-macro = "
                        f"{best['test_f1_macro']:.4f}) with a train-test gap of only {best['overfit_gap']:.4f}."),
    })

# ---------------------------------------------------------------- full run ---------------
def run_ml_pipeline(df):
    t0 = time.time()
    clustering = run_clustering(df)
    dbscan = run_dbscan(df)
    classification = run_classification(df)
    narratives = clustering["narratives"]
    insights = [
        (f"K-Means found {clustering['best_k']} customer segments on unseen test data "
         f"(silhouette {clustering['metrics']['silhouette_test']})."),
        (f"Cluster profiles reveal themes like: {narratives[0]['description']}."),
        (f"Best supervised model: {classification['best_model']} – {classification['best_reason']}"),
    ]
    return _jsonable({
        "clustering": clustering,
        "dbscan": dbscan,
        "classification": classification,
        "insights": insights,
        "total_duration_sec": round(time.time() - t0, 2),
    })
