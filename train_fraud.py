"""
Insurance claims fraud detection — public reproduction of the Shift Technology
mission (confidential client data replaced by the public 'Auto Insurance Claims'
dataset, ~1000 claims with a fraud_reported label).

Two volets, like the real mission:
  1. Supervised scoring (XGBoost) with class-imbalance handling, evaluated with
     SIU metrics: PR-AUC, Precision@K, Lift by decile.
  2. Unsupervised anomaly detection (Isolation Forest) to surface *novel* fraud.

Feature engineering mirrors the mission, notably the underwriting->claim delay
(policy_bind_date -> incident_date), a strong fraud indicator.

Outputs real metrics to results.json (feeds the mebarki.dev case-study page).
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

RNG = 42
df = pd.read_csv("data2/insurance_claims.csv")
df = df.replace("?", np.nan)

y = (df["fraud_reported"] == "Y").astype(int)

# --- Feature engineering: underwriting -> claim delay (mission signal) ---
for c in ("policy_bind_date", "incident_date"):
    df[c] = pd.to_datetime(df[c], errors="coerce")
df["days_bind_to_incident"] = (df["incident_date"] - df["policy_bind_date"]).dt.days

drop = [
    "fraud_reported", "policy_number", "policy_bind_date", "incident_date",
    "incident_location", "insured_zip",
]
# drop all-empty junk columns (e.g. trailing _c39)
drop += [c for c in df.columns if df[c].isna().all()]
X = df.drop(columns=[c for c in drop if c in df.columns])

# object cols -> category (unknown for missing), then one-hot
for c in X.select_dtypes("object").columns:
    X[c] = X[c].fillna("unknown")
X = pd.get_dummies(X)
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RNG
)

# ---- Volet 1: supervised scoring with imbalance handling ----
spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
clf = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, eval_metric="aucpr",
    random_state=RNG, n_jobs=4,
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]

pr_auc = average_precision_score(y_test, proba)
roc_auc = roc_auc_score(y_test, proba)
base_rate = y_test.mean()

order = np.argsort(-proba)
y_sorted = y_test.values[order]
n = len(y_sorted)
decile = max(int(n * 0.10), 1)

def p_at_k(k):
    k = min(k, n)
    return y_sorted[:k].sum() / k

prec_at_50 = p_at_k(50)
prec_decile = p_at_k(decile)
lift_decile1 = prec_decile / base_rate
recall_decile = y_sorted[:decile].sum() / max(y_test.sum(), 1)

imp = clf.get_booster().get_score(importance_type="gain")
top_features = [k for k, _ in sorted(imp.items(), key=lambda kv: -kv[1])[:10]]

# ---- Volet 2: unsupervised anomaly detection (novel fraud) ----
iso = IsolationForest(n_estimators=300, contamination=float(base_rate), random_state=RNG, n_jobs=4)
iso.fit(X_train)
iso_score = -iso.score_samples(X_test)
iso_order = np.argsort(-iso_score)
iso_top = set(iso_order[:decile].tolist())
sup_top = set(order[:decile].tolist())
iso_prec_decile = y_test.values[list(iso_top)].sum() / len(iso_top)
fraud_idx = set(np.where(y_test.values == 1)[0].tolist())
novel = (iso_top & fraud_idx) - sup_top
novel_rate = len(novel) / max(len(iso_top & fraud_idx), 1)

results = {
    "dataset": "Auto Insurance Claims (insurance_claims.csv), 1000 claims",
    "n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "fraud_rate_pct": round(float(y.mean()) * 100, 1),
    "supervised": {
        "model": f"XGBoost (scale_pos_weight={spw:.1f})",
        "pr_auc": round(float(pr_auc), 3),
        "roc_auc": round(float(roc_auc), 3),
        "precision_at_50": round(float(prec_at_50) * 100, 1),
        "precision_top_decile": round(float(prec_decile) * 100, 1),
        "lift_decile1": round(float(lift_decile1), 1),
        "recall_top_decile": round(float(recall_decile) * 100, 1),
        "base_rate_pct": round(float(base_rate) * 100, 1),
    },
    "unsupervised": {
        "model": "Isolation Forest",
        "precision_top_decile": round(float(iso_prec_decile) * 100, 1),
        "novel_fraud_rate_pct": round(float(novel_rate) * 100, 1),
    },
    "top_features_gain": top_features,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(json.dumps(results, indent=2, ensure_ascii=False))
