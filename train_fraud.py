"""
Insurance claims fraud detection — public reproduction of the Shift Technology
mission (confidential client data replaced by public datasets).

Honest comparative on TWO public datasets, because each illustrates a different
facet of the real problem:
  - carclaims (~15k, ~6% fraud): realistic class imbalance (close to the real
    1-3%), evaluated out-of-time. Weak but honest signal — real fraud is hard.
  - Auto Insurance Claims (~1k, 24.7% fraud): richer features (claim amounts,
    underwriting->claim delay), stronger signal. The synthetic 'insured_hobbies'
    leakage is DROPPED so the numbers reflect real modelling, not the artefact.

Fraud is a rare-positive problem, so we report PR-AUC, Precision@K and Lift
(what an SIU actually uses); ROC-AUC is kept only as a secondary reference
because it is misleadingly optimistic under class imbalance.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

RNG = 42


def evaluate(X_train, X_test, y_train, y_test, split_desc):
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    clf = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="aucpr", random_state=RNG, n_jobs=4,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]

    base = float(y_test.mean())
    order = np.argsort(-proba)
    ys = y_test.values[order]
    n = len(ys)
    dec = max(int(n * 0.10), 1)
    p50 = ys[:min(50, n)].sum() / min(50, n)
    p_dec = ys[:dec].sum() / dec

    iso = IsolationForest(n_estimators=300, contamination=base, random_state=RNG, n_jobs=4)
    iso.fit(X_train)
    iso_order = np.argsort(-(-iso.score_samples(X_test)))
    iso_top, sup_top = set(iso_order[:dec].tolist()), set(order[:dec].tolist())
    fraud_idx = set(np.where(y_test.values == 1)[0].tolist())
    novel = (iso_top & fraud_idx) - sup_top
    novel_rate = len(novel) / max(len(iso_top & fraud_idx), 1)

    return {
        "split": split_desc,
        "n_test": int(n),
        "fraud_rate_pct": round(base * 100, 1),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 3),
        "precision_at_50": round(float(p50) * 100, 1),
        "precision_top_decile": round(float(p_dec) * 100, 1),
        "lift_decile1": round(float(p_dec / base), 1),
        "recall_top_decile": round(float(ys[:dec].sum() / max(y_test.sum(), 1)) * 100, 1),
        "roc_auc_secondary": round(float(roc_auc_score(y_test, proba)), 3),
        "novel_fraud_rate_pct": round(float(novel_rate) * 100, 1),
    }


def carclaims():
    df = pd.read_csv("data/fraud_oracle.csv").drop(columns=["PolicyNumber"])
    y = df.pop("FraudFound_P")
    ty = sorted(df["Year"].unique())[-1]
    tr, te = df["Year"] < ty, df["Year"] == ty
    X = pd.get_dummies(df)
    return evaluate(X[tr], X[te], y[tr], y[te], f"out-of-time (test={ty})")


def autoclaims():
    df = pd.read_csv("data2/insurance_claims.csv").replace("?", np.nan)
    y = (df["fraud_reported"] == "Y").astype(int)
    for c in ("policy_bind_date", "incident_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["days_bind_to_incident"] = (df["incident_date"] - df["policy_bind_date"]).dt.days
    drop = ["fraud_reported", "policy_number", "policy_bind_date", "incident_date",
            "incident_location", "insured_zip", "insured_hobbies"]  # hobbies = synthetic leakage
    drop += [c for c in df.columns if df[c].isna().all()]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    for c in X.select_dtypes("object").columns:
        X[c] = X[c].fillna("unknown")
    X = pd.get_dummies(X).fillna(0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RNG)
    return evaluate(Xtr, Xte, ytr, yte, "stratified 70/30")


results = {
    "note": "Fraud is a rare-positive problem: PR-AUC / Precision@K / Lift are the decision metrics; ROC-AUC kept only as secondary reference. insured_hobbies dropped (synthetic leakage).",
    "auto_insurance_claims_deleaked": autoclaims(),
    "carclaims_realistic_imbalance": carclaims(),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(json.dumps(results, indent=2, ensure_ascii=False))
