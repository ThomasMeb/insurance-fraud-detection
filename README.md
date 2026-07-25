# Insurance Claims Fraud Detection

Public reproduction of an ML engineering mission I ran at **Shift Technology**
(insurtech, IARD fraud detection for insurers' SIU — Special Investigation Unit —
teams). The confidential client data is replaced here by a public **Auto
Insurance Claims** dataset (~1,000 claims, 24.7% fraud); the architecture,
methodology and evaluation mirror the real work.

## Two volets (like the mission)

### 1. Supervised scoring — XGBoost

Class-imbalance handling (`scale_pos_weight`), evaluated with the metrics an SIU
actually uses (Precision@K on the top alerts, Lift by decile), not just ROC-AUC.

| Metric | Value |
|---|---|
| ROC-AUC | 0.84 |
| PR-AUC | 0.58 |
| Precision@50 | 64.0% |
| Precision (top decile) | 60.0% |
| Lift (decile 1) | 2.4x |
| Recall (top decile) | 24.3% |
| Base rate | 24.7% |

Reading it: reviewing only the 50 highest-scored claims, ~64% are fraud (2.4x
the base rate) — the model concentrates the SIU's limited review capacity.

### 2. Unsupervised anomaly detection — Isolation Forest

Surfaces *novel* fraud with no historical label. ~50% of the frauds caught in the
anomaly model's top decile are ones the supervised model does **not** rank in its
own top decile — i.e. patterns not present in the labels.

## Feature engineering

Mirrors the mission — transactional and temporal signals, notably the
**underwriting→claim delay** (`policy_bind_date` → `incident_date`), a classic
fraud indicator, plus incident severity, claim amounts, past-claim frequency and
report/witness signals.

## Run

```
pip install -r requirements.txt
# place the public dataset at data2/insurance_claims.csv
python train_fraud.py     # trains both models, writes results.json
```

## Note

This is a **public proxy** for confidential Shift client data — the figures are
illustrative of the approach, not the production model. Dataset: Auto Insurance
Claims (Kaggle, public).
