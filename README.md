# Insurance Claims Fraud Detection

Public reproduction of an ML engineering mission I ran at **Shift Technology**
(insurtech, IARD fraud detection for insurers' SIU — Special Investigation Unit —
teams). Confidential client data is replaced by public datasets; the
architecture and methodology mirror the real work.

## Metric philosophy

Fraud is a **rare-positive** problem. The decision metrics are **PR-AUC**,
**Precision@K** (how much fraud sits in the top-K alerts an analyst actually
reviews) and **Lift by decile**. **ROC-AUC is misleadingly optimistic under
class imbalance**, so it is kept only as a secondary reference — not a headline.

## Honest comparative — two datasets

Each public dataset shows a different facet of the real problem, so both are
reported rather than cherry-picking the flattering one.

| Dataset | Fraud rate | Split | PR-AUC | Precision@50 | Lift (dec. 1) | ROC-AUC* |
|---|---|---|---|---|---|---|
| Auto Insurance Claims (richer features) | 24.7% | stratified 70/30 | **0.49** | **58%** | **2.3x** | 0.74 |
| carclaims (realistic imbalance) | 5.2% | out-of-time | 0.12 | 16% | 1.6x | 0.74 |

*ROC-AUC = secondary reference only.

**Reading it.** On the realistic-imbalance dataset (closer to the real 1-3%),
signal is genuinely hard: PR-AUC 0.12, but the top decile still concentrates
fraud at ~1.6x the base rate and Precision@50 is ~3x the base rate — that is the
real SIU value (focus scarce review capacity), not a shiny AUC. On the
feature-rich dataset, Lift reaches 2.3x and Precision@50 58%.

Notes on honesty:
- `insured_hobbies` in the Auto Insurance Claims set is a **synthetic leakage**
  feature (planted correlation with the label); it is **dropped** — with it, the
  numbers were inflated (ROC 0.84 vs 0.74, PR-AUC 0.58 vs 0.49).
- The public proxies do not reproduce the real 1-3% imbalance exactly; the
  metric choices (PR-AUC / Precision@K / Lift) are what transfer.

## Two volets (like the mission)

1. **Supervised scoring — XGBoost** with class-imbalance handling
   (`scale_pos_weight`), evaluated with the SIU metrics above.
2. **Unsupervised anomaly detection — Isolation Forest** for *novel* fraud with
   no historical label: **75-85%** of the anomaly top-decile frauds are ones the
   supervised model does not rank in its own top decile.

Feature engineering mirrors the mission — notably the **underwriting→claim
delay** (`policy_bind_date` → `incident_date`).

## Run

```
pip install -r requirements.txt
# data/fraud_oracle.csv  (carclaims)  +  data2/insurance_claims.csv  (auto claims)
python train_fraud.py     # trains both, writes results.json
```

## Note

Public proxy for confidential Shift client data — figures illustrate the
approach, not the production model. Datasets: Kaggle (public).
