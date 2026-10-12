# Results Report — Federated Architecture (synthetic (signal-bearing))

**What this shows:** the full pipeline runs end-to-end and the federated design
*works* — simulated hospitals train locally, a centre node aggregates their weights
with plain **FedAvg**, and the global model recovers most of the pooled-data ceiling
while clearly beating any hospital training alone.

**Defence layers are OFF** (no differential privacy, no Byzantine-robust
aggregation, no attacks) — this isolates the core learning loop.

---

## Setup

| Item | Value |
|---|---|
| Dataset | synthetic (signal-bearing), 48x48 px |
| Model | densenet121 (from scratch), dual head (pathology + severity) |
| Hospitals (nodes) | 3 |
| Partition | non-IID (Dirichlet alpha=0.5) |
| Aggregation | plain FedAvg |
| Defences | none (plain FedAvg, no DP, no robust aggregation) |
| Baseline epochs | 8 · Federated rounds | 12 |

---

## Headline result

| Configuration | Mean ROC-AUC | What it represents |
|---|---|---|
| Single-site (isolated) | **0.831** | floor — one hospital, its own data only |
| **Federated (FedAvg)** | **0.982** | our architecture |
| Centralised (pooled) | **0.989** | ceiling — all data in one place |

- Federated closes **95%** of the gap between the isolated floor and
  the pooled ceiling, **without any raw data leaving a node**.
- Federated global model mean accuracy: **0.979**.
- Severity head (proxy labels, n=32 test): MAE 0.00 classes, QWK 1.00. Secondary result — severity is a heuristic proxy on a small bbox subset.

![Floor vs Federated vs Ceiling](plots/03_floor_fed_ceiling.png)

---

## Federation learns over rounds

The global model starts near chance (AUC ~0.5) and climbs each round as client
updates are aggregated, settling near the centralised ceiling and above the floor.

![Learning curve](plots/01_learning_curve.png)

![Global loss](plots/02_global_loss.png)

---

## Per-hospital floor (why federation helps)

Under a non-IID split each hospital sees a skewed slice of diseases, so alone it
generalises worse on the shared test set. Aggregation combines their knowledge.

| Hospital node | Mean ROC-AUC (alone) |
|---|---|
| 0 | 0.860 |
| 1 | 0.864 |
| 2 | 0.768 |

![Per-pathology AUC](plots/04_per_pathology_auc.png)

---

## Takeaways

- The end-to-end architecture (data pipeline -> dual-head model -> local training ->
  FedAvg aggregation -> global evaluation) runs and produces the expected
  floor < federated <= ceiling ordering.
- Federation recovers most of the centralised performance while keeping data local.
- Next: switch the defence layers on (DP + robust aggregation) and measure the
  privacy-robustness-utility trade-off (Task 10).

*Generated from `summary.json` by `src/evaluation/report.py`.*
