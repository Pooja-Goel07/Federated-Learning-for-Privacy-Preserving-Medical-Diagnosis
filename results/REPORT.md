# Results Report — Federated Architecture (NIH ChestX-ray14 (subset, class-balanced))

**What this shows:** the full pipeline runs end-to-end and the federated design
*works* — simulated hospitals train locally, a centre node aggregates their weights
with plain **FedAvg**, and the global model approaches the pooled-data reference while
clearly beating any hospital training alone.

**Defence layers are OFF** (no differential privacy, no Byzantine-robust
aggregation, no attacks) — this isolates the core learning loop.

---

## Setup

| Item | Value |
|---|---|
| Dataset | NIH ChestX-ray14 (subset, class-balanced), 224x224 px |
| Model | densenet121, dual head (pathology + severity) |
| Hospitals (nodes) | 3 |
| Partition | dirichlet (alpha=0.5) |
| Aggregation | plain FedAvg |
| Defences | none (plain FedAvg, no DP, no robust aggregation) |
| Baseline epochs | 12 · Federated rounds | 12 |

---

## Headline result

| Configuration | Mean ROC-AUC | What it represents |
|---|---|---|
| Single-site (isolated) | **0.721** | floor — one hospital, its own data only |
| **Federated (FedAvg)** | **0.754** | our architecture |
| Centralised (pooled) | **0.772** | reference — all data in one place |

- Federated closes **66%** of the gap between the isolated floor and the pooled ceiling, **without any raw data leaving a node**.
- Federated global model mean accuracy: **0.757**.
- Severity head (proxy labels, n=187 test): MAE 0.63 classes, QWK 0.42. Secondary result — severity is a heuristic proxy on a small bbox subset.

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
| 0 | 0.713 |
| 1 | 0.731 |
| 2 | 0.718 |

![Per-pathology AUC](plots/04_per_pathology_auc.png)

---

## Takeaways

- The end-to-end architecture (data pipeline -> dual-head model -> local training ->
  FedAvg aggregation -> global evaluation) runs on real data.
- Federation clearly beats every isolated single-site hospital while keeping data local,
  and the global model improves across rounds.
- Next: switch the defence layers on (DP + robust aggregation) and measure the
  privacy-robustness-utility trade-off (Task 10).

*Generated from `summary.json` by `src/evaluation/report.py`.*
