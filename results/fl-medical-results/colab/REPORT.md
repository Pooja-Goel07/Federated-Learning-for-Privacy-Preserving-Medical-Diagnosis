# Results Report — Federated Architecture (NIH ChestX-ray14 (subset))

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
| Dataset | NIH ChestX-ray14 (subset), 224x224 px |
| Model | densenet121, dual head (pathology + severity) |
| Hospitals (nodes) | 3 |
| Partition | dirichlet (alpha=0.5) |
| Aggregation | plain FedAvg |
| Defences | none (plain FedAvg, no DP, no robust aggregation) |
| Baseline epochs | 3 · Federated rounds | 10 |

---

## Headline result

| Configuration | Mean ROC-AUC | What it represents |
|---|---|---|
| Single-site (isolated) | **0.557** | floor — one hospital, its own data only |
| **Federated (FedAvg)** | **0.674** | our architecture |
| Centralised (pooled) | **0.573** | reference — all data in one place |

- Federated (**0.674**) here exceeds the pooled centralised reference (**0.573**). This is a training-budget artefact, not a real effect: the centralised/single-site baselines ran **3** epochs while the federation accumulates **10** rounds of updates. For a strict ceiling, re-run with the baseline epochs matched to the number of rounds. Either way, federated still clearly beats every isolated hospital (**0.674** vs floor **0.557**).
- Federated global model mean accuracy: **0.926**.
- Severity head (proxy labels, n=179 test): MAE 0.61 classes, QWK 0.39. Secondary result — severity is a heuristic proxy on a small bbox subset.

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
| 0 | 0.577 |
| 1 | 0.550 |
| 2 | 0.544 |

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
