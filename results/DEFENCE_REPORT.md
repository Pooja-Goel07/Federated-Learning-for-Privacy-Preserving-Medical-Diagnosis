# Defence-Layer Results — DP, Robust Aggregation, and their Composition

Real federated runs on the NIH ChestX-ray14 subset (see `sweep_results.json`). Clean
FedAvg (no attack, no DP) reference AUC = **0.642**. Attack used for the
robustness/composition studies: **?** at fraction
**?** (? malicious nodes).

---

## 1. Privacy-utility trade-off (differential privacy)

Client-level DP-SGD (per-sample clipping + Gaussian noise), Renyi-DP accountant.

| Setting | Mean ROC-AUC | Privacy |
|---|---|---|
| FedAvg, no DP | 0.642 | no DP |
| FedAvg, DP sigma=1.0 | 0.493 | eps~4.5 |
| FedAvg, DP sigma=2.0 | 0.487 | eps~1.4 |

Higher noise (sigma) tightens the privacy budget (smaller epsilon) and lowers utility —
the expected privacy-utility trade-off.

![DP utility](plots/05_dp_utility_tradeoff.png)

---

## 2. Robustness under poisoning

A fraction of nodes run the **?** attack. Plain FedAvg has no defence;
the robust rules (Krum, trimmed mean, median) replace it as drop-in aggregators.

| Aggregation rule (under attack) | Mean ROC-AUC | |
|---|---|---|
| Krum | 0.545 | no DP |
| Trimmed mean | 0.571 | no DP |
| Median | 0.562 | no DP |

![Robust under attack](plots/06_robust_under_attack.png)

---

## 3. DP x robust aggregation — composition (research gap G1)

The central question: when DP and robust aggregation run **together** under attack, do
they help additively, or does DP's noise blunt the robust rule's ability to spot
malicious updates?

| Configuration (under attack) | Mean ROC-AUC | Privacy |
|---|---|---|
| FedAvg, DP on | 0.506 | eps~4.5 |
| Median, no DP | 0.562 | no DP |
| Median, DP on | 0.458 | eps~4.5 |

**Finding.** _(composition runs incomplete — rerun the sweep to populate this.)_

![Composition](plots/07_composition.png)

---

*Generated from `sweep_results.json` by `src/evaluation/defence_report.py`. All numbers
are from real runs on the stated subset; see study.md for configuration details.*
