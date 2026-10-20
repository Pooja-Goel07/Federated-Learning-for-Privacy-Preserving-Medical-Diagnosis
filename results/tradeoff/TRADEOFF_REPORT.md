# Privacy-Robustness-Utility Trade-off (regime map)

Federated global mean ROC-AUC as the fraction of malicious clients grows, for plain
FedAvg and a robust rule (median), each with and without differential privacy (σ=1.0).
Same defence-sweep scale as the other defence experiments (6 nodes, 128px, ~12k images,
8 rounds); see study.md.

| Malicious % | FedAvg no-DP | FedAvg DP | Median no-DP | Median DP |
|---|---|---|---|---|
| 0% | 0.651 | 0.493 | 0.611 | 0.488 |
| 17% | 0.500 | 0.491 | 0.602 | 0.472 |
| 34% | 0.500 | 0.482 | 0.581 | 0.450 |

**Reading the map.** Without DP, the robust rule's line stays high as the attack grows
while FedAvg's falls — the expected robustness benefit. With DP on, that gap narrows,
because DP's clipping and noise remove the update statistics the robust rule needs.

**Recommendation.** At 34% malicious clients, the robust rule (median) holds a +0.081 AUC advantage over FedAvg without DP, but only -0.032 with DP — DP erodes 0.113 of the robustness benefit (the advantage even reverses). **Recommendation:** when a tight privacy budget is mandatory, prefer aggregation rules that key on update *direction/similarity* rather than *magnitude*, or relax ε where strong Byzantine tolerance is the priority; do not assume magnitude-based robust rules retain their guarantee once DP noise is applied.

![Trade-off regime map](plots/08_tradeoff_regime_map.png)

*Generated from `sweep_results.json` by `src/evaluation/defence_report.py`.*
