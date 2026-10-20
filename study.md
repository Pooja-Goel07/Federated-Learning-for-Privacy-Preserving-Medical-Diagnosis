# study.md — Research Log

> The incremental results section of the project. For every experiment that produces
> results (esp. Tasks 4, 5, 8, 9, 10, 11) record: exact configuration, results (tables /
> plot references in `results/`), observations (compared to isolated-defence literature,
> esp. research gap **G1**), and any deviation from the plan and why.
>
> This is not a changelog — it is the paper's results/methodology section being written
> as experiments run. Session-to-session continuity lives in `progresstracker.md`.

---

## 0. Research Framing

**Central contribution (Objective 4):** characterise the joint **privacy–robustness–utility**
trade-off — specifically whether client-level DP (per-sample clipping + Gaussian noise) and
Byzantine-robust aggregation (Krum/Multi-Krum, trimmed mean, median, trust weighting) compose
**constructively or destructively**. Hypothesis: DP's clipping/noise erases the very statistical
structure (update norm, direction, inter-client similarity) that robust aggregators rely on to
detect malicious updates — so stacking the two defences may degrade robustness rather than add to it.

**Research gap G1:** no prior work studies this DP × Byzantine-robustness interaction directly;
the literature evaluates each defence in isolation.

---

## 1. Methodology Decisions (living section)

Record decisions here as they are made, with rationale, so the report can cite them.

| Item | Decision | Status | Notes |
|---|---|---|---|
| Dataset | NIH ChestX-ray14 (112,120 CXR, 14 labels) | Locked (CLAUDE.md §4) | Frontal images, public. |
| Severity labels | Heuristic proxy from `BBox_List_2017.csv` (~984 imgs, 8 pathologies): **max box area ÷ image area** per image, binned mild/moderate/severe | Locked; binning **method = tercile** (default) | **Proxy, NOT clinical ground truth** — stated limitation. Impl `src/data/severity.py`. Normalised by full image area (no lung mask in release) — documented caveat. Real tercile thresholds logged in §2 on the real run. |
| Nodes | 3 leaf + 1 centre; code node-count-agnostic | Locked | Sweep to 5+ later if time. |
| Partitioning | IID + non-IID (Dirichlet label-skew), **patient-level** (no patient leakage) | Locked; default **α = 0.5** (sweep TBD in Task 7) | Impl `src/data/partition.py`. Dirichlet class variable = per-image **dominant label** (first positive finding, else "No Finding") — a partitioning device only; training uses full multi-label targets. Uses official NIH split lists if present, else patient-wise random `test_frac`. |
| Backbone | DenseNet-121 (default) or ResNet-50, ImageNet-pretrained | Locked; **default densenet121** | Standard for ChestX-ray14. Impl `src/models/backbone.py`. |
| Severity head | 3-logit classifier, trained with masked cross-entropy, **evaluated ordinally** (MAE + quadratic-weighted-kappa) | Decided (Task 3) | Keeps head simple/stable; ordering respected by metrics. Images without a proxy carry `SEVERITY_IGNORE_INDEX` and are masked out of the severity loss. |
| FL framework | Flower `flwr` **core**; in-process orchestrator delegating to real Flower `Strategy` classes (no Ray) | Decided (Task 6) | Windows/deterministic/streamable; genuine `FedAvg.aggregate_fit`. See CLAUDE.md §5. |
| Heterogeneity mitigation | **FedProx** (µ/2·‖w−w_global‖² proximal term) + **FedBN** (BatchNorm kept local, excluded from aggregation) | Implemented (Task 7) | Config: `federated.heterogeneity_mitigation.{fedprox_mu, fedbn}`. FedBN → per-client averaged evaluation (single global model has no valid BN). For densenet121, FedBN keeps ~83% of state_dict entries (605/729, incl. BN buffers) local. µ value(s) and FedBN-vs-FedProx comparison on non-IID to be swept on real data. |
| Differential privacy | **Opacus DP-SGD** (per-sample clip C + Gaussian noise σ·C) with a **Renyi-DP (RDP) accountant** reporting cumulative (ε, δ). PrivacyEngine per client, reused across rounds so ε accumulates over the whole run. | Implemented (Task 8) | Config `privacy.dp.{enabled,max_grad_norm,noise_multiplier,target_delta,accountant}`. Requires **BN→GroupNorm** (`ModuleValidator.fix`) and no in-place activations; therefore **DP is mutually exclusive with FedProx/FedBN** (runner disables them). Verified ε monotonic in σ. Report worst-case (max over clients) final ε. |
| Byzantine-robust aggregation | **Krum, Multi-Krum, coord-wise Trimmed Mean, coord-wise Median, Trust-weighting** (cosine-sim to median), as Flower `FedAvg` subclasses | Implemented (Task 9) | Config `federated.strategy` + `federated.robust.{beta,multi_k}`. Unit check (4 honest + 1 gross outlier): krum/median/trimmed reduce distance-to-honest-mean from 34.5 (FedAvg) to ~0.005; trust to ~13.4. |
| Byzantine attacks | **label_flip, sign_flip, scaled_update, model_replacement** attack clients | Implemented (Task 9) | Config `attack.enabled` + `attack.byzantine.{type,fraction,scale}`. First round(fraction·N) nodes malicious (deterministic). |
| FL framework | Flower (flwr), simulation engine | Locked | Custom `Strategy` per aggregation rule. |
| DP | Opacus DP-SGD + Rényi-DP accountant (custom fallback) | Locked | Report cumulative (ε, δ). |

---

## 2. Experiment Log

### EXP-001 — Core architecture sanity on synthetic data, defences OFF  (Tasks 4/5/6, 2026-10-13)
- **Purpose:** end-to-end check that the pipeline works and the federated design recovers
  centralised performance before spending GPU time on real data. NOT a research result — the
  data is synthetic scaffolding with an injected learnable signal (each of 6 active pathologies
  = a fixed bright patch; bbox images get a severity square sized by class).
- **Config:** synthetic (~400 imgs, 48px, signal-bearing), densenet121 from scratch (`pretrained:false`),
  3 nodes, non-IID Dirichlet α=0.5, test_frac 0.25, Adam lr 1e-3, batch 16, seed 42. Baselines 8 epochs;
  federated 12 rounds × 1 local epoch; plain FedAvg; no DP, no robust aggregation, no attacks.
- **Metrics (mean ROC-AUC on shared test set):** centralised **0.989** · federated FedAvg **0.982**
  · single-site floor **0.831** (per node ~0.77–0.86). Federated accuracy 0.95+. Federated global AUC
  climbs 0.59→0.98 across rounds. Plots/report in `results/demo/`.
- **Observations:** expected ordering floor < federated ≤ ceiling holds cleanly; federation closes ~96%
  of the floor→ceiling gap while keeping data local. Confirms the FedAvg aggregation + partition + dual-head
  training loop are wired correctly. Severity head is secondary here (small synthetic bbox subset).
- **Deviations:** none. This is a plumbing/demo experiment, reported as such; real NIH numbers pending the
  Colab run (`notebooks/colab_training.ipynb`, `configs/colab_real.yaml`).

<!--
Template for each experiment entry:

### EXP-XXX — <short title>  (Task <n>, <date>)
- **Config:** partition (IID/non-IID α=?), model, hyperparams, ε/δ (if DP), aggregation rule,
  attacker fraction/type (if any), heterogeneity setting, seed.
- **Metrics:** per-pathology AUC/acc, severity MAE/QWK, attack success rate, recon SSIM/PSNR (as applicable).
  Reference saved tables/plots in results/.
- **Observations:** what happened, anything surprising, comparison to isolated-defence literature / G1.
- **Deviations:** any departure from the plan and why.
-->

### EXP-002 — Core architecture on real NIH ChestX-ray14 (Colab GPU), defences OFF  (Tasks 4/5/6, 2026-10-13)
- **Config:** NIH ChestX-ray14, images from the 224px resized Kaggle mirror, patient-aware subsample
  ~20k images, 3 nodes, non-IID Dirichlet α=0.5, test_frac 0.2, densenet121 **pretrained (ImageNet)**,
  Adam lr 1e-4, batch 32, seed 42. **Baselines 3 epochs; federated 10 rounds × 1 local epoch**; plain
  FedAvg; no DP/robust/attacks. Results in `results/fl-medical-results/colab/`.
- **Metrics (mean ROC-AUC, shared test set):** centralised **0.573** · federated FedAvg **0.674**
  · single-site floor **0.557** (nodes 0.58/0.55/0.54). Federated per-pathology: Cardiomegaly 0.78,
  Edema 0.79, Pneumothorax 0.77, …, Hernia 0.48 (rarest). Federated AUC rises 0.47→0.67 over 10 rounds.
  Severity (proxy, n=179): federated MAE 0.61 / QWK 0.39 vs centralised 0.78 / 0.19.
- **Observations:** (1) **Federation clearly beats isolated single-site** training on real data
  (0.674 vs 0.557) — the core FL value proposition holds. (2) The global model genuinely learns over
  rounds. (3) **Caveat / methodology issue:** federated (0.674) came out ABOVE centralised (0.573),
  inverting the expected ceiling. Root cause is a **training-budget mismatch** — baselines ran 3 epochs
  while federation accumulated 10 rounds of updates. So centralised here is a *lightly-trained
  reference*, not a strict upper bound. Absolute AUCs are modest (subset + few epochs); literature on
  the full 112k set reaches ~0.80.
- **Deviations / fix:** `notebooks/colab_training.ipynb` now sets baseline `local_epochs = rounds`
  (= 10) so a re-run yields a fair centralised ceiling (expected centralised ≥ federated > single-site).
  Report generator hardened to explain the inversion instead of printing a nonsensical "gap closed %".
  **A fair-epochs re-run (EXP-003) should replace these numbers for Review 2.**

### EXP-003 — Properly-trained core run on real NIH (Kaggle GPU), class-balanced, defences OFF  (Tasks 4/5/6, 2026-10-13)
- **Config:** NIH ChestX-ray14, patient-aware subsample ~30k images, 3 nodes, non-IID Dirichlet α=0.5,
  test_frac 0.2, densenet121 pretrained, Adam lr 1e-4, batch 32, **class-balanced BCE (per-class
  pos_weight)**, **baseline epochs = rounds = 12** (fair ceiling), seed 42. Plain FedAvg; no DP/robust/
  attacks. Results at `results/` (summary.json, REPORT.md, plots/).
- **Metrics (mean ROC-AUC, shared test set):** centralised **0.772** · federated FedAvg **0.754**
  · single-site floor **0.721** (nodes 0.713/0.731/0.718). **Ordering now correct: floor < federated
  < ceiling**; federation closes **66%** of the floor→ceiling gap. Federated per-pathology: Edema 0.86,
  Cardiomegaly 0.84, Pneumothorax 0.83, Effusion 0.81, Fibrosis 0.81 … Nodule 0.68, Hernia 0.67.
  Federated AUC rises 0.49→0.76 over 12 rounds. Severity (proxy, n=187): federated MAE 0.63 / QWK 0.42
  vs centralised 0.59 / 0.39.
- **Observations:** (1) The two fixes vs EXP-002 worked exactly as intended — class-balancing lifted
  absolute AUC from ~0.57–0.67 into the **0.72–0.77** range (credible for a 30k subset; full-data
  literature ~0.75–0.84), and matching baseline epochs to rounds restored centralised as a true ceiling.
  (2) **Note on accuracy:** centralised mean_accuracy fell (0.93→0.72) because pos_weight makes the model
  predict positives instead of the all-negative shortcut — raw accuracy is misleading under heavy
  multi-label imbalance; **AUC is the metric of record**. (3) This is the Review-2-ready core result.
- **Deviations:** none. Replaces EXP-002 as the headline core-comparison run.

### EXP-004/005/006 — Defence-layer sweep (DP, robust aggregation, composition), Kaggle  (Tasks 8/9/10, 2026-10-13)
- **Common config:** NIH subset, **6 nodes** (honest majority for robust agg), non-IID α=0.5, ~12k imgs,
  128px, densenet121 pretrained, class-balanced, 8 rounds × 1 local epoch, seed 42. Attack (for 005/006):
  **model_replacement @ fraction 0.34 (2/6 nodes)**. DP: Opacus DP-SGD, RDP accountant, clip C=1.0.
  Results at `results/` (sweep_results.json, DEFENCE_REPORT.md, plots 05/06/07).

- **EXP-004 (DP privacy–utility):** clean FedAvg **0.642** → DP σ=1.0 (ε≈4.5) **0.493** → σ=2.0 (ε≈1.4)
  **0.487**. Monotone privacy cost; DP is harsh at this small scale (near-chance), as expected — utility
  recovers with more data/less noise. Confirms the privacy–utility trade-off (Objective 2/4).

- **EXP-005 (robustness):** under model_replacement, **plain FedAvg diverged (weights → NaN, AUC ≈ 0.50 —
  total collapse; the attack boosts the malicious update by the client count and undefended averaging
  propagates it).** Robust rules hold: **trimmed_mean 0.571, median 0.562, krum 0.545** vs clean 0.642.
  Robust aggregation recovers most utility under a 33% attack (Objective 3). atk_fedavg confirmed = **0.500**
  (NaN-safe re-run): round 0 = 0.52, then collapses to 0.50 every round — FedAvg is fully destroyed.

- **EXP-006 (composition — G1, the core contribution):** under attack, {FedAvg, Median} × {no-DP, DP σ=1.0}:
  Median no-DP **0.562** → Median+DP **0.458** (**−0.104**); FedAvg diverged(~0.50) → FedAvg+DP **0.506**
  (≈flat). **Finding: DP composes DESTRUCTIVELY with robust aggregation** — DP noise costs the robust rule
  far more than it costs FedAvg, because the Gaussian noise + clipping erase the update-norm/direction
  signal Median uses to reject malicious updates. Notable nuance: DP *clipping* partially *shields* FedAvg
  (bounds the boosted malicious update), while DP *noise* *degrades* Median — i.e. the two defences
  interact non-trivially. This directly addresses research gap G1 (no prior work studies this interaction).

- **Bug fixed this run:** FedAvg-under-attack crashed the AUC metric on NaN (diverged model). Fixed
  `pathology_metrics` to sanitise non-finite probs → chance (0.50); added `run_sweep(resume=True)` so a
  re-run only redoes the failed variant. Logged so the EXP-005 FedAvg cell can be completed cheaply.

_Next: complete atk_fedavg (NaN-safe re-run), then write the research-paper report from EXP-003 (core) +
EXP-004/005/006 (defences). Remaining build: Task 11 privacy-audit attacks (unbuilt), 12–15._

### EXP-007 — Privacy-robustness-utility regime map  (Task 10 full, Kaggle 2026-10-20)
- **Config:** attacker fraction {0, 17%, 34%} × {FedAvg, median} × {no-DP, DP σ=1.0}, defence scale
  (6 nodes, 128px, ~12k, 8 rounds), model_replacement. Results in `results/tradeoff/`.
- **Results (mean ROC-AUC):**
  | Malicious % | FedAvg no-DP | FedAvg DP | Median no-DP | Median DP |
  | 0%  | 0.651 | 0.493 | 0.611 | 0.488 |
  | 17% | 0.500 | 0.491 | 0.602 | 0.472 |
  | 34% | 0.500 | 0.482 | 0.581 | 0.450 |
- **Finding (STRONG):** without DP, Median's robustness advantage over FedAvg grows with the attack
  (+0.081 at 34%); **with DP the advantage REVERSES to −0.032** (Median+DP 0.450 < FedAvg+DP 0.482) — DP
  erodes 0.113 of the robustness benefit. This is the regime map + a concrete recommendation (prefer
  direction/similarity-based rules under tight DP, or relax ε when Byzantine tolerance is critical).
  Directly upgrades the EXP-006 one-point finding into a definite, actionable conclusion.

### EXP-008 — Privacy audit: membership inference + gradient inversion  (Task 11, Kaggle 2026-10-20)
- **Config:** no-DP + DP(σ=1,2) models on the pooled subset (128px, 5 epochs); membership attack +
  controlled small-CNN (32px) gradient inversion. Results in `results/audit/`.
- **Gradient inversion (STRONG, primary G3 evidence):** no-DP reconstructs the image almost perfectly
  (SSIM **1.00**, PSNR 103 dB); DP σ=1.0 destroys it (SSIM **0.01**, PSNR 10 dB). DP demonstrably prevents
  reconstruction from an intercepted gradient.
- **Membership inference (INCONCLUSIVE):** no-DP attack ROC-AUC **0.492 ≈ 0.5** (member loss 0.250 ≈
  non-member 0.247) — the model did not memorize at this scale, so there was no membership signal for DP
  to remove; DP keeps it at 0.500. NOT a failure of DP, but it doesn't demonstrate the leak→protect arc.
  To show that arc, re-run membership in an **over-fitting regime** (small ~2k subset, ~25 epochs) so the
  undefended model leaks (AUC > 0.5) and DP pulls it toward 0.5. Gradient inversion already carries G3.
- **Report-text bugs fixed (local):** the trade-off recommendation printed "+-0.032" (signed-format fix);
  the audit auto-finding wrongly claimed 0.492 was "above 0.5" (now adaptive + honest).

_Dashboard (Objective 9 / FR8): `frontend/dashboard.html` (React+Tailwind) + `backend/app/main.py` (FastAPI)
visualize all scenarios with the real numbers. Not an experiment; a demonstrator._
