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

_No experiments run yet. First entries will appear after Task 4 (centralised baseline)._

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
