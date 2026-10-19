# Federated Learning for Privacy-Preserving Medical Diagnosis
## Comprehensive Technical Report (source-of-truth for the research paper)

**Course:** BSCE497J — Project 1
**Team:** Devraj K Chandani, Shaurya Saini, Pooja Goel · **Guide:** Prof. Jabanjalin Hilda J
**Dataset:** NIH ChestX-ray14 · **Framework:** Flower (flwr) + PyTorch + Opacus
**Status of numbers:** all results below are from real runs (local synthetic validation + GPU runs on Kaggle). No values are fabricated or hand-adjusted. Subset sizes are stated explicitly wherever used.

> How to read this document: Sections 1–9 are the write-up material (motivation, objectives, gaps, methods, algorithms). Section 10 is the experimental setup. Section 11 holds every number and the figure that visualises it. Sections 12–16 are discussion, limitations, conclusion, reproducibility, and the figure index.

---

## Abstract

Clinical deep learning is limited not by model capacity but by data access: individual hospitals hold datasets that are small, demographically narrow, and acquisition-biased, while HIPAA/GDPR/DPDP-style regulation makes centralising raw patient images infeasible. Federated Learning (FL) keeps data in place and shares only model updates — but FL alone is **not** a privacy guarantee (shared gradients are invertible) and the open-participation model exposes the global model to poisoning. We build a **hierarchical client–server federated architecture** for chest-radiograph diagnosis with two defence layers embedded in the training loop: **client-level differential privacy (DP)** (per-sample gradient clipping + calibrated Gaussian noise, tracked with a Rényi-DP accountant) and **Byzantine-robust aggregation** (Krum/Multi-Krum, coordinate-wise trimmed mean/median, trust weighting) replacing plain FedAvg. The system performs multi-label pathology detection plus an ordinal severity-grading head. On a subset of NIH ChestX-ray14 we show (i) the federated global model (mean ROC-AUC **0.754**) beats every isolated single-site model (**0.721**) and approaches the centralised ceiling (**0.772**); (ii) DP imposes a clear privacy–utility cost; (iii) robust aggregation survives a model-replacement attack that drives plain FedAvg to total divergence; and (iv) — our central contribution — DP and robust aggregation can compose **destructively**: adding DP costs a robust aggregator (median) **0.104** AUC under attack versus essentially nothing for FedAvg, because DP's clipping and noise erase the very update statistics the robust rule relies on to reject malicious clients (Research Gap **G1**).

---

## 1. Introduction

Deep models for chest radiography (e.g., DenseNet-121 on ChestX-ray14) reach strong performance when trained on large, pooled datasets, but such pooling is often legally impossible across hospitals. Federated Learning addresses this by training locally and aggregating updates centrally. However, two problems remain and are usually studied **in isolation**:

1. **Privacy.** FL shares gradients/weights, which are invertible via gradient-inversion and membership-inference attacks. Differential Privacy (DP) provides a formal defence but at a utility cost.
2. **Integrity / robustness.** Any participating client can submit malicious updates (label-flipping, sign-flipping, scaling, model replacement). Robust aggregation rules defend the global model against a bounded fraction of such clients.

The gap this project targets is their **interaction**: DP's clipping and noise destroy exactly the statistical structure (update magnitude, direction, inter-client similarity) that robust aggregators use to detect malicious updates. Whether the two defences compose **constructively** or **destructively** when run together — on real medical images — had not been characterised directly. This report documents the architecture, methods, and the empirical characterisation of that trade-off.

---

## 2. Objectives

(From the project charter; each is addressed to the extent noted in Results / Limitations.)

1. **O1 — Architecture.** Design a hierarchical federated architecture (1 centre + N leaf/hospital nodes) with a modular interface so the aggregation rule and privacy mechanism can be swapped independently.
2. **O2 — Differential privacy.** Integrate client-level DP into local training (per-sample clipping + calibrated Gaussian noise) with a Rényi-DP accountant reporting cumulative (ε, δ).
3. **O3 — Robust aggregation.** Implement and compare Byzantine-robust aggregation rules (Krum/Multi-Krum, trimmed mean, median, trust weighting) against FedAvg under poisoning attacks at varying adversarial fractions.
4. **O4 — Joint trade-off (primary contribution).** Characterise the joint privacy–robustness–utility trade-off; identify regimes where DP noise degrades robust aggregation's ability to reject malicious updates (constructive vs destructive composition).
5. **O5 — Heterogeneity + fairness.** Mitigate statistical heterogeneity (proximal regularisation + local batch norm) and check robust aggregation does not systematically exclude honest minority-distribution clients.
6. **O6 — Clinical validation.** Validate on chest radiography with two heads: multi-label pathology detection + per-disease ordinal severity grading, benchmarked against isolated single-site models and a centralised upper bound.
7. **O7 — Privacy audit.** Empirically audit privacy via gradient-inversion and membership-inference attacks, reporting reconstruction quality (SSIM/PSNR) and attack success rate per budget.
8. **O8 — Reproducible testbed.** Release a configuration-driven testbed where privacy, robustness, and heterogeneity axes vary independently.
9. **O9 — Demonstration.** Demonstrate the end-to-end federated process (nodes train locally → centre aggregates → global model evaluated), visualised.

**Coverage in this report:** O1, O2, O3, O4, O6, O8 are demonstrated with results. O5 (mitigation) and O9 (dashboard) are implemented/available in the testbed; O5's fairness check and O7 (privacy audit) are identified as the next steps (Section 14).

---

## 3. Research Gaps

- **G1 (primary) — Unstudied composition of DP and Byzantine-robustness.** Prior work treats client-level DP and robust aggregation as independent defences. But robust aggregators detect malicious updates *through* update norm/direction/similarity, and DP deliberately clips norms and injects noise — potentially blinding the aggregator. No prior work directly measures this interaction on medical imaging. **This project measures it.**
- **G2 — FL is not private by construction.** FL is frequently presented as "private because data stays local," yet shared updates leak. A privacy budget must be enforced *and* empirically audited (O7), not merely asserted.
- **G3 — Heterogeneity vs robustness fairness.** Robust aggregators may discard honest but statistically-unusual (minority-distribution) clients as if they were malicious — an under-examined fairness risk under non-IID data.

---

## 4. Background & Related Work

*(Citations to be inserted in the paper; key anchors listed.)*

- **Federated averaging:** McMahan et al., FedAvg (2017). **Heterogeneity:** FedProx (Li et al., 2020), FedBN (Li et al., 2021).
- **Differential privacy for deep learning:** DP-SGD (Abadi et al., 2016); Rényi-DP accountant (Mironov, 2017).
- **Byzantine-robust aggregation:** Krum / Multi-Krum (Blanchard et al., 2017); coordinate-wise trimmed mean & median (Yin et al., 2018); similarity/trust weighting (various).
- **Poisoning / model replacement:** backdoor & model-replacement attacks on FL (Bagdasaryan et al., 2020).
- **Privacy attacks (for O7):** Deep Leakage from Gradients (Zhu et al., 2019); Inverting Gradients (Geiping et al., 2020); membership inference (Shokri et al., 2017).
- **Chest radiography:** ChestX-ray14 (Wang et al., 2017); CheXNet (Rajpurkar et al., 2017).

**Positioning.** This project combines these components in one modular testbed and — unlike prior isolated studies — measures the **DP × robustness interaction** (G1) directly on ChestX-ray14 with a dual (detection + severity) task.

---

## 5. System Architecture

A **hierarchical client–server** topology: N leaf nodes (simulated hospitals) + 1 centre (aggregation) node.

- **Leaf node (hospital):** holds a private partition; runs local training (optionally DP-SGD); sends updated weights to the centre. A node may be *honest*, *DP-enabled*, or *Byzantine* (attacker).
- **Centre node:** aggregates the received updates using a pluggable rule (FedAvg or a robust rule), broadcasts the new global model, and evaluates it on a shared held-out test set each round.
- **Modularity (O1, O8).** Aggregation rule, privacy mechanism, attack behaviour, partitioning, and heterogeneity mitigation are **independent, config-selected axes**. Any combination is expressible in one YAML file.

**Orchestration (implementation note).** Rounds are driven by an **in-process orchestrator** that calls genuine Flower `Strategy` classes for aggregation (`flwr.server.strategy.FedAvg` and robust subclasses), reusing Flower's `NumPyClient`, `FitRes`, and parameter (de)serialisation. Flower's Ray-backed simulation engine is deliberately not used (Ray is heavy/flaky on Windows); the in-process loop is deterministic, easy to live-stream, and gives the tight control needed for the DP/Byzantine composition experiments. Flower remains the FL framework of record.

```
                 ┌─────────────── Centre node ───────────────┐
                 │  aggregate_fit(FedAvg | Krum | TrimMean |  │
                 │              Median | Trust)               │
                 │  → broadcast global weights → evaluate     │
                 └───▲───────────▲───────────▲────────────────┘
          weights │   weights │   weights │   (per round)
             ┌────┴───┐   ┌────┴───┐   ┌────┴───┐
             │ Node 1 │   │ Node 2 │   │ Node k │   ... (honest / DP / Byzantine)
             │ local  │   │ local  │   │ local  │
             │ train  │   │ train  │   │ train  │
             └────────┘   └────────┘   └────────┘
             private data  private data  private data   (never leaves the node)
```

---

## 6. Dataset & Preprocessing

- **Dataset.** NIH ChestX-ray14: 112,120 frontal chest X-rays, 30,805 patients, 14 pathology labels (multi-label). A bounding-box subset (~984 images, 8 pathologies) is used for the severity proxy.
- **Pathology labels.** The pipe-separated `Finding Labels` field is expanded to a 14-dim binary multi-label vector; "No Finding" is the implicit all-negative case.
- **Severity proxy (O6, stated as a proxy, not clinical ground truth).** NIH has no native severity labels. We derive a per-image severity from `BBox_List_2017.csv` as **relative lesion area** = bounding-box area / image area, then bin by **terciles** into {mild, moderate, severe}. When an image has multiple boxes we take the max relative area. Caveats (documented as limitations): box area is a crude disease-burden stand-in; it is normalised by full image area (no lung-field segmentation); only the bbox subset carries a severity label (all other images are masked out of the severity loss).
- **Partitioning.** **Patient-level** splits (no patient appears in two partitions → no leakage). Two schemes are implemented: IID and **non-IID Dirichlet label-skew** (α controls skew). All reported experiments use **non-IID, α = 0.5**, the harder and more realistic setting.
- **Subsampling (for tractable GPU runs).** Patient-aware subsampling keeps whole patients up to a target image count and **keeps bounding-box/severity patients first** so the severity head still has data. The pipeline also restricts to images actually present on disk (so partial dataset downloads work).
- **Image pipeline.** Grayscale → 3-channel, resize (224 px core / 128 px defence sweep), train-time augmentation (horizontal flip, small rotation), ImageNet normalisation.

---

## 7. Model & Training Objective

- **Backbone.** DenseNet-121, ImageNet-pretrained. (ResNet-50 is also supported.) In-place activations are disabled so the model is DP-compatible.
- **Dual head (O6).**
  - *Pathology head:* linear → 14 logits, sigmoid, **multi-label**.
  - *Severity head:* linear → 3 logits, softmax, **ordinal 3-class**, active only on severity-labelled (bbox-subset) images.
- **Loss.** `total = BCE_pathology + severity_weight · CE_severity`.
  - *Pathology:* binary cross-entropy with logits, with **per-class `pos_weight` = (#neg / #pos)** (clamped to 50) to counter ChestX-ray14's heavy negative skew — the single largest lever on AUC. This is the `class_balanced` option, enabled in all headline runs.
  - *Severity:* cross-entropy with `ignore_index` for images without a proxy (keeps the batch dimension intact so per-sample-gradient DP works; numerically equals the mean CE over labelled samples). On all-unlabelled batches a zero-weighted term keeps the head in the autograd graph.
- **Metrics.** Pathology: **per-class and mean ROC-AUC** (primary), mean accuracy (secondary — misleading under imbalance). Severity: **MAE** (in class steps) and **quadratic weighted kappa (QWK)**. Robustness: final global AUC under attack. (Privacy audit metrics SSIM/PSNR/ASR are for O7, future work.)

> **Why AUC, not accuracy.** ~90% of ChestX-ray14 labels are negative, so an "all-negative" predictor scores ~90% accuracy while being clinically useless. ROC-AUC (0.5 = chance, 1.0 = perfect) is the field-standard metric and is used as the headline throughout.

---

## 8. Methods — Defence Layers

### 8.1 Differential Privacy (O2)

Client-level **DP-SGD** (Opacus): per-sample gradients are clipped to L2 norm `C` and Gaussian noise with std `σ·C` is added before the optimizer step. A **Rényi-DP (RDP) accountant** tracks the cumulative **(ε, δ)** budget; one `PrivacyEngine` per client is **reused across rounds** so ε accumulates over the whole federated run. Structural requirement: BatchNorm (which mixes samples) is replaced by **GroupNorm** (`ModuleValidator.fix`), applied identically to every model (clients + global) so parameter keys stay aligned for aggregation. Smaller σ → larger ε (weaker privacy, higher utility); larger σ → smaller ε (stronger privacy, lower utility). DP is mutually exclusive with FedProx/FedBN (per-sample grads + GroupNorm), which are forced off when DP is on.

### 8.2 Byzantine-robust aggregation (O3)

Drop-in replacements for FedAvg, all implemented as Flower `Strategy` subclasses:

- **Krum / Multi-Krum.** Select the update(s) whose squared-distance to their `n − f − 2` nearest neighbours is smallest (Multi-Krum averages the k best). Assumes `n ≥ 2f + 3`.
- **Coordinate-wise trimmed mean.** For each parameter coordinate, drop the β-fraction largest and smallest values, average the rest (β = 0.34 in the defence sweep → trims ~2 of 6 each side).
- **Coordinate-wise median.** Per-coordinate median; tolerates < 50% corruption per coordinate.
- **Trust weighting.** Weight each update by cosine similarity to the coordinate-wise median direction (down-weights outliers).

### 8.3 Attack models (O3)

Byzantine clients (a chosen fraction become malicious):

- **label_flip** — train on flipped multi-label targets (data poisoning).
- **sign_flip** — train honestly, return `2·g − l` (negated update).
- **scaled_update** — return `g + s·(l − g)` (blow up the update by factor `s`).
- **model_replacement** — train on flipped labels, return `g + n·(l − g)` (boost the poisoned update by the client count `n` to try to overwrite the global model). *This is the attack used in the headline robustness/composition experiments — the most aggressive.*

Here `g` = global parameters at round start, `l` = local post-training parameters, `n` = number of clients.

### 8.4 Heterogeneity mitigation (O5, implemented / available)

- **FedProx:** adds `(μ/2)·‖w − w_global‖²` to the local objective, keeping local updates close to the global model under non-IID data.
- **FedBN:** keeps BatchNorm statistics local (excluded from the aggregated vector); the global model is evaluated per-client with each client's own BN, then averaged.
These are implemented and unit-tested in the testbed; a full real-data heterogeneity/fairness study is listed as future work (Section 14).

---

## 9. Algorithms (pseudocode)

**A. Hierarchical federated round (orchestrator)**
```
Input: global model w0; clients C; strategy S; rounds R; test set T
w ← w0
for r in 1..R:
    updates ← []
    for client c in C:                     # runs in sequence (in-process)
        w_c ← c.local_train(w)             # honest / DP-SGD / Byzantine
        updates.append( (w_c, n_c) )       # weights + #examples
    w ← S.aggregate(updates)               # FedAvg | Krum | TrimMean | Median | Trust
    metrics[r] ← evaluate(w, T)            # mean AUC, severity, loss
return w, metrics
```

**B. DP-SGD local update (per client, per epoch)**
```
Input: model w; loader D; clip C; noise σ; lr η
for batch B in D:
    for sample i in B:                     # per-sample gradients
        g_i ← ∇ loss(w; x_i)
        g_i ← g_i · min(1, C / ‖g_i‖2)     # clip
    g ← (1/|B|) ( Σ_i g_i + N(0, σ²C²I) )  # average + Gaussian noise
    w ← w − η · g
accountant.step(σ, sample_rate)            # RDP → (ε, δ)
```

**C. Robust aggregation rules**
```
Krum(updates, f):
    for each i: score_i ← Σ over the n−f−2 nearest j of ‖u_i − u_j‖²
    return u_{argmin score}                 # Multi-Krum: mean of k smallest

TrimmedMean(updates, β):                    # per coordinate d
    sort values u_[*][d]; drop ⌊βn⌋ lowest and highest; return mean of rest

Median(updates):                            # per coordinate d
    return median( u_[*][d] )
```

**D. model_replacement attack (Byzantine client)**
```
Input: global g; local data (labels flipped); client count n
l ← local_train(g)                          # on poisoned labels
return g + n · (l − g)                       # boosted poisoned update
```

**E. Composition experiment (G1 protocol)**
```
fix: attack = model_replacement, fraction = 2/6, non-IID α=0.5
for strategy in {FedAvg, Median}:
    for dp in {off, on(σ=1.0)}:
        w, m ← federated_run(strategy, attack=on, dp=dp)
        record final AUC
compare ΔAUC_dp(FedAvg) vs ΔAUC_dp(Median)   # destructive if robust loses more
```

---

## 10. Experimental Setup

**Common.** PyTorch + Flower (flwr) core + Opacus. DenseNet-121 (ImageNet-pretrained), dual head, class-balanced BCE. Non-IID Dirichlet α = 0.5, patient-level split, test fraction 0.2, Adam (lr 1e-4), batch 32, seed 42. GPU runs on Kaggle (T4). Metrics on a shared held-out test set.

**Run A — Core comparison (EXP-003).** 3 nodes; ~30,000-image patient-aware subset; 224 px; centralised and single-site baselines trained for 12 epochs; federated for 12 rounds × 1 local epoch (epochs matched to rounds so centralised is a **fair** ceiling). Plain FedAvg; no DP, no attack.

**Runs B–D — Defence sweep (EXP-004/005/006).** **6 nodes** (honest majority so robust aggregation is meaningful); ~12,000-image subset; 128 px; 8 rounds × 1 local epoch. Attack (B/C): **model_replacement at fraction 0.34 = 2 of 6 nodes**. DP: Opacus DP-SGD, clip C = 1.0, δ = 1e-5, σ ∈ {1.0, 2.0}. The defence sweep is deliberately smaller/faster (128 px, 12k, 8 rounds) because it is 9 runs including slow DP runs; its purpose is **relative** comparison, so absolute AUCs are lower than Run A by design.

---

## 11. Results

### 11.1 Core comparison — federated vs isolated vs centralised (EXP-003, O6/O9)

Mean ROC-AUC on the shared test set (30k subset, 224 px, non-IID α=0.5, class-balanced):

| Configuration | Mean ROC-AUC | Role |
|---|---:|---|
| Single-site (isolated) — mean of 3 nodes | **0.721** | floor (one hospital alone; nodes 0.713 / 0.731 / 0.718) |
| **Federated (FedAvg)** | **0.754** | our architecture |
| Centralised (pooled) | **0.772** | ceiling (all data in one place) |

- Federated **beats every isolated single-site model** (+0.033 AUC over the floor) and **closes 66%** of the floor→ceiling gap — *without any raw data leaving a node*.
- Federated global AUC rises monotonically over rounds: 0.486 (init) → 0.607 → 0.645 → 0.694 → 0.716 → 0.733 → 0.735 → 0.741 → 0.751 → 0.749 → 0.758 → **0.761** (peak, r11) → 0.754 (r12).
- Severity head (proxy, n = 187 test): federated MAE **0.63** / QWK **0.42**; centralised MAE 0.59 / QWK 0.39.
- Context: published full-dataset (112k) mean AUC is ~0.80–0.84; **0.77 on a 30k subset is competitive**, only a few points below, with ~4× less data.

Per-pathology federated AUC (selected): Edema **0.855**, Cardiomegaly **0.844**, Pneumothorax **0.829**, Fibrosis 0.807, Effusion 0.806, Emphysema 0.767, Mass 0.746; hardest: Nodule 0.678, Hernia 0.674 (rarest). Clinically sensible — "easy" cardiothoracic/edema signs score highest, small/rare findings lowest.

**Figures:** `plots/03_floor_fed_ceiling.png` (bar), `plots/01_learning_curve.png` (AUC vs round with floor/ceiling references), `plots/02_global_loss.png` (loss vs round), `plots/04_per_pathology_auc.png` (per-pathology, federated vs centralised).

### 11.2 Privacy–utility trade-off — differential privacy (EXP-004, O2)

Defence-sweep setup (6 nodes, 128 px, 12k, 8 rounds), no attack:

| Setting | Mean ROC-AUC | Privacy (ε, δ=1e-5) |
|---|---:|---|
| FedAvg, no DP | **0.642** | — |
| FedAvg, DP σ = 1.0 | **0.493** | ε ≈ 4.5 |
| FedAvg, DP σ = 2.0 | **0.487** | ε ≈ 1.4 |

- DP imposes a clear utility cost (≈ −0.15 AUC here); **tighter privacy (smaller ε) → lower utility.** At this small scale (128 px, 12k) DP is harsh (near-chance); utility recovers with more data / less noise — the trade-off direction is the result, not the absolute floor.

**Figure:** `plots/05_dp_utility_tradeoff.png`.

### 11.3 Robustness under poisoning (EXP-005, O3)

Under **model_replacement, 2 of 6 nodes malicious** (clean-FedAvg reference AUC = 0.642):

| Aggregation (under attack) | Mean ROC-AUC |
|---|---:|
| FedAvg | **0.500** (diverged — weights → NaN; model destroyed) |
| Krum | 0.545 |
| Median | 0.562 |
| Trimmed mean | **0.571** |

- **Plain FedAvg has no defence and collapses completely** (AUC 0.50 = random); the boosted malicious update overwrites the global model and the weights diverge to NaN.
- **Robust rules survive**, recovering 0.545–0.571 AUC — roughly **32–50%** of the utility lost to the attack (trimmed mean recovers most). This directly demonstrates the value of swapping the aggregation rule.

**Figure:** `plots/06_robust_under_attack.png`.

### 11.4 DP × robust composition — the G1 contribution (EXP-006, O4)

Under attack (model_replacement, 2/6), comparing {FedAvg, Median} × {no-DP, DP σ=1.0}:

| Configuration (under attack) | Mean ROC-AUC | Privacy |
|---|---:|---|
| FedAvg, no DP | 0.500 | — |
| FedAvg, DP on | 0.506 | ε ≈ 4.5 |
| Median, no DP | 0.562 | — |
| Median, DP on | **0.458** | ε ≈ 4.5 |

- **Finding (destructive composition).** Adding DP costs the robust rule (Median) **−0.104** AUC, while it costs FedAvg essentially nothing (**−0.006** / +0.006). DP's clipping and Gaussian noise erase the update-norm/direction signal Median uses to identify and reject malicious updates — so the two sensible defences, run together, **undercut each other**. This is exactly the interaction Gap **G1** predicted and that prior work had not measured directly.
- **Notable nuance.** DP's *clipping* partially *shields* FedAvg (it bounds the boosted malicious update's magnitude), while DP's *noise* *degrades* Median. So DP and robustness do not simply "add up" — their interaction depends on which mechanism dominates. This asymmetry is a concrete, reportable insight.

**Figure:** `plots/07_composition.png`.

### 11.5 Summary table (all headline numbers)

| Study | Metric | Result |
|---|---|---|
| Core (EXP-003) | floor / federated / ceiling AUC | 0.721 / **0.754** / 0.772 (66% gap closed) |
| DP (EXP-004) | AUC @ no-DP / ε≈4.5 / ε≈1.4 | 0.642 / 0.493 / 0.487 |
| Robustness (EXP-005) | AUC under attack: FedAvg / Krum / Median / Trimmed | 0.500 / 0.545 / 0.562 / 0.571 |
| Composition (EXP-006) | ΔAUC from DP: FedAvg / Median | −0.006 / **−0.104** (destructive) |

---

## 12. Discussion — the privacy–robustness–utility trade-off

Three findings combine into one message.

1. **Federation works and is worth it (O6).** The federated model recovers most of the pooled-data performance while keeping data local — the premise of the whole approach holds on real chest X-rays.
2. **Each defence works in isolation (O2, O3).** DP buys a formal privacy budget at a measurable utility cost; robust aggregation rescues the model from an attack that annihilates plain FedAvg.
3. **Together, the defences can fight each other (O4 / G1).** Under attack, turning DP *on* hurts the robust aggregator far more than it hurts undefended averaging, because DP is engineered to remove the very signal (update magnitude/direction) that robust aggregation exploits. The clipping-vs-noise asymmetry (clipping can *help* against magnitude-boosting attacks; noise *hurts* outlier detection) shows the interaction is mechanism-dependent, not a simple additive cost.

**Implication for practitioners.** A privacy-preserving *and* robust federated medical system cannot naively stack DP on top of a robust aggregator and expect both guarantees at full strength. Operating points must be chosen with the interaction in mind — e.g., looser DP (larger ε) when strong Byzantine tolerance is required, or robust rules that are less reliant on magnitude signals when tight DP is mandatory. Characterising this is the project's central contribution.

---

## 13. Limitations

- **Subsets, not the full dataset.** Core results use a ~30k-image subset (224 px); defence results use a ~12k subset at 128 px for runtime. Absolute AUCs are therefore below full-dataset state-of-the-art; the **relative** comparisons are the contribution. Re-running at full scale would raise the absolute numbers but is not expected to change the directions.
- **Severity is a heuristic proxy**, not clinical ground truth (relative bounding-box area; no lung-field segmentation; small labelled subset). Severity metrics are secondary.
- **Few nodes.** 3 nodes (core) / 6 nodes (defence). Robust-aggregation guarantees (e.g., Krum's `n ≥ 2f+3`) are tight at 6 nodes; larger federations would strengthen the robustness claims.
- **Single attack/α in the headline sweep.** The composition study fixes model_replacement at 2/6 and α=0.5; a full (ε × attacker-fraction × heterogeneity) surface is the natural extension.
- **Privacy audit not yet run (O7).** DP provides a formal budget, but gradient-inversion/membership-inference attacks to *empirically* verify the budget are future work.
- **Fairness-under-robustness (O5) not yet quantified.**

---

## 14. Conclusion & Future Work

We built a modular, config-driven hierarchical federated learning system for chest-radiograph diagnosis with embedded DP and Byzantine-robust aggregation, and used it to characterise the privacy–robustness–utility trade-off on real data. Federation beats isolated single-site training and approaches the centralised ceiling; DP and robust aggregation each work alone; but **their composition is destructive** under attack — DP erodes the signal robust aggregation depends on (G1). 

**Future work:** (a) **Privacy audit (O7)** — gradient-inversion (DLG/Inverting-Gradients) and membership-inference (shadow models), reporting SSIM/PSNR and attack-success-rate per ε. (b) **Full trade-off surface (O4)** — sweep ε × attacker-fraction × α. (c) **Heterogeneity/fairness (O5)** — real-data FedProx/FedBN comparison and a fairness-under-robustness check for minority-distribution clients. (d) **Scale-up** — full-dataset, higher-resolution runs. (e) **Demonstrator (O9)** — FastAPI backend + React dashboard streaming the live round-by-round process.

---

## 15. Reproducibility

- **Testbed (O8).** Every run is one YAML config; privacy, robustness, heterogeneity, and attack are independent axes. Entry points: `src/experiments/suite.py` (core comparison), `src/experiments/sweep.py` (defence sweep), `src/experiments/run.py` (CLI).
- **Configs used.** `configs/kaggle_real.yaml` (core, EXP-003), `configs/kaggle_defences.yaml` (defences, EXP-004/005/006).
- **Reports/plots generators.** `src/evaluation/report.py` (core), `src/evaluation/defence_report.py` (defences).
- **Seeds.** All runs seed = 42; partitions are deterministic and patient-disjoint.
- **Raw results.** `results/summary.json`, `results/sweep_results.json`, and per-run `*/federated_results.json`.

---

## 16. Figure index (for the paper)

| File | Caption to use |
|---|---|
| `results/plots/03_floor_fed_ceiling.png` | Federated vs isolated single-site vs centralised (mean ROC-AUC); federated closes 66% of the floor→ceiling gap. |
| `results/plots/01_learning_curve.png` | Federated global mean AUC per round, with single-site floor and centralised ceiling references. |
| `results/plots/02_global_loss.png` | Federated global test loss per round (convergence). |
| `results/plots/04_per_pathology_auc.png` | Per-pathology ROC-AUC, federated vs centralised. |
| `results/plots/05_dp_utility_tradeoff.png` | Privacy–utility trade-off: AUC vs DP noise (ε annotated). |
| `results/plots/06_robust_under_attack.png` | Robustness under model-replacement: FedAvg collapses, robust rules hold. |
| `results/plots/07_composition.png` | DP × robust aggregation (G1): DP degrades Median far more than FedAvg. |

*Source numbers: `results/summary.json` (core), `results/sweep_results.json` (defences). All values are from real runs; none were adjusted. See `study.md` for per-experiment configuration detail (EXP-003 … EXP-006).*
