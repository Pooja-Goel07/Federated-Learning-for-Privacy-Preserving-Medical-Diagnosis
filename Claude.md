# CLAUDE.md — Federated Learning for Privacy-Preserving Medical Diagnosis

This file is the persistent context document for Claude Code across sessions. Read this in full at the start of every session before touching code. Update it whenever architecture, scope, or major decisions change — it should always reflect the *current* state of intent, not the history of how we got here (that belongs in `progresstracker.md` and `study.md`).

---

## 1. Title

**Federated Learning for Privacy-Preserving Medical Diagnosis**
BSCE497J — Project 1, Review 2 in progress
Team: Devraj K Chandani, Shaurya Saini, Pooja Goel · Faculty Guide: Prof. Jabanjalin Hilda J

## 2. Abstract

Clinical deep learning is bottlenecked not by model capacity but by data access. Individual hospitals hold datasets that are small, demographically narrow and acquisition-biased, while regulatory frameworks (HIPAA, GDPR, India's DPDP Act) make raw-data centralisation infeasible. Federated learning (FL) addresses this by keeping data in place and aggregating only model updates — but FL alone is **not** a privacy guarantee. Shared gradients are invertible via reconstruction and membership-inference attacks, and the open participation model exposes the global model to poisoning and Byzantine faults from compromised or misconfigured clients.

This project builds a **hierarchical client–server federated architecture** for medical diagnosis, where each leaf node (simulated hospital) performs local training and a centre node performs secured aggregation. Two defence layers are embedded directly into the training loop:

1. **Client-level differential privacy (DP)** — per-sample gradient clipping + calibrated Gaussian noise, tracked with a Rényi-DP accountant, yielding a formal cumulative (ε, δ) budget.
2. **Byzantine-robust aggregation** — Krum/Multi-Krum, coordinate-wise trimmed mean/median, and similarity-based trust weighting, replacing plain FedAvg so a bounded fraction of adversarial updates cannot dominate the global model.

Statistical heterogeneity across simulated hospital nodes (non-IID label/feature skew) is handled via proximal regularisation (FedProx-style) and local batch normalisation (FedBN-style).

The framework is validated on chest radiography — multi-label pathology detection plus per-disease ordinal severity grading — comparing the federated global model against isolated single-site baselines and a centralised upper bound. The system is **audited empirically**: gradient-inversion and membership-inference attacks are executed against intercepted client updates to verify the claimed privacy is actually realised, not just asserted by construction.

**The central research contribution is characterising the joint privacy–robustness–utility trade-off** — specifically whether DP noise and Byzantine-robust aggregation compose *constructively* or *destructively* when run together, since DP's clipping/noise destroys exactly the statistical structure (update norm, direction, inter-client similarity) that robust aggregators rely on to detect malicious updates. No prior work in the literature studies this interaction directly (Research Gap G1).

## 3. Objectives

1. Design a hierarchical federated architecture (1 centre node + N leaf/hospital nodes) with a modular interface so the aggregation rule and privacy mechanism can be swapped independently.
2. Integrate client-level DP into local training (per-sample gradient clipping + calibrated Gaussian noise) with a Rényi-DP accountant reporting cumulative (ε, δ).
3. Implement and compare Byzantine-robust aggregation rules (Krum/Multi-Krum, trimmed mean, coordinate-wise median, trust weighting) against plain FedAvg under label-flipping, sign-flipping, scaled-update, and model-replacement attacks at varying adversarial client fractions.
4. **Characterise the joint privacy–robustness–utility trade-off** as the primary contribution — diagnostic performance as a function of (ε, attacker fraction, heterogeneity level); identify regimes where DP noise measurably degrades robust aggregation's ability to detect/reject malicious updates (constructive vs. destructive composition).
5. Mitigate statistical heterogeneity (proximal regularisation + local batch norm) and verify robust aggregation doesn't systematically exclude honest minority-distribution clients (fairness-under-robustness check).
6. Validate on chest radiography with two prediction heads: multi-label pathology detection + per-disease ordinal severity grading (mild/moderate/severe), benchmarked against (a) isolated single-site models and (b) a centralised upper bound.
7. Empirically audit privacy claims via gradient-inversion and membership-inference attacks against intercepted updates, reporting reconstruction quality (SSIM/PSNR) and attack success rate per privacy budget.
8. Release a reproducible, configuration-driven testbed where privacy, robustness, and heterogeneity axes vary independently.
9. **Demonstrate** the end-to-end federated process concretely: 3 simulated hospital nodes train locally, send updated weights to a root/centre node, the centre node aggregates, and the aggregated global model is evaluated — visualised via a live dashboard.

---

## 4. Locked-in Scope Decisions

These were explicitly decided with the user — do not silently change them. If a change is needed, flag it and update this section.

| Decision | Choice | Rationale |
|---|---|---|
| Dataset | **NIH ChestX-ray14** (112,120 frontal CXR images, 14 pathology labels) | Large, well-documented, public, includes a bounding-box subset needed for severity |
| Severity labels | **Heuristic proxy** derived from the NIH `BBox_List_2017.csv` bounding-box subset (~984 images, 8 pathologies) — relative lesion-box area vs. lung-field area, binned into mild/moderate/severe (e.g. terciles) | NIH/CheXpert have no native severity labels; this keeps severity task on the *same* dataset/disease space as detection instead of stitching in an unrelated severity-only dataset. **Must be documented in `study.md` as a heuristic proxy, not clinical ground truth** — this is a stated limitation, not a hidden one. |
| Number of hospital nodes | **3** (leaf nodes), configurable in code for later sweeps to 5+ if time permits | Matches the "demo the federated architecture" goal cleanly; keep the aggregation code node-count-agnostic |
| Node data partitioning | Both an **IID split** and a **non-IID split** (label-skew, e.g. Dirichlet partition) of ChestX-ray14 across the 3 nodes — non-IID is required for objective 5 (heterogeneity) | — |
| FL framework | **Flower (flwr)** | User requirement |
| DP implementation | Per-sample gradient clipping + Gaussian noise via **Opacus** (or custom if Opacus doesn't play well with Flower's client loop) + a Rényi-DP accountant | User requirement (objective 2) |
| Backend | **FastAPI** | User requirement |
| Frontend | **React + Tailwind CSS** | User requirement |
| Python environment | Always a **virtual environment** (`venv`) for local dev; **Google Colab** notebook(s) permitted for one-off / GPU-heavy training runs | User requirement |
| Research sweep orchestration | Config-driven Python scripts/CLI (not just notebooks) so runs are reproducible and loggable | Needed for the "rigorous study" requirement |

---

## 5. Tech Stack

**Core ML**
- Python 3.10+, PyTorch (backbone: DenseNet-121 or ResNet-50, ImageNet-pretrained, standard choice for ChestX-ray14 in the literature)
- **Flower (flwr)** — federated orchestration, custom `Strategy` subclasses for each aggregation rule
- **Opacus** — DP-SGD (per-sample clipping + Gaussian noise) and Rényi-DP accounting; fallback to a custom accountant if Opacus doesn't integrate cleanly with Flower's client `fit()` loop
- scikit-learn, NumPy, pandas — metrics, data wrangling, ordinal severity binning
- Pillow / OpenCV — image preprocessing

**Federated node simulation**
- 3 (or N) simulated clients on a single machine (Colab or local) rather than real network nodes — appropriate for a research/demo project.
- **Implementation note (decided Task 6):** we use an *in-process orchestrator* (`src/federated/simulation.py`) that drives rounds by hand but **delegates aggregation to real Flower `Strategy` classes** (`flwr.server.strategy.FedAvg`, and the robust rules in Task 9), reusing Flower's `NumPyClient`, `FitRes`, and parameter (de)serialisation. We deliberately do **not** use Flower's Ray-backed `Simulation` engine: Ray is heavy/flaky on Windows, and an in-process loop is deterministic, easy to live-stream to the dashboard, and gives tight control needed for the DP/Byzantine composition experiments. Flower (`flwr`) is still the FL framework of record — the aggregation strategies are genuine Flower `Strategy` subclasses. Installed `flwr` **core** (not `flwr[simulation]`), so Ray is not a dependency.

**Attacks & privacy audit**
- Custom implementations: label-flipping, sign-flipping, scaled-update, model-replacement (Byzantine attacks); gradient-inversion (deep leakage from gradients / "Inverting Gradients" style), membership-inference (shadow-model based)
- `torchmetrics` or custom SSIM/PSNR for reconstruction-quality scoring

**Backend**
- **FastAPI** — serves experiment results, exposes endpoints to trigger/monitor demo FL runs, streams round-by-round metrics (e.g. via WebSocket or Server-Sent Events) to the frontend

**Frontend**
- **React + Tailwind CSS** — live dashboard: per-node training status, weight-update flow visualisation (leaf → centre), aggregation step, global model evaluation curves, attack/defence toggles for the demo

**Experiment orchestration**
- Config-driven Python scripts (YAML/JSON configs + a small CLI, e.g. `argparse`/`typer`) so each sweep run (ε, attacker fraction, aggregation rule, heterogeneity level) is reproducible and logged, not run ad hoc from a notebook
- Simple experiment logging: CSV/JSON result files + optionally `matplotlib`/`plotly` for plots consumed by `study.md`

**Environment**
- `venv` for all local development; `requirements.txt` (pinned versions) or `pyproject.toml`
- A Colab notebook mirror of the training/sweep entry point for GPU-bound runs, reading/writing to the same data/results structure as local

---

## 6. Repository Structure (proposed)

```
fl-medical-diagnosis/
├── CLAUDE.md                  # this file
├── progresstracker.md         # session-to-session continuity log (see §8)
├── study.md                   # research log: methodology, results, observations (see §9)
├── README.md                  # public-facing project readme
├── requirements.txt
├── configs/                   # YAML configs for experiments (privacy, robustness, heterogeneity axes)
├── data/
│   ├── raw/                   # NIH ChestX-ray14 (gitignored)
│   ├── processed/             # preprocessed tensors/splits
│   └── partitions/            # per-node IID / non-IID partition manifests
├── src/
│   ├── data/                  # dataset loading, partitioning, severity-label derivation
│   ├── models/                # backbone + dual-head (pathology + severity) architecture
│   ├── federated/
│   │   ├── client.py          # Flower NumPyClient / DP-wrapped client
│   │   ├── strategies/        # FedAvg, Krum, Multi-Krum, trimmed-mean, median, trust-weighting
│   │   └── server.py          # Flower server / simulation entrypoint
│   ├── privacy/                # DP-SGD wrapper, Rényi-DP accountant
│   ├── attacks/
│   │   ├── byzantine/         # label-flip, sign-flip, scaled-update, model-replacement
│   │   └── inference/         # gradient-inversion, membership-inference audit attacks
│   ├── evaluation/            # metrics: AUC/accuracy per pathology, ordinal severity metrics, SSIM/PSNR, ASR
│   └── experiments/           # sweep runner (config-driven), CLI entrypoint
├── notebooks/
│   └── colab_training.ipynb   # Colab-mirrored training/sweep entrypoint
├── backend/                   # FastAPI app
│   └── app/
├── frontend/                  # React + Tailwind app
│   └── src/
└── results/                   # experiment outputs (csv/json/plots), consumed by study.md
```

---

## 7. Development Plan

Work through tasks roughly in order; some (7–9) can interleave. Each task should end with a `progresstracker.md` update. Mark tasks `[ ]` not started / `[~]` in progress / `[x]` done directly in `progresstracker.md`, not here.

**Task 1 — Environment & repo scaffolding**
Set up `venv`, pinned `requirements.txt`, repo folder structure per §6, git init, `.gitignore` for data/large artifacts. Set up a Colab notebook stub that mirrors the local training entrypoint.

**Task 2 — Data pipeline**
Download/prepare NIH ChestX-ray14 (images + `Data_Entry_2017.csv` + `BBox_List_2017.csv`). Build the multi-label pathology dataset loader. Derive the severity-proxy labels from the bounding-box subset (document the exact binning method in `study.md` as it's decided). Build IID and non-IID (Dirichlet label-skew) partitioners that split data across 3 simulated hospital nodes.

**Task 3 — Model architecture**
Backbone (DenseNet-121/ResNet-50) with two heads: multi-label pathology classification head + ordinal severity regression/classification head (only active on the bbox-subset images that have a severity proxy). Define loss functions for each head and a combined training objective.

**Task 4 — Centralised baseline (upper bound)**
Train the model centrally on the full dataset (Colab, GPU). This is the ceiling the federated setup is compared against. Log results into `study.md`.

**Task 5 — Isolated single-site baselines**
Train the same model separately on each node's local partition only (no federation) — the floor comparison. Run for both IID and non-IID partitions.

**Task 6 — Core federated learning loop (Flower, plain FedAvg)**
Implement the Flower client (local training loop) and server (FedAvg strategy) for 3 nodes using the virtual-client simulation engine. Get the basic "3 nodes train locally → send weights to centre → aggregate → evaluate global model" loop working end-to-end. **This is the core demo deliverable** — should be demonstrable/visualisable early.

**Task 7 — Heterogeneity mitigation**
Add FedProx-style proximal regularisation and FedBN-style local batch-norm to the client training loop; compare convergence against plain FedAvg on the non-IID partition.

**Task 8 — Differential privacy layer**
Integrate per-sample gradient clipping + calibrated Gaussian noise into the client's local training step (Opacus, or custom if needed). Implement/attach the Rényi-DP accountant to report cumulative (ε, δ) per training run. Sweep over a few ε budgets and measure the utility cost vs. the Task 6 baseline.

**Task 9 — Byzantine-robust aggregation strategies**
Implement Krum, Multi-Krum, coordinate-wise trimmed mean, coordinate-wise median, and similarity-based trust weighting as pluggable Flower `Strategy` subclasses (swap-in replacements for FedAvg). Implement the attack client behaviours: label-flipping, sign-flipping, scaled-update, model-replacement. Sweep over adversarial client fractions and compare each aggregation rule's resilience.

**Task 10 — The core research experiment: joint privacy–robustness composition**
Run DP (Task 8) and Byzantine-robust aggregation (Task 9) **together**, sweeping (ε, attacker fraction, heterogeneity level) jointly. This is the primary research contribution — measure whether DP noise degrades the robust aggregator's ability to distinguish honest from malicious updates (i.e., does the composition hold constructively, or does it break down). Also run the fairness-under-robustness check (objective 5): confirm robust aggregation isn't silently excluding honest minority-distribution (non-IID) clients.

**Task 11 — Privacy audit attacks**
Implement gradient-inversion (deep-leakage-from-gradients style) and membership-inference (shadow-model based) attacks against intercepted client updates. Run them at each DP budget from Task 8/10 and report reconstruction quality (SSIM/PSNR) and attack success rate — this empirically verifies the claimed ε actually protects against realistic attacks rather than just asserting it analytically.

**Task 12 — Evaluation & metrics consolidation**
Consolidate all metrics across tasks 4–11: AUC/accuracy per pathology, ordinal severity metrics (e.g. MAE/QWK on the mild/moderate/severe scale), attack success rate, reconstruction SSIM/PSNR, and the privacy–robustness–utility trade-off surfaces (ε × attacker-fraction × heterogeneity → utility). This feeds `study.md` directly.

**Task 13 — FastAPI backend**
Expose endpoints to trigger a demo FL run (3 nodes, chosen strategy/DP settings), stream round-by-round progress (per-node local training status, weight upload, aggregation event, global eval), and serve stored experiment results from Task 12 for the dashboard.

**Task 14 — React + Tailwind frontend demo dashboard**
Build the visual demo: 3 hospital nodes training locally → arrows showing weight updates flowing to the centre node → aggregation step → global model evaluation results. Include toggles to switch aggregation strategy, enable/disable DP, and inject a Byzantine attack, so the privacy/robustness composition story can be demonstrated live, not just read off a plot.

**Task 15 — Review 2 preparation**
Once you tell me what Review 2 requires (methodology slides, architecture diagram, partial results, etc.), we prepare that deliverable using whatever's been completed in `study.md`/`progresstracker.md` at that point.

---

## 8. `progresstracker.md` — session continuity (mandatory)

**Claude Code must create and continuously maintain `progresstracker.md` at the repo root.** Purpose: since development happens across many sessions, this file lets a fresh session (with no memory of prior ones) pick up exactly where the last one left off without re-deriving context.

At the **end of every working session**, update it with:
- Current task from §7 (which task number, what sub-part)
- What was just completed (files touched, what now works)
- What's broken / in-progress / blocked, with enough detail to resume immediately
- Any decisions made during the session that aren't yet reflected in this CLAUDE.md (flag these — CLAUDE.md itself should then be updated too)
- Exact next step to take when work resumes

Keep entries dated and reverse-chronological (newest on top). Keep old entries — this is a log, not a status snapshot; don't delete history from it. This file must be created before or during Task 1.

## 9. `study.md` — research log (mandatory, Claude Code maintains this itself)

**Claude Code must create and continuously maintain `study.md`** — this is the actual research record for the "study/experimentation" side of the project (as distinct from the "build a working demo" side), and must be kept complete enough that the Review 2/final report could be written directly from it.

For every experiment run (each task from §7 that produces results, especially Tasks 4, 5, 8, 9, 10, 11), log:
- Exact configuration (dataset partition used, model, hyperparameters, ε/δ if DP is on, aggregation rule, attacker fraction/type if any, heterogeneity setting)
- Results (metrics defined in Task 12), ideally with a table or saved plot reference in `results/`
- Observations — what happened, anything surprising, how it compares to the isolated-defence literature (§ research gap G1 in particular)
- Any deviation from the plan and why

This file should build up into the actual evidence base for objective 4 (the joint privacy–robustness–utility trade-off characterisation) — it's not a changelog, it's the paper's results section being written incrementally as experiments run.

---

## 10. Open Items / Things to Revisit

- Exact severity-proxy binning thresholds (terciles vs. fixed area cutoffs) — decide during Task 2, document the final method in `study.md`.
- Non-IID partitioning scheme specifics (Dirichlet α value(s) to sweep) — decide during Task 2/7.
- Whether to extend beyond 3 nodes for later robustness sweeps (Task 9/10) if time allows — keep code node-count-agnostic from the start so this is just a config change.
- Review 2 deliverable requirements — not yet provided by user, will update this file once known.