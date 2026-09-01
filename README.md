# Federated Learning for Privacy-Preserving Medical Diagnosis

**BSCE497J — Project 1** · Team: Devraj K Chandani, Shaurya Saini, Pooja Goel · Guide: Prof. Jabanjalin Hilda J

A hierarchical client–server **federated learning** testbed for chest-radiograph diagnosis that
embeds two defence layers directly into the training loop and studies how they interact:

1. **Client-level differential privacy** — per-sample gradient clipping + calibrated Gaussian noise,
   tracked with a Rényi-DP accountant reporting a cumulative (ε, δ) budget.
2. **Byzantine-robust aggregation** — Krum/Multi-Krum, coordinate-wise trimmed mean/median, and
   similarity-based trust weighting, replacing plain FedAvg.

**Central research question:** do differential privacy and Byzantine-robust aggregation compose
*constructively* or *destructively* when run together? DP's clipping/noise destroys the statistical
structure (update norm, direction, inter-client similarity) that robust aggregators rely on — so
stacking the two defences may not simply add their protections. No prior work studies this
interaction directly (research gap **G1**).

Validated on **NIH ChestX-ray14** with two prediction heads (multi-label pathology detection +
per-disease ordinal severity grading), benchmarked against isolated single-site models (floor) and
a centralised model (ceiling), and **audited empirically** via gradient-inversion and
membership-inference attacks.

> Full project context, locked scope decisions, and the task plan live in **`CLAUDE.md`**.
> Session continuity in **`progresstracker.md`**; the incremental research record in **`study.md`**.

---

## Repository layout

```
configs/     YAML experiment configs (privacy × robustness × heterogeneity axes)
data/        raw/ (NIH CXR, gitignored) · processed/ · partitions/ (IID + non-IID manifests)
src/
  data/          dataset loading, partitioning, severity-proxy derivation
  models/        backbone + dual-head (pathology + severity)
  federated/     Flower client/server + strategies/ (aggregation rules)
  privacy/       DP-SGD wrapper + Rényi-DP accountant
  attacks/       byzantine/ (poisoning) + inference/ (privacy-audit attacks)
  evaluation/    metrics (AUC, ordinal severity, SSIM/PSNR, attack success rate)
  experiments/   config-driven sweep runner + CLI
notebooks/   Colab-mirrored training/sweep entrypoint
backend/     FastAPI app (trigger/monitor demo runs, stream metrics, serve results)
frontend/    React + Tailwind live demo dashboard
results/     experiment outputs (csv/json/plots) consumed by study.md
```

## Setup

Primary environment is **Python 3.11** (matches Colab; the full stack incl. Flower/Opacus installs
cleanly, so smoke tests run locally and Colab is used only for GPU-scale training).

```bash
py -3.11 -m venv venv          # or: python3.11 -m venv venv
# Windows PowerShell:
venv\Scripts\Activate.ps1
# macOS/Linux:
# source venv/bin/activate

# torch/torchvision are platform-specific — install them first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # local CPU
# (Colab: torch is preinstalled — skip this line)

pip install -r requirements.txt
```

### Quick smoke test (no dataset download needed)

```bash
# generate a tiny synthetic dataset, then run the whole pipeline end-to-end on CPU:
python -c "from src.data.synthetic import make_synthetic_dataset as m; m('data/_smoke', write_images=True)"
python -m src.experiments.run centralized --config configs/smoke.yaml --data-root data/_smoke

# unit smoke tests:
python tests/test_data_pipeline.py       # torch-free: labels, severity proxy, partitioning
python tests/test_model_training.py      # end-to-end: model + centralized training
```

## Status

Tasks 1–4 of `CLAUDE.md §7` implemented and smoke-verified on synthetic data (data pipeline, dual-head
model, centralized-baseline runner + CLI). Real-dataset/GPU baseline runs happen on Colab. See
`progresstracker.md` for the live state and `study.md` for the research log.
