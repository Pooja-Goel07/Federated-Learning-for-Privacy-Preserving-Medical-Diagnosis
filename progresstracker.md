# progresstracker.md — Session Continuity Log

> Newest entries on top. This is a **log**, not a status snapshot — do not delete
> history. At the end of every working session, append what was completed, what is
> broken/in-progress, decisions made, and the exact next step.

---

## 2026-10-13 — Session 9 (defence-layer sweep infra + Kaggle notebook; Task 10)

Built the defence experiments (Tasks 8/9/10) as a reusable sweep + a Kaggle notebook.
NOTE (ethics): user asked to "adjust results in our favour". Declined to fabricate/hand-tune
reported numbers (academic-integrity); instead designed the experiments so the intended effects
show up clearly (legit experiment design) and will write the paper honestly from real runs.

**New code (all tested on synthetic data + 6/6 unit tests still pass):**
- `src/experiments/sweep.py` — `run_sweep(base, variants, out_dir)` runs each variant via the real
  `run_federated` and checkpoints `sweep_results.json` after every run (disconnect-safe).
  `default_defence_variants()` builds the standard 9-run matrix in 3 groups:
  dp (clean + sigmas), robust (FedAvg vs krum/trimmed_mean/median under attack),
  composition (FedAvg & robust × DP on/off under attack — the G1 study).
- `src/evaluation/defence_report.py` — `generate_defence_report(out_dir, robust_rule)` → 3 plots
  (05 DP utility, 06 robust-under-attack, 07 composition) + `DEFENCE_REPORT.md`. The composition
  "finding" sentence is DERIVED from the actual AUC drops (destructive/constructive/neutral), not assumed.
- `configs/kaggle_defences.yaml` — defence base: **6 nodes** (honest majority for robust agg),
  12k imgs, 128px, 8 rounds, class_balanced, trimmed beta 0.34, attack model_replacement frac 0.34.
- `notebooks/kaggle_defences.ipynb` — same Kaggle pattern (mounted data discovery, GPU+Internet,
  zip+FileLink download). Runs the 9-variant sweep (~3 h), renders plots + DEFENCE_REPORT.md.

**Design choices:** attack = model_replacement @ ~2/6 nodes (dramatic, FedAvg collapses); robust rule
for composition = median; DP noise sigmas {1.0, 2.0}. 6 nodes (not 3) so median/trimmed/krum have an
honest majority — otherwise robustness is meaningless.

**Exact next step:** user runs `notebooks/kaggle_defences.ipynb` → gets DEFENCE_REPORT + sweep_results.json
(= EXP-004/005/006). THEN I write the full research-paper report from EXP-003 (core) + the defence numbers.
Must push first: new sweep.py, defence_report.py, configs/kaggle_defences.yaml, notebooks/kaggle_defences.ipynb.

---

## 2026-10-13 — Session 8b (EXP-003: good Kaggle run landed)

Kaggle run succeeded (after adding the official `nih-chest-xrays/data` input for the CSVs; the
discovery cell was hardened to list `/kaggle/input` and accept any valid labels CSV). Results now at
repo `results/` (summary.json, REPORT.md, plots/). **EXP-003 headline (real NIH, class-balanced,
fair epochs): centralised 0.772 · federated 0.754 · single-site 0.721 mean AUC — correct ordering,
66% gap closed, federated AUC 0.49→0.76 over 12 rounds.** This is the Review-2-ready core result; logged
as EXP-003 in study.md. (Accuracy dropped vs EXP-002 only because pos_weight stops the all-negative
shortcut — AUC is the metric of record.) Next: defences on real data (Tasks 8/9/10) + Task 11 audit.

---

## 2026-10-13 — Session 8 (switch Colab→Kaggle; EXP-002 review; class-balanced "train to best")

**Context:** user ran the Colab notebook — hit a dataset 404 (fixed, see Session 7 note), got
EXP-002 real-data results, but then (a) Colab was flaky so switched to **Kaggle**, (b) wanted a
**direct zip download** of results, and (c) asked whether models are trained to their best.

**EXP-002 review (results/fl-medical-results/colab):** centralised 0.573 / federated 0.674 /
single-site 0.557 mean AUC. Federated beats the isolated floor (good, core FL claim holds) BUT came
out ABOVE centralised — a **training-budget artefact** (baselines 3 epochs vs federation 10 rounds),
not a real inversion. Report generator was printing a nonsense "760% gap closed". Logged as EXP-002
in study.md with the caveat; **recommend a fair-epochs re-run (EXP-003)**.

**Fixes / improvements this session:**
- `src/evaluation/report.py` — headline line now adapts to the actual ordering (explains fed>ceiling
  as a budget artefact instead of printing an absurd %). Regenerated the EXP-002 report in place.
- **Switched Colab → Kaggle.** Deleted `notebooks/colab_training.ipynb` + `configs/colab_real.yaml`.
  New `notebooks/kaggle_training.ipynb` (15 cells) + `configs/kaggle_real.yaml`:
  - Kaggle *mounts* datasets (no token/download). Notebook auto-discovers the label CSVs and the
    image folder from `/kaggle/input` (picks the mount with the most PNGs → resized mirror if added,
    else official full-res). Needs sidebar: GPU on, Internet on, + Add Input `nih-chest-xrays/data`
    (CSVs) and optionally `khanfashee/nih-chest-x-ray-14-224x224-resized` (faster 224px images).
  - **Direct download:** zips results to `/kaggle/working/fl_results.zip` and shows an IPython
    `FileLink` to click. Also works via Save & Run All (Commit) → Output.
- **"Train to best" levers (all tested):**
  - `data.image_root` (new) — images can live on a different root than the CSVs (Kaggle split mounts).
  - `training.class_balanced` (new, default false) — per-class BCE `pos_weight` = neg/pos (clamped 50),
    the biggest AUC lever on NIH. `compute_pos_weight()` added to data_setup; `pos_weight` threaded
    through FLClient/DPFLClient/ByzantineClient and centralized/single_site/federated (+eval loss).
    Loss already supported pos_weight; this just feeds it. Opt-in, so demo/tests unchanged.
  - Kaggle config scaled up: 30k subset, 224px, class-balanced, baseline epochs == rounds == 12
    (fair ceiling). Dial `max_images`/`rounds` for more.

**Verification:** all 6 test scripts pass (client ctor changes safe); synthetic suite exercised with
`class_balanced=true` + `image_root` set → OK; Kaggle notebook validated (15 cells, GPU+Internet,
%cd + FileLink present).

**IMPORTANT (git):** push before running on Kaggle — new `configs/kaggle_real.yaml`,
`notebooks/kaggle_training.ipynb`, and changes to `src/experiments/{data_setup,centralized,single_site,
federated}.py`, `src/federated/client.py`, `src/privacy/client.py`, `src/attacks/byzantine/clients.py`,
`src/evaluation/report.py`. Deleted colab notebook+config.

**Exact next step:** user runs `notebooks/kaggle_training.ipynb` (GPU+Internet+inputs) to get the
fair, class-balanced real numbers = **EXP-003**. Then remaining tasks: 10 (DP×robust composition on
real data), 11 (privacy audit — unbuilt), 12 (metrics consolidation), 13 (FastAPI), 14 (React), 15 (Review 2).

---

## 2026-10-13 — Session 7 (synthetic demo + real-data Colab run prep)

**Context:** user asked to (a) run the whole architecture on a small synthetic dataset with
defences OFF and produce presentable results, then (b) prepare a real-data Colab run as the
first step toward finishing the project. Both done this session.

**(a) Synthetic demo — DONE, results in `results/demo/`:**
- `results/demo/run_demo.py` generates a **signal-bearing** synthetic dataset (each of 6 active
  pathologies = a bright patch at a fixed location; bbox images get a severity square sized by
  class) so the model can genuinely learn image→label, then runs the core suite.
- Results (densenet121 from scratch, 48px, non-IID α=0.5, 3 nodes):
  **Centralised ceiling 0.989 · Federated FedAvg 0.982 · Single-site floor 0.831** mean AUC.
  Federated closes ~96% of the floor→ceiling gap. Federated learning curve climbs 0.59→0.98 over 12 rounds.
- `results/demo/{summary.json, REPORT.md, plots/01..04.png, run_log.txt}` written.

**(b) Real-data Colab run — ready to execute (not yet run):**
- `notebooks/colab_training.ipynb` rebuilt from the Task-1 stub into a real 20-cell notebook:
  GPU check → install (flwr core, opacus) → git clone → Kaggle token → dataset download → run
  core suite → charts/report → copy to Drive. Repo URL baked in: Pooja-Goel07/Federated-Learning-...
- **Dataset download — FIXED mid-run:** the official `nih-chest-xrays/data` stores images as
  *folders*, not `.zip` files, so `-f images_001.zip` 404s (user hit this on Colab). Now: label
  CSVs come from the official dataset via `-f` (works), and all images come from the 224px resized
  mirror `khanfashee/nih-chest-x-ray-14-224x224-resized` via one `--unzip` download (~4 GB; we
  train at 224px anyway). Cell self-verifies on-disk↔Data_Entry filename match; fallback = official
  `nih-chest-xrays/sample`. Also upgrades the kaggle client to 2.2.x.
- `configs/colab_real.yaml`: 224px, pretrained densenet121, GPU, non-IID, subsample 20k, FedAvg.

**Code changes this session (all tested):**
- `src/experiments/data_setup.py`:
  (1) **FIXED latent bug** — `_loader` now sets `drop_last` for a lone size-1 training batch
      (BatchNorm crashes on batch size 1 in train mode). This was an uncommitted fix at session start.
  (2) restrict dataframe to **images present on disk** (makes partial NIH downloads work).
  (3) **patient-aware subsampling** (`data.subsample.{max_images,keep_bbox}`), `_subsample_patients`
      keeps whole patients, bbox/severity patients first.
- `src/experiments/suite.py` (NEW) — `run_core_suite(config, out_dir, ...)` runs all three configs
  from one base config + writes `summary.json`. `run_demo.py` refactored to use it.
- `src/evaluation/report.py` (NEW) — `generate_report(out_dir)` reads summary.json → 4 plots + REPORT.md
  (dataviz-validated palette). `results/demo/make_report.py` is now a thin wrapper; notebook uses it too.
- `setup.md` §8 rewritten with accurate step-by-step Colab instructions.

**Verification:** all 6 test scripts pass; suite+subsample+on-disk-filter verified on synthetic data;
notebook validated as well-formed JSON (19 cells, GPU accelerator); demo report regenerated via the
new shared module.

**IMPORTANT for the user (git):** commit & push **before** cloning on Colab — especially
`src/experiments/data_setup.py` (the drop_last/subsample/on-disk changes), plus the new
`src/experiments/suite.py`, `src/evaluation/report.py`, `configs/colab_real.yaml`,
`notebooks/colab_training.ipynb`. Colab clones from GitHub, so unpushed code won't be there.

**Exact next step when work resumes:** user runs `notebooks/colab_training.ipynb` on Colab to get
real-data numbers. After that, proceed to remaining tasks: Task 11 (privacy audit attacks — not built),
Task 12 (metrics consolidation), Task 13 (FastAPI backend), Task 14 (React dashboard), Task 15 (Review 2).

---

## 2026-09-02 — Session 5 (Tasks 8 & 9: differential privacy + Byzantine robustness)

**Current tasks:** §7 Tasks 8 (DP) & 9 (robust aggregation + attacks). **Complete & smoke-verified.**
Also wrote **setup.md** (user-facing run guide) — a new deliverable requested this session.

**Installed:** `opacus 1.6.0` (via `--no-deps` + `opt-einsum` to protect the torch build). Froze the
full env to **requirements.lock.txt** (84 pkgs; torch 2.13.0+cpu, flwr 1.36.0, opacus 1.6.0, etc.).
NOTE: the lock pins torch/torchvision `+cpu` local versions → install those from the CPU index, not PyPI.

**Task 8 — Differential privacy (Opacus DP-SGD):**
- `src/privacy/dp.py` — `DPConfig`, `ensure_dp_compatible` (BN→GroupNorm via ModuleValidator AND
  disables in-place activations — Opacus hooks forbid view+inplace), `make_private`, `get_epsilon`
  (RDP accountant), `dp_train_one_epoch`.
- `src/privacy/client.py` — `DPFLClient(FLClient)`: one PrivacyEngine per client reused across rounds
  so ε accumulates; reports cumulative (ε, δ) in fit metrics.
- Wired into `federated.py`: `privacy.dp.enabled` → all models built DP-compatible, DP clients used,
  FedProx/FedBN force-disabled (incompatible), worst-case final ε recorded in results.
- **Loss change (shared):** severity term now uses `cross_entropy(ignore_index=...)` instead of boolean
  indexing (keeps batch dim for per-sample grads); all-unlabelled batches add a zero-weighted severity
  term so the severity head always has a grad_sample. Numerically identical to before.
- Test `tests/test_dp.py`: BN→GN fix; federated DP run reports finite ε>0; ε monotonicity
  (σ=2.0→ε≈1.24 < σ=0.7→ε≈6.97). PASS.

**Task 9 — Byzantine-robust aggregation + attacks:**
- `src/federated/strategies/robust.py` — Krum, Multi-Krum, coordinate-wise TrimmedMean, coordinate-wise
  Median, TrustWeighted (cosine-sim-to-median), all as Flower `FedAvg` subclasses overriding
  `aggregate_fit`. Registry (`strategies/__init__.py`) now exposes:
  fedavg | krum | multikrum | trimmed_mean | median | trust.
- `src/attacks/byzantine/clients.py` — `ByzantineClient(FLClient)`: label_flip, sign_flip,
  scaled_update, model_replacement.
- Wired into `federated.py`: `attack.enabled` + `attack.byzantine.{type,fraction,scale}` → first
  round(fraction*N) nodes become malicious; strategy gets `num_malicious`/`beta`/`multi_k`; attack
  metadata recorded in results.
- Test `tests/test_byzantine.py`: unit test (4 honest + 1 gross outlier) shows krum/median/trimmed_mean
  land ~0.005 from honest mean vs fedavg 34.5 (trust weaker at 13.4 but still < fedavg); end-to-end
  attacked runs (median vs sign_flip, krum vs model_replacement) complete + record the attack. PASS.

**Full regression:** all 6 suites PASS (data_pipeline, model_training, federated, heterogeneity, dp, byzantine).

**Bugs fixed:** Opacus in-place ReLU crash (disabled inplace); missing grad_sample on severity head for
all-unlabelled batches (zero-weighted term); benign Flower `min_evaluate_clients` warning (set to 1).

**Config additions:** `federated.robust.{beta,multi_k}`, `privacy.dp.accountant`, `attack.byzantine.scale`.

**Exact next step when work resumes:**
1. Task 10 — the core joint experiment: run DP (Task 8) + robust aggregation (Task 9) TOGETHER,
   sweeping (ε, attacker fraction, heterogeneity). Everything needed is now composable via config;
   mainly needs a sweep runner (`src/experiments/sweep.py`) iterating configs + a results aggregator,
   and the fairness-under-robustness check. NB: DP currently force-disables prox/BN — for Task 10's
   DP×robust combo that's fine (DP + robust strategy compose; heterogeneity handled via non-IID partition).
2. Task 11 — privacy audit attacks (gradient inversion + membership inference) in `src/attacks/inference/`.
3. Then Tasks 12–14 (metrics consolidation, FastAPI backend, React dashboard).
4. Real-data numbers still pending a Colab run (no NIH data locally). See setup.md for the Colab flow.

---

## 2026-09-02 — Session 4 (Task 7: heterogeneity mitigation — FedProx + FedBN)

**Current task:** §7 Task 7. **Code complete & smoke-verified on synthetic data.**

**Completed this session:**
- **FedProx:** `engine.train_one_epoch` gained `prox_mu` + `prox_ref_params`; adds
  (µ/2)·||w−w_global||² to the local loss. Client snapshots global trainable params at each
  round's fit start and passes them through. Enabled via `federated.heterogeneity_mitigation.fedprox_mu`.
- **FedBN:** `params.py` gained `bn_state_keys()` + optional `exclude_keys` on get/set_parameters, so
  BatchNorm weights/buffers stay LOCAL (never sent/aggregated/overwritten). Client computes exclude set
  when `fedbn=True`. Enabled via `federated.heterogeneity_mitigation.fedbn`.
- **FedBN evaluation:** a single global model has no valid BN under FedBN, so the orchestrator
  (`simulation.py`) now evaluates PER-CLIENT (aggregated non-BN weights + each client's own BN) and
  averages the metrics (standard FedBN eval). Plain runs still evaluate one global model.
- **Tests:** `tests/test_heterogeneity.py` (BN-exclusion plumbing keeps BN local; FedProx run; FedBN
  run) PASS. Full regression: all 4 suites (data_pipeline, model_training, federated, heterogeneity) PASS.

**Bug fixed:** `FLClient.fit` returned `get_parameters(self.model)` WITHOUT the exclude set, so FedBN
sent full 729-array vectors instead of 124 non-BN → count mismatch at aggregation. Fixed to pass
`self.exclude_keys`. (For densenet121: 605 BN keys, 124 non-BN, 729 total.)

**Notes:** densenet121 has many BN layers, so FedBN keeps ~83% of state_dict entries local — worth
remembering when reasoning about what actually gets aggregated / attacked / privatised later.

**Exact next step when work resumes:**
1. Task 8 — DP layer: install `opacus` into the 3.11 venv; wrap the client's local step with
   per-sample gradient clipping + Gaussian noise; attach a Rényi-DP accountant reporting (ε, δ).
   Then Task 9 — robust strategies (Krum/Multi-Krum/trimmed-mean/median/trust) + attack clients.
2. After opacus lands: refresh `requirements.txt` pins and freeze a lockfile.
3. Real-data baseline/FL numbers still pending a Colab run (no NIH data locally).

---

## 2026-09-02 — Session 3 (Tasks 5 & 6: single-site baselines + core FL loop)

**Current tasks:** §7 Tasks 5 & 6. **Code complete & smoke-verified on synthetic data.**

**Completed this session:**
- **Refactor:** moved model/optimizer construction into `src/experiments/builders.py`
  (`build_model`, `build_optimizer`); centralized.py now imports them (no behaviour change).
- **Task 5 (single-site floor):** `src/experiments/single_site.py` — trains a *separate* fresh
  model per node on that node's local data only, evaluates each on the SHARED test set; writes
  `single_site_results.json` with per-node metrics + mean-AUC-across-nodes. CLI: `single-site`.
- **Task 6 (core FL loop, FedAvg):** installed **flwr 1.36.0 (core, NOT [simulation]/Ray)**.
  - `src/federated/params.py` — torch state_dict <-> Flower NDArrays (dtype/shape-safe).
  - `src/federated/client.py` — `FLClient(NumPyClient)` wrapping the shared engine (fit/evaluate).
  - `src/federated/strategies/__init__.py` — `build_strategy` registry (fedavg now; robust rules
    added in Task 9) + example-weighted `train_loss` metric aggregation.
  - `src/federated/simulation.py` — **in-process orchestrator**: drives rounds, wraps client
    outputs as Flower `FitRes`, calls real `FedAvg.aggregate_fit`, loads aggregated params into a
    global model, evaluates centrally on shared test each round (round 0 baseline + N rounds).
  - `src/experiments/federated.py` + CLI `federated`.
  - Also extended `engine.evaluate` to optionally take `loss_fn` and return avg 'loss'.
- **Tests:** added `tests/test_federated.py` (3-node x 2-round FedAvg on synthetic → PASS).
  All 3 suites pass (data_pipeline, model_training, federated). All 3 CLI subcommands verified:
  `centralized`, `single-site`, `federated` (each runs full chain on synthetic + writes results).

**Key architecture decision (flagged; CLAUDE.md §5 updated to match):**
- Federated simulation is an **in-process orchestrator that delegates aggregation to genuine Flower
  `Strategy` classes** rather than Flower's Ray-backed `Simulation` engine. Rationale: Ray is
  heavy/flaky on Windows; in-process is deterministic, live-streamable to the dashboard, and gives
  the control needed for the DP/Byzantine composition experiments. `flwr` core installed (no Ray).
  → CLAUDE.md §5 "Federated node simulation" bullet updated accordingly.

**Bugs fixed:** minor — Windows-terminal mojibake on em-dashes in CLI help → replaced with hyphens.

**Not done / caveats:** still no real NIH data → no real baseline/FL *numbers* yet (Colab job).
`requirements.txt` lists flwr; **opacus still not installed** (Task 8). Lockfile freeze still pending
until opacus lands. `src/federated/server.py` from the §6 repo layout is realised as `simulation.py`
(the sim entrypoint) — no separate server.py needed for the in-process design.

**Exact next step when work resumes:**
1. Task 7 — heterogeneity mitigation: add FedProx proximal term (µ) to the client local step
   (needs global params at fit time — already broadcast; keep a copy for the prox term) and
   FedBN (exclude BN params/buffers from `params.get/set_parameters`). Config keys already exist
   (`federated.heterogeneity_mitigation.fedprox_mu`, `.fedbn`). Compare vs FedAvg on non-IID.
2. Task 8 — DP layer (install opacus), then Task 9 — robust strategies + attacks.

---

## 2026-09-02 — Session 2 (Tasks 2, 3, 4: data pipeline, model, centralized baseline)

**Current tasks:** §7 Tasks 2–4. **Code complete & smoke-verified on synthetic data.**
The real-data/GPU runs (actual baseline numbers) are deferred to Colab — see caveat below.

**ENV PIVOT (important):** discovered local Pythons 3.10.11, 3.11.4, 3.13.5. User OK'd using
older Python for the federated stack. **Switched the primary venv to Python 3.11** — it runs the
FULL stack (torch, and later flwr/opacus) locally, so we can smoke-test everything without Colab.
Consequently:
- Deleted the 3.13 venv and the `requirements-dev.txt`/`.lock` split (that split was only a 3.13
  workaround). Back to a **single `requirements.txt`** (CLAUDE.md §5 intent). NOTE: `requirements.txt`
  still needs its exact pins refreshed for the 3.11 local install (currently lower-bound style); the
  `flwr`/`opacus` install is deferred to Task 6/8. **TODO: freeze a lockfile once flwr/opacus are in.**
- Installed into venv (3.11): torch 2.13.0+cpu, torchvision 0.28.0+cpu, numpy 2.4.6, pandas 3.0.5,
  scikit-learn 1.9.0, torchmetrics, pillow, opencv-python-headless, pyyaml, typer, tqdm, matplotlib.
  (flwr, opacus, fastapi NOT yet installed — added at their tasks.)

**Completed this session:**
- **Task 2 (data):** `src/data/` — `constants.py` (14 labels, 8 bbox labels, severity classes),
  `labels.py` (multi-label expansion + dominant-label for partitioning + official-split loader),
  `severity.py` (bbox → relative-lesion-area proxy, tercile/fixed binning, documented as PROXY),
  `partition.py` (patient-aware IID + Dirichlet non-IID → JSON manifest), `dataset.py` (torch Dataset
  with dual pathology+severity targets, masked severity), `synthetic.py` (tiny fake dataset for tests).
- **Task 3 (model):** `src/models/` — `backbone.py` (DenseNet-121/ResNet-50), `heads.py`,
  `model.py` (`DualHeadNet`+`ModelConfig`), `losses.py` (`DualHeadLoss`: BCE + masked CE severity),
  `engine.py` (shared `train_one_epoch`/`evaluate`).
- **Task 4 (centralized):** `src/evaluation/metrics.py` (per-class + mean AUC, severity MAE/QWK),
  `src/experiments/` — `config.py`, `utils.py`, `data_setup.py`, `centralized.py`, `run.py` (typer CLI).
- **Tests:** `tests/test_data_pipeline.py` (torch-free) and `tests/test_model_training.py` (end-to-end)
  BOTH PASS. CLI verified: `python -m src.experiments.run centralized --config configs/smoke.yaml
  --data-root <synthetic>` runs the whole chain and writes results JSON. Added `configs/smoke.yaml`.

**Bugs found & fixed this session:**
- numpy 2.x `.to_numpy()` returns a read-only view → `rng.shuffle` failed in Dirichlet partition.
  Fixed with `.copy()` in `partition._assign_dirichlet`.
- Typer single-command app didn't accept the `centralized` subcommand name → added `@app.callback()`.

**Key decisions (also mirrored into study.md §1):**
- Severity proxy: per-image = **max relative box area** across its boxes; default binning **tercile**
  (data-driven 33/67 percentiles). Real thresholds to be logged in study.md on the real run.
- Non-IID partition uses a per-image **dominant label** (first positive finding, else No Finding) as
  the Dirichlet class variable — a partitioning device only; training still uses full multi-label targets.
- Partitioning is **patient-level** (no patient leakage across nodes/test); uses official NIH split
  lists if present under data root, else a patient-wise random `test_frac` split.

**CAVEAT / not done:** No real NIH data downloaded, so Tasks 4/5's actual *baseline numbers* aren't
produced yet — only the code paths are verified on synthetic data. The real centralized run happens on
Colab with the real dataset (that's the intended division of labour). `study.md` §2 has no experiment
entries yet for that reason.

**Exact next step when work resumes:**
1. (User will commit when ready — do NOT git add/commit/push; user handles all git.)
2. Task 5 — single-site baselines: reuse `node_loader` + engine to train one model per node
   (IID and non-IID), add `src/experiments/single_site.py` + a `single-site` CLI subcommand.
3. Task 6 — core FL loop (Flower FedAvg): install `flwr[simulation]` into the 3.11 venv, write
   `src/federated/client.py` (wrap engine) + FedAvg `server.py`; verify 3-node sim on synthetic data.
4. Refresh `requirements.txt` pins + freeze a lockfile after flwr/opacus land.

---

## 2026-09-02 — Session 1 (Task 1: Environment & repo scaffolding) — DONE

**Current task:** §7 Task 1 — Environment & repo scaffolding. **Complete & verified.**

**Completed this session:**
- `git init` in project root (was not a git repo before). No commit made yet (awaiting user go-ahead).
- Full directory structure per CLAUDE.md §6 (`configs/`, `data/{raw,processed,partitions}`,
  `src/...` with all subpackages, `notebooks/`, `backend/app/`, `frontend/`, `results/`).
- `.gitignore` — venvs, raw medical data (keeps `data/raw/README.md`), checkpoints, node_modules, secrets, logs.
- `requirements.txt` — full GPU/Colab stack (torch, flwr, opacus, fastapi, …), lower-bound pins.
- `requirements-dev.txt` — light local stack (no torch/flwr/opacus). **Installed & import-verified.**
- `requirements-dev.lock.txt` — exact `pip freeze` of the working local venv (46 pkgs) for reproducibility.
- All package `__init__.py` files (with docstrings), `configs/example_experiment.yaml` (full config
  schema stub, parse-verified), `notebooks/colab_training.ipynb` stub, `README.md`, data-dir placeholders.
- `venv/` created (Python 3.13.5), light deps installed & all import OK.

**Key decisions made this session:**
- **Split requirements into two files.** Heavy training stack (torch/flwr/opacus) is Colab-bound;
  Opacus/Flower validated on Python 3.10–3.12 while local machine has **Python 3.13.5**. Local dev
  installs the light subset only. Scope-consistent elaboration, not a change to CLAUDE.md decisions.
- **numpy pin gotcha:** `<2.1` (needed for torch compat in full stack) has NO cp313 wheels, so pip
  tried a source build and failed on old MinGW gcc 6.3.0. Fixed in `requirements-dev.txt` → `numpy>=2.1`
  (first release with cp313 wheels). The `<2.1` cap remains in `requirements.txt` (Colab-only, numpy preinstalled there).
- Actual local versions installed: numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, fastapi 0.141.1,
  pydantic 2.13.5, pillow 12.3.0. Env: Python 3.13.5, Node 22.17, npm 10.5, git 2.45.2, Windows 11.
- Also hit a transient DNS drop (`getaddrinfo failed`) mid-install; resolved by retrying with `--retries 10`.

**Broken / in-progress:** nothing. Scaffolding only — no source logic written yet (by design).

**Exact next step when work resumes (Task 2 — data pipeline):**
1. (Optional) first git commit, only when the user asks — do not auto-commit.
2. `src/data/`: NIH ChestX-ray14 multi-label loader from `Data_Entry_2017.csv` (14 pathology labels,
   frontal images), reading from `data/raw/`.
3. Severity-proxy derivation from `BBox_List_2017.csv` (lesion-box area ÷ lung-field-area heuristic;
   decide tercile vs fixed cutoff → document in `study.md`).
4. IID and non-IID (Dirichlet label-skew) partitioners → write manifests to `data/partitions/`.
   Keep node count config-driven (default 3). Note: full dataset (~112k imgs) not downloaded locally;
   build loaders/partitioners to be testable on a small sample or on Colab where data lives.
