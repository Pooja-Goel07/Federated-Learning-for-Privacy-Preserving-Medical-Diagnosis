# setup.md - How to set up and run the project

This guide explains, step by step, how to install the project and run every part of
it: the main model training (locally and on Google Colab), the federated learning
demo, the differential-privacy runs, the attack (Byzantine) simulations, and the
tests you use to check that everything works.

It is written to be followed top to bottom the first time, then used as a reference.
No prior familiarity with the codebase is assumed.

Contents:
1. What the project is (very short)
2. Prerequisites
3. One-time setup (virtual environment + dependencies)
4. Quick check that it works (no dataset needed)
5. The dataset (NIH ChestX-ray14)
6. How runs are structured (one command, many configs)
7. Running each part locally
   - 7.1 Centralized training (the upper-bound baseline)
   - 7.2 Single-site baselines (the floor)
   - 7.3 The federated demo (the core: 3 nodes + FedAvg)
   - 7.4 Heterogeneity mitigation (FedProx / FedBN)
   - 7.5 Differential privacy
   - 7.6 Attack simulation (Byzantine attacks + robust aggregation)
8. Main model training on Google Colab (GPU)
9. The configuration file explained
10. Where results go and how to read them
11. Running the tests / checking progress
12. Troubleshooting
13. What is not built yet

---

## 1. What the project is (very short)

A testbed for federated learning on chest X-rays. Several simulated hospitals train a
shared model without sharing raw images. On top of that, two defences are built in:
differential privacy (adds noise to protect patient data) and Byzantine-robust
aggregation (survives malicious/broken hospitals). The research goal is to see how
those two defences interact. Full context is in `CLAUDE.md`. The running research
notes are in `study.md`, and the session-by-session progress log is in
`progresstracker.md`.

---

## 2. Prerequisites

- Windows (these instructions use PowerShell), macOS, or Linux.
- Python 3.11 installed. Check with:
  ```
  py -3.11 --version
  ```
  (On macOS/Linux use `python3.11 --version`.) Python 3.11 is the supported version -
  the full stack, including Flower and Opacus, installs cleanly on it.
- Git (optional, for version control).
- Node.js (only needed later for the React dashboard, which is not built yet).

You do NOT need a GPU to run anything locally. Local runs use the CPU. The GPU is only
for the large real-data training, which is done on Colab (section 8).

---

## 3. One-time setup (virtual environment + dependencies)

Run these once, from the project root folder.

1. Create and activate a virtual environment with Python 3.11:
   ```
   py -3.11 -m venv venv
   venv\Scripts\Activate.ps1
   ```
   On macOS/Linux:
   ```
   python3.11 -m venv venv
   source venv/bin/activate
   ```
   After activation your prompt shows `(venv)`. Everything below assumes the venv is active.

2. Install PyTorch (CPU build) first. It is not pulled from the normal package index:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. Install the rest of the dependencies:
   ```
   pip install -r requirements.txt
   ```

That's it. To reproduce the exact versions used during development, you can instead use
`requirements.lock.txt` (a frozen snapshot). Note the lock file pins the CPU builds of
torch/torchvision, which still must come from the CPU index in step 2.

If you open a new terminal later, just re-activate the venv (`venv\Scripts\Activate.ps1`)
before running anything.

---

## 4. Quick check that it works (no dataset needed)

You can exercise the entire pipeline on a tiny synthetic (fake) dataset that the code
generates for you. This confirms your install is good before you touch the real data.

1. Generate a small synthetic dataset:
   ```
   python -c "from src.data.synthetic import make_synthetic_dataset as m; m('data/_smoke', write_images=True)"
   ```

2. Run a quick centralized training on it:
   ```
   python -m src.experiments.run centralized --config configs/smoke.yaml --data-root data/_smoke
   ```

3. Run the federated demo on it:
   ```
   python -m src.experiments.run federated --config configs/smoke.yaml --data-root data/_smoke
   ```

If both finish and print results, your setup is working. The `data/_smoke` folder is
temporary and safe to delete. `configs/smoke.yaml` is intentionally tiny (small images,
few batches) so it runs in seconds and never downloads anything.

---

## 5. The dataset (NIH ChestX-ray14)

The real experiments use the NIH ChestX-ray14 dataset (about 112,000 chest X-rays).

Download it from: https://nihcc.app.box.com/v/ChestXray-NIHCC

Place the files under `data/raw/` so the folder looks like this:
```
data/raw/
  Data_Entry_2017.csv        <- image labels (required)
  BBox_List_2017.csv         <- bounding boxes, used to derive severity (required)
  images_001/ ... images_012/   (or a single merged images folder)
  train_val_list.txt         <- official split (optional but recommended)
  test_list.txt              <- official split (optional but recommended)
```

The images can be in the original `images_001 ... images_012` subfolders or in one
merged folder - the loader searches subfolders automatically. The dataset is large
(tens of GB), so it is git-ignored and never committed.

For real runs you point the tools at `data/raw` (which is the default in the config),
so you simply omit the `--data-root` option. Because the full dataset is large, the
heavy real training is normally done on Colab (section 8), not on your laptop CPU.

---

## 6. How runs are structured (one command, many configs)

There is a single command with three sub-commands:

```
python -m src.experiments.run centralized   --config <config.yaml> [--data-root <dir>]
python -m src.experiments.run single-site    --config <config.yaml> [--data-root <dir>]
python -m src.experiments.run federated      --config <config.yaml> [--data-root <dir>]
```

- `centralized` and `single-site` are baselines (the ceiling and the floor).
- `federated` runs the actual federated learning. Everything else - heterogeneity
  mitigation, differential privacy, and attacks - is turned on through the config file
  for the `federated` command. There is no separate "attack" command; you enable
  attacks in the config and run `federated`.

`--config` selects a YAML file describing the run. `--data-root` overrides where the
data lives; use it to point at a synthetic dataset. For real data in `data/raw`, leave
it out.

The recommended workflow is: copy `configs/example_experiment.yaml` to a new file per
experiment (for example `configs/fed_dp_eps3.yaml`), edit the values, and run it. This
keeps every run reproducible. See section 9 for what each field means.

---

## 7. Running each part locally

All commands below assume the venv is active. They use `configs/example_experiment.yaml`
against real data in `data/raw`. To try them without real data, add
`--data-root data/_smoke` and use `configs/smoke.yaml` instead (after doing section 4).

### 7.1 Centralized training (the upper-bound baseline)

Trains one model on all the data pooled together. This is the best case the federated
model is compared against.
```
python -m src.experiments.run centralized --config configs/example_experiment.yaml
```
Output: `results/<experiment.output_dir>/centralized_results.json`.

### 7.2 Single-site baselines (the floor)

Trains a separate model on each hospital's data alone (no collaboration), then evaluates
each on the shared test set. This is the worst case - what a hospital gets without
federation.
```
python -m src.experiments.run single-site --config configs/example_experiment.yaml
```
Output: `single_site_results.json`, with per-node scores.

### 7.3 The federated demo (the core: 3 nodes + FedAvg)

This is the main deliverable: 3 simulated hospitals train locally, send their model
updates to a central node, the central node averages them (FedAvg), and the combined
global model is evaluated each round.
```
python -m src.experiments.run federated --config configs/example_experiment.yaml
```
Output: `federated_results.json`, containing a `history` list with the global model's
score after each round.

Relevant config fields:
```
partition.num_nodes: 3          # number of hospitals
partition.scheme: iid           # or non_iid for uneven data across hospitals
training.rounds: 20             # number of communication rounds
federated.strategy: fedavg      # the aggregation rule
```

### 7.4 Heterogeneity mitigation (FedProx / FedBN)

When hospitals have very different data (non-IID), plain FedAvg struggles. Two options
help. Set them in the config, then run the same `federated` command.

FedProx (keeps local training from drifting too far from the global model):
```
partition.scheme: non_iid
federated.heterogeneity_mitigation.fedprox_mu: 0.1     # 0 disables it; try 0.01 - 1.0
```

FedBN (each hospital keeps its own batch-normalization statistics):
```
partition.scheme: non_iid
federated.heterogeneity_mitigation.fedbn: true
```
Then:
```
python -m src.experiments.run federated --config configs/example_experiment.yaml
```

### 7.5 Differential privacy

Adds per-sample gradient clipping and Gaussian noise during local training, and reports
the cumulative privacy budget (epsilon, delta). Turn it on in the config:
```
privacy.dp.enabled: true
privacy.dp.max_grad_norm: 1.0        # clipping bound
privacy.dp.noise_multiplier: 1.0     # more noise = more privacy = smaller epsilon
privacy.dp.target_delta: 1.0e-5
```
Then run the federated command:
```
python -m src.experiments.run federated --config configs/example_experiment.yaml
```
The results file reports the final worst-case `epsilon` under `privacy.final_epsilon`.
A smaller epsilon means stronger privacy. Increasing `noise_multiplier` lowers epsilon
but usually lowers accuracy too.

Important: differential privacy automatically replaces batch-normalization with group
normalization and is therefore not combined with FedProx or FedBN. If you enable DP and
one of those at the same time, the tool disables FedProx/FedBN and prints a note.

### 7.6 Attack simulation (Byzantine attacks + robust aggregation)

This simulates malicious or broken hospitals and tests whether robust aggregation
survives them. Two things to set in the config: the attack, and the aggregation rule
that defends against it.

Enable an attack:
```
attack.enabled: true
attack.byzantine.fraction: 0.34      # fraction of hospitals that are malicious (0.34 of 3 = 1)
attack.byzantine.type: sign_flip     # label_flip | sign_flip | scaled_update | model_replacement
attack.byzantine.scale: 10.0         # strength for scaled_update / model_replacement
```

Choose a defence (the aggregation rule):
```
federated.strategy: median           # fedavg | krum | multikrum | trimmed_mean | median | trust
federated.robust.beta: 0.2           # trimmed_mean: fraction trimmed from each end
federated.robust.multi_k: 2          # multikrum: how many best updates to average
```

Then run:
```
python -m src.experiments.run federated --config configs/example_experiment.yaml
```

The typical experiment is to run the SAME attack twice - once with
`federated.strategy: fedavg` (no defence) and once with a robust rule such as `median`
or `krum` - and compare the final global accuracy. The robust rule should hold up much
better. The results file records which nodes were malicious under `attack`.

The four attack types:
- `label_flip`: the attacker trains on flipped labels (data poisoning).
- `sign_flip`: the attacker sends the negative of its honest update.
- `scaled_update`: the attacker multiplies its update by a large factor to dominate.
- `model_replacement`: the attacker trains a poisoned model and boosts it to try to
  overwrite the global model.

---

## 8. Main model training on Google Colab (GPU)

The real NIH dataset is large, so the real training runs on Colab with a GPU. The
notebook `notebooks/colab_training.ipynb` is self-contained: it installs everything,
downloads a slice of the dataset, runs the core suite (centralised + single-site +
federated), and produces the charts and report. You do not type any commands by hand.

What you need first:
- A GitHub repo with the latest code pushed (so Colab can clone it). Before cloning,
  commit and push any local changes, including `src/experiments/data_setup.py`.
- A free Kaggle account, to download the dataset (the notebook walks you through the
  API token).
- A Google account (Colab + Drive).

Steps (each is a numbered section in the notebook — run the cells top to bottom):
1. Open `notebooks/colab_training.ipynb` in Google Colab
   (colab.research.google.com -> File -> Open notebook -> GitHub, paste the repo URL).
2. Set the runtime to GPU: Runtime -> Change runtime type -> T4 GPU.
3. Section 1: checks the GPU is on.
4. Section 2: installs Flower, Opacus, etc. (torch is already on Colab).
5. Section 3: clones the repo. If your repo is private, edit the URL to the token form
   shown in the cell.
6. Section 4: upload your `kaggle.json` when prompted (Kaggle -> Settings -> API ->
   Create New Token).
7. Section 5: downloads the two label CSVs from the official dataset plus all images
   from a 224x224 resized mirror (~4 GB, not the full 45 GB). It prints how many on-disk
   images match the label file (should be ~112k); a fallback to the official sample is
   noted in the next cell.
8. Section 6: loads `configs/colab_real.yaml`; change `max_images`, `local_epochs`, or
   `rounds` here to trade runtime for quality.
9. Section 7: runs the full suite (the long cell, ~1-2 h on a T4). It prints per-epoch
   and per-round progress and the final floor / federated / ceiling AUCs.
10. Section 8: renders the four charts and the report inline.
11. Section 9: mounts Drive and copies `results/colab/` there so nothing is lost when
    the runtime disconnects.

To bring the results back into the project: download `results/colab/` from your Drive
and copy it into the repo's `results/` folder, then commit it. The files are the same
shape as the local demo (`summary.json`, `plots/`, `REPORT.md`).

Notes:
- The data pipeline uses only images actually present on disk and then subsamples to
  `data.subsample.max_images` patients (keeping the bounding-box/severity images it
  finds), so the partial download works without any code changes.
- Colab free sessions can disconnect after a while; section 9 (save to Drive) is why
  you will not lose a finished run. For very long runs consider Colab Pro.

---

## 9. The configuration file explained

Every run is fully described by one YAML file. Start from
`configs/example_experiment.yaml` and change what you need. The main sections:

```
experiment:
  seed: 42                     # reproducibility
  output_dir: results/my_run   # where this run's results are written

data:
  root: data/raw               # where the dataset lives
  image_size: 224              # input image size
  severity:
    enabled: true              # also train the severity (mild/moderate/severe) head
    binning: tercile           # how severity levels are derived

partition:
  num_nodes: 3                 # number of simulated hospitals
  scheme: iid                  # iid (even) or non_iid (uneven, more realistic)
  test_frac: 0.2               # held-out test fraction (ignored if official split files exist)
  non_iid:
    alpha: 0.5                 # lower = more uneven data across hospitals

model:
  backbone: densenet121        # densenet121 or resnet50
  pretrained: true             # use ImageNet weights (download; keep false for offline tests)
  heads:
    pathology: true            # 14-disease detection head
    severity: true             # severity grading head

training:
  local_epochs: 1              # local passes per round
  batch_size: 32
  optimizer: adam              # adam or sgd
  lr: 0.0001
  rounds: 20                   # federated communication rounds
  use_gpu: true                # falls back to CPU if no GPU
  max_batches: null            # cap batches per epoch for quick tests (null = use all)

federated:
  strategy: fedavg             # aggregation rule (see 7.6)
  heterogeneity_mitigation:
    fedprox_mu: 0.0            # >0 enables FedProx
    fedbn: false              # true enables FedBN
  robust:
    beta: 0.2                 # trimmed_mean tail fraction
    multi_k: 2                # multikrum count

privacy:
  dp:
    enabled: false            # true enables differential privacy
    max_grad_norm: 1.0
    noise_multiplier: 1.0
    target_delta: 1.0e-5

attack:
  enabled: false              # true enables a Byzantine attack
  byzantine:
    fraction: 0.0
    type: none
    scale: 10.0
```

You only need to change the parts relevant to your experiment; anything you leave out
uses a sensible default.

---

## 10. Where results go and how to read them

Each run writes a JSON file into the folder named by `experiment.output_dir` (default
under `results/`):
- `centralized_results.json` - final test metrics for the pooled model.
- `single_site_results.json` - per-hospital metrics.
- `federated_results.json` - a `history` list (global model after each round), plus the
  final metrics, and the privacy/attack settings that were used.

Key numbers inside these files:
- `pathology.mean_auc` - average AUC across the 14 diseases (higher is better; 0.5 is random).
- `severity.severity_mae` and `severity.severity_qwk` - severity grading quality.
- `privacy.final_epsilon` - the privacy budget spent (only when DP is on; smaller is more private).
- `attack` - which nodes were malicious and the attack type (only when an attack is on).

The results are plain JSON, so you can open them in any editor or load them in Python.

---

## 11. Running the tests / checking progress

The tests run on the synthetic dataset and need no download. From the project root with
the venv active:

```
python tests/test_data_pipeline.py     # data loading, severity proxy, partitioning
python tests/test_model_training.py     # model + centralized training
python tests/test_federated.py          # the 3-node FedAvg loop
python tests/test_heterogeneity.py      # FedProx and FedBN
python tests/test_dp.py                 # differential privacy (Opacus)
python tests/test_byzantine.py          # attacks + robust aggregation
```

Each prints a series of "OK" lines and ends with "ALL ... PASSED". If one fails, it
stops and shows the error. Running all of them is a good way to confirm nothing is
broken after a change.

To see the current state of the project and what is planned next:
- `progresstracker.md` - what has been done, what is in progress, and the exact next
  step (newest entry at the top).
- `study.md` - the research decisions and, eventually, experiment results.
- `CLAUDE.md` - the full plan and locked-in decisions.

---

## 12. Troubleshooting

- "torch not found" or a build error while installing numpy/torch: make sure you used
  Python 3.11 for the venv, and that you installed torch from the CPU index (section 3,
  step 2) before `pip install -r requirements.txt`.

- Slow runs on your laptop: that is expected on CPU, especially with differential
  privacy (per-sample gradients are heavy) or the real dataset. Use small settings for
  local checks: `training.max_batches: 2`, `training.batch_size: 8`,
  `model.pretrained: false`, `data.image_size: 64`. Do the heavy runs on Colab.

- On Windows, keep `training.num_workers: 0` (the default). Higher values can cause data
  loader errors on Windows.

- "DP enabled -> disabling FedBN/FedProx": this is expected. Differential privacy cannot
  be combined with those two; the tool turns them off and continues.

- Want a fast end-to-end check without the real data: use `configs/smoke.yaml` with
  `--data-root data/_smoke` after generating the synthetic data (section 4).

---

## 13. What is not built yet

These parts of the plan (`CLAUDE.md` section 7) are not implemented yet, so there are no
commands for them here:
- Task 10: the combined privacy-plus-robustness sweep runner.
- Task 11: the privacy audit attacks (gradient inversion, membership inference).
- Task 13: the FastAPI backend.
- Task 14: the React dashboard.

This file will be updated as those are added. For the current status at any time, read
`progresstracker.md`.
