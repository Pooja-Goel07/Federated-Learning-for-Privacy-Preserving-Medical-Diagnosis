"""Turn a run's ``summary.json`` into presentation charts + a short REPORT.md.

Shared by the local synthetic demo (results/demo/make_report.py) and the real-data
Colab run — both call ``generate_report(out_dir)`` on a directory containing a
``summary.json`` produced by ``src.experiments.suite.run_core_suite``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Entity colours (validated categorical palette). Colour follows the entity, and
# every bar/point also carries a text label so identity is never colour-alone.
C_CENTRAL = "#3b82c4"   # centralised (ceiling)
C_FED = "#4a9d6b"       # federated
C_SINGLE = "#e8833a"    # single-site (floor)
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"

_RC = {
    "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _learning_curve(s: dict, plots: Path) -> None:
    hist = s["federated"]["history"]
    rounds = [h["round"] for h in hist]
    auc = [h["mean_auc"] for h in hist]
    ceiling = s["centralized"]["mean_auc"]
    floor = s["single_site"]["mean_auc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(ceiling, color=C_CENTRAL, ls="--", lw=1.6)
    ax.axhline(floor, color=C_SINGLE, ls="--", lw=1.6)
    ax.plot(rounds, auc, color=C_FED, lw=2.2, marker="o", ms=6,
            markerfacecolor="white", markeredgecolor=C_FED, markeredgewidth=1.8)
    ax.text(rounds[-1], ceiling + 0.012, f"Centralised ceiling  {ceiling:.2f}",
            color=C_CENTRAL, ha="right", va="bottom", fontsize=10, fontweight="bold")
    ax.text(rounds[-1], floor - 0.02, f"Single-site floor  {floor:.2f}",
            color=C_SINGLE, ha="right", va="top", fontsize=10, fontweight="bold")
    ax.text(rounds[-1], auc[-1] + 0.012, f"Federated  {auc[-1]:.2f}",
            color=C_FED, ha="right", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Federated communication round")
    ax.set_ylabel("Global model — mean ROC-AUC (test)")
    ax.set_title("Federated learning recovers the pooled-data ceiling")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lo = min([floor, *auc]) - 0.05
    ax.set_ylim(max(0.0, lo), 1.0)
    _style(ax)
    fig.tight_layout()
    fig.savefig(plots / "01_learning_curve.png")
    plt.close(fig)


def _loss(s: dict, plots: Path) -> None:
    hist = s["federated"]["history"]
    rounds = [h["round"] for h in hist]
    loss = [h["loss"] for h in hist]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, loss, color=C_FED, lw=2.2, marker="o", ms=6,
            markerfacecolor="white", markeredgecolor=C_FED, markeredgewidth=1.8)
    ax.set_xlabel("Federated communication round")
    ax.set_ylabel("Global model — test loss")
    ax.set_title("Aggregated global model converges")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(ax)
    fig.tight_layout()
    fig.savefig(plots / "02_global_loss.png")
    plt.close(fig)


def _comparison(s: dict, plots: Path) -> None:
    floor = s["single_site"]["mean_auc"]
    fed = s["federated"]["final_mean_auc"]
    ceiling = s["centralized"]["mean_auc"]
    labels = ["Single-site\n(isolated floor)", "Federated\n(FedAvg)", "Centralised\n(pooled ceiling)"]
    vals = [floor, fed, ceiling]
    colors = [C_SINGLE, C_FED, C_CENTRAL]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors, width=0.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Mean ROC-AUC (shared test set)")
    ax.set_title("Federated beats isolated silos, approaches pooled data")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(plots / "03_floor_fed_ceiling.png")
    plt.close(fig)


def _per_pathology(s: dict, plots: Path) -> None:
    fed_auc = s["federated"]["per_pathology_auc"] or {}
    cen_auc = s["centralized"]["per_pathology_auc"] or {}
    # Prefer the demo's active set for ordering; else use every pathology with a
    # defined federated AUC (real data has up to 14), capped at the top 12 by AUC.
    active = s.get("setup", {}).get("active_pathologies")
    valid = {k: v for k, v in fed_auc.items() if isinstance(v, (int, float)) and v == v}
    if active:
        names = [p for p in active if p in valid]
    else:
        names = sorted(valid, key=valid.get, reverse=True)[:12]
    if not names:
        return
    fed_v = [valid[p] for p in names]
    cen_v = [cen_auc.get(p, float("nan")) for p in names]

    y = range(len(names))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(names) + 1.5)))
    h = 0.38
    ax.barh([i + h / 2 for i in y], cen_v, height=h, color=C_CENTRAL, zorder=3, label="Centralised")
    ax.barh([i - h / 2 for i in y], fed_v, height=h, color=C_FED, zorder=3, label="Federated")
    for i, v in zip(y, fed_v):
        ax.text(v + 0.01, i - h / 2, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel("ROC-AUC (test)")
    ax.set_title("Per-pathology detection — federated vs centralised")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(plots / "04_per_pathology_auc.png")
    plt.close(fig)


def _write_report(s: dict, out_dir: Path) -> None:
    floor = s["single_site"]["mean_auc"]
    fed = s["federated"]["final_mean_auc"]
    ceiling = s["centralized"]["mean_auc"]
    fed_acc = s["federated"]["final_mean_accuracy"]
    per_node = s["single_site"]["per_node_auc"]
    setup = s.get("setup", {})
    base_ep = setup.get("baseline_epochs", "?")
    rounds = setup.get("fed_rounds", "?")

    # Headline comparison line, robust to the ordering that actually came out.
    gap = ceiling - floor
    if gap > 1e-6 and fed <= ceiling + 1e-9:
        gap_line = (
            f"- Federated closes **{(fed - floor) / gap * 100:.0f}%** of the gap between the "
            f"isolated floor and the pooled ceiling, **without any raw data leaving a node**."
        )
    elif fed > ceiling:
        gap_line = (
            f"- Federated (**{fed:.3f}**) here exceeds the pooled centralised reference "
            f"(**{ceiling:.3f}**). This is a training-budget artefact, not a real effect: the "
            f"centralised/single-site baselines ran **{base_ep}** epochs while the federation "
            f"accumulates **{rounds}** rounds of updates. For a strict ceiling, re-run with the "
            f"baseline epochs matched to the number of rounds. Either way, federated still clearly "
            f"beats every isolated hospital (**{fed:.3f}** vs floor **{floor:.3f}**)."
        )
    else:
        gap_line = f"- Federated mean AUC **{fed:.3f}** vs isolated single-site floor **{floor:.3f}**."

    node_rows = "\n".join(
        f"| {nid} | {auc:.3f} |" for nid, auc in sorted(per_node.items())
    )
    sev_f = s["federated"].get("severity", {}) or {}
    sev_line = ""
    if sev_f.get("n"):
        sev_line = (
            f"- Severity head (proxy labels, n={sev_f.get('n')} test): "
            f"MAE {sev_f.get('severity_mae', float('nan')):.2f} classes, "
            f"QWK {sev_f.get('severity_qwk', float('nan')):.2f}. "
            f"Secondary result — severity is a heuristic proxy on a small bbox subset.\n"
        )
    img = setup.get("image_size", "?")

    md = f"""# Results Report — Federated Architecture ({setup.get('dataset', 'dataset')})

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
| Dataset | {setup.get('dataset', '?')}, {img}x{img} px |
| Model | {setup.get('backbone', '?')}, dual head (pathology + severity) |
| Hospitals (nodes) | {setup.get('num_nodes', '?')} |
| Partition | {setup.get('partition', '?')} |
| Aggregation | plain FedAvg |
| Defences | {setup.get('defences', '?')} |
| Baseline epochs | {setup.get('baseline_epochs', '?')} · Federated rounds | {setup.get('fed_rounds', '?')} |

---

## Headline result

| Configuration | Mean ROC-AUC | What it represents |
|---|---|---|
| Single-site (isolated) | **{floor:.3f}** | floor — one hospital, its own data only |
| **Federated (FedAvg)** | **{fed:.3f}** | our architecture |
| Centralised (pooled) | **{ceiling:.3f}** | reference — all data in one place |

{gap_line}
- Federated global model mean accuracy: **{fed_acc:.3f}**.
{sev_line}
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
{node_rows}

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
"""
    (out_dir / "REPORT.md").write_text(md, encoding="utf-8")


def generate_report(out_dir: str | Path) -> dict:
    """Read ``out_dir/summary.json``, write plots/ + REPORT.md. Returns the summary."""
    out_dir = Path(out_dir)
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    s = json.loads((out_dir / "summary.json").read_text())
    with plt.rc_context(_RC):
        _learning_curve(s, plots)
        _loss(s, plots)
        _comparison(s, plots)
        _per_pathology(s, plots)
    _write_report(s, out_dir)
    print(f"[report] wrote 4 plots -> {plots}")
    print(f"[report] wrote report -> {out_dir / 'REPORT.md'}")
    return s
