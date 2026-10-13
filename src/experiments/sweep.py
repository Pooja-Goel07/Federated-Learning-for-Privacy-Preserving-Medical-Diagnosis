"""Defence-layer sweep runner (Tasks 8, 9, 10).

Runs a list of federated configurations that differ only in the defence settings
(DP on/off + noise, aggregation rule, attack) and collects their final metrics into
one ``sweep_results.json`` for plotting. Three standard experiment groups are provided
by ``default_defence_variants``:

  * group "dp"          — utility cost of differential privacy (no-DP vs a few sigmas).
  * group "robust"      — aggregation rules under a poisoning attack (FedAvg vs robust).
  * group "composition" — DP x robust together under attack (the G1 contribution:
                          does DP noise degrade the robust aggregator's defence?).

Each run reuses the real federated pipeline (``run_federated``), so the numbers come
from genuine runs; only the config varies.
"""

from __future__ import annotations

import copy
import json
import math
import traceback
from pathlib import Path

from .config import cfg_get
from .federated import run_federated
from .utils import save_json


def _set(cfg: dict, dotted: str, value) -> None:
    """Set a nested config value by dotted key, creating dicts as needed."""
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _summarize(name: str, group: str, res: dict) -> dict:
    m = res.get("final_test_metrics", {})
    return {
        "name": name,
        "group": group,
        "strategy": res.get("strategy"),
        "dp_enabled": res.get("privacy", {}).get("dp_enabled", False),
        "noise_multiplier": res.get("privacy", {}).get("noise_multiplier"),
        "epsilon": res.get("privacy", {}).get("final_epsilon"),
        "attack": res.get("attack", {}),
        "mean_auc": m.get("pathology", {}).get("mean_auc", float("nan")),
        "mean_accuracy": m.get("pathology", {}).get("mean_accuracy", float("nan")),
        "severity": m.get("severity", {}),
        "history": [
            {"round": h["round"],
             "mean_auc": h["test_metrics"].get("pathology", {}).get("mean_auc", float("nan"))}
            for h in res.get("history", [])
        ],
    }


def default_defence_variants(
    attack_type: str = "model_replacement",
    attack_fraction: float = 0.34,     # ~2 of 6 nodes
    dp_noise: tuple[float, ...] = (1.0, 2.0),
    robust_rule: str = "median",       # headline robust rule for the composition study
) -> list[dict]:
    """Build the standard DP / robustness / composition experiment matrix.

    Returns a list of {name, group, overrides} dicts. The clean FedAvg run is shared
    as the reference point for all three groups.
    """
    def atk(on: bool) -> dict:
        return {
            "attack.enabled": on,
            "attack.byzantine.type": attack_type if on else "none",
            "attack.byzantine.fraction": attack_fraction if on else 0.0,
        }

    def dp(on: bool, sigma: float = 1.0) -> dict:
        return {"privacy.dp.enabled": on, "privacy.dp.noise_multiplier": sigma}

    variants: list[dict] = []

    # Reference: clean FedAvg, no attack, no DP (the ceiling for the defence plots).
    variants.append({"name": "clean_fedavg", "group": "dp",
                     "overrides": {"federated.strategy": "fedavg", **atk(False), **dp(False)}})

    # Group DP: FedAvg, no attack, increasing noise -> utility cost of privacy.
    for s in dp_noise:
        variants.append({"name": f"dp_sigma{s}", "group": "dp",
                         "overrides": {"federated.strategy": "fedavg", **atk(False), **dp(True, s)}})

    # Group robust: under attack, FedAvg vs robust aggregation rules (no DP).
    variants.append({"name": "atk_fedavg", "group": "robust",
                     "overrides": {"federated.strategy": "fedavg", **atk(True), **dp(False)}})
    for rule in ("krum", "trimmed_mean", "median"):
        variants.append({"name": f"atk_{rule}", "group": "robust",
                         "overrides": {"federated.strategy": rule, **atk(True), **dp(False)}})

    # Group composition: under attack, {FedAvg, robust} x {no-DP, DP}. FedAvg-noDP and
    # robust-noDP are already above (shared), so add the two DP-on cells here.
    variants.append({"name": f"atk_fedavg_dp", "group": "composition",
                     "overrides": {"federated.strategy": "fedavg", **atk(True), **dp(True, dp_noise[0])}})
    variants.append({"name": f"atk_{robust_rule}_dp", "group": "composition",
                     "overrides": {"federated.strategy": robust_rule, **atk(True), **dp(True, dp_noise[0])}})

    return variants


def _load_completed(out_dir: Path, name: str) -> dict | None:
    """Return a saved run's results dict if the variant already finished, else None."""
    path = out_dir / name / "federated_results.json"
    if not path.exists():
        return None
    try:
        res = json.loads(path.read_text())
    except Exception:
        return None
    auc = res.get("final_test_metrics", {}).get("pathology", {}).get("mean_auc")
    if auc is None or (isinstance(auc, float) and math.isnan(auc)):
        return None  # incomplete/invalid -> re-run
    return res


def run_sweep(
    base_config: dict,
    variants: list[dict],
    out_dir: str | Path,
    data_root: str | Path | None = None,
    resume: bool = True,
) -> dict:
    """Run every variant via run_federated; collect results into sweep_results.json.

    With ``resume=True`` (default), a variant whose ``federated_results.json`` already
    exists and is valid is loaded instead of re-run — so re-running the sweep after a
    crash only redoes the failed/missing variant.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    for i, v in enumerate(variants, 1):
        cfg = copy.deepcopy(base_config)
        for key, val in v["overrides"].items():
            _set(cfg, key, val)
        _set(cfg, "experiment.output_dir", str(out_dir / v["name"]))
        done = _load_completed(out_dir, v["name"]) if resume else None
        if done is not None:
            print(f"[sweep] ({i}/{len(variants)}) {v['name']}: reusing saved result "
                  f"(auc={done['final_test_metrics']['pathology']['mean_auc']:.3f})")
            runs.append(_summarize(v["name"], v["group"], done))
            save_json({"runs": runs}, out_dir / "sweep_results.json")
            continue
        print(f"\n{'='*70}\n[sweep] ({i}/{len(variants)}) {v['name']}  [group={v['group']}]  "
              f"strategy={cfg_get(cfg,'federated.strategy')} "
              f"attack={cfg_get(cfg,'attack.enabled')} dp={cfg_get(cfg,'privacy.dp.enabled')}\n{'='*70}")
        try:
            res = run_federated(cfg, data_root=data_root)
            runs.append(_summarize(v["name"], v["group"], res))
        except Exception:
            print(f"[sweep] RUN FAILED: {v['name']}\n{traceback.format_exc()}")
            runs.append({"name": v["name"], "group": v["group"], "error": traceback.format_exc()})
        save_json({"runs": runs}, out_dir / "sweep_results.json")  # checkpoint after each
    print(f"\n[sweep] wrote {out_dir / 'sweep_results.json'} ({len(runs)} runs)")
    return {"runs": runs}
