"""CLI entrypoint for config-driven experiment runs.

Usage:
    python -m src.experiments.run centralized --config configs/example_experiment.yaml
    python -m src.experiments.run centralized --config <cfg> --data-root <dir>   # e.g. synthetic

More run types (single-site, federated, sweeps) are added in later tasks.
"""

from __future__ import annotations

from typing import Optional

import typer

from .centralized import run_centralized
from .config import load_config
from .federated import run_federated
from .single_site import run_single_site

app = typer.Typer(add_completion=False, help="Federated medical-diagnosis testbed runner.")


@app.callback()
def _main() -> None:
    """Federated medical-diagnosis testbed runner (use a subcommand, e.g. `centralized`)."""


@app.command()
def centralized(
    config: str = typer.Option(..., "--config", "-c", help="Path to a YAML experiment config."),
    data_root: Optional[str] = typer.Option(
        None, "--data-root", help="Override data.root (e.g. a synthetic dataset dir)."
    ),
) -> None:
    """Train the centralised upper-bound baseline (Task 4)."""
    cfg = load_config(config)
    run_centralized(cfg, data_root=data_root)


@app.command("single-site")
def single_site(
    config: str = typer.Option(..., "--config", "-c", help="Path to a YAML experiment config."),
    data_root: Optional[str] = typer.Option(
        None, "--data-root", help="Override data.root (e.g. a synthetic dataset dir)."
    ),
) -> None:
    """Train isolated per-node baselines - the utility floor (Task 5)."""
    cfg = load_config(config)
    run_single_site(cfg, data_root=data_root)


@app.command()
def federated(
    config: str = typer.Option(..., "--config", "-c", help="Path to a YAML experiment config."),
    data_root: Optional[str] = typer.Option(
        None, "--data-root", help="Override data.root (e.g. a synthetic dataset dir)."
    ),
) -> None:
    """Run the core federated learning loop - 3 nodes + FedAvg aggregation (Task 6)."""
    cfg = load_config(config)
    run_federated(cfg, data_root=data_root)


if __name__ == "__main__":
    app()
