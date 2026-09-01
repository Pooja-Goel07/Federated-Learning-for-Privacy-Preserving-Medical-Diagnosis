"""Aggregation rules as Flower Strategy subclasses.

Selected by name from the experiment config so the aggregation rule swaps
independently of everything else:

    fedavg        -- plain federated averaging (baseline)
    krum          -- Krum (single best update)
    multikrum     -- Multi-Krum (average of the k best updates)
    trimmed_mean  -- coordinate-wise trimmed mean
    median        -- coordinate-wise median
    trust         -- similarity-based trust weighting

The robust rules (Task 9) live in ``robust.py``; the plain FedAvg baseline (Task 6)
is Flower's own ``FedAvg``.
"""

from __future__ import annotations

from typing import Sequence

from flwr.common import Scalar
from flwr.server.strategy import FedAvg, Strategy


def weighted_train_loss(metrics: Sequence[tuple[int, dict]]) -> dict[str, Scalar]:
    """Aggregate client fit metrics (example-weighted mean of train_loss)."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    loss = sum(n * float(m.get("train_loss", 0.0)) for n, m in metrics) / total
    return {"train_loss": loss}


# Imported after weighted_train_loss is defined (robust.py imports it from here).
from .robust import Krum, Median, TrimmedMean, TrustWeighted  # noqa: E402


def _fedavg_kwargs(kw: dict) -> dict:
    return dict(
        fraction_fit=kw.get("fraction_fit", 1.0),
        fraction_evaluate=0.0,
        min_fit_clients=kw.get("min_fit_clients", 1),
        min_evaluate_clients=1,  # we don't use Flower's federated-eval path
        min_available_clients=kw.get("min_available_clients", 1),
        accept_failures=True,
        fit_metrics_aggregation_fn=weighted_train_loss,
    )


_REGISTRY = {
    "fedavg": lambda **kw: FedAvg(**_fedavg_kwargs(kw)),
    "krum": lambda **kw: Krum(
        num_malicious=kw.get("num_malicious", 1), multi_k=1, **_fedavg_kwargs(kw)
    ),
    "multikrum": lambda **kw: Krum(
        num_malicious=kw.get("num_malicious", 1),
        multi_k=kw.get("multi_k", 2),
        **_fedavg_kwargs(kw),
    ),
    "trimmed_mean": lambda **kw: TrimmedMean(beta=kw.get("beta", 0.2), **_fedavg_kwargs(kw)),
    "median": lambda **kw: Median(**_fedavg_kwargs(kw)),
    "trust": lambda **kw: TrustWeighted(**_fedavg_kwargs(kw)),
}


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


def build_strategy(name: str, **kwargs) -> Strategy:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown aggregation strategy {name!r}; available: {available_strategies()}"
        )
    return _REGISTRY[key](**kwargs)
