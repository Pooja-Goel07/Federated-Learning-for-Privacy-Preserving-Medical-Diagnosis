"""Byzantine-robust aggregation rules as Flower Strategy subclasses (Task 9).

Each rule subclasses Flower's ``FedAvg`` and overrides ``aggregate_fit`` to combine
client updates robustly instead of by plain weighted mean, so a bounded fraction of
adversarial updates cannot dominate the global model. Implemented:

  * Krum / Multi-Krum   — pick the update(s) closest to their neighbours (Blanchard 2017)
  * Coordinate-wise TrimmedMean — drop the beta-fraction most extreme values per coordinate
  * Coordinate-wise Median      — elementwise median across clients
  * TrustWeighted        — weight updates by cosine similarity to the coordinate-wise
                           median direction (down-weights outliers)

All operate on the list of client parameter vectors (Flower NDArrays); they are
drop-in replacements for FedAvg selected by name from the config.
"""

from __future__ import annotations

import numpy as np
from flwr.common import FitRes, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def _flatten(update: NDArrays) -> np.ndarray:
    return np.concatenate([a.ravel() for a in update])


def _unflatten_like(flat: np.ndarray, template: NDArrays) -> NDArrays:
    out, i = [], 0
    for a in template:
        n = a.size
        out.append(flat[i : i + n].reshape(a.shape))
        i += n
    return out


def _weighted_average(updates: list[NDArrays], weights: list[float]) -> NDArrays:
    total = float(sum(weights)) or 1.0
    return [
        sum(w * u[layer] for u, w in zip(updates, weights)) / total
        for layer in range(len(updates[0]))
    ]


def _krum_scores(updates: list[NDArrays], num_malicious: int) -> np.ndarray:
    """Krum score per client: sum of squared distances to its closest neighbours."""
    flats = [_flatten(u) for u in updates]
    n = len(flats)
    # number of neighbours to sum over (excluding self): n - f - 2
    k = max(1, n - num_malicious - 2)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum((flats[i] - flats[j]) ** 2))
            dist[i, j] = dist[j, i] = d
    scores = np.zeros(n)
    for i in range(n):
        closest = np.sort(dist[i])[1 : k + 1]  # skip self (distance 0)
        scores[i] = float(np.sum(closest))
    return scores


class _RobustFedAvg(FedAvg):
    """Base: parse results into (updates, weights) then delegate to ``_aggregate``."""

    def _aggregate(self, updates: list[NDArrays], weights: list[float]) -> NDArrays:  # noqa: D401
        raise NotImplementedError

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}
        updates = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        weights = [float(fr.num_examples) for _, fr in results]
        aggregated = self._aggregate(updates, weights)
        metrics: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics = self.fit_metrics_aggregation_fn(
                [(fr.num_examples, fr.metrics) for _, fr in results]
            )
        return ndarrays_to_parameters(aggregated), metrics


class Krum(_RobustFedAvg):
    def __init__(self, num_malicious: int = 1, multi_k: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_malicious = num_malicious
        self.multi_k = multi_k  # >1 -> Multi-Krum (average the k best)

    def _aggregate(self, updates: list[NDArrays], weights: list[float]) -> NDArrays:
        scores = _krum_scores(updates, self.num_malicious)
        order = np.argsort(scores)
        k = min(self.multi_k, len(updates))
        chosen = order[:k]
        if k == 1:
            return updates[int(chosen[0])]
        sel_updates = [updates[i] for i in chosen]
        sel_weights = [weights[i] for i in chosen]
        return _weighted_average(sel_updates, sel_weights)


class TrimmedMean(_RobustFedAvg):
    def __init__(self, beta: float = 0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta  # fraction trimmed from EACH tail per coordinate

    def _aggregate(self, updates: list[NDArrays], weights: list[float]) -> NDArrays:
        n = len(updates)
        k = int(np.floor(self.beta * n))
        out = []
        for layer in range(len(updates[0])):
            stack = np.stack([u[layer] for u in updates], axis=0)  # (n, *shape)
            stack.sort(axis=0)
            trimmed = stack[k : n - k] if n - 2 * k > 0 else stack
            out.append(trimmed.mean(axis=0))
        return out


class Median(_RobustFedAvg):
    def _aggregate(self, updates: list[NDArrays], weights: list[float]) -> NDArrays:
        return [
            np.median(np.stack([u[layer] for u in updates], axis=0), axis=0)
            for layer in range(len(updates[0]))
        ]


class TrustWeighted(_RobustFedAvg):
    """Weight each update by max(0, cosine-sim to the coordinate-wise median direction)."""

    def _aggregate(self, updates: list[NDArrays], weights: list[float]) -> NDArrays:
        flats = [_flatten(u) for u in updates]
        ref = np.median(np.stack(flats, axis=0), axis=0)
        ref_norm = np.linalg.norm(ref) or 1.0
        trust = []
        for f, w in zip(flats, weights):
            cos = float(np.dot(f, ref) / ((np.linalg.norm(f) or 1.0) * ref_norm))
            trust.append(max(0.0, cos) * w)
        if sum(trust) == 0:  # degenerate -> fall back to example-weighted mean
            trust = weights
        return _weighted_average(updates, trust)
