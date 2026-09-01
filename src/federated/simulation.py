"""In-process federated orchestrator (Windows/Ray-free).

Drives federated rounds by hand while delegating aggregation to a real Flower
``Strategy`` (FedAvg now; robust rules in Task 9). Each round:

  1. broadcast the current global parameters to every client;
  2. each client trains locally (``client.fit``) and returns updated parameters;
  3. wrap results as Flower ``FitRes`` and call ``strategy.aggregate_fit`` — this is
     the genuine Flower aggregation code path, so swapping strategies is real;
  4. load the aggregated parameters and evaluate on the shared test set.

FedBN note: when clients keep BatchNorm local (``fedbn=True``) the aggregated vector
excludes BN, so a single global model has no valid BN stats. We then evaluate
per-client (aggregated non-BN weights + each client's own BN) and average — the
standard FedBN evaluation — instead of evaluating one global model.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import Strategy

from .client import FLClient
from .params import bn_state_keys, get_parameters, set_parameters


def _average_metrics(metrics_list: list[dict]) -> dict:
    """Average a list of nested metric dicts leaf-wise (ignoring NaNs)."""
    if not metrics_list:
        return {}
    out: dict = {}
    for key in metrics_list[0]:
        vals = [m[key] for m in metrics_list if key in m]
        if isinstance(vals[0], dict):
            out[key] = _average_metrics(vals)
        else:
            nums = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
            out[key] = sum(nums) / len(nums) if nums else float("nan")
    return out


def run_federated_rounds(
    global_model: torch.nn.Module,
    clients: Sequence[FLClient],
    strategy: Strategy,
    num_rounds: int,
    evaluate_model_fn: Callable[[torch.nn.Module], dict],
    fedbn: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[list[dict], list]:
    """Run ``num_rounds`` of federation. Returns (per-round history, final params)."""
    global_exclude = bn_state_keys(global_model) if fedbn else set()
    parameters = get_parameters(global_model, global_exclude)

    def eval_current(params) -> dict:
        set_parameters(global_model, params, global_exclude)
        if not fedbn:
            return evaluate_model_fn(global_model)
        # FedBN: evaluate each client's local model (aggregated non-BN + local BN).
        per_client = []
        for c in clients:
            set_parameters(c.model, params, c.exclude_keys)
            per_client.append(evaluate_model_fn(c.model))
        return _average_metrics(per_client)

    # Round 0: global model before any training (a useful baseline point).
    history: list[dict] = [{"round": 0, "test_metrics": evaluate_model_fn(global_model), "clients": []}]

    for rnd in range(1, num_rounds + 1):
        fit_results = []
        client_metrics = []
        for client in clients:
            new_params, num_examples, metrics = client.fit(parameters, {})
            fit_results.append(
                (
                    None,  # ClientProxy is unused by aggregate_fit
                    FitRes(
                        status=Status(Code.OK, "ok"),
                        parameters=ndarrays_to_parameters(new_params),
                        num_examples=num_examples,
                        metrics=metrics,
                    ),
                )
            )
            client_metrics.append({"num_examples": num_examples, **metrics})

        aggregated, agg_metrics = strategy.aggregate_fit(rnd, fit_results, [])
        if aggregated is None:
            raise RuntimeError(f"round {rnd}: strategy returned no aggregated parameters")
        parameters = parameters_to_ndarrays(aggregated)

        test_metrics = eval_current(parameters)
        auc = test_metrics.get("pathology", {}).get("mean_auc", float("nan"))
        log(f"[federated] round {rnd}/{num_rounds}: global mean_auc={auc:.4f}")
        history.append(
            {
                "round": rnd,
                "aggregate_metrics": dict(agg_metrics),
                "clients": client_metrics,
                "test_metrics": test_metrics,
            }
        )

    return history, parameters
