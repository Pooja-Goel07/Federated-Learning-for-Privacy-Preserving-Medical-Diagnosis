"""Flower client/server and federated orchestration.

    get_parameters, set_parameters        -- params.py
    FLClient                              -- client.py
    run_federated_rounds                  -- simulation.py
    build_strategy, available_strategies  -- strategies/
"""

from __future__ import annotations

from .client import FLClient
from .params import get_parameters, set_parameters
from .simulation import run_federated_rounds
from .strategies import available_strategies, build_strategy

__all__ = [
    "get_parameters",
    "set_parameters",
    "FLClient",
    "run_federated_rounds",
    "build_strategy",
    "available_strategies",
]
