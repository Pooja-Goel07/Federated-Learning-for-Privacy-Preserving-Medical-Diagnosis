"""Config loading for the config-driven testbed.

Every experiment is fully specified by a YAML file (see configs/example_experiment.yaml).
This module loads it into a plain dict and offers a tiny dotted-key accessor so
downstream code can read nested values with defaults without deep .get chains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment config into a dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    config.setdefault("_source", str(path))
    return config


def cfg_get(config: dict, dotted_key: str, default: Any = None) -> Any:
    """Read a nested value by dotted path, e.g. cfg_get(c, 'privacy.dp.enabled', False)."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
