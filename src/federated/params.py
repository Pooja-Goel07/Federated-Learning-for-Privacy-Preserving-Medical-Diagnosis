"""Convert between a torch model's state and Flower's parameter format (NDArrays).

Flower moves models around as an ordered list of numpy arrays. We use the full
``state_dict`` (weights AND buffers, e.g. BatchNorm running stats) so aggregation
covers everything by default; FedBN (Task 7) will later exclude BN layers here.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm


def bn_state_keys(model: torch.nn.Module) -> set[str]:
    """state_dict keys belonging to BatchNorm layers (weight, bias, running stats).

    Used by FedBN (Task 7): these keys are kept LOCAL to each client and excluded
    from federated aggregation.
    """
    keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            for pname, _ in module.named_parameters(recurse=False):
                keys.add(prefix + pname)
            for bname, _ in module.named_buffers(recurse=False):
                keys.add(prefix + bname)
    return keys


def _included_keys(model: torch.nn.Module, exclude_keys: set[str] | None) -> list[str]:
    exclude = exclude_keys or set()
    return [k for k in model.state_dict().keys() if k not in exclude]


def get_parameters(
    model: torch.nn.Module, exclude_keys: set[str] | None = None
) -> list[np.ndarray]:
    """Model state -> list of numpy arrays, in state_dict order.

    ``exclude_keys`` (e.g. BN keys under FedBN) are skipped so they are neither sent
    nor aggregated.
    """
    state = model.state_dict()
    return [state[k].detach().cpu().numpy() for k in _included_keys(model, exclude_keys)]


def set_parameters(
    model: torch.nn.Module, parameters: list[np.ndarray], exclude_keys: set[str] | None = None
) -> None:
    """Load numpy arrays back into the model, skipping ``exclude_keys``.

    Casts each array to the original tensor's dtype/device/shape so that, e.g.,
    averaged integer buffers (num_batches_tracked) load cleanly. Excluded keys keep
    their current (local) values.
    """
    keys = _included_keys(model, exclude_keys)
    if len(parameters) != len(keys):
        raise ValueError(
            f"parameter count mismatch: got {len(parameters)}, model expects {len(keys)}"
        )
    state = model.state_dict()
    new_state = OrderedDict(state)  # start from current (keeps excluded keys as-is)
    for key, new in zip(keys, parameters):
        old = state[key]
        new_state[key] = torch.as_tensor(new).to(dtype=old.dtype, device=old.device).reshape(old.shape)
    model.load_state_dict(new_state, strict=True)
