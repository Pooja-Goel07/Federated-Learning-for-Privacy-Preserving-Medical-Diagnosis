"""Combined training objective for the dual-head model.

  * Pathology: binary cross-entropy over 14 independent labels (multi-label),
    with optional per-class ``pos_weight`` to counter ChestX-ray14's heavy class
    imbalance.
  * Severity : cross-entropy over the 3 ordinal classes, masking out images that
    have no severity proxy (target == SEVERITY_IGNORE_INDEX). If a batch contains
    no severity-labelled samples, the severity term is exactly zero.

Total loss = pathology_loss + ``severity_weight`` * severity_loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.constants import SEVERITY_IGNORE_INDEX


@dataclass
class LossOutput:
    total: torch.Tensor
    pathology: torch.Tensor
    severity: torch.Tensor
    num_severity: int  # how many severity-labelled samples contributed this batch


class DualHeadLoss(nn.Module):
    def __init__(
        self,
        severity_weight: float = 1.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.severity_weight = severity_weight
        # pos_weight registered as buffer so it moves with .to(device).
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> LossOutput:
        device = next(iter(outputs.values())).device
        zero = torch.zeros((), device=device)

        pathology_loss = zero
        if "pathology" in outputs:
            pathology_loss = F.binary_cross_entropy_with_logits(
                outputs["pathology"], targets["pathology"].float(), pos_weight=self.pos_weight
            )

        severity_loss = zero
        num_sev = 0
        if "severity" in outputs and "severity" in targets:
            sev_target = targets["severity"]
            num_sev = int((sev_target != SEVERITY_IGNORE_INDEX).sum().item())
            if num_sev > 0:
                # ignore_index (rather than boolean indexing) keeps the batch dim
                # intact so per-sample-gradient DP (Opacus) works; numerically this
                # is still the mean CE over the labelled samples only.
                severity_loss = F.cross_entropy(
                    outputs["severity"], sev_target, ignore_index=SEVERITY_IGNORE_INDEX
                )
            else:
                # No severity labels in this batch: keep the severity head in the
                # autograd graph with a zero-valued term so per-sample-gradient DP
                # still gets a (zero) grad_sample for its parameters.
                severity_loss = 0.0 * outputs["severity"].sum()

        total = pathology_loss + self.severity_weight * severity_loss
        return LossOutput(
            total=total,
            pathology=pathology_loss.detach(),
            severity=severity_loss.detach() if num_sev > 0 else zero,
            num_severity=num_sev,
        )
