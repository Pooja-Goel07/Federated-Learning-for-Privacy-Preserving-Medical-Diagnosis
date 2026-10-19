"""Gradient-inversion audit (Task 11, Objective 7, gap G3).

"Deep Leakage from Gradients" (Zhu et al., 2019): an attacker who intercepts the
gradient a client computed for a sample can reconstruct that sample's image by
optimizing a random dummy image until its gradient matches the intercepted one. We run
this in a *controlled* setting — a small CNN on low-resolution images, with the true
label known (the iDLG simplification) — because full DenseNet-121 inversion at 224px
rarely converges even without any defence, which would obscure the comparison we care
about: reconstruction quality *without* DP versus *with* DP (per-sample clipping +
Gaussian noise applied to the intercepted gradient). DP should sharply degrade the
reconstruction, which is the privacy claim we are auditing.

The controlled setting is stated explicitly as such; it mirrors standard DLG practice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallNet(nn.Module):
    """Tiny CNN used only for the controlled gradient-inversion demo."""

    def __init__(self, in_ch: int = 3, size: int = 32, num_labels: int = 14) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 12, 5, padding=2)
        self.c2 = nn.Conv2d(12, 12, 5, padding=2)
        self.fc = nn.Linear(12 * size * size, num_labels)

    def forward(self, x):
        x = F.sigmoid(self.c1(x))          # smooth activations help gradient matching
        x = F.sigmoid(self.c2(x))
        return self.fc(x.flatten(1))


def _param_grads(model, x, y):
    out = model(x)
    loss = F.binary_cross_entropy_with_logits(out, y)
    return torch.autograd.grad(loss, model.parameters(), create_graph=True)


def _apply_dp(grads, clip: float, noise: float):
    """DP-style transform of an intercepted gradient: clip global L2 norm, add noise."""
    flat = torch.cat([g.reshape(-1) for g in grads])
    norm = flat.norm() + 1e-12
    scale = min(1.0, clip / norm)
    out = []
    for g in grads:
        gd = g.detach() * scale
        gd = gd + torch.randn_like(gd) * (noise * clip)
        out.append(gd)
    return out


def invert_gradient(
    target_image: torch.Tensor,     # (1, C, H, W), already normalized model input
    target_label: torch.Tensor,     # (1, num_labels) in {0,1}
    size: int = 32,
    steps: int = 300,
    lr: float = 0.1,
    dp_clip: float | None = None,
    dp_noise: float | None = None,
    seed: int = 0,
    device="cpu",
) -> dict:
    """Reconstruct ``target_image`` from its gradient (optionally DP-protected).

    Returns the reconstruction and the final gradient-match loss. Compare the returned
    image to the original with SSIM/PSNR to quantify leakage.
    """
    torch.manual_seed(seed)
    model = SmallNet(in_ch=target_image.shape[1], size=size,
                     num_labels=target_label.shape[1]).to(device)
    model.eval()
    x = target_image.to(device)
    y = target_label.to(device).float()

    true_grads = [g.detach() for g in _param_grads(model, x, y)]
    if dp_clip is not None and dp_noise is not None:
        true_grads = _apply_dp(true_grads, dp_clip, dp_noise)

    dummy = torch.randn_like(x, requires_grad=True)
    opt = torch.optim.Adam([dummy], lr=lr)
    last = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        g = _param_grads(model, dummy, y)
        loss = sum(((gi - ti) ** 2).sum() for gi, ti in zip(g, true_grads))
        loss.backward()
        opt.step()
        last = float(loss.item())
    return {"reconstruction": dummy.detach().cpu(), "grad_match_loss": last}
