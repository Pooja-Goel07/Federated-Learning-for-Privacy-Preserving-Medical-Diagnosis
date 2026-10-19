"""Privacy-audit runner (Task 11 / Objective 7): empirically verify the DP claim.

Runs two attacks against the model at several privacy budgets and writes
``audit_results.json`` for plotting:

  * Membership inference (loss-based): attack ROC-AUC separating training members from
    held-out non-members. 0.5 => no membership leakage. Expected to fall toward 0.5 as DP
    tightens.
  * Gradient inversion (controlled DLG): reconstruct one image from its intercepted
    gradient, with and without DP on that gradient; report SSIM/PSNR. DP should degrade
    the reconstruction.

This audits privacy rather than asserting it (research gap G3).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.attacks.inference.gradient_inversion import invert_gradient
from src.attacks.inference.membership import membership_attack
from src.models import DualHeadLoss
from src.models.engine import train_one_epoch
from src.privacy.dp import DPConfig, dp_train_one_epoch, ensure_dp_compatible, get_epsilon, make_private

from .builders import build_model, build_optimizer
from .config import cfg_get
from .data_setup import build_data_context, centralized_loaders, test_loader
from .utils import get_device, save_json, set_seed

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _minmax01(t: torch.Tensor) -> torch.Tensor:
    t = t.clone()
    lo, hi = t.amin(), t.amax()
    return (t - lo) / (hi - lo + 1e-8)


def run_privacy_audit(
    config: dict,
    out_dir: str | Path,
    data_root: str | Path | None = None,
    dp_noise_levels: tuple[float, ...] = (1.0, 2.0),
    inversion_steps: int = 300,
    inversion_size: int = 32,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg_get(config, "experiment.seed", 42))
    set_seed(seed)
    device = get_device(prefer_gpu=bool(cfg_get(config, "training.use_gpu", True)))

    ctx = build_data_context(config, data_root=data_root)
    train_loader, test_ld = centralized_loaders(ctx, config)
    epochs = int(cfg_get(config, "training.local_epochs", 3))
    max_batches = cfg_get(config, "training.max_batches")
    sev_w = float(cfg_get(config, "training.severity_weight", 1.0))
    loss_fn = DualHeadLoss(severity_weight=sev_w).to(device)

    results: dict = {"membership": {}, "gradient_inversion": {}, "setup": {
        "dataset": "NIH ChestX-ray14 (subset)",
        "image_size": int(cfg_get(config, "data.image_size", 224)),
        "epochs": epochs, "dp_noise_levels": list(dp_noise_levels),
        "inversion": {"controlled": True, "model": "small CNN", "size": inversion_size},
    }}

    # ---------- Membership inference: no-DP model ----------
    print("[audit] training no-DP model for membership attack")
    m = build_model(config).to(device)
    opt = build_optimizer(config, m)
    for e in range(epochs):
        train_one_epoch(m, train_loader, loss_fn, opt, device, max_batches=max_batches)
    results["membership"]["no_dp"] = membership_attack(m, train_loader, test_ld, device, max_batches)
    print(f"[audit] no-DP membership attack AUC = {results['membership']['no_dp']['attack_auc']:.3f}")

    # ---------- Membership inference: DP models ----------
    for sigma in dp_noise_levels:
        print(f"[audit] training DP model (sigma={sigma}) for membership attack")
        dp = DPConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=float(sigma),
                      target_delta=float(cfg_get(config, "privacy.dp.target_delta", 1e-5)))
        md = ensure_dp_compatible(build_model(config)).to(device)
        optd = build_optimizer(config, md)
        engine, dpm, dpopt, dploader = make_private(md, optd, train_loader, dp)
        for e in range(epochs):
            dp_train_one_epoch(dpm, dploader, loss_fn, dpopt, device, max_batches=max_batches)
        atk = membership_attack(md, train_loader, test_ld, device, max_batches)
        atk["epsilon"] = get_epsilon(engine, dp.target_delta)
        results["membership"][f"dp_sigma{sigma}"] = atk
        print(f"[audit] DP sigma={sigma} (eps~{atk['epsilon']:.2f}) membership attack AUC = {atk['attack_auc']:.3f}")

    # ---------- Gradient inversion (controlled) ----------
    print("[audit] running controlled gradient-inversion reconstruction")
    images, targets = next(iter(test_ld))
    img = images[:1]
    lab = targets["pathology"][:1].float()
    small = F.interpolate(img, size=(inversion_size, inversion_size), mode="bilinear", align_corners=False)

    recon_nodp = invert_gradient(small, lab, size=inversion_size, steps=inversion_steps, device=device)
    gi = {"no_dp": _recon_metrics(small, recon_nodp["reconstruction"], recon_nodp["grad_match_loss"])}
    saved = {"original": _minmax01(small), "no_dp": _minmax01(recon_nodp["reconstruction"])}
    for sigma in dp_noise_levels:
        rec = invert_gradient(small, lab, size=inversion_size, steps=inversion_steps,
                              dp_clip=1.0, dp_noise=float(sigma), device=device)
        gi[f"dp_sigma{sigma}"] = _recon_metrics(small, rec["reconstruction"], rec["grad_match_loss"])
        saved[f"dp_sigma{sigma}"] = _minmax01(rec["reconstruction"])
    results["gradient_inversion"] = gi
    torch.save(saved, out_dir / "inversion_images.pt")

    save_json(results, out_dir / "audit_results.json")
    print(f"[audit] wrote {out_dir / 'audit_results.json'}")
    return results


def _recon_metrics(original: torch.Tensor, recon: torch.Tensor, grad_loss: float) -> dict:
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )
    o = _minmax01(original)
    r = _minmax01(recon)
    try:
        ssim = float(structural_similarity_index_measure(r, o, data_range=1.0))
    except Exception:
        ssim = float("nan")
    try:
        psnr = float(peak_signal_noise_ratio(r, o, data_range=1.0))
    except Exception:
        psnr = float("nan")
    return {"ssim": ssim, "psnr": psnr, "grad_match_loss": grad_loss}
