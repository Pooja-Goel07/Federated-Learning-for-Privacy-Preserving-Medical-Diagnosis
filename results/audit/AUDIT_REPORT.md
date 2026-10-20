# Privacy Audit — Membership Inference & Gradient Inversion

Empirical verification of the differential-privacy claim (Objective 7, research gap G3):
privacy is *measured*, not asserted. Same subset scale as the defence experiments; see
study.md.

## 1. Membership inference (loss-based)

Attack ROC-AUC separating training members from held-out non-members. 0.5 = no leakage.

| Setting | Privacy (ε) | Attack ROC-AUC | Member loss | Non-member loss |
|---|---|---|---|---|
| No DP | — | 0.492 | 0.250 | 0.247 |
| σ=1.0 | 1.0 | 0.500 | 0.272 | 0.265 |
| σ=2.0 | 0.2 | 0.500 | 0.270 | 0.263 |

![Membership inference](plots/09_membership_inference.png)

## 2. Gradient inversion (controlled small-CNN DLG)

Reconstruction quality of one image recovered from its intercepted gradient, with and
without DP applied to that gradient. Higher SSIM/PSNR = better (worse for privacy). This
is a controlled setting (small CNN, 32px,
known label), standard for DLG demonstrations.

| Setting | SSIM | PSNR (dB) |
|---|---|---|
| No DP | 1.000 | 103.2 |
| σ=1.0 | 0.013 | 10.1 |
| σ=2.0 | 0.017 | 10.6 |

![Gradient inversion](plots/10_gradient_inversion.png)

## Finding

The membership-inference attack is essentially at chance even **without** DP (ROC-AUC 0.492 ≈ 0.5; member and non-member losses are nearly equal), meaning the model did not memorize its training set at this scale — so there is little membership signal for DP to remove here, and DP keeps the attack at chance (0.500). The decisive audit evidence is therefore the **gradient-inversion** result: without DP the intercepted gradient reconstructs the input almost perfectly (SSIM 1.00), whereas with DP the reconstruction collapses to noise (SSIM 0.01). This directly verifies that the DP budget prevents a realistic reconstruction attack (gap G3). A stronger membership demonstration would require an over-fitting regime (smaller training set, more epochs).

*Generated from `audit_results.json` by `src/evaluation/audit_report.py`.*
