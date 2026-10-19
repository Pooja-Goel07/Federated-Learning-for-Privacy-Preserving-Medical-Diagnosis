# Federated Learning for Privacy-Preserving Medical Diagnosis
### Review–1 Submission: Presentation Content & Literature Review

---

## 1. Aim

To design, implement and empirically evaluate a **privacy-preserving federated learning (FL) framework for collaborative medical diagnosis**, in which geographically separated healthcare institutions jointly train a shared deep learning model by exchanging only protected model updates — never raw patient records.

The framework integrates **differential privacy (DP)** for formal, quantifiable leakage guarantees and **Byzantine fault-tolerant aggregation** for integrity against faulty or malicious participating nodes. Chest radiography (multi-label disease detection with per-disease severity grading) is used as the **validation task** that demonstrates the architecture is clinically viable — it is not itself the research problem.

The central aim is to characterise and optimise the **privacy–robustness–utility trade-off** that determines whether federated medical AI can be deployed in practice.

---

## 2. Abstract

Clinical deep learning is bottlenecked not by model capacity but by **data access**. Individual hospitals hold datasets that are small, demographically narrow and acquisition-biased, while regulatory frameworks (HIPAA, GDPR, India's DPDP Act) make raw-data centralisation infeasible. Federated learning addresses this by keeping data in place and aggregating only model updates — but FL alone is **not** a privacy guarantee. Shared gradients are invertible via reconstruction and membership-inference attacks, and the open participation model exposes the global model to poisoning and Byzantine faults from compromised or misconfigured clients.

This work proposes a **hierarchical client–server federated architecture** for medical diagnosis, in which each leaf node (hospital) performs local training and a centre node performs secured aggregation. Two defence layers are embedded directly into the training loop: (i) **client-level differential privacy** via per-sample gradient clipping and calibrated Gaussian noise under a Rényi-DP accountant, yielding a stated privacy budget (ε, δ); and (ii) **Byzantine-robust aggregation** — Krum/Multi-Krum, coordinate-wise trimmed mean/median, and similarity-based trust weighting — replacing plain FedAvg, so that a bounded fraction of adversarial updates cannot dominate the global model. Statistical heterogeneity across sites (non-IID label and feature skew) is handled through proximal regularisation and local batch normalisation.

The framework is instantiated on multi-institutional chest radiograph data partitioned across simulated hospital nodes, performing multi-label pathology detection and ordinal severity grading. Evaluation compares the federated global model against isolated single-site baselines and a centralised upper bound, measured across varying privacy budgets, adversarial client fractions and heterogeneity levels. Crucially, the system is **audited empirically** — gradient-inversion and membership-inference attacks are executed against the shared updates to verify that the claimed privacy is actually realised, rather than assumed by construction. The expected outcome is a reproducible, threat-model-explicit blueprint for federated clinical AI, together with an empirical map of the regimes in which privacy, robustness and diagnostic accuracy can be jointly satisfied.

---

## 3. Literature Review

### Base Paper

**[1] Federated Learning for Thoracic Disease Classification Using Convolutional Neural Networks and Differential Privacy**
**Year:** 2026 — *Healthcare Technology Letters*

Proposes a federated framework with differential privacy for **multi-label thoracic disease classification** from chest radiographs, evaluated on NIH ChestX-ray14 and Stanford CheXpert. It explicitly targets resource-constrained clinical environments where data sharing is legally restricted, and studies the cost of adding DP on top of FL.

**Highlights**
- Directly quantifies the **privacy–utility penalty** of DP integration — both accuracy degradation and computational overhead — in a realistic multi-label radiology setting.
- Establishes CNN-based DP-FL as a workable, reproducible baseline for chest X-ray diagnosis under decentralised training.
- **Limitation relevant to this work:** all participating clients are assumed honest. No adversarial/Byzantine threat model, no robust aggregation, and no empirical privacy audit against reconstruction attacks — the privacy claim rests entirely on the analytic ε.

---

### Supporting Literature

**[2] OptiGradTrust: Byzantine-Robust Federated Learning with Multi-Feature Gradient Analysis and Reinforcement Learning-Based Trust Weighting**
**Year:** 2025

A defence framework for federated medical imaging that scores every client update using a six-dimensional gradient "fingerprint" (VAE reconstruction error, cosine-similarity metrics, L2 norm, sign-consistency ratio, Monte-Carlo Shapley value), fed into a hybrid reinforcement-learning/attention module producing adaptive trust weights before aggregation. Also introduces **FedBN-Prox**, combining federated batch normalisation with proximal regularisation for stable convergence under heterogeneity.

**Highlights**
- Evaluated on MNIST, CIFAR-10 and an Alzheimer's MRI dataset against five distinct Byzantine attack types, under both IID and non-IID splits, outperforming established defences including FLTrust and FLAME.
- Treats **heterogeneity and adversarial behaviour as coupled problems** — the optimizer is co-designed with the defence rather than bolted on.
- **Gap:** addresses integrity but not confidentiality — updates are aggregated in the clear, with no differential privacy guarantee.

---

**[3] Privacy-Preserving Chest X-ray Report Generation via Multimodal Federated Learning with ViT and GPT-2**
**Year:** 2025

A federated multimodal pipeline (Vision Transformer encoder + GPT-2 decoder) that generates radiology reports without centralising images, comparing three aggregation strategies: FedAvg, **Krum**, and a loss-aware federated averaging variant.

**Highlights**
- Demonstrates that a Byzantine-resilient aggregator (Krum) can *outperform* plain FedAvg on clinical output quality (ROUGE, BLEU, BERTScore, RaTEScore) even in a benign federation.
- Shows federated training matching or surpassing centralised training, countering the assumption that decentralisation necessarily costs accuracy.
- Establishes robust aggregation as a **first-class design choice** in medical FL rather than a pure security add-on. No DP layer is applied.

---

**[4] Fusion of Personalized Federated Learning (PFL) with Differential Privacy (DP) Learning for Diagnosis of Arrhythmia Disease**
**Year:** 2025 — *PLOS ONE*

A privacy-preserving architecture fusing personalised federated models with **local differential privacy** for 12-lead ECG arrhythmia diagnosis, using GRU-based temporal modelling and feature fusion across decentralised institutions.

**Highlights**
- Reports ~93.7% average accuracy over a 10,646-recording ECG corpus while enforcing local DP, outperforming both centralised and federated baselines.
- Evidence that the **FL + DP + personalisation stack is modality-agnostic**, generalising beyond imaging to signal-based diagnostics — supporting the framing of federated clinical AI as an architecture-level contribution rather than a task-specific one.

---

**[5] Efficient Differential Privacy Enabled Federated Learning Model for Detecting COVID-19 Disease Using Chest X-ray Images**
**Year:** 2024 — *Frontiers in Medicine*

Introduces an **adaptive** DP-FL model in which the privacy level is adjusted dynamically according to real-time data-sensitivity analysis, coupled with an early-stopping mechanism that limits privacy-budget consumption during training.

**Highlights**
- Moves beyond a single fixed ε: adaptive noise calibration improves the practical utility of DP-FL across diverse hospital environments.
- Explicitly analyses the joint effect of early stopping on both utility and privacy, i.e. budget spend as a controllable training hyperparameter.
- Motivates **adaptive rather than static budget allocation** in the proposed system.

---

**[6] Federated Learning Enables Big Data for Rare Cancer Boundary Detection**
**Year:** 2022 — *Nature Communications*

The largest federated study in medicine at the time of publication: **71 sites across 6 continents** collaboratively training a glioblastoma tumour-boundary detector (n = 6,314) by sharing only numerical model updates.

**Highlights**
- Reports a 33% delineation improvement for the surgically targetable tumour and 23% for the full tumour extent over a publicly trained model.
- Strongest existing evidence that federated aggregation genuinely improves **out-of-sample generalisability** for rare diseases and underrepresented populations — the core value proposition of this project.
- **Gap:** no formal DP guarantee and no adversarial threat model; mutual trust between the 71 sites is assumed by construction.

---

**[7] Defending Against Reconstruction Attacks Through Differentially Private Federated Learning for Classification of Heterogeneous Chest X-Ray Data**
**Year:** 2022 — *Sensors*

Evaluates DP-FL specifically as a **defence against privacy attacks**, comparing DenseNet121 and ResNet50 across an imbalanced, heterogeneous federation of 36 clients constructed from CheXpert and Mendeley chest X-ray data.

**Highlights**
- Non-private baselines reach AUC ≈ 0.94, but the authors demonstrate that **both architectures leak**: image-reconstruction attacks successfully recover training data from individual client updates.
- Identifies DenseNet121 as the more DP-friendly backbone, i.e. architecture choice interacts with the privacy mechanism.
- The clearest published evidence that **FL without DP is not privacy-preserving** — the direct motivation for this project's empirical privacy-audit component.

---

**[8] Federated Learning and Differential Privacy for Medical Image Analysis**
**Year:** 2022 — *Scientific Reports*

A systematic case study of DP-FL on histopathology whole-slide images from The Cancer Genome Atlas, varying the **number of participating healthcare providers, per-site dataset size, and IID vs non-IID distribution**.

**Highlights**
- Isolates **federation topology** as an independent experimental variable, showing that client count and data skew — not only the privacy budget — determine achievable accuracy.
- Provides the experimental-design template for the heterogeneity and client-count sweeps proposed here.
- Confirms that non-IID skew compounds the DP utility penalty, rather than acting independently of it.

---

**[9] End-to-End Privacy Preserving Deep Learning on Multi-Institutional Medical Imaging**
**Year:** 2021 — *Nature Machine Intelligence*

Presents **PriMIA**, an open-source framework combining differentially private federated training, **secure aggregation**, and encrypted remote inference, validated on expert-level paediatric chest X-ray classification.

**Highlights**
- Achieves classification performance on par with locally, non-securely trained models while providing both theoretical and empirical privacy guarantees.
- Demonstrates that the deployed protections **prevent recovery of usable data by a gradient-based model-inversion attack** — one of the few works to verify, not merely assert, its privacy claim.
- Closest existing precedent for an end-to-end secured medical FL stack; however, it models a curious server/adversary only, **not malicious participants**, so integrity under Byzantine clients is out of scope.

---

**[10] Federated Learning for Predicting Clinical Outcomes in Patients with COVID-19**
**Year:** 2021 — *Nature Medicine*

The **EXAM** study: a federated model trained across **20 institutes worldwide** on vital signs, laboratory values and chest X-rays, predicting future oxygen requirements of symptomatic COVID-19 patients.

**Highlights**
- Achieved average AUC > 0.92 for outcomes at 24 h and 72 h from emergency-room presentation.
- Reported a **16% average AUC improvement and a 38% average increase in generalisability** over models trained at a single site on that site's own data — the canonical quantitative justification for this project's core hypothesis that collaboration outperforms isolation.
- Showed FL enabling rapid multi-national collaboration across heterogeneous, unharmonised datasets without data exchange.
- **Gap:** standard federated averaging, no declared privacy budget, no robustness mechanism.

---

## 4. Research Gap

**G1 — Privacy and robustness are studied in isolation, never jointly.**
DP-FL work ([1], [5], [7], [8], [9]) assumes an honest-but-curious server with fully cooperative clients. Byzantine-robust work ([2], [3]) assumes plaintext, noise-free updates. No study systematically characterises their **interaction**, despite a direct conflict: DP gradient clipping and Gaussian noise deliberately destroy the very statistical structure (update norm, direction, inter-client similarity) that robust aggregators such as Krum, trimmed-mean and FLTrust rely on to identify malicious contributions. **A defence that is sound in isolation may fail under composition.**

**G2 — Clinically validated deployments carry no formal guarantees.**
The large-scale studies that prove FL works clinically ([6], [10]) use plain FedAvg with an implicit trust assumption, no stated ε, and no adversarial threat model. Conversely, frameworks with strong formal guarantees ([9]) are demonstrated only at small scale. There is no bridge between clinical credibility and cryptographic/statistical rigour.

**G3 — Privacy is asserted rather than audited.**
Most medical FL literature treats privacy preservation as an architectural property that follows automatically from not moving data. Only a minority ([7], [9]) execute reconstruction attacks against their own system. The field lacks the standard practice of reporting **measured residual leakage at a declared privacy budget**.

**G4 — Heterogeneity interacts with both defences, and is unquantified.**
Non-IID skew arising from scanner vendors, acquisition protocols and differing disease prevalence is handled by FedProx/FedBN-style methods ([2], [8]), but their behaviour **under simultaneous DP clipping and robust aggregation** is unstudied. Honest minority-distribution clients can be statistically indistinguishable from attackers, so a robust aggregator risks silently excluding exactly the underrepresented sites that federation was designed to include — a fairness failure with direct clinical consequences.

**G5 — Diagnostic granularity is limited to presence/absence.**
Federated radiology work is dominated by binary or multi-label pathology detection. **Ordinal severity grading**, which is what actually drives triage and clinical decision-making, remains essentially unexplored under federated and privacy-constrained training.

**G6 — No unified, reproducible testbed.**
Results across the literature are not comparable, because each work fixes a different subset of {privacy budget, attacker fraction, heterogeneity level, aggregation rule, backbone}. No open framework exists that allows these axes to be swept jointly on a common medical benchmark.

---

## 5. Objectives

1. **Design a hierarchical federated architecture** for multi-institutional medical diagnosis — a coordinating centre node with *N* leaf nodes — exposing a modular interface so that the aggregation rule and the privacy mechanism can be swapped independently.

2. **Integrate client-level differential privacy** into local training through per-sample gradient clipping and calibrated Gaussian noise, with a Rényi-DP accountant reporting a formal cumulative (ε, δ) budget for the complete training run.

3. **Implement and compare Byzantine fault-tolerant aggregation rules** — Krum/Multi-Krum, coordinate-wise trimmed mean and median, and similarity-based trust weighting — against FedAvg under label-flipping, sign-flipping, scaled-update and model-replacement attacks at varying adversarial client fractions.

4. **Characterise the joint privacy–robustness–utility trade-off** as the primary research contribution: measure diagnostic performance as a function of (ε, attacker fraction, heterogeneity level) and identify the regimes in which DP noise measurably degrades the detection capability of robust aggregation.

5. **Mitigate statistical heterogeneity** across simulated hospital nodes using proximal regularisation and local batch normalisation, and verify that robust aggregation does not systematically exclude honest clients holding minority data distributions (fairness-under-robustness check).

6. **Validate on chest radiography** with two prediction heads — multi-label pathology detection and per-disease ordinal severity grading (mild / moderate / severe) — benchmarking the federated global model against (a) isolated single-site models and (b) a centralised upper bound.

7. **Empirically audit the privacy claim** by executing gradient-inversion and membership-inference attacks against intercepted client updates, reporting measured reconstruction quality (SSIM/PSNR) and attack success rate at each privacy budget, instead of asserting privacy by construction.

8. **Release a reproducible, configuration-driven testbed** and reference implementation that allows the privacy, robustness and heterogeneity axes to be varied independently, enabling comparable evaluation of future federated clinical AI systems.

---

## 6. References

[1] M. Zulqarnain, S. J. Hussain, M. Z. Aslam, A. Fiaz, and M. Islam, "Federated learning for thoracic disease classification using convolutional neural networks and differential privacy," *Healthcare Technology Letters*, vol. 13, no. 1, art. e70080, Jan. 2026.

[2] M. Karami, F. Ghassemi, H. Kebriaei, and H. Azadegan, "OptiGradTrust: Byzantine-robust federated learning with multi-feature gradient analysis and reinforcement learning-based trust weighting," *arXiv preprint* arXiv:2507.23638, Jul. 2025.

[3] M. R. Islam, M. Z. Hossain, M. Ahmed, and M. S. S. Samu, "Privacy-preserving chest X-ray report generation via multimodal federated learning with ViT and GPT-2," *arXiv preprint* arXiv:2505.21715, May 2025.

[4] S. M. Bokhari, S. Sohaib, and M. Shafi, "Fusion of personalized federated learning (PFL) with differential privacy (DP) learning for diagnosis of arrhythmia disease," *PLOS ONE*, vol. 20, no. 7, art. e0327108, Jul. 2025.

[5] R. Ahmed, P. K. R. Maddikunta, T. R. Gadekallu, N. K. Alshammari, and F. A. Hendaoui, "Efficient differential privacy enabled federated learning model for detecting COVID-19 disease using chest X-ray images," *Frontiers in Medicine*, vol. 11, art. 1409314, Jun. 2024.

[6] S. Pati, U. Baid, B. Edwards, *et al.*, "Federated learning enables big data for rare cancer boundary detection," *Nature Communications*, vol. 13, no. 1, art. 7346, Dec. 2022.

[7] J. Ziegler, B. Pfitzner, H. Schulz, A. Saalbach, and B. Arnrich, "Defending against reconstruction attacks through differentially private federated learning for classification of heterogeneous chest X-ray data," *Sensors*, vol. 22, no. 14, art. 5195, Jul. 2022.

[8] M. Adnan, S. Kalra, J. C. Cresswell, G. W. Taylor, and H. R. Tizhoosh, "Federated learning and differential privacy for medical image analysis," *Scientific Reports*, vol. 12, no. 1, art. 1953, Feb. 2022.

[9] G. Kaissis, A. Ziller, J. Passerat-Palmbach, *et al.*, "End-to-end privacy preserving deep learning on multi-institutional medical imaging," *Nature Machine Intelligence*, vol. 3, no. 6, pp. 473–484, Jun. 2021.

[10] I. Dayan, H. R. Roth, A. Zhong, *et al.*, "Federated learning for predicting clinical outcomes in patients with COVID-19," *Nature Medicine*, vol. 27, no. 10, pp. 1735–1743, Oct. 2021.
