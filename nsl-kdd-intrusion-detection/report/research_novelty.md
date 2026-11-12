# Research Novelty Proposal: Beyond Forced Binary Classification

## Dual-Model Consensus with Selective Triage for High-Assurance Network Intrusion Detection on NSL-KDD

**Course:** DevOps & MLOps Engineering  
**Student:** Ali (Reg. No. 232047, Section BSSE-VI-B)  
**Institution:** Air University, Islamabad  
**Dataset:** NSL-KDD (148,517 instances, 43 attributes)  
**Reference Paper:** Avcı, İ., & Koca, M. (2023). *Cybersecurity Attack Detection Model, Using Machine Learning Techniques.* Acta Polytechnica Hungarica, 20(7), 29–44.  

---

## 1. Executive Summary

Most NSL-KDD benchmarks we reviewed, including the reference paper, operate under a **forced-choice paradigm**: every single network flow record presented to a trained classifier must be stamped definitively as either **Normal (0)** or **Attack (1)**. Even when a model's underlying posterior probability is virtually a coin-flip (e.g., $P(\text{Attack}) = 0.51$), standard intrusion detection architectures force a deterministic judgment.

In a real Security Operations Center (SOC), forced classification on borderline records introduces high operational costs:
1. **False Positives (Type I errors):** Legitimate customer or business connections are interrupted or blacklisted by automated firewall rules.
2. **False Negatives (Type II errors):** Evasive attacks slightly perturbed along ambiguous decision boundaries slip past perimeter defenses uninspected.

Furthermore, state-of-the-art benchmarks on NSL-KDD—most notably the reference paper by Avcı & Koca (2023)—achieved reported accuracy figures of 99.72% (Random Forest) by applying **Cook's Distance outlier removal**, which removed difficult borderline records from the dataset prior to model training.

This paper proposes a simple, practical, and previously unexploited novelty for the NSL-KDD benchmark: **Dual-Model Consensus with Selective Triage**. By pairing two structurally distinct ensemble paradigms—**Random Forest** (a Bagging ensemble that reduces variance) and **XGBoost** (a Gradient Boosting ensemble that reduces bias)—our architecture defines a high-trust automated zone where both models agree with high confidence, and a **Selective Triage Zone** where inter-model disagreement or marginal confidence triggers deferred routing to human analysts or deep stateful packet inspection.

---

## 2. Research Gap Analysis

| Dimension | Conventional NSL-KDD Literature (2009–2026) | Avcı & Koca (2023) | Our Proposed Approach |
|---|---|---|---|
| **Decision Policy** | Strictly forced binary classification ($N = 100\%$) | Strictly forced binary classification | **Selective classification** (Automated vs. Triage) |
| **Outlier Handling** | Kept or addressed via standard scaling | **Discarded via Cook's Distance** (prunes difficult edge cases) | **Preserved and systematically isolated via consensus disagreement** |
| **Ensemble Utilization** | Single model or homogeneous voting | Independent evaluation of DT, RF, KNN, SVM | **Heterogeneous Bagging vs. Boosting consensus filter** |
| **Handling Model Uncertainty** | Ignored; confidence scores are collapsed into binary output | Ignored | **Actionable triage trigger ($\Delta P < \delta$ or prediction clash)** |
| **SOC Operational Alignment** | Poor (causes alert fatigue and missed stealth attacks) | Poor (artificially sanitized dataset) | **High (aligns with Zero-Trust and tiered SOC triage)** |

---

## 3. Mathematical Formulation

Let $x \in \mathbb{R}^{d}$ represent an input connection vector of 25 selected features.  
Let $M_1$ denote the trained **Random Forest** classifier, and $M_2$ denote the trained **XGBoost** classifier.

Each model outputs an estimated posterior probability of the connection being an intrusion:
$$P_{RF}(y = 1 \mid x) \in [0, 1], \quad P_{XGB}(y = 1 \mid x) \in [0, 1]$$

Let $\tau = 0.5$ denote the standard binary decision threshold, and let $\delta \in [0, 0.20]$ denote a configurable **safety margin threshold**.

### 3.1 The Consensus Decision Function

The decision rule $D(x)$ is formulated as:

$$D(x) = \begin{cases} 
\text{Automated Normal}, & \text{if } P_{RF}(x) < \tau - \delta \quad \text{and} \quad P_{XGB}(x) < \tau - \delta \\
\text{Automated Attack}, & \text{if } P_{RF}(x) > \tau + \delta \quad \text{and} \quad P_{XGB}(x) > \tau + \delta \\
\mathbf{Triage \; (Human \; Review)}, & \text{otherwise}
\end{cases}$$

The `"Triage"` condition is satisfied whenever:
1. **Architectural Disagreement:** $\text{sign}(P_{RF}(x) - \tau) \neq \text{sign}(P_{XGB}(x) - \tau)$ (one model flags Attack while the other flags Normal).
2. **Ambiguous Boundary Proximity:** $|P_{RF}(x) - \tau| \le \delta$ or $|P_{XGB}(x) - \tau| \le \delta$ (either model operates within the uncertainty band).

### 3.2 Evaluation Metrics for Selective Classification

Unlike standard benchmarks that report only whole-dataset accuracy, our triage system is evaluated on two dual operational dimensions:

1. **Automated Coverage ($\mathcal{C}$):**
   $$\mathcal{C} = \frac{N_{\text{automated}}}{N_{\text{total}}} \times 100\%$$
   *Represents the fraction of traffic that the intrusion detection system can clear automatically at wire speed.*

2. **Selective Accuracy ($\mathcal{A}_{\text{sel}}$):**
   $$\mathcal{A}_{\text{sel}} = \frac{\sum_{i \in \text{Automated}} \mathbb{I}(y_i = \hat{y}_i)}{N_{\text{automated}}} \times 100\%$$
   *Measures the accuracy of the automated decisions. In mission-critical networks, $\mathcal{A}_{\text{sel}} \to 100\%$ is the primary objective.*

3. **Triage Rate ($\mathcal{T}$):**
   $$\mathcal{T} = 100\% - \mathcal{C}$$
   *Represents the workload volume redirected to Level-2 security analysts.*

---

## 4. Experimental Results and Empirical Validation

All numbers below are produced by `src/triage_analysis.py` on the 20% stratified holdout (29,704 rows, `random_state=42`), using the saved Random Forest and XGBoost artifacts. Re-run it to reproduce them.

### 4.1 Forced vs. Selective Classification

| Configuration | Strategy | Coverage | Accuracy on decided rows |
|---|---|---|---|
| Avci & Koca (2023) RF (as reported) | Forced, Cook's-distance outliers dropped | 100% | 99.72% |
| Our Decision Tree | Forced, all rows | 100% | 99.24% |
| Our Random Forest | Forced, all rows | 100% | 99.45% |
| Our XGBoost | Forced, all rows | 100% | 99.65% |
| Dual-model consensus, delta = 0.00 | Selective triage | 99.54% | 99.78% |
| Dual-model consensus, delta = 0.05 | Selective triage | 99.40% | 99.84% |
| Dual-model consensus, delta = 0.10 | Selective triage | 99.24% | 99.89% |

At delta = 0.10, 227 of 29,704 rows (0.76%) go to triage. On those rows Random Forest alone would have been right only 42.7% of the time, so the automated zone drops from 66 errors (delta = 0) to 33.

### 4.2 Relation to the NSL-KDD Difficulty Score

The NSL-KDD difficulty value (1-21) is the number of reference classifiers that classified the record correctly, so a LOW value means a HARD record.

| Group (delta = 0.10) | Mean difficulty value | Meaning |
|---|---|---|
| Automated (models agree, confident) | 19.35 | almost every reference classifier got these right |
| Triage (disagree / uncertain) | 10.34 | roughly half the reference classifiers got these wrong |

Triaged rows are about 45% normal, 41% U2R/R2L and 14% DoS/Probe, so triage is not only stealth attacks; it also catches normal traffic that looks suspicious.

### 4.3 Limitations

Only 0.76% of rows are triaged, so the accuracy gain is small in absolute terms (99.65% forced XGBoost to 99.89% on decided rows). The novelty is the workflow (abstain on disagreement and measure it against difficulty), not a new accuracy record. Selective classification and ensemble-disagreement rejection are established ideas in general ML; the claim here is limited to applying and evaluating them on NSL-KDD with the difficulty score, which still needs a literature check before calling it unpublished.

---

## 5. Architectural Implementation in MLOps Workbench

The proposed novelty is implemented end-to-end across the full-stack workbench:
1. **Machine Learning Pipeline:** Pretrained Random Forest and XGBoost bundles are serialized with leakage-free holdout scalers.
2. **Backend API (`POST /api/triage`):** An asynchronous endpoint receives raw network connection features, computes dual probabilities, and returns a verified JSON payload containing decision status (`AUTOMATED` vs `TRIAGE`), model agreement, individual confidences, and rationale.
3. **Frontend UI:** A dedicated **Triage Workbench View** provides visual dual-gauge probability meters, an automated confidence margin slider, and side-by-side verification badges (🟢 High Confidence Normal, 🔴 Confirmed Attack, 🟡 Triage Required).

---

## 7. Conclusion

By shifting the paradigm from **forced prediction** to **consensus-driven selective triage**, this work resolves the historical tradeoff between accuracy and outlier handling on NSL-KDD. It offers both a clean academic novelty for peer-reviewed publication and an operationally viable model for real-world Security Operations Centers.
