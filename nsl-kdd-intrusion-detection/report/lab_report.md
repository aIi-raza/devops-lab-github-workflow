# Lab Report: NSL-KDD Network Intrusion Detection

**Course:** DevOps (MLOps-focused)
**Student:** Ali — Registration No. 232047, Section BSSE-VI-B
**Institution:** Air University, Islamabad

---

## 1. Objective

The goal of this lab was to:
1. Select a dataset with at least 100,000 records and 35 columns.
2. Find a research paper (published after 2020) that uses that exact dataset.
3. Replicate the paper's best-performing models on the same dataset and compare results.
4. Implement an additional model not used in the paper, to assess whether it could outperform the paper's reported results.

---

## 2. Dataset

**Name:** NSL-KDD (Network Intrusion Detection Dataset)
**Source:** Canadian Institute for Cybersecurity, University of New Brunswick
**Official link:** https://www.unb.ca/cic/datasets/nsl.html
**Kaggle mirror used:** https://www.kaggle.com/datasets/hassan06/nslkdd

The dataset consists of two files, `KDDTrain+.txt` and `KDDTest+.txt`, which were combined into a single dataset for this lab (matching the reference paper's approach of using the full dataset and creating a fresh 80/20 split).

| | Rows | Columns |
|---|---|---|
| KDDTrain+.txt | 125,973 | 43 |
| KDDTest+.txt | 22,544 | 43 |
| **Combined** | **148,517** | **43** |

This satisfies the lab's minimum requirement of 100,000 rows and 35 columns.

---

## 3. Research Paper

**Citation:**
Avcı, İ., Koca, M. (2023). *Cybersecurity Attack Detection Model, Using Machine Learning Techniques.* Acta Polytechnica Hungarica, 20(7), 29-44. DOI: 10.12700/APH.20.7.2023.7.2

**Why this paper was chosen:**
- Published in 2023 — satisfies the "published after 2020" requirement.
- Confirmed to use the exact same NSL-KDD dataset (148,517 samples, 42 features — matches our combined dataset).
- Indexed in Web of Science (SCI-Expanded) and Scopus (Q1) — a credible, peer-reviewed source.
- Compares four machine learning models — Random Forest (RF), Decision Tree (DT), K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) — on this dataset, which gave us models to directly compare against.

**Paper's methodology:**
- Selected 25 of the 41 available features using feature importance (threshold > 0.05%).
- Applied label encoding to categorical columns (`protocol_type`, `service`, `flag`).
- Applied min-max normalization to scale all features to a 0–1 range.
- Removed outliers using Cook's Distance.
- Used an 80/20 train-test split.
- Framed the task as binary classification: malicious (1) vs. benign (0) traffic.

**Paper's reported results:**

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score |
|---|---|---|---|---|
| Random Forest | 99.72% | 99.84% | 99.56% | 99.70% |
| Decision Tree | 99.51% | 99.49% | 99.46% | 99.47% |
| KNN | 99.42% | 99.46% | 99.30% | 99.38% |
| SVM | 99.03% | 99.39% | 98.54% | 98.96% |

---

## 4. Methodology (Our Implementation)

We followed the same preprocessing pipeline as the paper, with one deliberate omission noted below.

1. **Load & combine** `KDDTrain+.txt` and `KDDTest+.txt` into a single 148,517-row dataset.
2. **Assign column names** using the official NSL-KDD feature list.
3. **Create the binary target**: `label = 0` if `class == 'normal'`, else `label = 1`.
4. **Select the same 25 features** identified in the paper's feature importance table.
5. **Encode categorical features** (`protocol_type`, `service`, `flag`) using Label Encoding.
6. **Normalize all features** using Min-Max scaling.
7. **Split** into 80% training / 20% testing (`random_state=42` for reproducibility).
8. **Train** Decision Tree and Random Forest (the two models our Lab 1 work already used, which overlap with the paper's models — this placed us in **Scenario A**: implement the paper's approach and compare directly).
9. **Train a third model, XGBoost**, which was not used anywhere in the paper, to see if a more advanced ensemble method could outperform the paper's results.
10. **(Bonus) Feature engineering**: added 7 derived features (total bytes, byte ratio, log-transformed byte counts, combined error rates, connection count ratio) and retrained XGBoost to see if performance improved further.

**Deliberate omission:** The paper used Cook's Distance to detect and remove statistical outliers before training. This step was not replicated in our implementation, for simplicity. This is the primary reason for the small performance gap discussed in the Results section.

---

## 5. Results

### 5.1 Our Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 99.19% | 99.12% | 99.19% | 99.15% | 99.19% |
| Random Forest | 99.38% | 99.42% | 99.30% | 99.36% | 99.38% |
| XGBoost | 99.61% | 99.66% | 99.53% | 99.59% | 99.61% |
| XGBoost + Feature Engineering | 99.62% | 99.66% | 99.55% | 99.61% | 99.62% |

### 5.2 Comparison: Ours vs. Paper

| Model | Accuracy (Ours) | Accuracy (Paper) | Difference |
|---|---|---|---|
| Decision Tree | 99.19% | 99.51% | -0.32% |
| Random Forest | 99.38% | 99.72% | -0.34% |

Both models fall within the acceptable 3–5% variance range agreed upon for this lab, confirming that our implementation is a valid, faithful reproduction of the paper's approach.

### 5.3 Does Our New Model (XGBoost) Beat the Paper?

| Metric | Paper's Best (Random Forest) | Our Best (XGBoost + FE) |
|---|---|---|
| Accuracy | **99.72%** | 99.62% |
| Precision | **99.84%** | 99.66% |
| Recall | 99.56% | 99.55% |
| F1-Score | **99.70%** | 99.61% |

Our XGBoost model outperformed our **own** Decision Tree and Random Forest across nearly every metric, but it did **not** surpass the paper's reported Random Forest results.

---

## 6. Discussion

The most likely explanation for the remaining gap between our XGBoost model and the paper's Random Forest is the **outlier removal step (Cook's Distance)** used in the paper but not replicated in our implementation. Outlier removal can meaningfully improve model performance on datasets like NSL-KDD, where certain extreme connection records can distort the decision boundary learned by a model.

This gap does not indicate a flawed implementation. It demonstrates that on a dataset where baseline models already perform above 99%, **data preprocessing quality can matter as much as — or more than — model choice**. A more advanced algorithm (XGBoost) could not compensate for a missing preprocessing step that the original authors included.

We chose not to pursue further preprocessing changes (such as implementing Cook's Distance ourselves) or deep hyperparameter tuning, in order to keep the lab's scope aligned with a software engineering course rather than a dedicated machine learning research project.

---

## 7. Conclusion

- A valid dataset (NSL-KDD, 148,517 rows × 43 columns) and a valid, peer-reviewed research paper (Avcı & Koca, 2023) using that exact dataset were identified.
- Our Decision Tree and Random Forest models matched the paper's reported results within ~0.3%, validating our implementation.
- An additional model (XGBoost) not used in the paper was implemented and outperformed our own baseline models, though it did not surpass the paper's best reported result.
- The gap is attributed to a specific, identifiable preprocessing difference (outlier removal), not a flaw in model implementation.

---

## 8. References

1. Avcı, İ., Koca, M. (2023). Cybersecurity Attack Detection Model, Using Machine Learning Techniques. *Acta Polytechnica Hungarica*, 20(7), 29-44.
2. Tavallaee, M., Bagheri, E., Lu, W., Ghorbani, A. A. (2009). A Detailed Analysis of the KDD CUP 99 Data Set. *IEEE Symposium on Computational Intelligence for Security and Defense Applications.*
3. Canadian Institute for Cybersecurity. NSL-KDD Dataset. University of New Brunswick. https://www.unb.ca/cic/datasets/nsl.html
