# NSL-KDD Intrusion Detection

DevOps/MLOps lab project: binary classification (normal vs. attack) on the NSL-KDD
network intrusion dataset, benchmarked against a published research paper.

## Dataset

- **NSL-KDD** — 148,517 records, 41 features (+ class label)
- See `data/README.md` for download links

## Reference Paper

Avcı, İ., Koca, M. (2023). *Cybersecurity Attack Detection Model, Using Machine
Learning Techniques.* Acta Polytechnica Hungarica, 20(7), 29-44.

The paper compares Decision Tree, Random Forest, KNN, and SVM on this exact
dataset. We replicate their Decision Tree and Random Forest results, then add
XGBoost as a model not used in the paper.

## Project Structure

```
nsl-kdd-intrusion-detection/
├── data/                   # dataset files go here (not included, see data/README.md)
├── src/
│   ├── data_loader.py      # load + combine + column naming
│   ├── preprocessing.py    # cleaning, target creation, feature selection, encoding, scaling, split
│   ├── feature_engineering.py  # bonus: 7 derived features
│   ├── train_models.py     # DT, RF, XGBoost training
│   ├── evaluate.py         # metrics, comparison table, conclusion
│   └── visualize.py        # confusion matrices, comparison chart
├── main.py                 # runs the full pipeline
├── requirements.txt
├── research_paper/         # reference paper PDF
└── report/                 # lab report
```

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset files into `data/` (see `data/README.md`), then run:

```bash
python main.py
```

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| DT (Yours) | 99.19% | 99.12% | 99.19% | 99.15% |
| RF (Yours) | 99.38% | 99.42% | 99.30% | 99.36% |
| XGBoost (Yours) | 99.61% | 99.66% | 99.53% | 99.59% |
| XGBoost + Feature Engineering (Yours) | 99.62% | 99.66% | 99.55% | 99.61% |
| DT (Paper) | 99.51% | 99.49% | 99.46% | 99.47% |
| RF (Paper) | 99.72% | 99.84% | 99.56% | 99.70% |

Our DT and RF results matched the paper's within ~0.2%, validating the implementation.
XGBoost outperformed our own DT/RF but did not surpass the paper's RF — likely due to
the paper's Cook's Distance outlier removal step, which was not replicated here.
