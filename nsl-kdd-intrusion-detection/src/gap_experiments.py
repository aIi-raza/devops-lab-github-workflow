"""
gap_experiments.py

The reference paper (Avci & Koca, 2023) reports its results from ONE random
80/20 split of the merged dataset. This file tests two things it did not report:

1. Stability: does the result hold across 5 different splits? (cross-validation)
2. Realism: how do the models do on the official, separate test file (KDDTest+)?
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_models(random_state: int = 42) -> dict:
    """The two models the paper used (we reuse the same ones for a fair comparison)."""
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
    }


def run_cross_validation(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> dict:
    """
    Test 1: 5-fold cross-validation on the merged dataset.
    The data is split into 5 parts. Each part takes a turn as the test set,
    so we get 5 accuracy scores instead of 1. Then we report the average
    and the spread (standard deviation).
    """
    print("\n" + "=" * 60)
    print("GAP TEST 1: 5-FOLD CROSS-VALIDATION (stability check)")
    print("=" * 60)

    X = df.drop('label', axis=1)
    y = df['label']

    # StratifiedKFold keeps the normal/attack ratio the same in every fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}
    for name, model in get_models(random_state).items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy') * 100
        results[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"  {name}: {scores.mean():.2f}% +/- {scores.std():.2f}%")
    return results


def run_official_split_test(df: pd.DataFrame, n_train_rows: int, random_state: int = 42) -> dict:
    """
    Test 2: train on the official KDDTrain+ rows only, test on the official
    KDDTest+ rows only. The merged data keeps its original order (train rows
    first, then test rows), so we can split it back by position.
    KDDTest+ contains attack types that never appear in KDDTrain+, so this is
    a harder and more realistic test.
    """
    print("\n" + "=" * 60)
    print("GAP TEST 2: OFFICIAL TRAIN/TEST FILES (realism check)")
    print("=" * 60)

    train_part = df.iloc[:n_train_rows]
    test_part = df.iloc[n_train_rows:]

    X_train, y_train = train_part.drop('label', axis=1), train_part['label']
    X_test, y_test = test_part.drop('label', axis=1), test_part['label']
    print(f"  Train rows: {len(train_part)}  |  Test rows: {len(test_part)}")

    results = {}
    for name, model in get_models(random_state).items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds) * 100,
            "precision": precision_score(y_test, preds) * 100,
            "recall": recall_score(y_test, preds) * 100,
            "f1": f1_score(y_test, preds) * 100,
        }
        r = results[name]
        print(f"\n  --- {name} ---")
        print(f"  Accuracy:  {r['accuracy']:.2f}%")
        print(f"  Precision: {r['precision']:.2f}%")
        print(f"  Recall:    {r['recall']:.2f}%")
        print(f"  F1-Score:  {r['f1']:.2f}%")
    return results


def print_gap_summary(dt_results: dict, rf_results: dict, cv_results: dict, official_results: dict):
    """Print one table putting the paper, our single split, CV, and the official split side by side."""
    paper = {"Decision Tree": 99.51, "Random Forest": 99.72}
    ours = {"Decision Tree": dt_results["accuracy"], "Random Forest": rf_results["accuracy"]}

    rows = []
    for name in ["Decision Tree", "Random Forest"]:
        rows.append({
            "Model": name,
            "Paper (1 random split)": round(paper[name], 2),
            "Ours (1 random split)": round(ours[name], 2),
            "Ours (5-fold CV)": f"{cv_results[name]['mean']:.2f} +/- {cv_results[name]['std']:.2f}",
            "Ours (official KDDTest+)": round(official_results[name]["accuracy"], 2),
        })

    print("\n" + "=" * 60)
    print("GAP SUMMARY: ACCURACY (%) UNDER DIFFERENT EVALUATION SETUPS")
    print("=" * 60)
    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 60 + "\n")


def print_gap_conclusion(cv_results: dict, official_results: dict):
    """Print the conclusion for the gap tests. The numbers are filled in from the actual results."""
    dt_cv, rf_cv = cv_results["Decision Tree"], cv_results["Random Forest"]
    dt_off = official_results["Decision Tree"]["accuracy"]
    rf_off = official_results["Random Forest"]["accuracy"]

    # If the 5 scores barely differ from each other, the random-split result is stable
    stable = max(dt_cv["std"], rf_cv["std"]) < 0.5
    stability_text = ("stable, so the paper's random-split result is not due to a lucky split"
                      if stable else "variable, so a single random split may be misleading")

    print("\n" + "=" * 60)
    print("GAP CONCLUSION")
    print("=" * 60)
    print(f"""
The reference paper reports results from one random 80/20 split of the
merged dataset. We tested two things it did not report.

1. Stability (5-fold cross-validation):
   Decision Tree {dt_cv['mean']:.2f}% +/- {dt_cv['std']:.2f}%, Random Forest {rf_cv['mean']:.2f}% +/- {rf_cv['std']:.2f}%.
   The scores are {stability_text}.

2. Realism (official KDDTrain+ -> KDDTest+):
   Decision Tree {dt_off:.2f}%, Random Forest {rf_off:.2f}%.
   That is a drop of about {dt_cv['mean'] - dt_off:.1f} points (DT) and {rf_cv['mean'] - rf_off:.1f} points (RF)
   compared with cross-validation.

Meaning: accuracy of about 99% under a random split of the merged data
does not carry over to the official test file. KDDTest+ contains attack
types that do not appear in KDDTrain+, and its attack mix is different.
A likely reason is that a random split puts similar records in both the
training and testing parts, which makes the test easier.
Results from the two setups should not be compared with each other.
""")