"""
train_models.py

Trains the three models used in this lab:
- Decision Tree
- Random Forest
- XGBoost (the extra model, not used in the reference paper)
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_decision_tree(X_train, y_train, random_state: int = 42) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier."""
    print("\n" + "-" * 60)
    print("Training Decision Tree...")
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    print("  Done.")
    return model


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    print("\n" + "-" * 60)
    print("Training Random Forest...")
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    print("  Done.")
    return model


def train_xgboost(X_train, y_train, random_state: int = 42) -> XGBClassifier:
    """Train an XGBoost classifier (the model not used by the paper's authors)."""
    print("\n" + "-" * 60)
    print("Training XGBoost...")
    model = XGBClassifier(random_state=random_state, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("  Done.")
    return model
