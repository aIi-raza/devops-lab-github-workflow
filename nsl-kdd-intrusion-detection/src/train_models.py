"""
train_models.py

Trains the three models used in this lab:
- Decision Tree
- Random Forest
- XGBoost (the extra model, not used in the reference paper)
"""

from sklearn.tree import DecisionTreeClassifier       # single decision tree model
from sklearn.ensemble import RandomForestClassifier   # many trees combined (ensemble)
from xgboost import XGBClassifier                      # boosted trees model (our 3rd, extra model)


def train_decision_tree(X_train, y_train, random_state: int = 42) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier."""
    print("\n" + "-" * 60)
    print("Training Decision Tree...")
    # Create an untrained model object
    # random_state=42 makes results reproducible (same tree every time we run this)
    model = DecisionTreeClassifier(random_state=random_state)
    # .fit() is the actual training step - the model studies X_train (features)
    # against y_train (correct answers) and learns the patterns
    model.fit(X_train, y_train)
    print("  Done.")
    return model


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    print("\n" + "-" * 60)
    print("Training Random Forest...")
    # Same idea as Decision Tree, but this trains many trees on random
    # subsets of the data and combines their votes for a more stable result
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    print("  Done.")
    return model


def train_xgboost(X_train, y_train, random_state: int = 42) -> XGBClassifier:
    """Train an XGBoost classifier (the model not used by the paper's authors)."""
    print("\n" + "-" * 60)
    print("Training XGBoost...")
    # eval_metric='logloss' just tells XGBoost which internal scoring method
    # to use while it trains - a standard choice for binary (0/1) classification
    model = XGBClassifier(random_state=random_state, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("  Done.")
    return model