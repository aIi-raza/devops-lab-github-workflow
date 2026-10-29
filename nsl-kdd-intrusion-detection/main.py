"""
main.py

Runs the full NSL-KDD intrusion detection pipeline end to end:
1. Load and combine the dataset
2. Clean data and create the binary target
3. Select the paper's 25 features, encode, and normalize
4. Train/test split
5. Train Decision Tree and Random Forest, evaluate, compare to the paper
6. Train XGBoost (extra model not used by the paper), evaluate
7. Add engineered features, retrain XGBoost, evaluate
8. Print the final comparison table and conclusion

Update DATA_DIR below to point at wherever you placed KDDTrain+.txt and
KDDTest+.txt on your machine.
"""

import os

from src.data_loader import load_and_combine_data, assign_column_names
from src.preprocessing import (
    drop_unused_column_and_create_target, select_features,
    encode_categorical_features, normalize_features, split_data
)
from src.feature_engineering import add_engineered_features
from src.train_models import train_decision_tree, train_random_forest, train_xgboost
from src.evaluate import evaluate_model, build_comparison_table, print_explanation
from src.visualize import plot_confusion_matrices, plot_comparison_chart

# ----------------------------------------------------------------
# Update this path to wherever your dataset files are stored locally
# ----------------------------------------------------------------
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")


def main():
    # --- 1. Load data ---
    full_df = load_and_combine_data(TRAIN_FILE, TEST_FILE)
    full_df = assign_column_names(full_df)

    # --- 2. Clean + target ---
    full_df = drop_unused_column_and_create_target(full_df)
    df_selected = select_features(full_df)

    # --- 3. Encode + normalize ---
    df_selected = encode_categorical_features(df_selected)
    X_scaled, y, scaler = normalize_features(df_selected)

    # --- 4. Split ---
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # --- 5. Train + evaluate DT and RF ---
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    dt_results = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    plot_confusion_matrices(y_test, dt_model.predict(X_test), rf_model.predict(X_test))
    plot_comparison_chart(dt_results, rf_results)

    # --- 6. Train + evaluate XGBoost (extra model) ---
    xgb_model = train_xgboost(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # --- 7. Feature engineering + retrain XGBoost (bonus) ---
    df_fe = add_engineered_features(df_selected)
    X_fe, y_fe, _ = normalize_features(df_fe)
    X_train_fe, X_test_fe, y_train_fe, y_test_fe = split_data(X_fe, y_fe)

    xgb_fe_model = train_xgboost(X_train_fe, y_train_fe)
    xgb_fe_results = evaluate_model(xgb_fe_model, X_test_fe, y_test_fe, "XGBoost + Feature Engineering")

    # --- 8. Final comparison + conclusion ---
    your_results = {
        "DT (Yours)": dt_results,
        "RF (Yours)": rf_results,
        "XGB (Yours)": xgb_results,
        "XGB_Tuned (Yours)": xgb_fe_results,
    }
    build_comparison_table(your_results)
    print_explanation()


if __name__ == "__main__":
    main()
