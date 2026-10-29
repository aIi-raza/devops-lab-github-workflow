"""
preprocessing.py

Cleans the raw NSL-KDD data and prepares it for model training:
- drops the unused 'difficulty' column
- creates a binary target (normal vs. attack)
- keeps only the 25 features identified as important in the reference paper
- encodes categorical columns
- normalizes numeric columns
- splits into train/test sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split  # splits data into train/test sets
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # LabelEncoder: text -> numbers, MinMaxScaler: scales numbers to 0-1

# The 25 features identified as important in the paper (>0.05% feature importance)
# We only keep these, instead of all 41, to match the paper's exact setup
SELECTED_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "wrong_fragment", "hot", "num_failed_logins", "logged_in",
    "count", "srv_count", "serror_rate", "same_srv_rate", "diff_srv_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

# These 3 columns contain text (e.g. 'tcp', 'http') instead of numbers
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']


def drop_unused_column_and_create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop 'difficulty' (not a feature, just a difficulty score for the sample)
    and create a binary target column: 0 = normal, 1 = attack.
    """
    print("\n" + "=" * 60)
    print("STEP 2: CLEANING DATA & CREATING TARGET")
    print("=" * 60)

    # axis=1 means "drop a column" (axis=0 would mean "drop a row")
    df = df.drop('difficulty', axis=1)

    # 'class' has attack names like 'normal', 'neptune', 'smurf', etc.
    # We collapse this into a binary label matching the paper's malicious/benign setup
    # .apply runs this tiny check on every row: normal -> 0, anything else -> 1
    df['label'] = df['class'].apply(lambda x: 0 if x == 'normal' else 1)

    # Count how many rows ended up as 0 (normal) vs 1 (attack), just to check the balance
    counts = df['label'].value_counts()
    print(f"  Normal (0): {counts.get(0, 0)}")
    print(f"  Attack (1): {counts.get(1, 0)}")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the 25 selected features plus the target column."""
    print("\n" + "=" * 60)
    print("STEP 3: SELECTING FEATURES (paper's 25-feature subset)")
    print("=" * 60)
    # df[list_of_columns] keeps only those columns and drops everything else
    # we add 'label' to the list so our target column comes along too
    df_selected = df[SELECTED_FEATURES + ['label']]
    print(f"  Shape after feature selection: {df_selected.shape}")
    return df_selected


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text columns (protocol_type, service, flag) into numbers using
    Label Encoding, since models need numeric input.
    """
    print("\n" + "=" * 60)
    print("STEP 4: ENCODING CATEGORICAL FEATURES")
    print("=" * 60)
    # Copy the dataframe first so we don't accidentally change the original
    df = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()  # creates a new converter for this column
        # fit_transform does two things: learns the unique text values (fit),
        # then replaces each one with a number (transform) - e.g. 'tcp' -> 0, 'udp' -> 1
        df[col] = le.fit_transform(df[col])
    print(f"  Encoded columns: {CATEGORICAL_COLS}")
    return df


def normalize_features(df: pd.DataFrame):
    """
    Separate features (X) from target (y), then scale all feature values
    into a 0-1 range using Min-Max scaling (matches the paper's method).

    Returns X_scaled (DataFrame), y (Series), and the fitted scaler.
    """
    print("\n" + "=" * 60)
    print("STEP 5: NORMALIZING FEATURES (min-max scaling)")
    print("=" * 60)

    # X = all the input features, y = the answer column we're trying to predict
    X = df.drop('label', axis=1)
    y = df['label']

    scaler = MinMaxScaler()
    # fit_transform learns the min/max of each column, then squashes every
    # value into the range 0 to 1 using those min/max values
    X_scaled = scaler.fit_transform(X)
    # MinMaxScaler returns a plain array and loses column names, so we
    # rebuild it as a proper DataFrame and reattach the original column names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    print(f"  Scaled {X_scaled.shape[1]} feature columns to range [0, 1]")

    return X_scaled, y, scaler


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split features and target into train/test sets.
    80/20 split with a fixed random_state matches the paper's setup and
    keeps results reproducible.
    """
    print("\n" + "=" * 60)
    print("STEP 6: TRAIN-TEST SPLIT (80/20)")
    print("=" * 60)
    # test_size=0.2 means 20% goes to testing, 80% to training
    # random_state=42 is a fixed "seed" so the random split is identical every time we run this
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  Training set: {X_train.shape}")
    print(f"  Testing set:  {X_test.shape}")
    return X_train, X_test, y_train, y_test