"""
data_loader.py

Loads the NSL-KDD train and test files, combines them into one dataset,
and assigns proper column names (the raw files have no headers).
"""

import pandas as pd

# Official NSL-KDD column names (41 features + class label + difficulty level)
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "class", "difficulty"
]


def load_and_combine_data(train_path: str, test_path: str) -> pd.DataFrame:
    """
    Load the train and test .txt files and combine them into a single dataframe.

    We combine them because the reference paper (Avci & Koca, 2023) used the
    full 148,517-sample dataset and then created its own 80/20 split, rather
    than using the official pre-split files directly.
    """
    print("\n" + "=" * 60)
    print("STEP 1: LOADING & COMBINING DATA")
    print("=" * 60)

    # header=None because the raw files don't include column names
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    print(f"  Train shape:    {train_df.shape}")
    print(f"  Test shape:     {test_df.shape}")

    # Stack train and test on top of each other into one dataset
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    print(f"  Combined shape: {full_df.shape}")

    return full_df


def assign_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the official NSL-KDD column names to the dataframe."""
    df.columns = COLUMN_NAMES
    print(f"  Columns assigned: {len(COLUMN_NAMES)}")
    return df
