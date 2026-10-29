"""
feature_engineering.py

Adds extra, derived features on top of the paper's original 25 features.
This was done as a bonus step (not required by the paper) to see whether
it could help close the performance gap with the paper's best model.
"""

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7 new features derived from existing ones:
    1. total_bytes               - combined traffic volume (src + dst)
    2. byte_ratio                - ratio of outgoing to incoming bytes
    3. log_src_bytes             - log-transformed src_bytes (reduces skew)
    4. log_dst_bytes             - log-transformed dst_bytes (reduces skew)
    5. error_rate_sum            - combined connection-level error signal
    6. dst_host_error_sum        - combined host-level error signal
    7. count_ratio               - service-specific vs total connection ratio
    """
    df = df.copy()

    df['total_bytes'] = df['src_bytes'] + df['dst_bytes']

    # +1 avoids divide-by-zero when dst_bytes is 0
    df['byte_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)

    # log1p handles zero values safely (log(0) is undefined, log1p(0) = 0)
    df['log_src_bytes'] = np.log1p(df['src_bytes'])
    df['log_dst_bytes'] = np.log1p(df['dst_bytes'])

    df['error_rate_sum'] = df['serror_rate'] + df['dst_host_serror_rate']

    df['dst_host_error_sum'] = (
        df['dst_host_serror_rate'] + df['dst_host_srv_serror_rate'] +
        df['dst_host_rerror_rate'] + df['dst_host_srv_rerror_rate']
    )

    df['count_ratio'] = df['srv_count'] / (df['count'] + 1)

    print("\n" + "=" * 60)
    print("BONUS STEP: FEATURE ENGINEERING (+7 derived features)")
    print("=" * 60)
    print(f"  Shape after feature engineering: {df.shape}")
    return df
