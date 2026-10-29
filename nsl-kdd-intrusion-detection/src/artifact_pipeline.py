"""Leakage-aware training and reusable inference preprocessing for NSL-KDD."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .data_loader import load_and_combine_data, assign_column_names
from .preprocessing import SELECTED_FEATURES, CATEGORICAL_COLS, drop_unused_column_and_create_target
from .feature_engineering import add_engineered_features
from .train_models import train_decision_tree, train_random_forest, train_xgboost

MODEL_NAMES = ['decision_tree', 'random_forest', 'xgboost', 'xgboost_fe']
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / 'artifacts'


def prepare_raw_frame(frame: pd.DataFrame, encoders: dict, *, fit=False, allow_unknown=False):
    """Validate and encode raw selected features; never learns during inference.

    Unknown categories are rejected by default. During holdout evaluation only,
    allow_unknown=True maps unseen categories to -1 so a valid rare test category
    does not abort training. The production inference path keeps the strict default.
    """
    missing = [c for c in SELECTED_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f'Missing required feature columns: {missing}')
    data = frame[SELECTED_FEATURES].copy()
    for col in SELECTED_FEATURES:
        if col in CATEGORICAL_COLS:
            vals = data[col].astype(str)
            if fit:
                encoders[col] = LabelEncoder().fit(vals)
            enc = encoders[col]
            known = set(enc.classes_)
            unknown = sorted(set(vals) - known)
            if unknown and not allow_unknown:
                raise ValueError(f'Unsupported category in {col}: {unknown[:10]}')
            # Preserve fitted LabelEncoder mappings for known categories. -1 is a
            # deliberate sentinel for unseen values in evaluation data only.
            mapping = {category: i for i, category in enumerate(enc.classes_)}
            data[col] = vals.map(mapping).fillna(-1).astype(int)
        else:
            data[col] = pd.to_numeric(data[col], errors='raise')
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Feature values must not contain NaN or infinity')
    return data


def build_model_input(frame: pd.DataFrame, bundle: dict, input_format='raw') -> pd.DataFrame:
    expected = bundle['feature_names']
    if input_format == 'raw':
        encoded = prepare_raw_frame(frame, bundle['encoders'])
        base = encoded.copy()
        if bundle['engineered']:
            base = add_engineered_features(base)
        # Each model's scaler is fitted only on its training partition.
        transformed = bundle['scaler'].transform(base[expected])
        return pd.DataFrame(transformed, columns=expected)
    if input_format == 'preprocessed':
        missing = [c for c in expected if c not in frame.columns]
        extra = [c for c in frame.columns if c not in expected]
        if missing:
            raise ValueError(f'Missing preprocessed feature columns: {missing}')
        if extra:
            raise ValueError(f"Unexpected columns in preprocessed input: {extra}. Keep only this model numeric feature columns; the API removes an optional actual_label before validation.")
        if list(frame.columns) != list(expected):
            raise ValueError('Preprocessed feature columns are not in the exact order expected by this model. Use the downloaded preprocessed CSV unchanged.')
        try:
            numeric = frame[expected].apply(pd.to_numeric, errors='raise')
        except (ValueError, TypeError) as exc:
            raw_examples = [c for c in ('protocol_type', 'service', 'flag') if c in frame.columns and frame[c].astype(str).str.lower().isin(['tcp', 'udp', 'icmp', 'http', 'private', 'sf', 's0']).any()]
            if raw_examples:
                raise ValueError(f'This CSV contains raw categorical values in {raw_examples}. Select "Raw selected features" or use Download preprocessed copy first.') from exc
            raise ValueError('Preprocessed mode requires numeric encoded and scaled values in the exact model feature columns. No encoding or scaling is applied in this mode.') from exc
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Preprocessed values must not contain NaN or infinity')
        return pd.DataFrame(values, columns=expected)
    raise ValueError("input_format must be 'raw' or 'preprocessed'")


def train_and_save(train_path=None, test_path=None, artifact_dir=None):
    root = Path(__file__).resolve().parents[1]
    train_path = Path(train_path or root / 'data/KDDTrain+.txt')
    test_path = Path(test_path or root / 'data/KDDTest+.txt')
    out = Path(artifact_dir or ARTIFACT_DIR); out.mkdir(parents=True, exist_ok=True)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f'Dataset files required: {train_path} and {test_path}. See data/README.md')
    full = assign_column_names(load_and_combine_data(str(train_path), str(test_path)))
    full = drop_unused_column_and_create_target(full)
    frame = full[SELECTED_FEATURES].copy()
    y = full['label'].astype(int)
    # Split raw rows first; fit encoders/scalers on training data only.
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(frame, y, test_size=.2, random_state=42, stratify=y)
    encoders = {}
    Xtr = prepare_raw_frame(Xtr_raw, encoders, fit=True)
    # A random holdout may contain rare categories absent from the training
    # partition (e.g. NSL-KDD service="http_2784"). Map these holdout-only
    # categories to -1 for evaluation; API inference remains strict and reports
    # unsupported input categories clearly.
    Xte = prepare_raw_frame(Xte_raw, encoders, allow_unknown=True)
    records = {}
    for name, engineered, trainer in [
        ('decision_tree', False, train_decision_tree), ('random_forest', False, train_random_forest),
        ('xgboost', False, train_xgboost), ('xgboost_fe', True, train_xgboost)]:
        train_base = add_engineered_features(Xtr) if engineered else Xtr.copy()
        test_base = add_engineered_features(Xte) if engineered else Xte.copy()
        feature_names = list(train_base.columns)
        scaler = MinMaxScaler().fit(train_base[feature_names])
        X_train = pd.DataFrame(scaler.transform(train_base[feature_names]), columns=feature_names)
        X_test = pd.DataFrame(scaler.transform(test_base[feature_names]), columns=feature_names)
        model = trainer(X_train, ytr)
        pred = model.predict(X_test)
        bundle = {'model': model, 'encoders': encoders, 'scaler': scaler, 'feature_names': feature_names,
                  'raw_feature_names': SELECTED_FEATURES, 'engineered': engineered, 'target_mapping': {'Normal': 0, 'Attack': 1},
                  'artifact_version': 1, 'input_scaling': 'MinMaxScaler fitted on training partition'}
        joblib.dump(bundle, out / f'{name}.pkl')
        records[name] = {'feature_names': feature_names, 'engineered': engineered,
                         'metrics': {'accuracy': float(accuracy_score(yte, pred)), 'precision': float(precision_score(yte, pred, zero_division=0)),
                                     'recall': float(recall_score(yte, pred, zero_division=0)), 'f1': float(f1_score(yte, pred, zero_division=0))}}
        print(f'{name}: saved; leakage-free holdout accuracy={records[name]["metrics"]["accuracy"]:.4f}')
    joblib.dump({'encoders': encoders, 'raw_feature_names': SELECTED_FEATURES, 'categorical_features': CATEGORICAL_COLS}, out / 'preprocessing.pkl')
    metadata = {'artifact_version': 1, 'dataset': 'NSL-KDD', 'target_mapping': {'Normal': 0, 'Attack': 1},
                'raw_feature_names': SELECTED_FEATURES, 'categorical_features': CATEGORICAL_COLS,
                'split': {'test_size': .2, 'random_state': 42, 'stratify': True, 'preprocessing_fit_on': 'training partition'}, 'models': records}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Artifacts saved to {out}')
    return metadata
