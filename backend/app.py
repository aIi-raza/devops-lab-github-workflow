"""Flask API for recorded research results and real NSL-KDD model inference."""
from __future__ import annotations
import csv, io, os, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'nsl-kdd-intrusion-detection'
sys.path.insert(0, str(RESEARCH))
from src.artifact_pipeline import ARTIFACT_DIR, MODEL_NAMES, build_model_input

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
MAX_ROWS = 5000
MODELS = [
    {"name":"Decision Tree", "short":"DT", "source":"ours", "accuracy":99.19,"precision":99.12,"recall":99.19,"f1":99.15},
    {"name":"Random Forest", "short":"RF", "source":"ours", "accuracy":99.38,"precision":99.42,"recall":99.30,"f1":99.36},
    {"name":"XGBoost", "short":"XGB", "source":"ours", "accuracy":99.61,"precision":99.66,"recall":99.53,"f1":99.59},
    {"name":"XGBoost + feature engineering", "short":"XGB+FE", "source":"ours", "accuracy":99.62,"precision":99.66,"recall":99.55,"f1":99.61},
    {"name":"Decision Tree (paper)", "short":"DT", "source":"paper", "accuracy":99.51,"precision":99.49,"recall":99.46,"f1":99.47},
    {"name":"Random Forest (paper)", "short":"RF", "source":"paper", "accuracy":99.72,"precision":99.84,"recall":99.56,"f1":99.70},
]
_bundles = {}
_load_errors = {}


def load_bundles():
    global _bundles, _load_errors
    _bundles, _load_errors = {}, {}
    for name in MODEL_NAMES:
        path = Path(os.environ.get('NSLKDD_ARTIFACT_DIR', ARTIFACT_DIR)) / f'{name}.pkl'
        try:
            if not path.exists():
                _load_errors[name] = 'Artifact missing; run python train_artifacts.py in nsl-kdd-intrusion-detection.'
                continue
            bundle = joblib.load(path)  # trusted locally generated artifacts only
            if bundle.get('artifact_version') != 1 or not hasattr(bundle.get('model'), 'predict'):
                raise ValueError('Unsupported artifact version or missing model')
            _bundles[name] = bundle
        except Exception as exc:
            _load_errors[name] = f'Artifact could not be loaded: {exc}'

load_bundles()

@app.after_request
def cors(response):
    origin = request.headers.get('Origin', '')
    allowed = os.environ.get('CORS_ORIGINS', '*').split(',')
    if '*' in allowed or origin in allowed:
        response.headers['Access-Control-Allow-Origin'] = origin if origin else '*'
        response.headers['Vary'] = 'Origin'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.errorhandler(413)
def too_large(_): return jsonify(error='Upload exceeds the 5 MB limit.'), 413

@app.get('/')
def home(): return jsonify(service='nsl-kdd-api', status='ok', prediction_ready=bool(_bundles))

@app.get('/api/health')
def health():
    return jsonify(status='ok', prediction_ready=bool(_bundles), available_models=list(_bundles), missing_or_failed=_load_errors)

@app.get('/api/models')
def models():
    return jsonify(models=[{'id': n, 'name': {'decision_tree':'Decision Tree','random_forest':'Random Forest','xgboost':'XGBoost','xgboost_fe':'XGBoost + Feature Engineering'}[n], 'available': n in _bundles, 'error': _load_errors.get(n)} for n in MODEL_NAMES])

@app.get('/api/results')
def results(): return jsonify(models=MODELS)

def parse_ground_truth(value, row_number):
    val = str(value).strip().lower()
    if val in ('normal', '0'):
        return 0
    if val in ('attack', '1'):
        return 1
    raise ValueError(f'Invalid ground-truth label on row {row_number}: {value!r}. Use Normal/Attack or 0/1.')


def read_csv_upload(file):
    if not file or not file.filename:
        raise ValueError("Upload a CSV in the 'file' field.")
    if not file.filename.lower().endswith('.csv'):
        raise ValueError('Only .csv files are accepted.')
    raw = file.stream.read(5 * 1024 * 1024 + 1)
    if len(raw) > 5 * 1024 * 1024:
        raise OverflowError('Upload exceeds the 5 MB limit.')
    frame = pd.read_csv(io.BytesIO(raw))
    if frame.empty:
        raise ValueError('CSV contains no records.')
    if len(frame) > MAX_ROWS:
        raise ValueError(f'CSV exceeds the {MAX_ROWS}-row limit.')
    return frame


def extract_ground_truth(frame):
    for col in ('actual_label', 'label', 'class'):
        if col in frame.columns:
            actual = frame[col].copy()
            frame = frame.drop(columns=[col])
            return frame, actual
    return frame, None


def predict_frame(name, frame, fmt):
    if name not in MODEL_NAMES: raise KeyError(f'Unsupported model: {name}')
    if name not in _bundles: raise RuntimeError(_load_errors.get(name, 'Model artifact is unavailable.'))
    bundle = _bundles[name]
    X = build_model_input(frame, bundle, fmt)
    pred = bundle['model'].predict(X).astype(int)
    probs = bundle['model'].predict_proba(X) if hasattr(bundle['model'], 'predict_proba') else None
    rows = []
    for i, label in enumerate(pred):
        item = {'row': i + 1, 'predicted_class': int(label), 'prediction': 'Attack' if label else 'Normal'}
        if probs is not None:
            classes = list(bundle['model'].classes_)
            item['probabilities'] = {'Normal': float(probs[i][classes.index(0)]), 'Attack': float(probs[i][classes.index(1)])}
        rows.append(item)
    return rows

@app.post('/api/predict')
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get('features'), dict): return jsonify(error="JSON body must contain a 'features' object."), 400
    try:
        name = body.get('model', 'random_forest'); fmt = body.get('input_format', 'raw')
        frame = pd.DataFrame([body['features']])
        rows = predict_frame(name, frame, fmt)
        return jsonify(model=name, input_format=fmt, result=rows[0])
    except KeyError as e: return jsonify(error=str(e).strip("'")), 400
    except RuntimeError as e: return jsonify(error=str(e), prediction_ready=False), 503
    except Exception as e: return jsonify(error=str(e)), 400

@app.post('/api/preprocess/csv')
def preprocess_csv():
    """Convert a raw-feature CSV to this model's exact numeric model-input schema."""
    try:
        frame = read_csv_upload(request.files.get('file'))
        name = request.form.get('model', 'random_forest')
        if name not in MODEL_NAMES:
            return jsonify(error=f'Unsupported model: {name}'), 400
        if name not in _bundles:
            return jsonify(error=_load_errors.get(name, 'Model artifact is unavailable.'), prediction_ready=False), 503
        raw_frame, actual = extract_ground_truth(frame)
        bundle = _bundles[name]
        if actual is not None:
            # Validate before preserving labels in the output.
            labels = [parse_ground_truth(v, i + 2) for i, v in enumerate(actual)]
        else:
            labels = None
        transformed = build_model_input(raw_frame, bundle, 'raw')
        if labels is not None:
            transformed['actual_label'] = ['Normal' if y == 0 else 'Attack' for y in labels]
        output = transformed.to_csv(index=False)
        return Response(output, mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename="{name}_preprocessed.csv"'})
    except OverflowError as e:
        return jsonify(error=str(e)), 413
    except RuntimeError as e:
        return jsonify(error=str(e), prediction_ready=False), 503
    except Exception as e:
        return jsonify(error=f'Unable to preprocess CSV: {e}'), 400


@app.post('/api/predict/csv')
def predict_csv():
    try:
        frame = read_csv_upload(request.files.get('file'))
        frame, actual = extract_ground_truth(frame)
        fmt = request.form.get('input_format', 'raw'); name = request.form.get('model', 'random_forest')
        # Validate all rows at once; errors fail the request instead of silently dropping records.
        rows = predict_frame(name, frame, fmt)
        for i, row in enumerate(rows):
            row['row'] = i + 1
            if actual is not None:
                truth = parse_ground_truth(actual.iloc[i], i + 2)
                row['actual_class'] = truth; row['actual_label'] = 'Attack' if truth else 'Normal'
        predictions = [r['predicted_class'] for r in rows]
        response = {'model': name, 'input_format': fmt, 'total_records': len(rows), 'normal_count': predictions.count(0), 'attack_count': predictions.count(1), 'records': rows}
        if actual is not None:
            y = [r['actual_class'] for r in rows]
            response['metrics'] = {'accuracy': accuracy_score(y, predictions), 'precision': precision_score(y, predictions, zero_division=0), 'recall': recall_score(y, predictions, zero_division=0), 'f1': f1_score(y, predictions, zero_division=0)}
            response['evaluation_note'] = 'Metrics describe only the uploaded labelled records; small or non-independent samples are not a reliable estimate of generalization.'
        return jsonify(response)
    except OverflowError as e:
        return jsonify(error=str(e)), 413
    except RuntimeError as e:
        return jsonify(error=str(e), prediction_ready=False), 503
    except Exception as e:
        return jsonify(error=f'Invalid CSV/input: {e}'), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)), debug=False)
