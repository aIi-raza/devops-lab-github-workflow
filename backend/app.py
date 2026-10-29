"""Flask API for recorded research results and real NSL-KDD model inference."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "nsl-kdd-intrusion-detection"
sys.path.insert(0, str(RESEARCH))

from src import artifact_pipeline
from src.artifact_pipeline import ARTIFACT_DIR, MODEL_NAMES, build_model_input

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MAX_ROWS = 5000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

MODELS = [
    {
        "name": "Decision Tree",
        "short": "DT",
        "source": "ours",
        "accuracy": 99.19,
        "precision": 99.12,
        "recall": 99.19,
        "f1": 99.15,
    },
    {
        "name": "Random Forest",
        "short": "RF",
        "source": "ours",
        "accuracy": 99.38,
        "precision": 99.42,
        "recall": 99.30,
        "f1": 99.36,
    },
    {
        "name": "XGBoost",
        "short": "XGB",
        "source": "ours",
        "accuracy": 99.61,
        "precision": 99.66,
        "recall": 99.53,
        "f1": 99.59,
    },
    {
        "name": "XGBoost + feature engineering",
        "short": "XGB+FE",
        "source": "ours",
        "accuracy": 99.62,
        "precision": 99.66,
        "recall": 99.55,
        "f1": 99.61,
    },
    {
        "name": "Decision Tree (paper)",
        "short": "DT",
        "source": "paper",
        "accuracy": 99.51,
        "precision": 99.49,
        "recall": 99.46,
        "f1": 99.47,
    },
    {
        "name": "Random Forest (paper)",
        "short": "RF",
        "source": "paper",
        "accuracy": 99.72,
        "precision": 99.84,
        "recall": 99.56,
        "f1": 99.70,
    },
]

_bundles = {}
_load_errors = {}


def load_bundles():
    """Load locally trained model artifacts without crashing if they are absent."""
    global _bundles, _load_errors

    _bundles = {}
    _load_errors = {}

    artifact_dir = Path(
        os.environ.get("NSLKDD_ARTIFACT_DIR", str(ARTIFACT_DIR))
    )

    for name in MODEL_NAMES:
        path = artifact_dir / f"{name}.pkl"

        try:
            if not path.exists():
                _load_errors[name] = (
                    "Artifact missing; run python train_artifacts.py "
                    "in nsl-kdd-intrusion-detection."
                )
                continue

            bundle = joblib.load(path)

            if (
                bundle.get("artifact_version") != 1
                or not hasattr(bundle.get("model"), "predict")
            ):
                raise ValueError(
                    "Unsupported artifact version or missing model."
                )

            _bundles[name] = bundle

        except Exception as exc:
            _load_errors[name] = f"Artifact could not be loaded: {exc}"


load_bundles()


def get_expected_features(input_format):
    """
    Find the feature schema exported by artifact_pipeline.

    This lets the API validate incomplete requests even when the trained
    model artifacts are unavailable, such as in a CI test environment.
    """
    candidate_names = (
        "RAW_FEATURES",
        "RAW_FEATURE_COLUMNS",
        "SELECTED_FEATURES",
        "FEATURE_COLUMNS",
        "MODEL_FEATURES",
        "ENGINEERED_FEATURES",
        "ENGINEERED_FEATURE_COLUMNS",
    )

    discovered = {}

    for name in candidate_names:
        value = getattr(artifact_pipeline, name, None)

        if isinstance(value, (list, tuple, set)):
            discovered[name] = set(value)

    if input_format == "raw":
        for name in (
            "RAW_FEATURES",
            "RAW_FEATURE_COLUMNS",
            "SELECTED_FEATURES",
            "FEATURE_COLUMNS",
            "MODEL_FEATURES",
        ):
            if name in discovered:
                return discovered[name]

    elif input_format == "engineered":
        for name in (
            "ENGINEERED_FEATURES",
            "ENGINEERED_FEATURE_COLUMNS",
            "MODEL_FEATURES",
            "FEATURE_COLUMNS",
        ):
            if name in discovered:
                return discovered[name]

    return None


def validate_features_without_bundle(features, input_format):
    """
    Validate the request schema without loading a model.

    Returns an error message when the input is invalid, otherwise None.
    """
    if input_format not in ("raw", "engineered"):
        return f"Unsupported input format: {input_format}"

    if not isinstance(features, dict) or not features:
        return "The features object must contain feature values."

    expected = get_expected_features(input_format)

    if expected:
        missing = expected - set(features.keys())

        if missing:
            missing_names = ", ".join(sorted(missing))
            return f"Missing required features: {missing_names}"

        return None

    # Fallback for pipelines that do not export their schema as a constant.
    # This project uses 25 selected raw input features and 32 features for
    # the feature-engineered input format.
    minimum_features = 25 if input_format == "raw" else 32

    if len(features) < minimum_features:
        return (
            f"Incomplete feature input: expected at least "
            f"{minimum_features} features for input format "
            f"'{input_format}', received {len(features)}."
        )

    return None


@app.after_request
def cors(response):
    origin = request.headers.get("Origin", "")
    allowed = [
        value.strip()
        for value in os.environ.get("CORS_ORIGINS", "*").split(",")
    ]

    if "*" in allowed or origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = (
            origin if origin else "*"
        )
        response.headers["Vary"] = "Origin"

    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"

    return response


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Upload exceeds the 5 MB limit."), 413


@app.get("/")
def home():
    return jsonify(
        service="nsl-kdd-api",
        status="ok",
        prediction_ready=bool(_bundles),
    )


@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        prediction_ready=bool(_bundles),
        available_models=list(_bundles),
        missing_or_failed=_load_errors,
    )


@app.get("/api/models")
def models():
    names = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "xgboost_fe": "XGBoost + Feature Engineering",
    }

    return jsonify(
        models=[
            {
                "id": name,
                "name": names[name],
                "available": name in _bundles,
                "error": _load_errors.get(name),
            }
            for name in MODEL_NAMES
        ]
    )


@app.get("/api/results")
def results():
    return jsonify(models=MODELS)


def parse_ground_truth(value, row_number):
    val = str(value).strip().lower()

    if val in ("normal", "0"):
        return 0

    if val in ("attack", "1"):
        return 1

    raise ValueError(
        f"Invalid ground-truth label on row {row_number}: {value!r}. "
        "Use Normal/Attack or 0/1."
    )


def read_csv_upload(file):
    if not file or not file.filename:
        raise ValueError("Upload a CSV in the 'file' field.")

    if not file.filename.lower().endswith(".csv"):
        raise ValueError("Only .csv files are accepted.")

    raw = file.stream.read(MAX_UPLOAD_BYTES + 1)

    if len(raw) > MAX_UPLOAD_BYTES:
        raise OverflowError("Upload exceeds the 5 MB limit.")

    frame = pd.read_csv(io.BytesIO(raw))

    if frame.empty:
        raise ValueError("CSV contains no records.")

    if len(frame) > MAX_ROWS:
        raise ValueError(f"CSV exceeds the {MAX_ROWS}-row limit.")

    return frame


def extract_ground_truth(frame):
    for col in ("actual_label", "label", "class"):
        if col in frame.columns:
            actual = frame[col].copy()
            frame = frame.drop(columns=[col])
            return frame, actual

    return frame, None


def predict_frame(name, frame, fmt):
    if name not in MODEL_NAMES:
        raise KeyError(f"Unsupported model: {name}")

    if name not in _bundles:
        raise RuntimeError(
            _load_errors.get(name, "Model artifact is unavailable.")
        )

    bundle = _bundles[name]
    X = build_model_input(frame, bundle, fmt)

    pred = bundle["model"].predict(X).astype(int)

    probs = (
        bundle["model"].predict_proba(X)
        if hasattr(bundle["model"], "predict_proba")
        else None
    )

    rows = []

    for i, label in enumerate(pred):
        item = {
            "row": i + 1,
            "predicted_class": int(label),
            "prediction": "Attack" if label else "Normal",
        }

        if probs is not None:
            classes = list(bundle["model"].classes_)

            item["probabilities"] = {
                "Normal": float(probs[i][classes.index(0)]),
                "Attack": float(probs[i][classes.index(1)]),
            }

        rows.append(item)

    return rows


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True)

    if not isinstance(body, dict) or not isinstance(
        body.get("features"), dict
    ):
        return (
            jsonify(error="JSON body must contain a 'features' object."),
            400,
        )

    name = body.get("model", "random_forest")
    fmt = body.get("input_format", "raw")
    features = body["features"]

    # Validate the model name before checking artifact availability.
    if name not in MODEL_NAMES:
        return jsonify(error=f"Unsupported model: {name}"), 400

    # Validate the input format and required features before checking whether
    # the model artifact is available. This is important for CI environments.
    validation_error = validate_features_without_bundle(features, fmt)

    if validation_error:
        return jsonify(error=validation_error), 400

    # A complete request cannot be predicted without a trained artifact.
    if name not in _bundles:
        return (
            jsonify(
                error=_load_errors.get(
                    name, "Model artifact is unavailable."
                ),
                prediction_ready=False,
            ),
            503,
        )

    try:
        frame = pd.DataFrame([features])
        rows = predict_frame(name, frame, fmt)

        return jsonify(
            model=name,
            input_format=fmt,
            result=rows[0],
        )

    except KeyError as exc:
        return jsonify(error=str(exc).strip("'")), 400

    except RuntimeError as exc:
        return jsonify(error=str(exc), prediction_ready=False), 503

    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/preprocess/csv")
def preprocess_csv():
    """Convert a raw-feature CSV to the model's numeric input schema."""
    try:
        frame = read_csv_upload(request.files.get("file"))
        name = request.form.get("model", "random_forest")

        if name not in MODEL_NAMES:
            return jsonify(error=f"Unsupported model: {name}"), 400

        if name not in _bundles:
            return (
                jsonify(
                    error=_load_errors.get(
                        name, "Model artifact is unavailable."
                    ),
                    prediction_ready=False,
                ),
                503,
            )

        raw_frame, actual = extract_ground_truth(frame)

        if actual is not None:
            labels = [
                parse_ground_truth(value, i + 2)
                for i, value in enumerate(actual)
            ]
        else:
            labels = None

        bundle = _bundles[name]
        transformed = build_model_input(raw_frame, bundle, "raw")

        if labels is not None:
            transformed["actual_label"] = [
                "Normal" if label == 0 else "Attack"
                for label in labels
            ]

        output = transformed.to_csv(index=False)

        return Response(
            output,
            mimetype="text/csv",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{name}_preprocessed.csv"'
                )
            },
        )

    except OverflowError as exc:
        return jsonify(error=str(exc)), 413

    except RuntimeError as exc:
        return jsonify(error=str(exc), prediction_ready=False), 503

    except Exception as exc:
        return jsonify(error=f"Unable to preprocess CSV: {exc}"), 400


@app.post("/api/predict/csv")
def predict_csv():
    try:
        frame = read_csv_upload(request.files.get("file"))
        frame, actual = extract_ground_truth(frame)

        fmt = request.form.get("input_format", "raw")
        name = request.form.get("model", "random_forest")

        if name not in MODEL_NAMES:
            return jsonify(error=f"Unsupported model: {name}"), 400

        if fmt not in ("raw", "engineered"):
            return jsonify(error=f"Unsupported input format: {fmt}"), 400

        # Validate labels before running prediction.
        if actual is not None:
            labels = [
                parse_ground_truth(value, i + 2)
                for i, value in enumerate(actual)
            ]
        else:
            labels = None

        rows = predict_frame(name, frame, fmt)

        for i, row in enumerate(rows):
            row["row"] = i + 1

            if labels is not None:
                truth = labels[i]
                row["actual_class"] = truth
                row["actual_label"] = "Attack" if truth else "Normal"

        predictions = [row["predicted_class"] for row in rows]

        response = {
            "model": name,
            "input_format": fmt,
            "total_records": len(rows),
            "normal_count": predictions.count(0),
            "attack_count": predictions.count(1),
            "records": rows,
        }

        if labels is not None:
            response["metrics"] = {
                "accuracy": accuracy_score(labels, predictions),
                "precision": precision_score(
                    labels, predictions, zero_division=0
                ),
                "recall": recall_score(
                    labels, predictions, zero_division=0
                ),
                "f1": f1_score(
                    labels, predictions, zero_division=0
                ),
            }

            response["evaluation_note"] = (
                "Metrics describe only the uploaded labelled records; "
                "small or non-independent samples are not a reliable "
                "estimate of generalization."
            )

        return jsonify(response)

    except OverflowError as exc:
        return jsonify(error=str(exc)), 413

    except RuntimeError as exc:
        return jsonify(error=str(exc), prediction_ready=False), 503

    except Exception as exc:
        return jsonify(error=f"Invalid CSV/input: {exc}"), 400


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
