"""
Small Flask API for the NSL-KDD project page.

It serves the model results so the Angular page can load them from a backend.
It does not need the trained model files. A /predict endpoint that loads a
saved model can be added later.

Run it with:  python app.py   (it listens on http://localhost:5000)
"""

from flask import Flask, jsonify

app = Flask(__name__)

# Results from our runs (accuracy, precision, recall and F1, in percent).
# "ours" rows are our models, "paper" rows are the numbers the paper reports.
MODELS = [
    {"name": "Decision Tree", "short": "DT", "source": "ours",
     "accuracy": 99.19, "precision": 99.12, "recall": 99.19, "f1": 99.15},
    {"name": "Random Forest", "short": "RF", "source": "ours",
     "accuracy": 99.38, "precision": 99.42, "recall": 99.30, "f1": 99.36},
    {"name": "XGBoost", "short": "XGB", "source": "ours",
     "accuracy": 99.61, "precision": 99.66, "recall": 99.53, "f1": 99.59},
    {"name": "XGBoost + feature engineering", "short": "XGB+FE", "source": "ours",
     "accuracy": 99.62, "precision": 99.66, "recall": 99.55, "f1": 99.61},
    {"name": "Decision Tree (paper)", "short": "DT", "source": "paper",
     "accuracy": 99.51, "precision": 99.49, "recall": 99.46, "f1": 99.47},
    {"name": "Random Forest (paper)", "short": "RF", "source": "paper",
     "accuracy": 99.72, "precision": 99.84, "recall": 99.56, "f1": 99.70},
]


@app.after_request
def add_cors_headers(response):
    """
    The page is served from github.io and this API runs on localhost, so the
    browser needs these headers before it lets the page read the answer.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.get("/")
def home():
    """Health check: tells you the server is running."""
    return jsonify(service="nsl-kdd-api", status="ok")


@app.get("/api/results")
def results():
    """The results table shown on the page."""
    return jsonify(models=MODELS)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)