# NSL-KDD Workbench

A research prototype for network intrusion detection on the NSL-KDD dataset. It combines an Angular frontend, a Flask inference API, and a Python model-training pipeline, with CI/CD on GitHub Actions.

**Live site:** https://aii-raza.github.io/devops-lab-github-workflow/

With the workbench you can:
- **Predict a single record**: fill in 25 NSL-KDD features (or start from a preset) and get Normal/Attack with class probabilities
- **Analyse a CSV**: upload a file for per-row predictions, counts, and accuracy/precision/recall/F1 if labels are included
- **Compare models**: run 2-3 trained models on the same CSV, side by side

> GitHub Pages only hosts the static frontend. Predictions need the Flask backend running with locally trained model artifacts, so the live site shows "API unavailable" unless you run the backend yourself. The interface never invents predictions.

## Models

Four models are trained for binary classification (Normal vs Attack):

| Model | API id | Notes |
|---|---|---|
| Decision Tree | `decision_tree` | Single tree, baseline |
| Random Forest | `random_forest` | Tree ensemble (default model) |
| XGBoost | `xgboost` | Boosted trees; extra model not used in the reference paper |
| XGBoost + Feature Engineering | `xgboost_fe` | XGBoost on an engineered feature set |

All models use `random_state=42`.

### Recorded results (%)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 99.19 | 99.12 | 99.19 | 99.15 |
| Random Forest | 99.38 | 99.42 | 99.30 | 99.36 |
| XGBoost | 99.61 | 99.66 | 99.53 | 99.59 |
| XGBoost + feature engineering | 99.62 | 99.66 | 99.55 | 99.61 |
| Decision Tree (paper) | 99.51 | 99.49 | 99.46 | 99.47 |
| Random Forest (paper) | 99.72 | 99.84 | 99.56 | 99.70 |

These figures are stored in `backend/app.py` and served by `GET /api/results`. The "(paper)" rows are the figures reported in the reference paper (`research_paper/DRL-GAN.pdf`). If you retrain and get different numbers, update them in both places.

## Project structure

```
.github/workflows/   ci.yml (tests + build), deploy.yml (GitHub Pages)
backend/             Flask API (app.py) and unit tests (test_app.py)
frontend/            Angular app + sample CSVs (public/samples/)
nsl-kdd-intrusion-detection/
  train_artifacts.py            trains and saves the model artifacts
  main.py                       research pipeline entry point
  src/                          data loading, preprocessing, feature engineering,
                                training, evaluation, gap experiments, visualisation
  samples/                      sample CSV generator and files
  report/lab_report.md          lab report
  research_paper/DRL-GAN.pdf    reference paper
```

## Getting started

### 1. Train the models (once, locally)

```powershell
cd nsl-kdd-intrusion-detection
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Put KDDTrain+.txt and KDDTest+.txt in data/ first
python train_artifacts.py
```

Artifacts are saved as `.pkl` files in `nsl-kdd-intrusion-detection/artifacts/`. They are gitignored, so every machine must train its own. See `nsl-kdd-intrusion-detection/README.md` for details.

### 2. Run the Flask API

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

The API runs at http://127.0.0.1:5000. Check `/api/health` to see which models loaded. The API starts even without artifacts, but prediction routes return `503` until they exist.

| Environment variable | Purpose | Default |
|---|---|---|
| `PORT` | API port | `5000` |
| `NSLKDD_ARTIFACT_DIR` | Folder to load model artifacts from | `nsl-kdd-intrusion-detection/artifacts` |
| `CORS_ORIGINS` | Comma-separated allowed origins | `*` |

### 3. Run the Angular frontend

```powershell
cd frontend
npm ci
npm start
```

Open http://localhost:4200. The frontend calls `http://localhost:5000` by default; set `globalThis.NSLKDD_API_URL` in a deployment bootstrap to point it elsewhere.

## Frontend

The Angular app (v1.1) is a single-page workbench:

- **Hero:** an animated 3D-style globe, floating model cards for choosing the active model, a pointer-following spotlight, and animated headline.
- **Single prediction:** a form of 25 raw NSL-KDD features in four groups (Connection details, Traffic volume, Connection patterns, Destination host patterns). Three presets (Normal, DoS-like, UDP) fill the form. The result shows the predicted class and Normal/Attack probability bars.
- **CSV analysis:** drag-and-drop upload (CSV only, up to 5 MB and 5,000 rows) with bundled sample files. Results show animated record counts, evaluation metrics when the file has an `actual_label` column, and a table of the first 100 records with confidence. All processed rows can be exported to CSV.
- **Model comparison:** pick 2 or 3 models, run them on the same CSV, and compare counts and metrics in one table. A failing model is flagged without blocking the others.
- **Interface:** full-width navigation that collapses into a floating pill on scroll, light/dark theme (remembered in `localStorage`), a live API-status indicator, a custom cursor, count-up numbers and scroll-reveal animations.
- **Honest limits:** a "Good to know" section states that it classifies supplied records only, does not monitor networks, and that NSL-KDD is an older benchmark.

The web form and CSV upload both use the **raw** NSL-KDD features. Encoding, scaling and feature engineering happen in the backend.

## CSV format

- Include the 25 selected raw NSL-KDD feature columns, with original categorical values such as `tcp`, `http`, `SF`.
- Optionally add an `actual_label` column (`Normal`/`Attack` or `0`/`1`) to get accuracy, precision, recall and F1. The API also accepts a column named `label` or `class`. Labels are never used as model features.
- Samples are in the workbench and in `frontend/public/samples/`: `sample_normal.csv`, `sample_attack.csv`, `sample_mixed.csv`.

**Note on evaluation:** a 100% score on a few rows only means those rows matched their labels. The training pipeline merges KDDTrain+ and KDDTest+ before its own 80/20 split, so KDDTest+ rows are not guaranteed to be unseen. Use a large, independent labelled set for a credible evaluation.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service status and whether prediction is ready |
| GET | `/api/health` | Loaded models and any artifact load errors |
| GET | `/api/models` | The four models and whether each is available |
| GET | `/api/results` | Recorded research results (table above) |
| POST | `/api/predict` | Predict a single record (JSON) |
| POST | `/api/predict/csv` | Predict every row of an uploaded CSV |
| POST | `/api/preprocess/csv` | Convert a raw-feature CSV to a model's numeric input format (API only, not in the UI) |

**Single record** (`POST /api/predict`):

```json
{
  "model": "random_forest",
  "input_format": "raw",
  "features": { "...": "..." }
}
```

`model` defaults to `random_forest`. `input_format` defaults to `raw`; the API also accepts `engineered`, but the UI only uses `raw`. The response contains the predicted class and probabilities.

**CSV upload** (`POST /api/predict/csv`, multipart form): fields `file`, `model`, `input_format`. The response contains total, normal and attack counts, per-row predictions, and metrics if the CSV has labels.

**Validation and limits**
- CSV uploads must be `.csv`, at most 5 MB and 5,000 rows.
- Unknown models or input formats return `400`.
- Incomplete feature input returns `400` naming the missing features. This check runs before the model loads, so it also works in CI where no artifacts exist.
- A valid request for a model with no artifact returns `503`.

## Tests

```powershell
cd backend
python -m unittest -v
cd ..\frontend
npm ci
npm run build
```

## CI/CD

Two GitHub Actions workflows handle testing and deployment.

**CI** (`ci.yml`) runs on every push and pull request to `main`, with two parallel jobs:
- **Backend (Flask):** Python 3.12, installs dependencies, syntax-checks `app.py`, runs the unit tests.
- **Frontend (Angular):** Node 24, `npm ci`, production build, then verifies `dist/frontend/browser/index.html` exists.

**Deploy** (`deploy.yml`) runs only after CI completes successfully on `main`:
1. Checks out the exact commit that passed CI.
2. Builds Angular with `--base-href /devops-lab-github-workflow/`.
3. Uploads the build as a Pages artifact and deploys it to GitHub Pages.

A concurrency group ensures a newer deployment cancels one still in progress.