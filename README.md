# NSL-KDD intrusion-detection research dashboard

Angular research dashboard + Flask API + Python model-training pipeline. The dashboard preserves the original Data, Methodology, Results, Research Gap, and CI/CD sections and adds a Model Playground for CSV inference.

## Train models (required once, locally)

```powershell
cd nsl-kdd-intrusion-detection
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Put KDDTrain+.txt and KDDTest+.txt in data/ first
python main.py --train-artifacts
```

The command creates ignored local artifacts in `nsl-kdd-intrusion-detection/artifacts/`. See `nsl-kdd-intrusion-detection/README.md` for full details. Artifacts are not committed, so each machine must train them locally.

## Run Flask

In a second terminal from the repository root:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Check `http://127.0.0.1:5000/api/health`. The API starts without artifacts to preserve the existing research-results endpoint, but prediction routes remain unavailable until artifacts exist.

## Run Angular

```powershell
cd frontend
npm ci
npm start
```

Open `http://localhost:4200`. The Model Playground offers sample CSV downloads and CSV inference. `globalThis.NSLKDD_API_URL` can override the default local API URL through a deployment bootstrap. GitHub Pages only hosts static frontend files; it does not run the Python model.

## Tests

```powershell
cd backend
python -m unittest -v
cd ..\frontend
npm ci
npm run build
```

CI preserves the existing GitHub Pages deployment and runs backend tests and the Angular build. Generated model artifacts and dataset files remain untracked/ignored.


## CSV input formats and evaluating predictions

- **Raw selected features** accepts the original 25 NSL-KDD feature columns, including categorical values such as `tcp`, `http`, and `SF`. The backend applies the saved encoders, model-specific feature engineering, and fitted scaler. This is the recommended format for a professor's own NSL-KDD-format CSV.
- **Preprocessed numeric features** accepts only numeric values already encoded and scaled using this project's saved artifacts, with the exact model-specific feature columns and order. Use **Download preprocessed copy** in the workbench to create a compatible file from a raw CSV. The conversion is model-specific, so keep the same model selected when using that converted file.
- An optional `actual_label` column may contain `Normal`/`Attack` or `0`/`1`. It is used only to compare predictions and calculate metrics; it is never a model feature. Leave it out if the labels are unknown.
- A 100% score on a handful of rows means only that those particular rows matched their supplied labels. It is not comparable to the accuracy calculated over a large holdout set. For a credible evaluation, use a sufficiently large independent labelled set and ensure it was not used for training. The current artifact-training pipeline combines KDDTrain+ and KDDTest+ before making its own 80/20 split, so the official KDDTest+ samples are not guaranteed to be unseen by the resulting model.
