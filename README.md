# Heart Disease Classification Lab

A FastAPI and HTML/CSS/JavaScript project comparing the five classifiers taught before the
midterm: Decision Tree, Naive Bayes, K-Nearest Neighbors, Artificial Neural Network (MLP),
and Support Vector Machine. Later-course techniques and slide activities are not included.

## Dataset and preprocessing

The project uses the supplied Heart Failure Prediction `heart.csv` at the repository root.
It contains 918 records, 11 input features, and the binary `HeartDisease` target. The target
counts in the inspected data are 410 class 0 and 508 class 1. No generated or synthetic
records are used.

The model pipeline:

- Validates the expected fields, categorical values, numeric ranges, and binary target.
- Reports read, usable, blank, incomplete, and invalid row counts.
- Treats zero `RestingBP` and `Cholesterol` values as missing. Numeric columns are median
  imputed and standardized; categorical fields are most-frequent-imputed and one-hot encoded.
- Fits every imputer, scaler, and encoder inside the model pipeline on the training partition
  only, including each cross-validation fold.
- Uses five-fold shuffled stratified cross-validation with fixed random state 42. The classes
  are reasonably balanced, so the models do not use class weights or oversampling.
- Reports pooled out-of-fold accuracy, balanced accuracy, macro/weighted precision, recall and
  F1, ROC-AUC, per-class results, confusion matrices, and fold mean ± standard deviation.
- Compares the required models with a most-frequent-class baseline.
- Uses documented fixed model settings rather than tuning on evaluation folds: tree depth 5
  and leaf size 8; GaussianNB defaults; 9 distance-weighted neighbors; an L2-regularized
  `(24, 12)` MLP with early stopping and a 1,500-iteration limit; and an RBF SVM with
  `C=1.0` and probability estimates enabled.

The reproducible model training command regenerates the JSON comparison, Markdown report, and
compressed model bundles from `heart.csv`.

## Project files

```text
app.py                  FastAPI endpoints and model loading
game_ml/data.py         Heart CSV validation and audit
game_ml/modeling.py     Fold-local preprocessing and five classifiers
game_ml/reporting.py    Generated JSON metrics and Markdown report
train_models.py         Training, evaluation, reports, and model bundles
heart.csv               Dataset used by training and inference
static/index.html       Connected frontend and prediction form
tests/                  Data, preprocessing, and model tests
render.yaml             Existing Render service, build and start configuration
```

## Run locally on Windows

Requires Python 3.13.5. From the repository directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train_models.py
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000> for the frontend, <http://127.0.0.1:8000/docs> for interactive
API docs, <http://127.0.0.1:8000/health> for health, <http://127.0.0.1:8000/models> for
evaluation JSON, and <http://127.0.0.1:8000/report> for the generated report.

`POST /predict/classification` accepts `features` containing the 11 named input fields and
`technique` set to `all` or one of `decision_tree`, `naive_bayes`, `knn`, `ann`, or `svm`.
Predictions include the 0/1 estimate, positive-class probability, and class probabilities.
The service is educational only and must not be used for diagnosis or treatment decisions.

## Deploy

The existing Render service URL remains <https://lab-mid.onrender.com>. Its Blueprint keeps
the current `app:app` entry point and uses:

```text
Build: pip install -r requirements.txt && python train_models.py
Start: uvicorn app:app --host 0.0.0.0 --port $PORT
```

Training on the Render build is the reliable deployment option: the source CSV and pinned
requirements are installed with the repository, and model artifacts are generated using the
same scikit-learn version as the service. This avoids committing machine-specific serialized
models. A free Render service may spin down while idle, so its first request after inactivity
can take longer.

GitHub Pages is an optional static frontend. It uses the same Render API URL, which is
pre-filled in the connection field. The Pages origin is allowed by the API's CORS settings.
