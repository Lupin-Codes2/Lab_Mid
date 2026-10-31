# Lab Midterm Machine Learning Project

A FastAPI and HTML/CSS/JavaScript app that compares the five techniques listed before the
midterm in the **AIC354 Machine Learning Fundamentals lab manual**:

1. Decision Tree
2. Naive Bayes
3. K-Nearest Neighbors
4. Artificial Neural Network (multilayer perceptron)
5. Support Vector Machine

The project brief requests a suitable dataset, preprocessing, model evaluation, a FastAPI
backend, a connected frontend, setup instructions, and a brief report. Activities/examples
from the slides are intentionally excluded as requested. The manual places K-Means, PCA,
Genetic Algorithms, Bagging, and Reinforcement Learning after the midterm; those techniques
are not included.

## Dataset and preprocessing

The included `Top 1000 games of all time.csv` has ranking, game name, year, genre tags, and
vote count. The task is to classify a game's **first listed genre** using ranking, release
year, and votes. Game name and genre tags are not model inputs. The app supports searching
the dataset by game name to fill in those three feature values.

The training pipeline:

- Reads UTF-8 CSV data, parses comma-formatted vote counts and extracts the four-digit year.
- Skips fully blank, incomplete, and invalid rows with counts recorded in the report.
- Groups genres with fewer than 15 examples into `Other` to avoid evaluating classes with
  too little support.
- Applies `log1p` to votes and standardizes numeric features inside each model pipeline.
- Randomly oversamples minority classes only within each training fold; validation folds keep
  the original class distribution.
- Uses five-fold stratified group cross-validation, grouping duplicate game titles so they
  cannot appear in both training and validation in a fold.
- Reports accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, a confusion
  matrix, and per-class results. The most-frequent-class baseline shows why accuracy alone is
  misleading on this imbalanced dataset.

The report is generated at `reports/lab_midterm_report.md`; machine-readable metrics are
written to `reports/model_comparison.json`.

The web interface provides a model dropdown for focused predictions or a one-click
comparison of all five models. Each technique can also be opened from the comparison table
to review its rationale, tradeoff, and per-genre precision/recall/F1. A checklist on the page
maps the implementation to the project brief requirements.

## Project structure

```text
app.py                  FastAPI endpoints and artifact loading
game_ml/data.py         CSV parsing, validation, and dataset audit
game_ml/modeling.py     Preprocessing, five classifiers, and cross-validation
game_ml/reporting.py    JSON results and brief Markdown report
train_models.py         Reproducible model-training entry point
static/index.html       Frontend, game search, prediction, and score comparison
tests/                  Focused data and model tests
models/                 Generated model bundles (not committed)
reports/                Generated brief report and evaluation results
```

## Run locally on Windows

Requires Python 3.10 or newer. From the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train_models.py
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000> for the frontend, <http://127.0.0.1:8000/docs> for interactive
API docs, <http://127.0.0.1:8000/health> for service health, and
<http://127.0.0.1:8000/report> for the generated report. The API also exposes `/models`,
`/games?query=last%20of%20us`, and `POST /predict/classification`.

Training must run before starting the API because model bundles and evaluation reports are
generated locally from the CSV. Render's build command does this automatically.

## Deploy the GUI with Render

The source repository is [Lupin-Codes2/Lab_Mid](https://github.com/Lupin-Codes2/Lab_Mid).
The frontend is served by FastAPI from the same origin as its API, so the search, prediction,
and results UI work together after deployment. GitHub Pages is static-only and cannot run this
Python backend.

1. Push or connect the repository's `main` branch on [Render](https://render.com/).
2. Choose **New** > **Blueprint**, select `Lupin-Codes2/Lab_Mid`, and apply the service defined
   in [`render.yaml`](render.yaml).
3. The Blueprint pins Python 3.13.5, installs `requirements.txt`, trains all models from the
   included CSV during the build, and starts Uvicorn on Render's `$PORT`.
4. Open the Render `onrender.com` service URL. The GUI is at `/`, its health check is `/health`,
   and the interactive API docs are at `/docs`.

The Blueprint uses Render's free web-service plan. Free services can spin down while idle, so
the first visit after inactivity may take a little longer to respond.

Model binaries, virtual environments, caches, and course PDFs are excluded from Git. The CSV
remains in the repository because deployment training and game search require it. Confirm you
have permission to redistribute the dataset before making the repository public. The prediction
API has no authentication, so do not add private data without access controls.
