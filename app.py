import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from game_ml.data import load_dataset
from game_ml.modeling import TECHNIQUES


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
STATIC_DIR = ROOT_DIR / "static"
DATASET_PATH = ROOT_DIR / "Top 1000 games of all time.csv"
REPORT_PATH = REPORT_DIR / "model_comparison.json"
BRIEF_REPORT_PATH = REPORT_DIR / "lab_midterm_report.md"


def load_model_bundle(model_key: str) -> dict[str, Any]:
    model_path = MODEL_DIR / f"{model_key}.joblib"
    if not model_path.is_file():
        raise RuntimeError(
            f"Model file not found: {model_path}. Run `python train_models.py` first."
        )
    return joblib.load(model_path)


def load_comparison_report() -> dict[str, Any]:
    if not REPORT_PATH.is_file():
        raise RuntimeError(
            f"Model report not found: {REPORT_PATH}. Run `python train_models.py` first."
        )
    with REPORT_PATH.open(encoding="utf-8") as report_file:
        return json.load(report_file)


game_dataset = load_dataset(DATASET_PATH)
model_bundles = {
    model_key: load_model_bundle(model_key)
    for model_key in TECHNIQUES
}
comparison_report = load_comparison_report()

app = FastAPI(title="Pre-Midterm Machine Learning Techniques", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lupin-codes2.github.io",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ClassificationFeatures(BaseModel):
    ranking: int = Field(ge=1, le=1000)
    year: int = Field(ge=1950, le=2030)
    votes: int = Field(ge=0, le=10_000_000)


class ClassificationRequest(BaseModel):
    features: ClassificationFeatures
    technique: Literal[
        "all",
        "decision_tree",
        "naive_bayes",
        "knn",
        "ann",
        "svm",
    ] = "all"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "models_loaded": len(model_bundles)}


@app.get("/games")
def search_games(
    query: str = Query(min_length=2, max_length=100),
    limit: int = Query(default=10, ge=1, le=20),
) -> list[dict[str, int | str]]:
    search_term = query.strip().casefold()
    if len(search_term) < 2:
        return []

    matches = []
    for game in game_dataset.records:
        if search_term not in game["name"].casefold():
            continue
        matches.append(
            {
                "name": game["name"],
                "ranking": game["ranking"],
                "year": game["year"],
                "votes": game["votes"],
                "genre": game["genre"],
            }
        )
        if len(matches) >= limit:
            break
    return matches


@app.get("/models")
def get_model_comparison() -> dict[str, Any]:
    return comparison_report


@app.get("/report")
def get_brief_report() -> FileResponse:
    return FileResponse(
        BRIEF_REPORT_PATH,
        media_type="text/markdown",
        filename=BRIEF_REPORT_PATH.name,
    )


@app.post("/predict/classification")
def predict_genres(request: ClassificationRequest) -> dict[str, Any]:
    features = request.features
    values = np.asarray(
        [[features.ranking, features.year, features.votes]],
        dtype=float,
    )
    predictions = []
    selected_models = (
        model_bundles.items()
        if request.technique == "all"
        else [(request.technique, model_bundles[request.technique])]
    )
    for model_key, bundle in selected_models:
        estimator = bundle["estimator"]
        predicted_genre = str(estimator.predict(values)[0])
        predictions.append(
            {
                "model": model_key,
                "name": bundle["name"],
                "prediction": predicted_genre,
                "metrics": bundle["metrics"],
                "why_used": bundle["why_used"],
                "tradeoff": bundle["tradeoff"],
                "per_class_metrics": bundle["per_class_metrics"],
            }
        )

    return {
        "predictions": predictions,
        "note": (
            "Predictions use ranking, release year, and votes only. "
            "They are estimates, not reliable game metadata."
        ),
    }
