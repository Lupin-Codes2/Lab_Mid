import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from game_ml.data import FEATURE_COLUMNS, HeartDataset, load_dataset
from game_ml.modeling import TECHNIQUES


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
STATIC_DIR = ROOT_DIR / "static"
DATASET_PATH = ROOT_DIR / "heart.csv"
REPORT_PATH = REPORT_DIR / "model_comparison.json"
BRIEF_REPORT_PATH = REPORT_DIR / "lab_midterm_report.md"


def load_model_bundle(model_key: str) -> dict[str, Any]:
    model_path = MODEL_DIR / f"{model_key}.joblib"
    if not model_path.is_file():
        raise RuntimeError(
            f"Model file not found: {model_path}. Run `python train_models.py` "
            "to generate the heart-disease models."
        )
    return joblib.load(model_path)


def load_comparison_report() -> dict[str, Any]:
    if not REPORT_PATH.is_file():
        raise RuntimeError(
            f"Model report not found: {REPORT_PATH}. "
            "Run `python train_models.py` to generate it."
        )
    with REPORT_PATH.open(encoding="utf-8") as report_file:
        return json.load(report_file)


heart_dataset: HeartDataset = load_dataset(DATASET_PATH)
model_bundles = {
    model_key: load_model_bundle(model_key)
    for model_key in TECHNIQUES
}
comparison_report = load_comparison_report()

app = FastAPI(title="Heart Disease Classification Lab", version="3.0.0")
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


class HeartFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Age: int = Field(strict=True, ge=18, le=120)
    Sex: Literal["F", "M"]
    ChestPainType: Literal["ASY", "ATA", "NAP", "TA"]
    RestingBP: int = Field(strict=True, ge=0, le=300)
    Cholesterol: int = Field(strict=True, ge=0, le=1000)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["LVH", "Normal", "ST"]
    MaxHR: int = Field(strict=True, ge=1, le=250)
    ExerciseAngina: Literal["N", "Y"]
    Oldpeak: float = Field(ge=-10, le=20)
    ST_Slope: Literal["Down", "Flat", "Up"]


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: HeartFeatures
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


@app.get("/models")
def get_model_comparison() -> dict[str, Any]:
    return comparison_report


@app.get("/report")
def get_brief_report() -> FileResponse:
    if not BRIEF_REPORT_PATH.is_file():
        raise RuntimeError(
            f"Project report not found: {BRIEF_REPORT_PATH}. "
            "Run `python train_models.py` to generate it."
        )
    return FileResponse(
        BRIEF_REPORT_PATH,
        media_type="text/markdown",
        filename=BRIEF_REPORT_PATH.name,
    )


@app.post("/predict/classification")
def predict_classification(request: ClassificationRequest) -> dict[str, Any]:
    feature_data = request.features.model_dump()
    feature_values: list[float | str] = [
        feature_data[column] for column in FEATURE_COLUMNS
    ]
    for column in ("RestingBP", "Cholesterol"):
        index = FEATURE_COLUMNS.index(column)
        if feature_values[index] == 0:
            feature_values[index] = float("nan")
    values = np.asarray([feature_values], dtype=object)
    selected_models = (
        model_bundles.items()
        if request.technique == "all"
        else [(request.technique, model_bundles[request.technique])]
    )
    predictions: list[dict[str, Any]] = []
    for model_key, bundle in selected_models:
        estimator = bundle["estimator"]
        predicted_class = int(estimator.predict(values)[0])
        class_probabilities = estimator.predict_proba(values)[0]
        positive_index = list(estimator.classes_).index(1)
        predictions.append(
            {
                "model": model_key,
                "name": bundle["name"],
                "prediction": predicted_class,
                "probability": float(class_probabilities[positive_index]),
                "class_probabilities": {
                    str(int(label)): float(probability)
                    for label, probability in zip(
                        estimator.classes_,
                        class_probabilities,
                    )
                },
                "metrics": bundle["metrics"],
                "cross_validation_mean_std": bundle["cross_validation_mean_std"],
                "confusion_matrix": bundle["confusion_matrix"],
                "why_used": bundle["why_used"],
                "tradeoff": bundle["tradeoff"],
                "per_class_metrics": bundle["per_class_metrics"],
            }
        )

    return {
        "predictions": predictions,
        "note": (
            "Educational estimates only. This model is not validated for clinical "
            "use and must not be used to diagnose or make treatment decisions."
        ),
    }
