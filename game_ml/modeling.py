from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from game_ml.data import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    HeartDataset,
)


RANDOM_STATE = 42
CV_FOLDS = 5
NUMERIC_INDICES = [FEATURE_COLUMNS.index(column) for column in NUMERIC_COLUMNS]
CATEGORICAL_INDICES = [
    FEATURE_COLUMNS.index(column) for column in CATEGORICAL_COLUMNS
]

TECHNIQUES = {
    "decision_tree": {
        "name": "Decision Tree",
        "description": "A shallow tree captures threshold-based interactions and is easy to interpret.",
        "why_used": (
            "A depth-limited decision tree can model nonlinear interactions between "
            "clinical measurements and encoded symptom categories while keeping its "
            "decision structure relatively interpretable."
        ),
        "tradeoff": (
            "A single tree can be unstable and may miss patterns outside its limited "
            "depth. Its thresholds are educational signals, not clinical rules."
        ),
    },
    "naive_bayes": {
        "name": "Gaussian Naive Bayes",
        "description": "A probabilistic classifier offers a fast baseline on mixed tabular features.",
        "why_used": (
            "Gaussian Naive Bayes is a lightweight probabilistic classifier and a "
            "useful contrast to boundary-based methods. Median imputation, scaling, "
            "and one-hot encoding make the mixed inputs usable."
        ),
        "tradeoff": (
            "It assumes conditional independence and Gaussian numeric feature "
            "distributions, assumptions that are unlikely to fully hold for related "
            "health measurements."
        ),
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "description": "Scaled measurements and one-hot categories let distance-weighted neighbors classify records.",
        "why_used": (
            "KNN tests whether nearby records in standardized numeric and one-hot "
            "feature space have similar target outcomes. Scaling is essential so "
            "large-unit measurements do not dominate distance."
        ),
        "tradeoff": (
            "Distance can be less meaningful in a mixed, one-hot feature space; "
            "predictions also depend on neighborhood size and the dataset's sampling."
        ),
    },
    "ann": {
        "name": "Artificial Neural Network",
        "description": "A regularized multilayer perceptron learns nonlinear feature interactions.",
        "why_used": (
            "A compact, L2-regularized multilayer perceptron can learn nonlinear "
            "interactions across the encoded categorical and scaled numeric inputs."
        ),
        "tradeoff": (
            "With fewer than one thousand records, a neural network can be sensitive "
            "to initialization and overfit. Cross-validation variation is important "
            "alongside its average score."
        ),
    },
    "svm": {
        "name": "Support Vector Machine",
        "description": "An RBF-kernel SVM models nonlinear boundaries and provides class probabilities.",
        "why_used": (
            "An RBF-kernel support vector machine is a strong small-tabular-data "
            "comparison method. It uses scaled and one-hot encoded measurements and "
            "probability=True enables a probability estimate for the interface."
        ),
        "tradeoff": (
            "The nonlinear boundary is not directly interpretable, and probability "
            "estimates are calibrated internally. Cross-validation results do not "
            "establish clinical usefulness."
        ),
    },
}

MODEL_SETTINGS = {
    "decision_tree": {
        "max_depth": 5,
        "min_samples_leaf": 8,
        "random_state": RANDOM_STATE,
    },
    "naive_bayes": "GaussianNB defaults",
    "knn": {"n_neighbors": 9, "weights": "distance"},
    "ann": {
        "hidden_layer_sizes": (24, 12),
        "alpha": 0.05,
        "max_iter": 1500,
        "early_stopping": True,
        "n_iter_no_change": 30,
        "random_state": RANDOM_STATE,
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": RANDOM_STATE,
    },
}


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_INDICES),
            ("categorical", categorical_pipeline, CATEGORICAL_INDICES),
        ],
        sparse_threshold=0.0,
    )


def build_model(name: str) -> Pipeline:
    if name not in TECHNIQUES:
        raise ValueError(f"Unknown classifier: {name}")

    classifiers = {
        "decision_tree": DecisionTreeClassifier(
            **MODEL_SETTINGS["decision_tree"],
        ),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(**MODEL_SETTINGS["knn"]),
        "ann": MLPClassifier(**MODEL_SETTINGS["ann"]),
        "svm": SVC(**MODEL_SETTINGS["svm"]),
    }
    return Pipeline(
        [
            ("preprocessing", build_preprocessor()),
            ("classifier", classifiers[name]),
        ]
    )


def _metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probability_positive: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision_macro": float(
            precision_score(actual, predicted, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(actual, predicted, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(actual, predicted, average="weighted", zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(actual, probability_positive)),
    }


def _cv_summary(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([fold[metric] for fold in fold_metrics])),
            "std": float(np.std([fold[metric] for fold in fold_metrics])),
        }
        for metric in fold_metrics[0]
    }


def _evaluate_estimator(
    estimator: Any,
    dataset: HeartDataset,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    out_of_fold_predictions = np.empty_like(dataset.labels)
    out_of_fold_probabilities = np.empty(len(dataset.labels), dtype=float)
    fold_metrics: list[dict[str, float]] = []

    for training_indices, validation_indices in folds:
        model = estimator()
        model.fit(
            dataset.features[training_indices],
            dataset.labels[training_indices],
        )
        predictions = model.predict(dataset.features[validation_indices])
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(
            dataset.features[validation_indices]
        )[:, positive_index]
        out_of_fold_predictions[validation_indices] = predictions
        out_of_fold_probabilities[validation_indices] = probabilities
        fold_metrics.append(
            _metrics(
                dataset.labels[validation_indices],
                predictions,
                probabilities,
            )
        )

    return {
        "out_of_fold_metrics": _metrics(
            dataset.labels,
            out_of_fold_predictions,
            out_of_fold_probabilities,
        ),
        "cross_validation_mean_std": _cv_summary(fold_metrics),
        "confusion_matrix": confusion_matrix(
            dataset.labels,
            out_of_fold_predictions,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            dataset.labels,
            out_of_fold_predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_and_train(
    dataset: HeartDataset,
) -> tuple[dict[str, Any], dict[str, Pipeline]]:
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    folds = list(splitter.split(dataset.features, dataset.labels))
    results: dict[str, Any] = {}

    baseline_result = _evaluate_estimator(
        lambda: DummyClassifier(strategy="most_frequent"),
        dataset,
        folds,
    )
    results["baseline"] = {
        "name": "Most-frequent baseline",
        **baseline_result,
    }

    trained_models: dict[str, Pipeline] = {}
    for model_key, technique in TECHNIQUES.items():
        result = _evaluate_estimator(
            lambda key=model_key: build_model(key),
            dataset,
            folds,
        )
        final_model = build_model(model_key)
        final_model.fit(dataset.features, dataset.labels)
        trained_models[model_key] = final_model
        results[model_key] = {
            "name": technique["name"],
            "description": technique["description"],
            "why_used": technique["why_used"],
            "tradeoff": technique["tradeoff"],
            **result,
        }

    results["evaluation"] = {
        "method": "5-fold stratified cross-validation",
        "folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "class_names": {"0": "No heart disease", "1": "Heart disease"},
        "primary_metric": "f1_macro",
        "feature_columns": list(FEATURE_COLUMNS),
        "model_settings": MODEL_SETTINGS,
        "preprocessing": (
            "Median imputation for numeric values, standard scaling for numeric "
            "features, and most-frequent imputation plus one-hot encoding for "
            "categorical features. All transformations are fitted within each fold."
        ),
    }
    return results, trained_models


def make_model_bundle(
    model_key: str,
    model: Pipeline,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_key": model_key,
        "name": TECHNIQUES[model_key]["name"],
        "why_used": TECHNIQUES[model_key]["why_used"],
        "tradeoff": TECHNIQUES[model_key]["tradeoff"],
        "estimator": model,
        "feature_names": list(FEATURE_COLUMNS),
        "metrics": result["out_of_fold_metrics"],
        "cross_validation_mean_std": result["cross_validation_mean_std"],
        "confusion_matrix": result["confusion_matrix"],
        "per_class_metrics": {
            class_name: result["classification_report"][class_name]
            for class_name in ("0", "1")
        },
    }
