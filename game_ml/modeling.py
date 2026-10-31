from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from game_ml.data import GameDataset


RANDOM_STATE = 42
CV_FOLDS = 5

TECHNIQUES = {
    "decision_tree": {
        "name": "Decision Tree",
        "description": "A bounded-depth tree captures nonlinear feature splits.",
        "why_used": (
            "The release year, popularity rank, and vote count may separate genres "
            "through threshold-like interactions. A shallow tree models those splits "
            "and is straightforward to interpret on tabular data."
        ),
        "tradeoff": (
            "Individual trees have high variance; depth and leaf-size limits reduce "
            "overfitting, but group cross-validation still checks generalization."
        ),
    },
    "naive_bayes": {
        "name": "Naive Bayes",
        "description": "Gaussian Naive Bayes models numeric features per genre.",
        "why_used": (
            "Gaussian Naive Bayes is a fast probabilistic baseline that naturally "
            "accepts these numeric predictors and provides a useful contrast with "
            "boundary-based classifiers."
        ),
        "tradeoff": (
            "It assumes features are conditionally independent within each genre. "
            "Rank and votes both reflect popularity, so this assumption is unlikely "
            "to hold exactly."
        ),
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "description": "Distance-weighted neighbors use scaled numeric features.",
        "why_used": (
            "Games with similar ranks, years, and vote counts may have similar "
            "primary genres. KNN tests this local-similarity idea directly, and "
            "standardization is essential because it uses distances."
        ),
        "tradeoff": (
            "Its local decisions are sensitive to class imbalance, the chosen "
            "neighborhood size, and whether these three fields meaningfully encode "
            "genre similarity."
        ),
    },
    "ann": {
        "name": "Artificial Neural Network",
        "description": "A regularized multilayer perceptron learns nonlinear patterns.",
        "why_used": (
            "A compact multilayer perceptron can model nonlinear interactions "
            "between the numeric inputs and covers the neural-network technique "
            "listed before the midterm."
        ),
        "tradeoff": (
            "With only three predictors and about one thousand usable rows, a "
            "network can overfit or fail to learn stable patterns; regularization "
            "and grouped cross-validation expose that risk."
        ),
    },
    "svm": {
        "name": "Support Vector Machine",
        "description": "An RBF-kernel SVM learns nonlinear class boundaries.",
        "why_used": (
            "An RBF SVM is a standard classifier for small-to-medium numeric "
            "datasets and can form nonlinear class boundaries after feature scaling."
        ),
        "tradeoff": (
            "Its predictions depend on kernel and regularization choices, and class "
            "imbalance can bias decision boundaries even when the features are scaled."
        ),
    },
}


def _log_vote_feature_transform(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float).copy()
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected feature columns in ranking, year, votes order.")
    if np.any(values[:, 2] < 0):
        raise ValueError("Vote counts must be non-negative.")
    values[:, 2] = np.log1p(values[:, 2])
    return values


class BalancedTrainingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator: Any, random_state: int = RANDOM_STATE) -> None:
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "BalancedTrainingClassifier":
        values = np.asarray(features)
        self.target_encoder_ = LabelEncoder().fit(labels)
        target = self.target_encoder_.transform(labels)
        classes, counts = np.unique(target, return_counts=True)
        if len(classes) < 2:
            raise ValueError("At least two target classes are required for training.")

        rng = np.random.default_rng(self.random_state)
        balanced_indices = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(target == label),
                    size=int(counts.max()),
                    replace=count < counts.max(),
                )
                for label, count in zip(classes, counts)
            ]
        )
        rng.shuffle(balanced_indices)
        self.estimator_ = clone(self.estimator).fit(
            values[balanced_indices],
            target[balanced_indices],
        )
        self.classes_ = self.target_encoder_.classes_
        self.n_features_in_ = values.shape[1]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        encoded_predictions = self.estimator_.predict(features)
        return self.target_encoder_.inverse_transform(encoded_predictions)

def build_model(name: str) -> Pipeline:
    if name not in TECHNIQUES:
        raise ValueError(f"Unknown classifier: {name}")

    estimators = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "ann": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.01,
            max_iter=400,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        "svm": SVC(
            C=1.0,
            kernel="rbf",
            random_state=RANDOM_STATE,
        ),
    }
    return Pipeline(
        [
            (
                "feature_engineering",
                FunctionTransformer(_log_vote_feature_transform, validate=False),
            ),
            ("scaling", StandardScaler()),
            (
                "classifier",
                BalancedTrainingClassifier(
                    estimator=estimators[name],
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "precision_macro": float(
            precision_score(
                actual,
                predicted,
                labels=class_names,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                actual,
                predicted,
                labels=class_names,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                actual,
                predicted,
                labels=class_names,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                actual,
                predicted,
                labels=class_names,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def _cross_validation_summary(
    folds: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        metric_name: {
            "mean": float(np.mean([fold[metric_name] for fold in folds])),
            "std": float(np.std([fold[metric_name] for fold in folds])),
        }
        for metric_name in folds[0]
    }


def evaluate_and_train(
    dataset: GameDataset,
) -> tuple[dict[str, Any], dict[str, Pipeline]]:
    class_names = sorted(np.unique(dataset.labels).tolist())
    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_indices = list(
        splitter.split(dataset.features, dataset.labels, groups=dataset.groups)
    )
    results: dict[str, Any] = {}
    trained_models: dict[str, Pipeline] = {}

    baseline_predictions = np.empty_like(dataset.labels)
    baseline_folds: list[dict[str, float]] = []
    for training_indices, validation_indices in fold_indices:
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(
            dataset.features[training_indices],
            dataset.labels[training_indices],
        )
        predictions = baseline.predict(dataset.features[validation_indices])
        baseline_predictions[validation_indices] = predictions
        baseline_folds.append(
            _metrics(
                dataset.labels[validation_indices],
                predictions,
                class_names,
            )
        )
    results["baseline"] = {
        "name": "Most-frequent baseline",
        "cross_validation_mean_std": _cross_validation_summary(baseline_folds),
        "out_of_fold_metrics": _metrics(
            dataset.labels,
            baseline_predictions,
            class_names,
        ),
    }

    for model_key, technique in TECHNIQUES.items():
        out_of_fold_predictions = np.empty_like(dataset.labels)
        fold_metrics: list[dict[str, float]] = []
        for training_indices, validation_indices in fold_indices:
            model = build_model(model_key)
            model.fit(
                dataset.features[training_indices],
                dataset.labels[training_indices],
            )
            predictions = model.predict(dataset.features[validation_indices])
            out_of_fold_predictions[validation_indices] = predictions
            fold_metrics.append(
                _metrics(
                    dataset.labels[validation_indices],
                    predictions,
                    class_names,
                )
            )

        final_model = build_model(model_key)
        final_model.fit(dataset.features, dataset.labels)
        trained_models[model_key] = final_model
        results[model_key] = {
            "name": technique["name"],
            "description": technique["description"],
            "why_used": technique["why_used"],
            "tradeoff": technique["tradeoff"],
            "cross_validation_mean_std": _cross_validation_summary(fold_metrics),
            "out_of_fold_metrics": _metrics(
                dataset.labels,
                out_of_fold_predictions,
                class_names,
            ),
            "confusion_matrix": confusion_matrix(
                dataset.labels,
                out_of_fold_predictions,
                labels=class_names,
            ).tolist(),
            "classification_report": classification_report(
                dataset.labels,
                out_of_fold_predictions,
                labels=class_names,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            ),
        }

    results["evaluation"] = {
        "method": "5-fold stratified group cross-validation",
        "folds": CV_FOLDS,
        "grouped_by": "case-insensitive game title",
        "random_state": RANDOM_STATE,
        "training_imbalance_handling": (
            "Random oversampling of minority genres within each training fold only."
        ),
        "class_names": class_names,
        "primary_metric": "f1_macro",
    }
    return results, trained_models


def make_model_bundle(
    model_key: str,
    model: Pipeline,
    result: dict[str, Any],
    dataset: GameDataset,
) -> dict[str, Any]:
    return {
        "model_key": model_key,
        "name": TECHNIQUES[model_key]["name"],
        "why_used": TECHNIQUES[model_key]["why_used"],
        "tradeoff": TECHNIQUES[model_key]["tradeoff"],
        "estimator": model,
        "feature_names": list(("ranking", "year", "votes")),
        "class_names": sorted(np.unique(dataset.labels).tolist()),
        "metrics": result["out_of_fold_metrics"],
        "per_class_metrics": {
            class_name: result["classification_report"][class_name]
            for class_name in sorted(np.unique(dataset.labels).tolist())
        },
    }
