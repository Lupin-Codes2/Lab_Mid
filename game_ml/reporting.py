import json
from pathlib import Path
from typing import Any

from game_ml.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, HeartDataset
from game_ml.modeling import MODEL_SETTINGS, TECHNIQUES


METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("precision_macro", "Macro precision"),
    ("recall_macro", "Macro recall"),
    ("f1_macro", "Macro F1"),
    ("f1_weighted", "Weighted F1"),
    ("roc_auc", "ROC-AUC"),
)


def _render_markdown(results: dict[str, Any], dataset: HeartDataset) -> str:
    audit = dataset.audit
    evaluated_keys = ("baseline", *TECHNIQUES.keys())
    lines = [
        "# Heart Disease Classification Lab Report",
        "",
        "## Dataset and task",
        "",
        "This educational project predicts the `HeartDisease` 0/1 label from the 11 "
        "measurements and categories in the [Heart Failure Prediction dataset]"
        "(https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) "
        "(Kaggle: `fedesoriano/heart-failure-prediction`). The CSV is used as supplied; "
        "no synthetic records are created.",
        "",
        f"- **File:** `{audit['dataset_name']}`",
        f"- **Rows read / usable:** {audit['rows_read']} / {audit['rows_usable']}",
        f"- **Excluded:** {audit['rows_skipped_blank']} blank, "
        f"{audit['rows_skipped_incomplete']} incomplete, "
        f"{audit['rows_skipped_invalid']} invalid.",
        f"- **Target counts:** no heart disease = {audit['target_counts']['0']}; "
        f"heart disease = {audit['target_counts']['1']}.",
        "- **Inputs:** age, sex, chest-pain type, resting blood pressure, cholesterol, "
        "fasting blood sugar, resting ECG, maximum heart rate, exercise angina, oldpeak, "
        "and ST slope.",
        "",
        "## Preprocessing and evaluation",
        "",
        "- Zero `RestingBP` and `Cholesterol` values are treated as missing. Their "
        "medians are learned from each training fold only and applied to its validation "
        f"fold; {audit['missing_values_imputed']['RestingBP']} blood-pressure and "
        f"{audit['missing_values_imputed']['Cholesterol']} cholesterol zeros were found.",
        "- Numeric features are median-imputed and standardized. Categorical features "
        "are most-frequent-imputed and one-hot encoded. The full preprocessing pipeline "
        "is refit within each fold to avoid leakage.",
        "- Five-fold shuffled `StratifiedKFold` uses random state 42. The class ratio is "
        f"{audit['class_balance_ratio']:.3f}; no oversampling or class weights were used.",
        "- Classifier settings are fixed and documented, not selected on the evaluation "
        "folds; no hyperparameter grid search was performed.",
        "- Reported pooled out-of-fold metrics use predictions for held-out rows. "
        "Mean ± standard deviation summarizes each fold separately. ROC-AUC measures "
        "the ranking of the positive-class probabilities.",
        "",
        "## Pooled out-of-fold model comparison",
        "",
        "| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key in evaluated_keys:
        result = results[model_key]
        metrics = result["out_of_fold_metrics"]
        lines.append(
            f"| {result['name']} | "
            + " | ".join(f"{metrics[key]:.3f}" for key, _ in METRICS)
            + " |"
        )

    lines.extend(
        [
            "",
            "### Fold mean ± standard deviation",
            "",
            "| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC-AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_key in evaluated_keys:
        result = results[model_key]
        fold_summary = result["cross_validation_mean_std"]
        cells = [
            f"{fold_summary[key]['mean']:.3f} ± {fold_summary[key]['std']:.3f}"
            for key, _ in METRICS
        ]
        lines.append(f"| {result['name']} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "### Fixed classifier settings",
            "",
            "| Technique | Settings |",
            "|---|---|",
        ]
    )
    for model_key, technique in TECHNIQUES.items():
        settings = json.dumps(MODEL_SETTINGS[model_key], separators=(",", ":"))
        lines.append(f"| {technique['name']} | `{settings}` |")

    lines.extend(["", "## Class metrics and confusion matrices", ""])
    for model_key in evaluated_keys:
        result = results[model_key]
        report = result["classification_report"]
        lines.extend(
            [
                f"### {result['name']}",
                "",
                "Confusion-matrix order: actual rows and predicted columns `[0, 1]`.",
                "",
                f"Confusion matrix: `{result['confusion_matrix']}`",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
                f"| 0 — No heart disease | {report['0']['precision']:.3f} | "
                f"{report['0']['recall']:.3f} | {report['0']['f1-score']:.3f} | "
                f"{int(report['0']['support'])} |",
                f"| 1 — Heart disease | {report['1']['precision']:.3f} | "
                f"{report['1']['recall']:.3f} | {report['1']['f1-score']:.3f} | "
                f"{int(report['1']['support'])} |",
                "",
            ]
        )

    lines.extend(["## Why these techniques", ""])
    for model_key, technique in TECHNIQUES.items():
        result = results[model_key]
        lines.extend(
            [
                f"### {technique['name']}",
                "",
                technique["why_used"],
                "",
                f"**Tradeoff:** {technique['tradeoff']}",
                "",
                f"**Measured pooled out-of-fold macro F1 / ROC-AUC:** "
                f"{result['out_of_fold_metrics']['f1_macro']:.3f} / "
                f"{result['out_of_fold_metrics']['roc_auc']:.3f}.",
                "",
            ]
        )

    macro_f1_ranking = sorted(
        TECHNIQUES,
        key=lambda key: results[key]["out_of_fold_metrics"]["f1_macro"],
    )
    best_model_key = macro_f1_ranking[-1]
    runner_up_key = macro_f1_ranking[-2]
    best_result = results[best_model_key]
    runner_up_result = results[runner_up_key]
    baseline_f1 = results["baseline"]["out_of_fold_metrics"]["f1_macro"]
    macro_f1_gap = (
        best_result["out_of_fold_metrics"]["f1_macro"]
        - runner_up_result["out_of_fold_metrics"]["f1_macro"]
    )
    lines.extend(
        [
            "## Findings and limitations",
            "",
            f"- **Highest pooled out-of-fold macro F1:** {best_result['name']} "
            f"({best_result['out_of_fold_metrics']['f1_macro']:.3f}); majority baseline "
            f"macro F1 is {baseline_f1:.3f}. The ROC-AUC leader can differ from the "
            "macro-F1 leader, so no single metric tells the whole story.",
            f"- One possible reason {best_result['name']} ranked highest is its design: "
            f"{TECHNIQUES[best_model_key]['description']} The measured comparison does not "
            "establish that this is the cause. Its pooled ROC-AUC was "
            f"{best_result['out_of_fold_metrics']['roc_auc']:.3f}; the macro-F1 gap over "
            f"{runner_up_result['name']} was only {macro_f1_gap:.3f}, with overlapping fold "
            "variation. This comparison does not establish statistical or clinical "
            "superiority.",
            "- The dataset has 918 records and represents a limited sample. Fold-to-fold "
            "variation, data provenance, selection bias, and the absence of external "
            "validation limit generalization.",
            "- Imputation of zero cholesterol and blood-pressure values is a stated "
            "preprocessing assumption; zero does not reveal the true measurement.",
            "- This project is for education only. It is not a medical device, is not "
            "validated for clinical use, and must not be used for diagnosis or treatment.",
            "",
            "## Reproducing the project",
            "",
            "On Windows, install `requirements.txt`, run `python train_models.py`, then "
            "start the service with `uvicorn app:app --reload`. Render runs the same "
            "training script in its build command before starting `uvicorn app:app`.",
            f"**Feature types:** {', '.join(NUMERIC_COLUMNS)} are numeric; "
            f"{', '.join(CATEGORICAL_COLUMNS)} are categorical.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    results: dict[str, Any],
    dataset: HeartDataset,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "dataset": dataset.audit,
        "baseline": results["baseline"],
        "models": {
            key: results[key]
            for key in TECHNIQUES
        },
        "evaluation": results["evaluation"],
        "model_settings": MODEL_SETTINGS,
    }
    with (report_dir / "model_comparison.json").open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(comparison, json_file, indent=2)
    (report_dir / "lab_midterm_report.md").write_text(
        _render_markdown(results, dataset),
        encoding="utf-8",
    )
