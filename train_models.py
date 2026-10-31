from pathlib import Path

import joblib

from heart_ml.data import load_dataset
from heart_ml.modeling import TECHNIQUES, evaluate_and_train, make_model_bundle
from heart_ml.reporting import write_reports


ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "heart.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("precision_macro", "Macro precision"),
    ("recall_macro", "Macro recall"),
    ("f1_macro", "Macro F1"),
    ("f1_weighted", "Weighted F1"),
    ("roc_auc", "ROC-AUC"),
)


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results, trained_models = evaluate_and_train(dataset)
    write_reports(results, dataset, REPORT_DIR)

    for model_key in TECHNIQUES:
        bundle = make_model_bundle(
            model_key,
            trained_models[model_key],
            results[model_key],
        )
        joblib.dump(bundle, MODEL_DIR / f"{model_key}.joblib", compress=3)

    print(
        f"Rows read: {dataset.audit['rows_read']}; "
        f"usable: {dataset.audit['rows_usable']}; "
        f"excluded: {dataset.audit['rows_skipped_blank']} blank, "
        f"{dataset.audit['rows_skipped_incomplete']} incomplete, "
        f"{dataset.audit['rows_skipped_invalid']} invalid."
    )
    print(
        "Zero-value missing fields imputed inside training folds: "
        f"RestingBP={dataset.audit['missing_values_imputed']['RestingBP']}, "
        f"Cholesterol={dataset.audit['missing_values_imputed']['Cholesterol']}."
    )
    print("Pooled out-of-fold metrics (5-fold stratified cross-validation):")
    header = f"{'Model':<27}" + "".join(f"{label:>19}" for _, label in METRICS)
    print(header)
    for model_key in ("baseline", *TECHNIQUES.keys()):
        result = results[model_key]
        metrics = result["out_of_fold_metrics"]
        row = f"{result['name']:<27}" + "".join(
            f"{metrics[key]:>19.3f}" for key, _ in METRICS
        )
        print(row)
    print("Per-fold mean +/- standard deviation:")
    print(f"{'Model':<27}{'Macro F1':>24}{'ROC-AUC':>24}")
    for model_key in ("baseline", *TECHNIQUES.keys()):
        result = results[model_key]
        summary = result["cross_validation_mean_std"]
        print(
            f"{result['name']:<27}"
            f"{summary['f1_macro']['mean']:.3f} +/- {summary['f1_macro']['std']:.3f}"
            f"{summary['roc_auc']['mean']:>12.3f} +/- "
            f"{summary['roc_auc']['std']:.3f}"
        )
    print(f"Saved trained model bundles to {MODEL_DIR}")
    print(f"Saved comparison metrics and project report to {REPORT_DIR}")


if __name__ == "__main__":
    main()
