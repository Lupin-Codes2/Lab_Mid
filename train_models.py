from pathlib import Path

import joblib

from game_ml.data import load_dataset
from game_ml.modeling import TECHNIQUES, evaluate_and_train, make_model_bundle
from game_ml.reporting import write_reports


ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "Top 1000 games of all time.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"


def main() -> None:
    dataset = load_dataset(DATASET_PATH)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results, trained_models = evaluate_and_train(dataset)
    write_reports(results, dataset, REPORT_DIR)

    for model_key in TECHNIQUES:
        bundle = make_model_bundle(
            model_key,
            trained_models[model_key],
            results[model_key],
            dataset,
        )
        joblib.dump(bundle, MODEL_DIR / f"{model_key}.joblib")

    print(f"Usable records: {dataset.audit['rows_usable']} / {dataset.audit['rows_read']}")
    print(
        "Skipped rows: "
        f"{dataset.audit['rows_skipped_blank']} blank, "
        f"{dataset.audit['rows_skipped_incomplete']} incomplete, "
        f"{dataset.audit['rows_skipped_invalid']} invalid"
    )
    print("Five-fold stratified group cross-validation (pooled out-of-fold metrics):")
    baseline_metrics = results["baseline"]["out_of_fold_metrics"]
    print(
        f"  Most-frequent baseline: "
        f"accuracy={baseline_metrics['accuracy']:.4f}, "
        f"balanced_accuracy={baseline_metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={baseline_metrics['f1_macro']:.4f}"
    )
    for model_key in TECHNIQUES:
        metrics = results[model_key]["out_of_fold_metrics"]
        print(
            f"  {TECHNIQUES[model_key]['name']}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['f1_macro']:.4f}"
        )
    print(f"Saved model bundles to {MODEL_DIR}")
    print(f"Saved comparison and brief report to {REPORT_DIR}")


if __name__ == "__main__":
    main()
