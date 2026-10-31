import json
from pathlib import Path
from typing import Any

from game_ml.data import GameDataset
from game_ml.modeling import TECHNIQUES


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision_macro": "Macro precision",
    "recall_macro": "Macro recall",
    "f1_macro": "Macro F1",
    "f1_weighted": "Weighted F1",
}


def _render_markdown(results: dict[str, Any], dataset: GameDataset) -> str:
    audit = dataset.audit
    technique_results = [
        (key, results[key])
        for key in TECHNIQUES
        if key in results
    ]
    top_model = max(
        technique_results,
        key=lambda item: item[1]["out_of_fold_metrics"]["f1_macro"],
    )
    grouped_counts = audit["grouped_genre_counts"]

    lines = [
        "# Lab Midterm Project Report",
        "",
        "## Scope",
        "",
        "This project applies the five techniques listed before the midterm in the lab manual: "
        "Decision Tree, Naive Bayes, K-Nearest Neighbors, Artificial Neural Network, and "
        "Support Vector Machine. Activities/examples from the slides were intentionally not "
        "reproduced. K-Means, PCA, Genetic Algorithms, Bagging, and Reinforcement Learning "
        "are listed after the midterm in the manual and are not included.",
        "",
        "## Dataset inspection",
        "",
        f"- **File:** `{audit['dataset_name']}`",
        f"- **Rows read:** {audit['rows_read']}",
        f"- **Usable records:** {audit['rows_usable']}",
        f"- **Blank rows skipped:** {audit['rows_skipped_blank']}",
        f"- **Incomplete rows skipped:** {audit['rows_skipped_incomplete']}",
        f"- **Invalid rows skipped:** {audit['rows_skipped_invalid']}",
        f"- **Duplicate title rows retained:** {audit['duplicate_title_rows']} "
        "(they remain separate ranked entries; same titles are grouped into one CV fold)",
        f"- **Ranking range:** {audit['ranking_range'][0]}–{audit['ranking_range'][1]}",
        f"- **Release-year range:** {audit['year_range'][0]}–{audit['year_range'][1]}",
        f"- **Vote-count range:** {audit['votes_range'][0]:,}–{audit['votes_range'][1]:,}",
        "",
        "The target is the first genre tag. Genres with fewer than "
        f"{audit['rare_genre_threshold']} examples are grouped as `Other` to keep the "
        "multiclass comparison supportable. The resulting target distribution is:",
        "",
        "| Primary genre | Records |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {genre} | {count} |" for genre, count in grouped_counts.items()
    )
    lines.extend(
        [
            "",
            "## Preprocessing",
            "",
            "- Ranking, release year, and votes are the only predictors. Game title and genre "
            "tags are not model inputs.",
            "- Rows with blank required values, incomplete records, or invalid numeric values "
            "are excluded; comma-separated votes are parsed as integers and the year is "
            "extracted from the source text.",
            "- Vote counts are transformed with `log1p`, then all three numeric predictors "
            "are standardized inside each model pipeline.",
            "- Minority genres are randomly oversampled **only in each training fold**. "
            "Validation data is left at its original distribution.",
            "- Five-fold stratified group cross-validation keeps rows with the same game title "
            "in one fold, reducing duplicate-title leakage.",
            "",
            "## Model comparison",
            "",
            "Scores below are pooled out-of-fold predictions from the five-fold evaluation. "
            "Macro F1 and balanced accuracy are emphasized because Action is the dominant "
            "class. The most-frequent baseline is included as a reference, not as one of the "
            "five required techniques.",
            "",
            "| Technique | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    baseline = results["baseline"]["out_of_fold_metrics"]
    lines.append(
        "| Most-frequent baseline | "
        + " | ".join(f"{baseline[key]:.3f}" for key in METRIC_LABELS)
        + " |"
    )
    for model_key, result in technique_results:
        metrics = result["out_of_fold_metrics"]
        lines.append(
            f"| {TECHNIQUES[model_key]['name']} | "
            + " | ".join(f"{metrics[key]:.3f}" for key in METRIC_LABELS)
            + " |"
        )
    lines.extend(
        [
            "",
            "### Cross-validation mean (± standard deviation)",
            "",
            "| Technique | Balanced accuracy | Macro F1 |",
            "|---|---:|---:|",
        ]
    )
    lines.append(
        "| Most-frequent baseline | "
        f"{results['baseline']['cross_validation_mean_std']['balanced_accuracy']['mean']:.3f} "
        f"± {results['baseline']['cross_validation_mean_std']['balanced_accuracy']['std']:.3f} | "
        f"{results['baseline']['cross_validation_mean_std']['f1_macro']['mean']:.3f} "
        f"± {results['baseline']['cross_validation_mean_std']['f1_macro']['std']:.3f} |"
    )
    for model_key, result in technique_results:
        summary = result["cross_validation_mean_std"]
        lines.append(
            f"| {TECHNIQUES[model_key]['name']} | "
            f"{summary['balanced_accuracy']['mean']:.3f} "
            f"± {summary['balanced_accuracy']['std']:.3f} | "
            f"{summary['f1_macro']['mean']:.3f} "
            f"± {summary['f1_macro']['std']:.3f} |"
        )
    _, best_result = top_model
    baseline_macro_f1 = baseline["f1_macro"]
    best_macro_f1 = best_result["out_of_fold_metrics"]["f1_macro"]
    lines.extend(
        [
            "",
            "### Why each technique was included",
            "",
        ]
    )
    for model_key, result in technique_results:
        lines.extend(
            [
                f"#### {result['name']}",
                "",
                result["why_used"],
                "",
                f"**Tradeoff:** {result['tradeoff']}",
                "",
                f"**Observed:** pooled macro F1 "
                f"{result['out_of_fold_metrics']['f1_macro']:.3f}; "
                f"balanced accuracy "
                f"{result['out_of_fold_metrics']['balanced_accuracy']:.3f}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Findings and limitations",
            "",
            f"- **Highest pooled out-of-fold macro F1:** {best_result['name']} "
            f"({best_macro_f1:.3f}), compared with baseline macro F1 "
            f"{baseline_macro_f1:.3f}.",
            f"- **Interpretation:** {best_result['name']} ranks highest on macro F1, "
            f"but its score ({best_macro_f1:.3f}) and balanced accuracy "
            f"({best_result['out_of_fold_metrics']['balanced_accuracy']:.3f}) remain "
            "modest; none of the tested models demonstrates strong genre prediction "
            "from the available inputs.",
            "- The majority baseline and minority-sensitive metrics should be compared "
            "together; raw accuracy alone can hide poor performance on smaller genres.",
            "- Results use only three numeric popularity/time/rank signals. Genre is not "
            "strongly identifiable from these fields alone, so predictions are educational "
            "estimates, not reliable game metadata.",
            "- Oversampling can increase minority recall but also overfit a small dataset. "
            "The grouped cross-validation scores are reported to make that tradeoff visible.",
            "",
            "## Reproducing the results",
            "",
            "Install `requirements.txt`, run `python train_models.py`, then start the service "
            "with `uvicorn app:app --reload`. Model bundles and this report are regenerated "
            "from the CSV by the training command.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    results: dict[str, Any],
    dataset: GameDataset,
    report_directory: Path,
) -> None:
    report_directory.mkdir(parents=True, exist_ok=True)
    comparison = {
        "dataset": dataset.audit,
        "evaluation": results["evaluation"],
        "baseline": results["baseline"],
        "models": {
            key: results[key]
            for key in TECHNIQUES
        },
    }
    (report_directory / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    (report_directory / "lab_midterm_report.md").write_text(
        _render_markdown(results, dataset),
        encoding="utf-8",
    )
