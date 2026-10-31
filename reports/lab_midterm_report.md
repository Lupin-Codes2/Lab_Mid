# Lab Midterm Project Report

## Scope

This project applies the five techniques listed before the midterm in the lab manual: Decision Tree, Naive Bayes, K-Nearest Neighbors, Artificial Neural Network, and Support Vector Machine. Activities/examples from the slides were intentionally not reproduced. K-Means, PCA, Genetic Algorithms, Bagging, and Reinforcement Learning are listed after the midterm in the manual and are not included.

## Dataset inspection

- **File:** `Top 1000 games of all time.csv`
- **Rows read:** 1100
- **Usable records:** 996
- **Blank rows skipped:** 100
- **Incomplete rows skipped:** 3
- **Invalid rows skipped:** 1
- **Duplicate title rows retained:** 21 (they remain separate ranked entries; same titles are grouped into one CV fold)
- **Ranking range:** 1–999
- **Release-year range:** 1980–2022
- **Vote-count range:** 938–63,580

The target is the first genre tag. Genres with fewer than 15 examples are grouped as `Other` to keep the multiclass comparison supportable. The resulting target distribution is:

| Primary genre | Records |
|---|---:|
| Action | 836 |
| Adventure | 64 |
| Other | 38 |
| Drama | 21 |
| Sport | 19 |
| Animation | 18 |

## Preprocessing

- Ranking, release year, and votes are the only predictors. Game title and genre tags are not model inputs.
- Rows with blank required values, incomplete records, or invalid numeric values are excluded; comma-separated votes are parsed as integers and the year is extracted from the source text.
- Vote counts are transformed with `log1p`, then all three numeric predictors are standardized inside each model pipeline.
- Minority genres are randomly oversampled **only in each training fold**. Validation data is left at its original distribution.
- Five-fold stratified group cross-validation keeps rows with the same game title in one fold, reducing duplicate-title leakage.

## Model comparison

Scores below are pooled out-of-fold predictions from the five-fold evaluation. Macro F1 and balanced accuracy are emphasized because Action is the dominant class. The most-frequent baseline is included as a reference, not as one of the five required techniques.

| Technique | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Most-frequent baseline | 0.839 | 0.167 | 0.140 | 0.167 | 0.152 | 0.766 |
| Decision Tree | 0.368 | 0.146 | 0.170 | 0.146 | 0.131 | 0.483 |
| Naive Bayes | 0.141 | 0.226 | 0.200 | 0.226 | 0.079 | 0.207 |
| K-Nearest Neighbors | 0.584 | 0.233 | 0.204 | 0.233 | 0.205 | 0.650 |
| Artificial Neural Network | 0.194 | 0.229 | 0.182 | 0.229 | 0.101 | 0.279 |
| Support Vector Machine | 0.266 | 0.244 | 0.195 | 0.244 | 0.132 | 0.374 |

### Cross-validation mean (± standard deviation)

| Technique | Balanced accuracy | Macro F1 |
|---|---:|---:|
| Most-frequent baseline | 0.167 ± 0.000 | 0.152 ± 0.000 |
| Decision Tree | 0.148 ± 0.059 | 0.127 ± 0.029 |
| Naive Bayes | 0.232 ± 0.031 | 0.075 ± 0.009 |
| K-Nearest Neighbors | 0.236 ± 0.073 | 0.206 ± 0.044 |
| Artificial Neural Network | 0.224 ± 0.082 | 0.097 ± 0.014 |
| Support Vector Machine | 0.241 ± 0.075 | 0.131 ± 0.024 |

### Why each technique was included

#### Decision Tree

The release year, popularity rank, and vote count may separate genres through threshold-like interactions. A shallow tree models those splits and is straightforward to interpret on tabular data.

**Tradeoff:** Individual trees have high variance; depth and leaf-size limits reduce overfitting, but group cross-validation still checks generalization.

**Observed:** pooled macro F1 0.131; balanced accuracy 0.146.

#### Naive Bayes

Gaussian Naive Bayes is a fast probabilistic baseline that naturally accepts these numeric predictors and provides a useful contrast with boundary-based classifiers.

**Tradeoff:** It assumes features are conditionally independent within each genre. Rank and votes both reflect popularity, so this assumption is unlikely to hold exactly.

**Observed:** pooled macro F1 0.079; balanced accuracy 0.226.

#### K-Nearest Neighbors

Games with similar ranks, years, and vote counts may have similar primary genres. KNN tests this local-similarity idea directly, and standardization is essential because it uses distances.

**Tradeoff:** Its local decisions are sensitive to class imbalance, the chosen neighborhood size, and whether these three fields meaningfully encode genre similarity.

**Observed:** pooled macro F1 0.205; balanced accuracy 0.233.

#### Artificial Neural Network

A compact multilayer perceptron can model nonlinear interactions between the numeric inputs and covers the neural-network technique listed before the midterm.

**Tradeoff:** With only three predictors and about one thousand usable rows, a network can overfit or fail to learn stable patterns; regularization and grouped cross-validation expose that risk.

**Observed:** pooled macro F1 0.101; balanced accuracy 0.229.

#### Support Vector Machine

An RBF SVM is a standard classifier for small-to-medium numeric datasets and can form nonlinear class boundaries after feature scaling.

**Tradeoff:** Its predictions depend on kernel and regularization choices, and class imbalance can bias decision boundaries even when the features are scaled.

**Observed:** pooled macro F1 0.132; balanced accuracy 0.244.

## Findings and limitations

- **Highest pooled out-of-fold macro F1:** K-Nearest Neighbors (0.205), compared with baseline macro F1 0.152.
- **Interpretation:** K-Nearest Neighbors ranks highest on macro F1, but its score (0.205) and balanced accuracy (0.233) remain modest; none of the tested models demonstrates strong genre prediction from the available inputs.
- The majority baseline and minority-sensitive metrics should be compared together; raw accuracy alone can hide poor performance on smaller genres.
- Results use only three numeric popularity/time/rank signals. Genre is not strongly identifiable from these fields alone, so predictions are educational estimates, not reliable game metadata.
- Oversampling can increase minority recall but also overfit a small dataset. The grouped cross-validation scores are reported to make that tradeoff visible.

## Reproducing the results

Install `requirements.txt`, run `python train_models.py`, then start the service with `uvicorn app:app --reload`. Model bundles and this report are regenerated from the CSV by the training command.
