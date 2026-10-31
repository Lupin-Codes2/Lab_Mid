# Heart Disease Classification Lab Report

## Dataset and task

This educational project predicts the `HeartDisease` 0/1 label from the 11 measurements and categories in the [Heart Failure Prediction dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle: `fedesoriano/heart-failure-prediction`). The CSV is used as supplied; no synthetic records are created.

- **File:** `heart.csv`
- **Rows read / usable:** 918 / 918
- **Excluded:** 0 blank, 0 incomplete, 0 invalid.
- **Target counts:** no heart disease = 410; heart disease = 508.
- **Inputs:** age, sex, chest-pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise angina, oldpeak, and ST slope.

## Preprocessing and evaluation

- Zero `RestingBP` and `Cholesterol` values are treated as missing. Their medians are learned from each training fold only and applied to its validation fold; 1 blood-pressure and 172 cholesterol zeros were found.
- Numeric features are median-imputed and standardized. Categorical features are most-frequent-imputed and one-hot encoded. The full preprocessing pipeline is refit within each fold to avoid leakage.
- Five-fold shuffled `StratifiedKFold` uses random state 42. The class ratio is 0.807; no oversampling or class weights were used.
- Classifier settings are fixed and documented, not selected on the evaluation folds; no hyperparameter grid search was performed.
- Reported pooled out-of-fold metrics use predictions for held-out rows. Mean ± standard deviation summarizes each fold separately. ROC-AUC measures the ranking of the positive-class probabilities.

## Pooled out-of-fold model comparison

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Most-frequent baseline | 0.553 | 0.500 | 0.277 | 0.500 | 0.356 | 0.394 | 0.500 |
| Decision Tree | 0.809 | 0.807 | 0.807 | 0.807 | 0.807 | 0.809 | 0.879 |
| Gaussian Naive Bayes | 0.846 | 0.845 | 0.844 | 0.845 | 0.845 | 0.847 | 0.910 |
| K-Nearest Neighbors | 0.862 | 0.856 | 0.863 | 0.856 | 0.859 | 0.861 | 0.905 |
| Artificial Neural Network | 0.833 | 0.832 | 0.831 | 0.832 | 0.832 | 0.833 | 0.900 |
| Support Vector Machine | 0.865 | 0.859 | 0.867 | 0.859 | 0.862 | 0.864 | 0.918 |

### Fold mean ± standard deviation

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Most-frequent baseline | 0.553 ± 0.001 | 0.500 ± 0.000 | 0.277 ± 0.001 | 0.500 ± 0.000 | 0.356 ± 0.000 | 0.394 ± 0.001 | 0.500 ± 0.000 |
| Decision Tree | 0.809 ± 0.014 | 0.807 ± 0.015 | 0.809 ± 0.014 | 0.807 ± 0.015 | 0.807 ± 0.014 | 0.809 ± 0.014 | 0.882 ± 0.026 |
| Gaussian Naive Bayes | 0.846 ± 0.014 | 0.846 ± 0.015 | 0.846 ± 0.014 | 0.846 ± 0.015 | 0.845 ± 0.014 | 0.846 ± 0.014 | 0.912 ± 0.018 |
| K-Nearest Neighbors | 0.862 ± 0.027 | 0.856 ± 0.030 | 0.866 ± 0.026 | 0.856 ± 0.030 | 0.858 ± 0.028 | 0.861 ± 0.028 | 0.908 ± 0.027 |
| Artificial Neural Network | 0.833 ± 0.028 | 0.832 ± 0.028 | 0.832 ± 0.028 | 0.832 ± 0.028 | 0.832 ± 0.028 | 0.833 ± 0.028 | 0.902 ± 0.017 |
| Support Vector Machine | 0.865 ± 0.029 | 0.859 ± 0.031 | 0.868 ± 0.027 | 0.859 ± 0.031 | 0.862 ± 0.030 | 0.864 ± 0.029 | 0.920 ± 0.022 |

### Fixed classifier settings

| Technique | Settings |
|---|---|
| Decision Tree | `{"max_depth":5,"min_samples_leaf":8,"random_state":42}` |
| Gaussian Naive Bayes | `"GaussianNB defaults"` |
| K-Nearest Neighbors | `{"n_neighbors":9,"weights":"distance"}` |
| Artificial Neural Network | `{"hidden_layer_sizes":[24,12],"alpha":0.05,"max_iter":1500,"early_stopping":true,"n_iter_no_change":30,"random_state":42}` |
| Support Vector Machine | `{"C":1.0,"kernel":"rbf","probability":true,"random_state":42}` |

## Class metrics and confusion matrices

### Most-frequent baseline

Confusion-matrix order: actual rows and predicted columns `[0, 1]`.

Confusion matrix: `[[0, 410], [0, 508]]`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 — No heart disease | 0.000 | 0.000 | 0.000 | 410 |
| 1 — Heart disease | 0.553 | 1.000 | 0.712 | 508 |

### Decision Tree

Confusion-matrix order: actual rows and predicted columns `[0, 1]`.

Confusion matrix: `[[321, 89], [86, 422]]`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 — No heart disease | 0.789 | 0.783 | 0.786 | 410 |
| 1 — Heart disease | 0.826 | 0.831 | 0.828 | 508 |

### Gaussian Naive Bayes

Confusion-matrix order: actual rows and predicted columns `[0, 1]`.

Confusion matrix: `[[343, 67], [74, 434]]`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 — No heart disease | 0.823 | 0.837 | 0.830 | 410 |
| 1 — Heart disease | 0.866 | 0.854 | 0.860 | 508 |

### K-Nearest Neighbors

Confusion-matrix order: actual rows and predicted columns `[0, 1]`.

Confusion matrix: `[[331, 79], [48, 460]]`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 — No heart disease | 0.873 | 0.807 | 0.839 | 410 |
| 1 — Heart disease | 0.853 | 0.906 | 0.879 | 508 |

### Artificial Neural Network

Confusion-matrix order: actual rows and predicted columns `[0, 1]`.

Confusion matrix: `[[336, 74], [79, 429]]`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 — No heart disease | 0.810 | 0.820 | 0.815 | 410 |
| 1 — Heart disease | 0.853 | 0.844 | 0.849 | 508 |

### Support Vector Machine

Confusion-matrix order: actual rows and predicted columns `[0, 1]`.

Confusion matrix: `[[331, 79], [45, 463]]`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 — No heart disease | 0.880 | 0.807 | 0.842 | 410 |
| 1 — Heart disease | 0.854 | 0.911 | 0.882 | 508 |

## Why these techniques

### Decision Tree

A depth-limited decision tree can model nonlinear interactions between clinical measurements and encoded symptom categories while keeping its decision structure relatively interpretable.

**Tradeoff:** A single tree can be unstable and may miss patterns outside its limited depth. Its thresholds are educational signals, not clinical rules.

**Measured pooled out-of-fold macro F1 / ROC-AUC:** 0.807 / 0.879.

### Gaussian Naive Bayes

Gaussian Naive Bayes is a lightweight probabilistic classifier and a useful contrast to boundary-based methods. Median imputation, scaling, and one-hot encoding make the mixed inputs usable.

**Tradeoff:** It assumes conditional independence and Gaussian numeric feature distributions, assumptions that are unlikely to fully hold for related health measurements.

**Measured pooled out-of-fold macro F1 / ROC-AUC:** 0.845 / 0.910.

### K-Nearest Neighbors

KNN tests whether nearby records in standardized numeric and one-hot feature space have similar target outcomes. Scaling is essential so large-unit measurements do not dominate distance.

**Tradeoff:** Distance can be less meaningful in a mixed, one-hot feature space; predictions also depend on neighborhood size and the dataset's sampling.

**Measured pooled out-of-fold macro F1 / ROC-AUC:** 0.859 / 0.905.

### Artificial Neural Network

A compact, L2-regularized multilayer perceptron can learn nonlinear interactions across the encoded categorical and scaled numeric inputs.

**Tradeoff:** With fewer than one thousand records, a neural network can be sensitive to initialization and overfit. Cross-validation variation is important alongside its average score.

**Measured pooled out-of-fold macro F1 / ROC-AUC:** 0.832 / 0.900.

### Support Vector Machine

An RBF-kernel support vector machine is a strong small-tabular-data comparison method. It uses scaled and one-hot encoded measurements and probability=True enables a probability estimate for the interface.

**Tradeoff:** The nonlinear boundary is not directly interpretable, and probability estimates are calibrated internally. Cross-validation results do not establish clinical usefulness.

**Measured pooled out-of-fold macro F1 / ROC-AUC:** 0.862 / 0.918.

## Findings and limitations

- **Highest pooled out-of-fold macro F1:** Support Vector Machine (0.862); majority baseline macro F1 is 0.356. The ROC-AUC leader can differ from the macro-F1 leader, so no single metric tells the whole story.
- One possible reason Support Vector Machine ranked highest is its design: An RBF-kernel SVM models nonlinear boundaries and provides class probabilities. The measured comparison does not establish that this is the cause. Its pooled ROC-AUC was 0.918; the macro-F1 gap over K-Nearest Neighbors was only 0.003, with overlapping fold variation. This comparison does not establish statistical or clinical superiority.
- The dataset has 918 records and represents a limited sample. Fold-to-fold variation, data provenance, selection bias, and the absence of external validation limit generalization.
- Imputation of zero cholesterol and blood-pressure values is a stated preprocessing assumption; zero does not reveal the true measurement.
- This project is for education only. It is not a medical device, is not validated for clinical use, and must not be used for diagnosis or treatment.

## Reproducing the project

On Windows, install `requirements.txt`, run `python train_models.py`, then start the service with `uvicorn app:app --reload`. Render runs the same training script in its build command before starting `uvicorn app:app`.
**Feature types:** Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak are numeric; Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope are categorical.
