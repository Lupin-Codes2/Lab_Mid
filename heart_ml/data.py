import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_COLUMNS = (
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
)
TARGET_COLUMN = "HeartDisease"
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
NUMERIC_COLUMNS = (
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
)
CATEGORICAL_COLUMNS = (
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
)
CATEGORY_VALUES = {
    "Sex": {"F", "M"},
    "ChestPainType": {"ASY", "ATA", "NAP", "TA"},
    "RestingECG": {"LVH", "Normal", "ST"},
    "ExerciseAngina": {"N", "Y"},
    "ST_Slope": {"Down", "Flat", "Up"},
}
NUMERIC_RANGES = {
    "Age": (18.0, 120.0),
    "RestingBP": (0.0, 300.0),
    "Cholesterol": (0.0, 1000.0),
    "FastingBS": (0.0, 1.0),
    "MaxHR": (1.0, 250.0),
    "Oldpeak": (-10.0, 20.0),
}


@dataclass(frozen=True)
class HeartDataset:
    features: np.ndarray
    labels: np.ndarray
    audit: dict[str, Any]


def _validate_numeric(name: str, raw_value: str) -> float:
    value = float(raw_value)
    lower, upper = NUMERIC_RANGES[name]
    if not lower <= value <= upper:
        raise ValueError(f"{name} value outside the accepted range")
    if name == "FastingBS" and value not in (0.0, 1.0):
        raise ValueError("FastingBS must be 0 or 1")
    return value


def load_dataset(path: Path) -> HeartDataset:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    feature_rows: list[list[float | str]] = []
    labels: list[int] = []
    rows_read = 0
    skipped_blank = 0
    skipped_incomplete = 0
    skipped_invalid = 0
    missing_resting_bp = 0
    missing_cholesterol = 0

    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no CSV header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Dataset contains duplicate column names: {path}")

        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {', '.join(missing_columns)}"
            )

        for row in reader:
            rows_read += 1
            values = [(row.get(column) or "").strip() for column in REQUIRED_COLUMNS]
            if not any(values):
                skipped_blank += 1
                continue
            if not all(values):
                skipped_incomplete += 1
                continue

            try:
                parsed: dict[str, float | str] = {}
                for column in NUMERIC_COLUMNS:
                    value = _validate_numeric(column, row[column].strip())
                    if column == "RestingBP" and value == 0:
                        missing_resting_bp += 1
                        parsed[column] = float("nan")
                    elif column == "Cholesterol" and value == 0:
                        missing_cholesterol += 1
                        parsed[column] = float("nan")
                    else:
                        parsed[column] = value

                for column in CATEGORICAL_COLUMNS:
                    value = row[column].strip()
                    if value not in CATEGORY_VALUES[column]:
                        raise ValueError(f"Unexpected {column} category")
                    parsed[column] = value

                label = int(row[TARGET_COLUMN].strip())
                if label not in (0, 1):
                    raise ValueError("HeartDisease target must be 0 or 1")
            except (TypeError, ValueError):
                skipped_invalid += 1
                continue

            feature_rows.append([parsed[column] for column in FEATURE_COLUMNS])
            labels.append(label)

    if not feature_rows:
        raise ValueError(f"No usable heart records found in {path}")
    if set(labels) != {0, 1}:
        raise ValueError("HeartDisease must contain both target classes 0 and 1")

    numeric_indices = [FEATURE_COLUMNS.index(column) for column in NUMERIC_COLUMNS]
    numeric_values = np.asarray(feature_rows, dtype=object)[:, numeric_indices].astype(float)
    target_counts = Counter(labels)
    audit = {
        "dataset_name": path.name,
        "rows_read": rows_read,
        "rows_usable": len(feature_rows),
        "rows_skipped_blank": skipped_blank,
        "rows_skipped_incomplete": skipped_incomplete,
        "rows_skipped_invalid": skipped_invalid,
        "missing_values_imputed": {
            "RestingBP": missing_resting_bp,
            "Cholesterol": missing_cholesterol,
        },
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "target_counts": {str(key): target_counts[key] for key in (0, 1)},
        "numeric_ranges": {
            column: [
                float(np.nanmin(numeric_values[:, index])),
                float(np.nanmax(numeric_values[:, index])),
            ]
            for index, column in enumerate(NUMERIC_COLUMNS)
        },
        "categorical_values": {
            column: dict(Counter(str(row[FEATURE_COLUMNS.index(column)]) for row in feature_rows))
            for column in CATEGORICAL_COLUMNS
        },
        "class_balance_ratio": float(min(target_counts.values()) / max(target_counts.values())),
    }
    return HeartDataset(
        features=np.asarray(feature_rows, dtype=object),
        labels=np.asarray(labels, dtype=int),
        audit=audit,
    )
