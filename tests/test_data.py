import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from game_ml.data import FEATURE_COLUMNS, load_dataset


ROOT_DIR = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_project_dataset_is_audited_and_usable(self) -> None:
        dataset = load_dataset(ROOT_DIR / "heart.csv")

        self.assertEqual(dataset.audit["rows_read"], 918)
        self.assertEqual(dataset.audit["rows_usable"], 918)
        self.assertEqual(dataset.audit["feature_count"], 11)
        self.assertEqual(dataset.audit["target_counts"], {"0": 410, "1": 508})
        self.assertEqual(dataset.features.shape, (918, len(FEATURE_COLUMNS)))
        self.assertEqual(dataset.labels.shape, (918,))
        self.assertEqual(
            dataset.audit["rows_read"],
            dataset.audit["rows_usable"]
            + dataset.audit["rows_skipped_blank"]
            + dataset.audit["rows_skipped_incomplete"]
            + dataset.audit["rows_skipped_invalid"],
        )
        self.assertEqual(dataset.audit["missing_values_imputed"]["RestingBP"], 1)
        self.assertEqual(dataset.audit["missing_values_imputed"]["Cholesterol"], 172)

    def test_zero_measurements_are_nan_for_fold_local_imputation(self) -> None:
        dataset = load_dataset(ROOT_DIR / "heart.csv")
        resting_bp_index = FEATURE_COLUMNS.index("RestingBP")
        cholesterol_index = FEATURE_COLUMNS.index("Cholesterol")

        self.assertTrue(np.isnan(dataset.features[:, resting_bp_index].astype(float)).any())
        self.assertEqual(
            int(np.isnan(dataset.features[:, cholesterol_index].astype(float)).sum()),
            172,
        )

    def test_incomplete_and_invalid_rows_are_counted(self) -> None:
        records = [
            [40, "M", "ATA", 140, 289, 0, "Normal", 172, "N", 0, "Up", 0],
            [49, "F", "NAP", 160, 180, 0, "Normal", 156, "N", 1, "Flat", 1],
            [""] * 12,
            [54, "M", "ATA", 0, 0, 0, "Normal", 140, "N", 0, "Up", ""],
            [54, "M", "ATA", 301, 208, 0, "Normal", 142, "N", 0, "Up", 1],
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "heart.csv"
            with path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([*FEATURE_COLUMNS, "HeartDisease"])
                writer.writerows(records)

            dataset = load_dataset(path)

        self.assertEqual(dataset.audit["rows_read"], 5)
        self.assertEqual(dataset.audit["rows_usable"], 2)
        self.assertEqual(dataset.audit["rows_skipped_blank"], 1)
        self.assertEqual(dataset.audit["rows_skipped_incomplete"], 1)
        self.assertEqual(dataset.audit["rows_skipped_invalid"], 1)


if __name__ == "__main__":
    unittest.main()
