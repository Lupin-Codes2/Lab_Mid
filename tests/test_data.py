import csv
import tempfile
import unittest
from pathlib import Path

from game_ml.data import load_dataset


ROOT_DIR = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_project_dataset_is_audited_and_usable(self) -> None:
        dataset = load_dataset(ROOT_DIR / "Top 1000 games of all time.csv")

        self.assertGreater(dataset.audit["rows_read"], 1000)
        self.assertGreater(dataset.audit["rows_usable"], 900)
        self.assertEqual(len(dataset.features), len(dataset.labels))
        self.assertEqual(len(dataset.labels), len(dataset.groups))
        self.assertEqual(dataset.features.shape[1], 3)
        self.assertGreater(dataset.audit["rows_skipped_blank"], 0)
        self.assertEqual(
            dataset.audit["rows_read"],
            dataset.audit["rows_usable"]
            + dataset.audit["rows_skipped_blank"]
            + dataset.audit["rows_skipped_incomplete"]
            + dataset.audit["rows_skipped_invalid"],
        )
        self.assertIn("Other", dataset.audit["grouped_genre_counts"])

    def test_incomplete_and_invalid_rows_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "games.csv"
            with path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Ranking", "Name", "Year", "Genre", "Votes"])
                writer.writerow([1, "Game A", "(2001 Video Game)", "Action", "1,200"])
                writer.writerow([2, "Game B", "(2002 Video Game)", "Adventure", "900"])
                writer.writerow(["", "", "", "", ""])
                writer.writerow([3, "Incomplete", "(2003 Video Game)", "", "800"])
                writer.writerow([4, "Invalid", "(2004 Video Game)", "Action", "many"])

            dataset = load_dataset(path)

        self.assertEqual(dataset.audit["rows_read"], 5)
        self.assertEqual(dataset.audit["rows_usable"], 2)
        self.assertEqual(dataset.audit["rows_skipped_blank"], 1)
        self.assertEqual(dataset.audit["rows_skipped_incomplete"], 1)
        self.assertEqual(dataset.audit["rows_skipped_invalid"], 1)
        self.assertEqual(dataset.features[0].tolist(), [1.0, 2001.0, 1200.0])


if __name__ == "__main__":
    unittest.main()
