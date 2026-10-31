import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = ("ranking", "year", "votes")
REQUIRED_COLUMNS = ("Ranking", "Name", "Year", "Genre", "Votes")
MIN_GENRE_SAMPLES = 15
YEAR_PATTERN = re.compile(r"\d{4}")


@dataclass(frozen=True)
class GameDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    records: list[dict[str, Any]]
    audit: dict[str, Any]


def _parse_integer(value: str) -> int:
    return int(value.replace(",", "").strip())


def load_dataset(path: Path) -> GameDataset:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: list[dict[str, Any]] = []
    skipped_blank = 0
    skipped_incomplete = 0
    skipped_invalid = 0

    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no CSV header: {path}")

        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {', '.join(missing_columns)}"
            )

        for row in reader:
            values = [(row.get(column) or "").strip() for column in REQUIRED_COLUMNS]
            if not any(values):
                skipped_blank += 1
                continue
            if not all(values):
                skipped_incomplete += 1
                continue

            try:
                ranking = _parse_integer(row["Ranking"])
                votes = _parse_integer(row["Votes"])
                year_match = YEAR_PATTERN.search(row["Year"])
                primary_genre = row["Genre"].split(",", maxsplit=1)[0].strip()
                if year_match is None or not primary_genre:
                    skipped_incomplete += 1
                    continue
                year = int(year_match.group())
            except (ValueError, AttributeError):
                skipped_invalid += 1
                continue

            if ranking < 1 or votes < 0:
                skipped_invalid += 1
                continue

            records.append(
                {
                    "ranking": ranking,
                    "name": row["Name"].strip(),
                    "year": year,
                    "genre": row["Genre"].strip(),
                    "primary_genre": primary_genre,
                    "votes": votes,
                }
            )

    if not records:
        raise ValueError(f"No usable game records found in {path}")

    raw_genre_counts = Counter(record["primary_genre"] for record in records)
    labels = np.asarray(
        [
            record["primary_genre"]
            if raw_genre_counts[record["primary_genre"]] >= MIN_GENRE_SAMPLES
            else "Other"
            for record in records
        ],
        dtype=str,
    )
    grouped_genre_counts = Counter(labels.tolist())
    title_groups = np.asarray(
        [record["name"].casefold() for record in records],
        dtype=str,
    )
    features = np.asarray(
        [
            [record["ranking"], record["year"], record["votes"]]
            for record in records
        ],
        dtype=float,
    )

    total_rows = len(records) + skipped_blank + skipped_incomplete + skipped_invalid
    unique_titles = len(set(title_groups.tolist()))
    audit = {
        "dataset_name": path.name,
        "rows_read": total_rows,
        "rows_usable": len(records),
        "rows_skipped_blank": skipped_blank,
        "rows_skipped_incomplete": skipped_incomplete,
        "rows_skipped_invalid": skipped_invalid,
        "unique_titles": unique_titles,
        "duplicate_title_rows": len(records) - unique_titles,
        "ranking_range": [int(features[:, 0].min()), int(features[:, 0].max())],
        "year_range": [int(features[:, 1].min()), int(features[:, 1].max())],
        "votes_range": [int(features[:, 2].min()), int(features[:, 2].max())],
        "primary_genre_counts": dict(raw_genre_counts.most_common()),
        "grouped_genre_counts": dict(grouped_genre_counts.most_common()),
        "rare_genre_threshold": MIN_GENRE_SAMPLES,
    }
    return GameDataset(
        features=features,
        labels=labels,
        groups=title_groups,
        records=records,
        audit=audit,
    )
