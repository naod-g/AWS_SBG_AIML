"""Load raw datasets from a CSV file or from an in-memory list."""

from __future__ import annotations

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "raw_students.csv"

# Fallback dataset kept in code so the assignment still runs if the CSV is missing.
SAMPLE_ROWS: list[dict[str, str]] = [
    {"name": "Ada Lovelace", "age": "36", "city": "london", "score": "88", "hours_studied": "12"},
    {"name": "Alan Turing", "age": "41", "city": "CAMBRIDGE", "score": "91", "hours_studied": "14"},
    {"name": "Grace Hopper", "age": "85", "city": "New York", "score": "95", "hours_studied": "10"},
]


def load_csv(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load a raw CSV file as a list of dictionaries (one dict per row)."""
    csv_path = Path(path) if path is not None else DEFAULT_CSV
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_from_list(rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Load a raw dataset that is already a list of row dictionaries."""
    source = SAMPLE_ROWS if rows is None else rows
    return [dict(row) for row in source]
