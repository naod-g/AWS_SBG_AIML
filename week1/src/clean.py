"""Named cleaning functions for the student dataset."""

from __future__ import annotations

MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan"}


def strip_text_fields(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trim whitespace from every string field."""
    cleaned: list[dict[str, str]] = []
    for row in rows:
        cleaned.append({key: value.strip() if isinstance(value, str) else value for key, value in row.items()})
    return cleaned


def normalize_missing_values(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Turn empty strings and common missing tokens into None."""
    cleaned: list[dict[str, str | None]] = []
    for row in rows:
        next_row: dict[str, str | None] = {}
        for key, value in row.items():
            if value is None or str(value).strip().lower() in MISSING_TOKENS:
                next_row[key] = None
            else:
                next_row[key] = value
        cleaned.append(next_row)
    return cleaned  # type: ignore[return-value]


def fix_text_case(rows: list[dict]) -> list[dict]:
    """Title-case names and cities so duplicates and labels match."""
    cleaned = []
    for row in rows:
        next_row = dict(row)
        if next_row.get("name"):
            next_row["name"] = str(next_row["name"]).title()
        if next_row.get("city"):
            next_row["city"] = str(next_row["city"]).title()
        cleaned.append(next_row)
    return cleaned


def convert_column_types(rows: list[dict]) -> list[dict]:
    """Convert age, score, and hours_studied to numbers when possible."""
    numeric_fields = ("age", "score", "hours_studied")
    cleaned = []
    for row in rows:
        next_row = dict(row)
        for field in numeric_fields:
            value = next_row.get(field)
            if value is None or value == "":
                next_row[field] = None
                continue
            try:
                next_row[field] = float(value)
            except (TypeError, ValueError):
                next_row[field] = None
        if next_row.get("age") is not None:
            next_row["age"] = int(next_row["age"])
        cleaned.append(next_row)
    return cleaned


def drop_duplicate_rows(rows: list[dict]) -> list[dict]:
    """Remove exact duplicate people (same name, age, and city)."""
    seen: set[tuple] = set()
    unique_rows: list[dict] = []
    for row in rows:
        key = (row.get("name"), row.get("age"), row.get("city"))
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def remove_outliers(rows: list[dict], score_min: float = 0, score_max: float = 100) -> list[dict]:
    """Drop rows whose score is missing or outside a valid 0–100 range."""
    kept: list[dict] = []
    for row in rows:
        score = row.get("score")
        if score is None:
            continue
        if score < score_min or score > score_max:
            continue
        kept.append(row)
    return kept


def drop_incomplete_records(rows: list[dict], required: tuple[str, ...] = ("name", "age", "city", "score")) -> list[dict]:
    """Keep only rows that have the required fields after cleaning."""
    return [row for row in rows if all(row.get(field) not in (None, "") for field in required)]


def clean_dataset(rows: list[dict]) -> list[dict]:
    """Run the full cleaning pipeline in a fixed order."""
    data = strip_text_fields(rows)
    data = normalize_missing_values(data)
    data = fix_text_case(data)
    data = convert_column_types(data)
    data = drop_duplicate_rows(data)
    data = remove_outliers(data)
    data = drop_incomplete_records(data)
    return data
