"""NumPy calculations on cleaned numeric columns."""

from __future__ import annotations

import numpy as np


def scores_array(rows: list[dict]) -> np.ndarray:
    """Turn the cleaned score column into a NumPy array."""
    return np.array([row["score"] for row in rows], dtype=float)


def hours_array(rows: list[dict]) -> np.ndarray:
    """Turn hours_studied into a NumPy array (missing values become nan)."""
    values = [row.get("hours_studied") if row.get("hours_studied") is not None else np.nan for row in rows]
    return np.array(values, dtype=float)


def mean_score(scores: np.ndarray) -> float:
    return float(np.mean(scores))


def median_score(scores: np.ndarray) -> float:
    return float(np.median(scores))


def std_score(scores: np.ndarray) -> float:
    return float(np.std(scores, ddof=0))


def score_percentile(scores: np.ndarray, q: float = 75) -> float:
    return float(np.percentile(scores, q))


def high_scores(scores: np.ndarray, threshold: float = 85) -> np.ndarray:
    """Return scores at or above a threshold using np.where."""
    return scores[np.where(scores >= threshold)]


def summarize(rows: list[dict]) -> dict[str, float | int]:
    """Run several NumPy stats on the cleaned dataset."""
    scores = scores_array(rows)
    hours = hours_array(rows)
    return {
        "count": int(scores.size),
        "mean_score": mean_score(scores),
        "median_score": median_score(scores),
        "std_score": std_score(scores),
        "p75_score": score_percentile(scores, 75),
        "high_score_count": int(high_scores(scores).size),
        "mean_hours": float(np.nanmean(hours)),
    }
