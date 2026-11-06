"""Class-count tables used before any model is trained."""

from __future__ import annotations

import pandas as pd


def class_distribution(labels: pd.Series) -> pd.DataFrame:
    """Return class, count, and fraction. Missing labels are counted, not dropped."""
    if len(labels) == 0:
        return pd.DataFrame(columns=["class", "count", "fraction"])
    cleaned = labels.map(lambda value: "(missing)" if pd.isna(value) else str(value))
    counts = cleaned.value_counts()
    total = int(counts.sum())
    frame = pd.DataFrame({"class": counts.index.astype(str), "count": counts.to_numpy()})
    frame["fraction"] = frame["count"] / total
    return frame.sort_values(["count", "class"], ascending=[False, True]).reset_index(drop=True)
