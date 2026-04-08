from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

_COMMON_DIR = Path(__file__).resolve().parent
_DEFAULT_CSV = _COMMON_DIR / "TMDB_movie_dataset_v11.csv"


def default_tmdb_csv_path() -> Path:
    """Absolute path to the bundled TMDB CSV (same directory as this module)."""
    return _DEFAULT_CSV


def load_tmdb_movies(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the TMDB movie dataset from CSV."""
    path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV
    return pd.read_csv(path)


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with zero runtime or missing genres."""
    return df.loc[(df["runtime"] != 0) & df["genres"].notna()].copy()


def normalize_popularity_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """Min–max scale ``popularity`` to [0, 1] (handles constant column)."""
    out = df.copy()
    col = out["popularity"]
    lo, hi = col.min(), col.max()
    if hi == lo:
        out["popularity"] = 0.0
    else:
        out["popularity"] = (col - lo) / (hi - lo)
    return out


def load_prepared_tmdb_movies(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load, clean, and normalize in one step (for workflows that skip exploratory steps)."""
    df = load_tmdb_movies(csv_path)
    df = clean_movies(df)
    return normalize_popularity_minmax(df)
