# Synthetic sub-dataset generator: builds small, medium, and large CSV subsets from
# TMDB_movie_dataset_v11.csv using the same cleaning + popularity scaling as
# ``data_loader.load_prepared_tmdb_movies``, then adds weeks_since_release, min_shows,
# genre_peak_slot, and genre_sigma (see GENRE_AUDIENCE).

from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from data_loader import default_tmdb_csv_path, load_prepared_tmdb_movies

_REPO_COMMON = Path(__file__).resolve().parent
_DEFAULT_INPUT = default_tmdb_csv_path()
# Fixed reference so weeks_since_release is reproducible and aligned with a typical summer release window.
_DEFAULT_WEEKS_AS_OF = date(2023, 10, 1)

# Audience curve parameters by genre family. ``genre_peak_slot`` is a 15-minute slot index
# (same grid as cineplex configs). ``genre_sigma`` controls spread (wider = flatter demand).
# TMDB ``genres`` strings are matched in list order: first profile that hits any tag wins.
#
# | Family              | peak (slot) | σ   | note (informal)                         |
# |---------------------|------------:|----:|----------------------------------------|
# | Animation / Family  | 12 (~1 PM)  |  4  | Kids peak early; narrow evening cutoff |
# | Comedy / Romance    | 32 (~6 PM)  |  6  | Pre-dinner / early evening             |
# | Action / Adventure  | 40 (~8 PM)  |  8  | Prime-time blockbuster, broad          |
# | Horror / Thriller   | 48 (~10 PM) |  5  | Late peak; low morning demand          |
# | Documentary         | 20 (~3 PM)  | 10  | Broad, lower-intensity daytime curve   |
GENRE_AUDIENCE: tuple[tuple[str, int, int, frozenset[str]], ...] = (
    # label, peak_slot, sigma, TMDB genre names (exact string as in CSV)
    ("Horror / Thriller", 48, 5, frozenset({"Horror", "Thriller", "Mystery", "Crime"})),
    ("Animation / Family", 12, 4, frozenset({"Animation", "Family"})),
    ("Comedy / Romance", 32, 6, frozenset({"Comedy", "Romance"})),
    ("Documentary", 20, 10, frozenset({"Documentary", "Music"})),
    (
        "Action / Adventure",
        40,
        8,
        frozenset(
            {
                "Action",
                "Adventure",
                "Science Fiction",
                "Fantasy",
                "Drama",
                "War",
                "Western",
                "History",
                "TV Movie",
            }
        ),
    ),
)
_DEFAULT_PEAK_SLOT = 40
_DEFAULT_SIGMA = 8


def _tmdb_genre_tokens(genres_cell: object) -> list[str]:
    if genres_cell is None or (isinstance(genres_cell, float) and pd.isna(genres_cell)):
        return []
    s = str(genres_cell).strip()
    if not s:
        return []
    return [g.strip() for g in s.split(",") if g.strip()]


def _genre_peak_slot_and_sigma(genres_cell: object) -> tuple[int, int]:
    tokens = _tmdb_genre_tokens(genres_cell)
    if not tokens:
        return _DEFAULT_PEAK_SLOT, _DEFAULT_SIGMA
    token_set = set(tokens)
    for _label, peak, sigma, keywords in GENRE_AUDIENCE:
        if token_set & keywords:
            return peak, sigma
    return _DEFAULT_PEAK_SLOT, _DEFAULT_SIGMA


def _apply_genre_audience(genres_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    peaks: list[int] = []
    sigmas: list[int] = []
    for v in genres_series:
        p, s = _genre_peak_slot_and_sigma(v)
        peaks.append(p)
        sigmas.append(s)
    return pd.Series(peaks, index=genres_series.index, dtype="int64"), pd.Series(
        sigmas, index=genres_series.index, dtype="int64"
    )


def _weeks_since_release(release_dates: pd.Series, as_of: date) -> pd.Series:
    rd = pd.to_datetime(release_dates, errors="coerce").dt.normalize()
    as_of_ts = pd.Timestamp(as_of)
    delta_days = (as_of_ts - rd).dt.days
    weeks = delta_days // 7
    return weeks.clip(lower=0).astype("Int64")


def _pick_top_movies(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return up to ``n`` rows with valid release dates, most popular first (by prepared popularity)."""
    work = df.copy()
    work["popularity"] = pd.to_numeric(work["popularity"], errors="coerce")
    work["vote_count"] = pd.to_numeric(work["vote_count"], errors="coerce").fillna(0).astype(int)
    work["runtime"] = pd.to_numeric(work["runtime"], errors="coerce")
    work = work.loc[work["runtime"] >= 60]
    work = work.loc[work["release_date"].notna() & (work["release_date"].astype(str).str.strip() != "")]
    work["_release_parsed"] = pd.to_datetime(work["release_date"], errors="coerce")
    work = work.loc[work["_release_parsed"].notna()].drop(columns=["_release_parsed"])
    work = work.sort_values(
        by=["popularity", "vote_count"],
        ascending=[False, False],
        kind="mergesort",
    )
    return work.drop_duplicates(subset=["id"], keep="first").head(n).copy()


def _augment(df: pd.DataFrame, as_of: date, rng: random.Random) -> pd.DataFrame:
    out = df.copy()
    out["weeks_since_release"] = _weeks_since_release(out["release_date"], as_of)
    out["min_shows"] = [rng.randint(1, 3) for _ in range(len(out))]
    peak, sigma = _apply_genre_audience(out["genres"])
    out["genre_peak_slot"] = peak
    out["genre_sigma"] = sigma
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build small/medium/large movie CSVs from the main TMDB CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Path to TMDB_movie_dataset_v11.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_COMMON,
        help="Directory for movies_small.csv, movies_medium.csv, and movies_large.csv",
    )
    parser.add_argument(
        "--as-of",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help=(
            "Reference date for weeks_since_release (ISO YYYY-MM-DD). "
            f"Default: {_DEFAULT_WEEKS_AS_OF.isoformat()}."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for min_shows and subset sizes.")
    args = parser.parse_args()

    as_of = args.as_of if args.as_of is not None else _DEFAULT_WEEKS_AS_OF
    rng = random.Random(args.seed)

    small_n = rng.randint(4, 6)
    medium_n = rng.randint(12, 18)
    large_n = rng.randint(30, 50)
    need = max(small_n, medium_n, large_n)

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    df = load_prepared_tmdb_movies(args.input)
    required = ("id", "title", "release_date", "popularity", "vote_count", "genres", "runtime")
    for col in required:
        if col not in df.columns:
            print(f"Missing required column {col!r} after load (expected for prepared TMDB data).", file=sys.stderr)
            return 1

    top = _pick_top_movies(df, need)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("movies_small.csv", small_n),
        ("movies_medium.csv", medium_n),
        ("movies_large.csv", large_n),
    ]
    for name, n in outputs:
        subset = _augment(top.head(n), as_of, rng)
        path = args.output_dir / name
        subset.to_csv(path, index=False)
        print(f"Wrote {path} ({len(subset)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
