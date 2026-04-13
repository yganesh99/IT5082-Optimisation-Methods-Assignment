#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ga_path = _REPO_ROOT / "heuristic(GA)" / "ga_solve.py"
_ga_spec = importlib.util.spec_from_file_location("ga_solve", _ga_path)
if _ga_spec is None or _ga_spec.loader is None:
    raise ImportError(f"Cannot load GA solver from {_ga_path}")
_ga_mod = importlib.util.module_from_spec(_ga_spec)
sys.modules["ga_solve"] = _ga_mod
_ga_spec.loader.exec_module(_ga_mod)
GAProblem = _ga_mod.GAProblem
from exact_method.exact_scheduler import solve_schedule_ilp
from validator import validate


def _parse_int_list(value: str) -> list[int]:
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in parts]


def _tier_paths(repo_root: Path) -> dict[str, tuple[Path, Path]]:
    return {
        "small": (repo_root / "common" / "movies_small.csv", repo_root / "common" / "small_config.json"),
        "medium": (repo_root / "common" / "movies_medium.csv", repo_root / "common" / "med_config.json"),
        "large": (repo_root / "common" / "movies_large.csv", repo_root / "common" / "large_config.json"),
        "xlarge": (repo_root / "common" / "movies_xlarge.csv", repo_root / "common" / "xlarge_config.json"),
    }


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    tier_map = _tier_paths(_REPO_ROOT)
    ilp_limits = _parse_int_list(args.ilp_time_limits_seconds)
    ga_seeds = _parse_int_list(args.ga_seeds)
    selected_tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    rows: list[dict[str, Any]] = []

    for tier in selected_tiers:
        if tier not in tier_map:
            raise ValueError(f"unknown tier {tier!r}. Supported tiers: {sorted(tier_map)}")
        movies_csv, config_json = tier_map[tier]
        if not movies_csv.is_file():
            print(f"Skipping tier={tier}: missing movies CSV at {movies_csv}")
            continue
        if not config_json.is_file():
            print(f"Skipping tier={tier}: missing config JSON at {config_json}")
            continue

        print(f"\n=== Tier: {tier} ===")

        # ILP runs for each time limit
        for limit in ilp_limits:
            print(f"[ILP] limit={limit}s")
            ilp_result = solve_schedule_ilp(
                movies_csv=movies_csv,
                config_json=config_json,
                day_type=args.day_type,
                time_limit_seconds=limit,
            )
            out_path = args.output_dir / tier / f"ilp_t{limit}s.json"
            _write_json(out_path, ilp_result)
            is_valid, issues = validate(
                out_path,
                movies_csv=movies_csv,
                config_json=config_json,
                day_type=args.day_type,
            )
            rows.append(
                {
                    "tier": tier,
                    "algorithm": "ILP",
                    "seed": None,
                    "ilp_time_limit_seconds": limit,
                    "ga_population_size": None,
                    "ga_generations": None,
                    "solver_status": ilp_result["metadata"].get("solver_status"),
                    "execution_time_seconds": ilp_result["metadata"].get("execution_time_seconds"),
                    "total_revenue": ilp_result["metadata"].get("total_revenue"),
                    "constraints_violated": ilp_result["metadata"].get("constraints_violated"),
                    "num_movies": ilp_result["metadata"].get("num_movies"),
                    "num_halls": ilp_result["metadata"].get("num_halls"),
                    "valid_schedule": is_valid,
                    "validation_issue_count": len(issues),
                    "output_json": str(out_path),
                }
            )

        # GA runs for each seed
        for seed in ga_seeds:
            print(f"[GA] seed={seed}")
            problem = GAProblem.from_files(movies_csv, config_json, day_type=args.day_type)
            ga_result, _history, diagnostics = problem.solve(
                pop_size=args.ga_population_size,
                generations=args.ga_generations,
                mutation_rate=args.ga_mutation_rate,
                crossover_rate=args.ga_crossover_rate,
                tournament_size=args.ga_tournament_size,
                elitism_count=args.ga_elitism_count,
                seed=seed,
                verbose=False,
                trace_memory=True,
            )
            out_path = args.output_dir / tier / f"ga_seed{seed}.json"
            _write_json(out_path, ga_result)
            is_valid, issues = validate(
                out_path,
                movies_csv=movies_csv,
                config_json=config_json,
                day_type=args.day_type,
            )
            rows.append(
                {
                    "tier": tier,
                    "algorithm": "GA",
                    "seed": seed,
                    "ilp_time_limit_seconds": None,
                    "ga_population_size": args.ga_population_size,
                    "ga_generations": args.ga_generations,
                    "solver_status": ga_result["metadata"].get("solver_status"),
                    "execution_time_seconds": ga_result["metadata"].get("execution_time_seconds"),
                    "total_revenue": ga_result["metadata"].get("total_revenue"),
                    "constraints_violated": ga_result["metadata"].get("constraints_violated"),
                    "num_movies": ga_result["metadata"].get("num_movies"),
                    "num_halls": ga_result["metadata"].get("num_halls"),
                    "valid_schedule": is_valid,
                    "validation_issue_count": len(issues),
                    "peak_rss_bytes": diagnostics.get("peak_rss_bytes"),
                    "tracemalloc_peak_bytes": diagnostics.get("tracemalloc_peak_bytes"),
                    "last_improvement_generation": diagnostics.get("last_improvement_generation"),
                    "output_json": str(out_path),
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GA vs ILP benchmarks across dataset tiers and export a comparison CSV."
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="small,medium,large",
        help="Comma-separated tiers from: small,medium,large,xlarge",
    )
    parser.add_argument(
        "--day-type",
        type=str,
        default="weekday",
        choices=["weekday", "weekend"],
        help="Demand profile used by both GA and ILP.",
    )
    parser.add_argument(
        "--ilp-time-limits-seconds",
        type=str,
        default="60,180,600",
        help="Comma-separated ILP time limits in seconds.",
    )
    parser.add_argument(
        "--ga-seeds",
        type=str,
        default="11,22,33,44,55",
        help="Comma-separated GA random seeds.",
    )
    parser.add_argument("--ga-population-size", type=int, default=150)
    parser.add_argument("--ga-generations", type=int, default=500)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.4)
    parser.add_argument("--ga-crossover-rate", type=float, default=0.85)
    parser.add_argument("--ga-tournament-size", type=int, default=5)
    parser.add_argument("--ga-elitism-count", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "benchmark_outputs",
        help="Directory to store benchmark JSON outputs and summary CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = run_benchmark(args)
    if df.empty:
        print("No benchmark rows generated. Check selected tiers and file paths.")
        return 1
    summary_csv = args.output_dir / "benchmark_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"\nWrote summary CSV: {summary_csv}")
    print(df.groupby(["tier", "algorithm"])["execution_time_seconds"].agg(["count", "median", "max"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
