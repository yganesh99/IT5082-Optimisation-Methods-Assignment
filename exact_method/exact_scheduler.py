from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.utils import calculate_screening_revenue, generate_robust_demand_matrix, parse_hhmm


@dataclass(frozen=True)
class Hall:
    id: str
    capacity: int
    average_ticket_price: float


def _clock_to_minutes(value: str) -> int:
    h, m = parse_hhmm(value)
    return h * 60 + m


def _compute_num_slots(config: dict[str, Any]) -> int:
    constraints = config["constraints"]
    slot_duration = int(constraints["slot_duration_minutes"])
    opening = _clock_to_minutes(constraints["opening_time"])
    closing = _clock_to_minutes(constraints["closing_time"])
    closing_next_day = bool(constraints.get("closing_time_is_next_calendar_day", True))

    if closing_next_day and closing <= opening:
        closing += 24 * 60
    if closing <= opening:
        raise ValueError("closing time must be after opening time")

    span = closing - opening
    if span % slot_duration != 0:
        raise ValueError("operating window is not divisible by slot duration")
    return span // slot_duration


def _slot_to_time(slot_index: int, opening_time: str, slot_duration_minutes: int) -> str:
    opening = _clock_to_minutes(opening_time)
    total = opening + slot_index * slot_duration_minutes
    total %= 24 * 60
    h, m = divmod(total, 60)
    return f"{h:02d}:{m:02d}"


def _load_halls(config: dict[str, Any]) -> list[Hall]:
    hall_type_map = {h["id"]: h for h in config["hall_types"]}
    halls: list[Hall] = []
    for hall in config["halls"]:
        hall_type_id = hall["hall_type_id"]
        hall_type = hall_type_map[hall_type_id]
        halls.append(
            Hall(
                id=str(hall["id"]),
                capacity=int(hall_type["capacity"]),
                average_ticket_price=float(hall_type["average_ticket_price"]),
            )
        )
    return halls


def _build_demand_matrix(movies: pd.DataFrame, day_type: str, num_slots: int) -> np.ndarray:
    movie_records: list[dict[str, Any]] = []
    for i, row in movies.iterrows():
        movie_records.append(
            {
                "index": int(i),
                "base_popularity": float(row["popularity"]),
                "weeks_since_release": int(row["weeks_since_release"]),
                "genre_peak_slot": int(row["genre_peak_slot"]),
                "genre_sigma": int(row["genre_sigma"]),
            }
        )
    return generate_robust_demand_matrix(movie_records, day_type=day_type, num_slots=num_slots)


def solve_schedule_ilp(
    movies_csv: Path,
    config_json: Path,
    day_type: str = "weekday",
    time_limit_seconds: int = 60,
    min_shows_penalty: float | None = None,
) -> dict[str, Any]:
    movies = pd.read_csv(movies_csv).reset_index(drop=True)
    config = json.loads(config_json.read_text(encoding="utf-8"))

    required_cols = [
        "id",
        "title",
        "runtime",
        "popularity",
        "weeks_since_release",
        "min_shows",
        "genre_peak_slot",
        "genre_sigma",
    ]
    missing = [c for c in required_cols if c not in movies.columns]
    if missing:
        raise ValueError(f"movies CSV is missing required columns: {missing}")

    constraints = config["constraints"]
    slot_duration = int(constraints["slot_duration_minutes"])
    opening_time = str(constraints["opening_time"])
    cleaning_buffer = int(constraints.get("cleaning_buffer_slots", 1))
    lobby_max_starts = int(constraints.get("lobby_max_simultaneous_starts", 2))

    num_slots = _compute_num_slots(config)
    halls = _load_halls(config)

    movies = movies.copy()
    movies["runtime_slots"] = movies["runtime"].apply(
        lambda x: int(math.ceil(float(x) / slot_duration))
    )
    movies["min_shows"] = movies["min_shows"].astype(int)

    demand_matrix = _build_demand_matrix(movies, day_type=day_type, num_slots=num_slots)

    hall_dicts = [
        {
            "id": hall.id,
            "capacity": hall.capacity,
            "average_ticket_price": hall.average_ticket_price,
        }
        for hall in halls
    ]

    movie_meta_for_revenue: list[dict[str, Any]] = []
    for i, row in movies.iterrows():
        movie_meta_for_revenue.append(
            {
                "index": int(i),
                "runtime_slots": int(row["runtime_slots"]),
            }
        )

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("Failed to initialize CBC solver")
    solver.SetTimeLimit(int(time_limit_seconds * 1000))

    x: dict[tuple[int, int, int], pywraplp.Variable] = {}
    revenue: dict[tuple[int, int, int], float] = {}
    valid_starts_by_movie: dict[int, list[int]] = {}

    for m in range(len(movies)):
        runtime_slots = int(movies.at[m, "runtime_slots"])
        valid_starts = [t for t in range(num_slots) if t + runtime_slots <= num_slots]
        valid_starts_by_movie[m] = valid_starts

        for h_idx, hall in enumerate(halls):
            for t in valid_starts:
                key = (m, h_idx, t)
                x[key] = solver.BoolVar(f"x_m{m}_h{h_idx}_t{t}")
                revenue[key] = float(
                    calculate_screening_revenue(
                        movie=movie_meta_for_revenue[m],
                        hall=hall_dicts[h_idx],
                        start_slot=t,
                        demand_matrix=demand_matrix,
                    )
                )

    max_rev = max(revenue.values(), default=0.0)
    if min_shows_penalty is None:
        penalty_per_undershot = max(max_rev * 50.0, 1e-4)
    else:
        penalty_per_undershot = float(min_shows_penalty)

    undershot: dict[int, pywraplp.Variable] = {}
    for m in range(len(movies)):
        cap = int(movies.at[m, "min_shows"])
        undershot[m] = solver.IntVar(0, cap, f"undershot_m{m}")

    objective = solver.Objective()
    for key, var in x.items():
        objective.SetCoefficient(var, revenue[key])
    for m, uvar in undershot.items():
        objective.SetCoefficient(uvar, -penalty_per_undershot)
    objective.SetMaximization()

    # Soft contractual minimum shows: sum(x) + undershot_m >= min_shows_m.
    for m in range(len(movies)):
        ct = solver.Constraint(float(movies.at[m, "min_shows"]), solver.infinity())
        ct.SetCoefficient(undershot[m], 1)
        for h_idx, _ in enumerate(halls):
            for t in valid_starts_by_movie[m]:
                ct.SetCoefficient(x[(m, h_idx, t)], 1)

    # Lobby throughput: maximum simultaneous starts in any slot.
    for t in range(num_slots):
        ct = solver.Constraint(-solver.infinity(), lobby_max_starts)
        for m in range(len(movies)):
            if t not in valid_starts_by_movie[m]:
                continue
            for h_idx, _ in enumerate(halls):
                ct.SetCoefficient(x[(m, h_idx, t)], 1)

    # No overlap in each hall, including cleaning buffer after each screening.
    for h_idx, _ in enumerate(halls):
        for s in range(num_slots):
            ct = solver.Constraint(-solver.infinity(), 1)
            for m in range(len(movies)):
                runtime_slots = int(movies.at[m, "runtime_slots"])
                blocked_duration = runtime_slots + cleaning_buffer
                for t in valid_starts_by_movie[m]:
                    if t <= s < t + blocked_duration:
                        ct.SetCoefficient(x[(m, h_idx, t)], 1)

    start_time = time.perf_counter()
    status = solver.Solve()
    elapsed = time.perf_counter() - start_time

    schedule: list[dict[str, Any]] = []
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for (m, h_idx, t), var in x.items():
            if var.solution_value() < 0.5:
                continue
            runtime_slots = int(movies.at[m, "runtime_slots"])
            end_slot = t + runtime_slots
            schedule.append(
                {
                    "hall_id": halls[h_idx].id,
                    "movie_id": str(movies.at[m, "id"]),
                    "movie_title": str(movies.at[m, "title"]),
                    "weeks_since_release": int(movies.at[m, "weeks_since_release"]),
                    "start_slot": int(t),
                    "start_time": _slot_to_time(t, opening_time, slot_duration),
                    "end_slot": int(end_slot),
                    "end_time": _slot_to_time(end_slot, opening_time, slot_duration),
                    "expected_revenue": float(revenue[(m, h_idx, t)]),
                }
            )

    schedule.sort(key=lambda r: (r["hall_id"], r["start_slot"]))

    status_name = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(status, "UNKNOWN")

    total_revenue = float(sum(r["expected_revenue"] for r in schedule))
    min_shows_deficit_by_movie: list[dict[str, Any]] = []
    min_shows_deficit_total = 0
    objective_value: float | None = None

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for m in range(len(movies)):
            d = int(round(undershot[m].solution_value()))
            min_shows_deficit_total += d
            if d > 0:
                min_shows_deficit_by_movie.append(
                    {
                        "movie_id": str(movies.at[m, "id"]),
                        "movie_title": str(movies.at[m, "title"]),
                        "deficit": d,
                    }
                )
        objective_value = total_revenue - penalty_per_undershot * min_shows_deficit_total

    return {
        "metadata": {
            "algorithm": "ILP (OR-Tools CBC)",
            "solver_status": status_name,
            "execution_time_seconds": elapsed,
            "total_revenue": total_revenue,
            "objective_value": objective_value,
            "min_shows_penalty_per_unit": penalty_per_undershot,
            "fitness_score": None,
            "constraints_violated": min_shows_deficit_total
            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
            else None,
            "min_shows_deficit_total": min_shows_deficit_total
            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
            else None,
            "min_shows_deficit_by_movie": min_shows_deficit_by_movie
            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
            else None,
            "day_type": day_type,
            "num_slots": num_slots,
            "num_halls": len(halls),
            "num_movies": len(movies),
        },
        "schedule": schedule,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact ILP movie scheduler")
    parser.add_argument(
        "--movies-csv",
        type=Path,
        required=True,
        help="Path to movies_small.csv / movies_medium.csv / movies_large.csv",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        required=True,
        help="Path to small_config.json / med_config.json / large_config.json",
    )
    parser.add_argument(
        "--day-type",
        type=str,
        default="weekday",
        choices=["weekday", "weekend"],
        help="Demand profile scalar used in demand matrix generation",
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=int,
        default=60,
        help="Solver time limit in seconds",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional file path to write JSON output",
    )
    parser.add_argument(
        "--min-shows-penalty",
        type=float,
        default=None,
        help="Penalty per missed min_shows unit (default: max_screening_revenue * 50, min 1e-4)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = solve_schedule_ilp(
        movies_csv=args.movies_csv,
        config_json=args.config_json,
        day_type=args.day_type,
        time_limit_seconds=args.time_limit_seconds,
        min_shows_penalty=args.min_shows_penalty,
    )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output_json}")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
