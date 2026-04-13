#!/usr/bin/env python3
"""
Validate a schedule JSON (GA or ILP) against cineplex rules and revenue consistency.

Checks:
  - Hall overlaps + cleaning buffer (hall locked until end_slot + buffer)
  - Lobby congestion (simultaneous starts per slot <= lobby cap)
  - Operating horizon (slots within [0, num_slots); end_slot <= num_slots)
  - min_shows per movie from the CSV
  - Expected revenue matches recomputation from common demand/revenue logic
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.ga_solve import GAProblem  # noqa: E402


def _compute_num_slots_from_config(config: dict) -> int:
    constraints = config["constraints"]
    slot_duration = int(constraints["slot_duration_minutes"])

    def clock(hhmm: str) -> int:
        h, m = (int(x) for x in hhmm.strip().split(":"))
        return h * 60 + m

    opening = clock(constraints["opening_time"])
    closing = clock(constraints["closing_time"])
    if bool(constraints.get("closing_time_is_next_calendar_day", True)) and closing <= opening:
        closing += 24 * 60
    return (closing - opening) // slot_duration


def validate(
    schedule_json: Path,
    movies_csv: Path,
    config_json: Path,
    *,
    day_type: str | None = None,
    revenue_rtol: float = 1e-5,
    revenue_atol: float = 1e-4,
) -> tuple[bool, list[str]]:
    data = json.loads(schedule_json.read_text(encoding="utf-8"))
    meta = data.get("metadata") or {}
    schedule = data.get("schedule") or []

    if day_type is None:
        day_type = str(meta.get("day_type") or "weekend")

    config = json.loads(config_json.read_text(encoding="utf-8"))
    constraints = config["constraints"]
    cleaning_buffer = int(constraints.get("cleaning_buffer_slots", 1))
    lobby_max = int(constraints.get("lobby_max_simultaneous_starts", 2))
    num_slots = _compute_num_slots_from_config(config)

    problem = GAProblem.from_files(movies_csv, config_json, day_type=day_type)
    if problem.num_slots != num_slots:
        return False, [f"internal num_slots mismatch: problem={problem.num_slots} config={num_slots}"]

    issues: list[str] = []

    # --- Structural: slots & end within horizon (1:00 AM closure encoded as num_slots) ---
    for i, row in enumerate(schedule):
        t0 = int(row["start_slot"])
        end = int(row["end_slot"])
        if t0 < 0 or t0 >= num_slots:
            issues.append(f"row {i}: start_slot {t0} out of range [0, {num_slots})")
        if end < 0 or end > num_slots:
            issues.append(
                f"row {i}: end_slot {end} exceeds horizon (max end_slot {num_slots} — exclusive end after last slot)"
            )

    # --- Runtime consistency & overlap + buffer per hall ---
    by_hall: dict[str, list[dict]] = {}
    for row in schedule:
        by_hall.setdefault(row["hall_id"], []).append(row)

    slot_duration = int(constraints["slot_duration_minutes"])

    import pandas as pd

    movies_df = pd.read_csv(movies_csv).reset_index(drop=True)
    id_to_runtime_slots = {
        str(movies_df.at[i, "id"]): int(math.ceil(float(movies_df.at[i, "runtime"]) / slot_duration))
        for i in range(len(movies_df))
    }

    for hall_id, rows in by_hall.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["start_slot"]))
        for i, row in enumerate(rows_sorted):
            mid = str(row["movie_id"])
            rs = id_to_runtime_slots.get(mid)
            if rs is None:
                issues.append(f"unknown movie_id {mid!r} in hall {hall_id}")
                continue
            t0 = int(row["start_slot"])
            end = int(row["end_slot"])
            if end != t0 + rs:
                issues.append(
                    f"{hall_id} movie {mid}: end_slot {end} != start_slot+runtime_slots ({t0}+{rs}={t0 + rs})"
                )
        for j in range(1, len(rows_sorted)):
            prev = rows_sorted[j - 1]
            cur = rows_sorted[j]
            p_end = int(prev["end_slot"])
            c_start = int(cur["start_slot"])
            if c_start < p_end + cleaning_buffer:
                issues.append(
                    f"overlap/buffer violation in {hall_id}: screening after "
                    f"{prev.get('movie_id')} ends at slot {p_end} (need gap {cleaning_buffer}) "
                    f"but next starts at {c_start}"
                )

    # --- Lobby: simultaneous starts ---
    starts = Counter(int(r["start_slot"]) for r in schedule)
    for t, cnt in sorted(starts.items()):
        if cnt > lobby_max:
            issues.append(
                f"lobby congestion: {cnt} screenings start at slot {t} (max allowed {lobby_max})"
            )

    # --- min_shows ---
    min_by_id = {str(movies_df.at[i, "id"]): int(movies_df.at[i, "min_shows"]) for i in range(len(movies_df))}
    show_counts = Counter(str(r["movie_id"]) for r in schedule)
    for mid, need in min_by_id.items():
        got = show_counts.get(mid, 0)
        if got < need:
            issues.append(f"movie {mid}: min_shows {need} not met (got {got})")

    # --- Revenue ---
    max_rev = problem.max_screening_revenue
    sum_json = float(sum(float(r["expected_revenue"]) for r in schedule))
    meta_rev = meta.get("total_revenue")
    if meta_rev is not None and abs(sum_json - float(meta_rev)) > revenue_atol + revenue_rtol * abs(float(meta_rev)):
        issues.append(f"sum(expected_revenue)={sum_json} != metadata.total_revenue={meta_rev}")

    for i, row in enumerate(schedule):
        exp = float(row["expected_revenue"])
        if exp < -revenue_atol:
            issues.append(f"row {i}: negative expected_revenue {exp}")
        if exp > max_rev * (1 + 10 * revenue_rtol) + revenue_atol:
            issues.append(f"row {i}: expected_revenue {exp} above plausible max single screening {max_rev}")

        recomputed = problem.expected_revenue_for_screening(
            str(row["movie_id"]), str(row["hall_id"]), int(row["start_slot"])
        )
        if recomputed is None:
            issues.append(
                f"row {i}: could not recompute revenue for movie={row['movie_id']} hall={row['hall_id']} "
                f"start={row['start_slot']}"
            )
        elif abs(recomputed - exp) > revenue_atol + revenue_rtol * max(abs(recomputed), abs(exp), 1.0):
            issues.append(
                f"row {i}: expected_revenue {exp} != recomputed {recomputed} "
                f"(movie={row['movie_id']} hall={row['hall_id']} start={row['start_slot']})"
            )

    return len(issues) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate schedule JSON against scheduling rules.")
    parser.add_argument("schedule_json", type=Path, help="Path to schedule_*.json")
    parser.add_argument("--movies-csv", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument(
        "--day-type",
        type=str,
        default=None,
        choices=["weekday", "weekend"],
        help="Demand profile (default: read from JSON metadata)",
    )
    args = parser.parse_args()

    ok, issues = validate(args.schedule_json, args.movies_csv, args.config_json, day_type=args.day_type)
    if ok:
        print("OK: schedule passes all checks.")
        return 0
    print("FAILED:")
    for line in issues:
        print(f"  - {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
