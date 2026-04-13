from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pygad

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.scheduling_config import (
    num_slots_from_config,
    opening_minutes_from_constraints,
    slot_to_wall_clock,
)
from common.utils import calculate_screening_revenue, generate_robust_demand_matrix


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HallInfo:
    id: str
    index: int
    capacity: int
    average_ticket_price: float


@dataclass(frozen=True)
class MovieInfo:
    id: str
    index: int
    title: str
    runtime_slots: int
    min_shows: int
    weeks_since_release: int


# ---------------------------------------------------------------------------
# GAProblem — shared problem context (also consumed by validator.py)
# ---------------------------------------------------------------------------

@dataclass
class GAProblem:
    movies: list[MovieInfo]
    halls: list[HallInfo]
    num_slots: int
    cleaning_buffer: int
    lobby_max_starts: int
    slot_duration: int
    opening_minutes: int
    demand_matrix: np.ndarray
    day_type: str

    _movie_id_to_index: dict[str, int] = field(init=False, repr=False)
    _hall_id_to_index: dict[str, int] = field(init=False, repr=False)
    _revenue_table: np.ndarray = field(init=False, repr=False)
    _max_screening_revenue: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._movie_id_to_index = {m.id: m.index for m in self.movies}
        self._hall_id_to_index = {h.id: h.index for h in self.halls}
        self._revenue_table = self._build_revenue_table()
        self._max_screening_revenue = float(self._revenue_table.max())

    def _build_revenue_table(self) -> np.ndarray:
        n_movies = len(self.movies)
        n_halls = len(self.halls)
        table = np.zeros((n_movies, n_halls, self.num_slots))
        hall_dicts = [
            {"id": h.id, "capacity": h.capacity, "average_ticket_price": h.average_ticket_price}
            for h in self.halls
        ]
        movie_dicts = [
            {"index": m.index, "runtime_slots": m.runtime_slots}
            for m in self.movies
        ]
        for m in self.movies:
            for h in self.halls:
                for t in range(self.num_slots - m.runtime_slots + 1):
                    table[m.index, h.index, t] = float(
                        calculate_screening_revenue(
                            movie=movie_dicts[m.index],
                            hall=hall_dicts[h.index],
                            start_slot=t,
                            demand_matrix=self.demand_matrix,
                        )
                    )
        return table

    @property
    def max_screening_revenue(self) -> float:
        return self._max_screening_revenue

    def expected_revenue_for_screening(
        self, movie_id: str, hall_id: str, start_slot: int
    ) -> float | None:
        m_idx = self._movie_id_to_index.get(str(movie_id))
        h_idx = self._hall_id_to_index.get(str(hall_id))
        if m_idx is None or h_idx is None:
            return None
        movie = self.movies[m_idx]
        if start_slot < 0 or start_slot + movie.runtime_slots > self.num_slots:
            return None
        return float(self._revenue_table[m_idx, h_idx, start_slot])

    def revenue(self, movie_index: int, hall_index: int, start_slot: int) -> float:
        return float(self._revenue_table[movie_index, hall_index, start_slot])

    @classmethod
    def from_files(
        cls,
        movies_csv: str | Path,
        config_json: str | Path,
        *,
        day_type: str = "weekday",
    ) -> GAProblem:
        movies_csv = Path(movies_csv)
        config_json = Path(config_json)

        df = pd.read_csv(movies_csv).reset_index(drop=True)
        config = json.loads(config_json.read_text(encoding="utf-8"))
        constraints = config["constraints"]

        slot_duration = int(constraints["slot_duration_minutes"])
        opening_min = opening_minutes_from_constraints(constraints)
        cleaning_buffer = int(constraints.get("cleaning_buffer_slots", 1))
        lobby_max = int(constraints.get("lobby_max_simultaneous_starts", 2))
        n_slots = num_slots_from_config(config)

        hall_type_map = {ht["id"]: ht for ht in config["hall_types"]}
        halls: list[HallInfo] = []
        for idx, h in enumerate(config["halls"]):
            ht = hall_type_map[h["hall_type_id"]]
            halls.append(
                HallInfo(
                    id=str(h["id"]),
                    index=idx,
                    capacity=int(ht["capacity"]),
                    average_ticket_price=float(ht["average_ticket_price"]),
                )
            )

        movies: list[MovieInfo] = []
        for i in range(len(df)):
            movies.append(
                MovieInfo(
                    id=str(df.at[i, "id"]),
                    index=i,
                    title=str(df.at[i, "title"]),
                    runtime_slots=int(math.ceil(float(df.at[i, "runtime"]) / slot_duration)),
                    min_shows=int(df.at[i, "min_shows"]),
                    weeks_since_release=int(df.at[i, "weeks_since_release"]),
                )
            )

        movie_records = [
            {
                "index": m.index,
                "base_popularity": float(df.at[m.index, "popularity"]),
                "weeks_since_release": m.weeks_since_release,
                "genre_peak_slot": int(df.at[m.index, "genre_peak_slot"]),
                "genre_sigma": int(df.at[m.index, "genre_sigma"]),
            }
            for m in movies
        ]
        demand_matrix = generate_robust_demand_matrix(movie_records, day_type=day_type, num_slots=n_slots)

        return cls(
            movies=movies,
            halls=halls,
            num_slots=n_slots,
            cleaning_buffer=cleaning_buffer,
            lobby_max_starts=lobby_max,
            slot_duration=slot_duration,
            opening_minutes=opening_min,
            demand_matrix=demand_matrix,
            day_type=day_type,
        )


# ---------------------------------------------------------------------------
# Chromosome encoding helpers
# ---------------------------------------------------------------------------

def _max_screenings_per_hall(problem: GAProblem) -> int:
    min_block = min(m.runtime_slots for m in problem.movies) + problem.cleaning_buffer
    return problem.num_slots // min_block + 1


def _num_genes(problem: GAProblem) -> int:
    return len(problem.halls) * _max_screenings_per_hall(problem)


# ---------------------------------------------------------------------------
# Greedy seed: build a high-quality initial chromosome
# ---------------------------------------------------------------------------

def _build_greedy_seed(problem: GAProblem) -> np.ndarray:
    """Construct one chromosome via decoder-aware greedy placement.

    Process halls in the same revenue-potential order the decoder uses.
    For each hall, at each time position the decoder would consider,
    pick the movie that maximises revenue at that slot -- but prioritise
    movies that still need min_shows fulfilment.  This ensures the
    chromosome decodes to the exact schedule we intended and satisfies
    min_shows without relying on the penalty.
    """
    sph = _max_screenings_per_hall(problem)
    num_halls = len(problem.halls)
    num_genes = num_halls * sph
    genes = np.full(num_genes, -1, dtype=int)

    decode_order = sorted(
        range(num_halls),
        key=lambda i: problem.halls[i].capacity * problem.halls[i].average_ticket_price,
        reverse=True,
    )

    next_free = [0] * num_halls
    gene_pos = [0] * num_halls
    starts_at: dict[int, int] = {}
    show_counts = [0] * len(problem.movies)

    for h_idx in decode_order:
        while gene_pos[h_idx] < sph:
            slot = next_free[h_idx]
            while slot < problem.num_slots and starts_at.get(slot, 0) >= problem.lobby_max_starts:
                slot += 1
            if slot >= problem.num_slots:
                break

            best_m = -1
            best_score = -float("inf")
            for m in problem.movies:
                if slot + m.runtime_slots > problem.num_slots:
                    continue
                rev = problem.revenue(m.index, h_idx, slot)
                deficit = max(0, m.min_shows - show_counts[m.index])
                score = rev + deficit * problem.max_screening_revenue
                if score > best_score:
                    best_score = score
                    best_m = m.index

            if best_m < 0:
                break

            genes[h_idx * sph + gene_pos[h_idx]] = best_m
            gene_pos[h_idx] += 1
            movie = problem.movies[best_m]
            show_counts[best_m] += 1
            next_free[h_idx] = slot + movie.runtime_slots + problem.cleaning_buffer
            starts_at[slot] = starts_at.get(slot, 0) + 1

    return genes


# ---------------------------------------------------------------------------
# Decoder: chromosome -> feasible schedule
# ---------------------------------------------------------------------------

def decode_chromosome(
    genes: np.ndarray,
    problem: GAProblem,
) -> list[dict[str, Any]]:
    """Greedy hall-by-hall decoder.

    For each hall's segment of genes, read movie indices left-to-right and pack
    each movie at the earliest available slot.  -1 (sentinel) means "skip".
    Overlap + buffer + horizon + lobby cap are satisfied by construction.
    Halls are processed highest-revenue-potential first (capacity * price descending).
    """
    num_halls = len(problem.halls)
    sph = _max_screenings_per_hall(problem)
    num_movies = len(problem.movies)
    lobby_cap = problem.lobby_max_starts

    decode_order = sorted(
        range(num_halls),
        key=lambda i: problem.halls[i].capacity * problem.halls[i].average_ticket_price,
        reverse=True,
    )

    starts_at: dict[int, int] = {}
    next_free: list[int] = [0] * num_halls

    schedule: list[dict[str, Any]] = []
    for h_idx in decode_order:
        hall = problem.halls[h_idx]
        segment = genes[h_idx * sph : (h_idx + 1) * sph]

        for gene_val in segment:
            m_idx = int(gene_val)
            if m_idx < 0 or m_idx >= num_movies:
                continue
            movie = problem.movies[m_idx]

            start = next_free[h_idx]
            while start + movie.runtime_slots <= problem.num_slots:
                if starts_at.get(start, 0) < lobby_cap:
                    break
                start += 1
            end = start + movie.runtime_slots
            if end > problem.num_slots:
                continue

            starts_at[start] = starts_at.get(start, 0) + 1
            rev = problem.revenue(m_idx, h_idx, start)
            schedule.append(
                {
                    "hall_id": hall.id,
                    "hall_index": h_idx,
                    "movie_id": movie.id,
                    "movie_index": m_idx,
                    "movie_title": movie.title,
                    "weeks_since_release": movie.weeks_since_release,
                    "start_slot": start,
                    "end_slot": end,
                    "expected_revenue": rev,
                }
            )
            next_free[h_idx] = end + problem.cleaning_buffer

    return schedule


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def _evaluate(
    genes: np.ndarray,
    problem: GAProblem,
    min_shows_penalty: float,
) -> tuple[float, list[dict[str, Any]], int]:
    """Return (fitness, schedule, min_shows_deficit_total)."""
    schedule = decode_chromosome(genes, problem)

    total_revenue = sum(s["expected_revenue"] for s in schedule)

    show_counts: Counter[int] = Counter(s["movie_index"] for s in schedule)
    min_shows_deficit = sum(
        max(0, m.min_shows - show_counts.get(m.index, 0)) for m in problem.movies
    )

    fitness = total_revenue - min_shows_penalty * min_shows_deficit
    return fitness, schedule, min_shows_deficit


# ---------------------------------------------------------------------------
# solve_schedule_ga — main entry point
# ---------------------------------------------------------------------------

def solve_schedule_ga(
    movies_csv: str | Path,
    config_json: str | Path,
    *,
    day_type: str = "weekday",
    seed: int = 42,
    # PyGAD hyperparameters
    num_generations: int = 300,
    sol_per_pop: int = 150,
    num_parents_mating: int = 40,
    parent_selection_type: str = "tournament",
    K_tournament: int = 5,
    crossover_type: str = "two_points",
    mutation_type: str = "random",
    mutation_percent_genes: float = 15.0,
    keep_elitism: int = 5,
    # Penalty weights
    min_shows_penalty: float | None = None,
) -> dict[str, Any]:
    movies_csv = Path(movies_csv)
    config_json = Path(config_json)

    problem = GAProblem.from_files(movies_csv, config_json, day_type=day_type)

    max_rev = problem.max_screening_revenue
    if min_shows_penalty is None:
        min_shows_penalty = max_rev * 50.0

    num_genes = _num_genes(problem)
    num_movies = len(problem.movies)

    gene_space = [range(-1, num_movies) for _ in range(num_genes)]

    # Seed half the population with greedy heuristic + mutated variants
    rng = np.random.RandomState(seed)
    greedy_chromosome = _build_greedy_seed(problem)
    n_seeded = max(1, sol_per_pop // 2)
    initial_pop = np.array(
        [rng.randint(-1, num_movies, size=num_genes) for _ in range(sol_per_pop)],
        dtype=int,
    )
    initial_pop[0] = greedy_chromosome.copy()
    for i in range(1, n_seeded):
        variant = greedy_chromosome.copy()
        n_flip = max(1, num_genes // 10)
        flip_idx = rng.choice(num_genes, size=n_flip, replace=False)
        variant[flip_idx] = rng.randint(-1, num_movies, size=n_flip)
        initial_pop[i] = variant

    def fitness_func(ga_inst: pygad.GA, solution: np.ndarray, sol_idx: int) -> float:
        fit, *_ = _evaluate(solution, problem, min_shows_penalty)
        return float(fit)

    avg_fitness_history: list[float] = []

    def _on_gen(ga_inst: pygad.GA) -> None:
        avg_fitness_history.append(float(np.mean(ga_inst.last_generation_fitness)))

    ga_kwargs: dict[str, Any] = dict(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        initial_population=initial_pop,
        gene_type=int,
        gene_space=gene_space,
        parent_selection_type=parent_selection_type,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        keep_elitism=keep_elitism,
        random_seed=seed,
        suppress_warnings=True,
        save_best_solutions=True,
        on_generation=_on_gen,
    )
    if parent_selection_type == "tournament":
        ga_kwargs["K_tournament"] = K_tournament

    ga_instance = pygad.GA(**ga_kwargs)

    start_t = time.perf_counter()
    ga_instance.run()
    elapsed = time.perf_counter() - start_t

    best_solution, best_fitness, _best_idx = ga_instance.best_solution()
    fitness, schedule, ms_deficit = _evaluate(
        best_solution, problem, min_shows_penalty
    )

    # Build min_shows deficit detail
    show_counts: Counter[int] = Counter(s["movie_index"] for s in schedule)
    deficit_by_movie: list[dict[str, Any]] = []
    for m in problem.movies:
        d = max(0, m.min_shows - show_counts.get(m.index, 0))
        if d > 0:
            deficit_by_movie.append(
                {"movie_id": m.id, "movie_title": m.title, "deficit": d}
            )

    total_revenue = float(sum(s["expected_revenue"] for s in schedule))

    # Build output schedule rows (strip internal keys, add wall-clock times)
    output_schedule: list[dict[str, Any]] = []
    for s in schedule:
        output_schedule.append(
            {
                "hall_id": s["hall_id"],
                "movie_id": s["movie_id"],
                "movie_title": s["movie_title"],
                "weeks_since_release": s["weeks_since_release"],
                "start_slot": s["start_slot"],
                "start_time": slot_to_wall_clock(
                    s["start_slot"], problem.opening_minutes, problem.slot_duration
                ),
                "end_slot": s["end_slot"],
                "end_time": slot_to_wall_clock(
                    s["end_slot"], problem.opening_minutes, problem.slot_duration
                ),
                "expected_revenue": s["expected_revenue"],
            }
        )
    output_schedule.sort(key=lambda r: (r["hall_id"], r["start_slot"]))

    raw_best = ga_instance.best_solutions_fitness
    best_fitness_history = [float(f) for f in raw_best[-len(avg_fitness_history):]]

    return {
        "metadata": {
            "algorithm": "Genetic Algorithm (PyGAD)",
            "execution_time_seconds": elapsed,
            "total_revenue": total_revenue,
            "fitness_score": float(fitness),
            "constraints_violated": ms_deficit,
            "min_shows_deficit_total": ms_deficit,
            "min_shows_deficit_by_movie": deficit_by_movie,
            "day_type": day_type,
            "num_slots": problem.num_slots,
            "num_halls": len(problem.halls),
            "num_movies": len(problem.movies),
            "seed": seed,
            "num_generations": num_generations,
            "sol_per_pop": sol_per_pop,
        },
        "convergence": {
            "best_fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history,
        },
        "schedule": output_schedule,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GA movie scheduler (PyGAD)")
    parser.add_argument("--movies-csv", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument(
        "--day-type", type=str, default="weekday", choices=["weekday", "weekend"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--num-generations", type=int, default=300)
    parser.add_argument("--sol-per-pop", type=int, default=150)
    parser.add_argument("--num-parents-mating", type=int, default=40)
    parser.add_argument("--parent-selection-type", type=str, default="tournament")
    parser.add_argument("--K-tournament", type=int, default=5)
    parser.add_argument("--crossover-type", type=str, default="two_points")
    parser.add_argument("--mutation-type", type=str, default="random")
    parser.add_argument("--mutation-percent-genes", type=float, default=15.0)
    parser.add_argument("--keep-elitism", type=int, default=5)
    parser.add_argument("--min-shows-penalty", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = solve_schedule_ga(
        movies_csv=args.movies_csv,
        config_json=args.config_json,
        day_type=args.day_type,
        seed=args.seed,
        num_generations=args.num_generations,
        sol_per_pop=args.sol_per_pop,
        num_parents_mating=args.num_parents_mating,
        parent_selection_type=args.parent_selection_type,
        K_tournament=args.K_tournament,
        crossover_type=args.crossover_type,
        mutation_type=args.mutation_type,
        mutation_percent_genes=args.mutation_percent_genes,
        keep_elitism=args.keep_elitism,
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
