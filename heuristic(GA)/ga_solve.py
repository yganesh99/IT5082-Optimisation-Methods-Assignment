from __future__ import annotations

import copy
import json
import math
import random
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.utils import calculate_screening_revenue, generate_robust_demand_matrix

Screening = tuple[int, int, int]  # (movie_idx, hall_idx, start_slot)
Chromosome = list[Screening]


@dataclass(frozen=True)
class Hall:
    id: str
    capacity: int
    average_ticket_price: float


def _clock_to_minutes(hhmm: str) -> int:
    h, m = (int(x) for x in hhmm.strip().split(":"))
    return h * 60 + m


class GAProblem:
    """Loads a scheduling instance and runs the genetic algorithm (same logic as the course notebook)."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        config: dict[str, Any],
        day_type: str,
    ) -> None:
        self.day_type = day_type
        self.config = config
        constraints = config["constraints"]
        self.slot_duration = int(constraints["slot_duration_minutes"])
        self.cleaning_buffer = int(constraints.get("cleaning_buffer_slots", 1))
        self.lobby_max_starts = int(constraints.get("lobby_max_simultaneous_starts", 2))

        self.opening_min = _clock_to_minutes(constraints["opening_time"])
        closing_min = _clock_to_minutes(constraints["closing_time"])
        if bool(constraints.get("closing_time_is_next_calendar_day", True)) and closing_min <= self.opening_min:
            closing_min += 24 * 60
        self.num_slots = (closing_min - self.opening_min) // self.slot_duration

        hall_type_map = {ht["id"]: ht for ht in config["hall_types"]}
        self.halls = [
            Hall(
                id=str(h["id"]),
                capacity=int(hall_type_map[h["hall_type_id"]]["capacity"]),
                average_ticket_price=float(hall_type_map[h["hall_type_id"]]["average_ticket_price"]),
            )
            for h in config["halls"]
        ]
        self.hall_dicts = [
            {"id": h.id, "capacity": h.capacity, "average_ticket_price": h.average_ticket_price}
            for h in self.halls
        ]
        self.num_halls = len(self.halls)

        self.movies_df = movies_df.reset_index(drop=True).copy()
        self.movies_df["runtime_slots"] = self.movies_df["runtime"].apply(
            lambda x: int(math.ceil(float(x) / self.slot_duration))
        )
        self.movies_df["min_shows"] = self.movies_df["min_shows"].astype(int)
        self.num_movies = len(self.movies_df)

        movie_records = [
            {
                "index": int(i),
                "base_popularity": float(row["popularity"]),
                "weeks_since_release": int(row["weeks_since_release"]),
                "genre_peak_slot": int(row["genre_peak_slot"]),
                "genre_sigma": int(row["genre_sigma"]),
            }
            for i, row in self.movies_df.iterrows()
        ]
        self.demand_matrix = generate_robust_demand_matrix(
            movie_records, day_type=day_type, num_slots=self.num_slots
        )

        self.movie_meta = [
            {"index": int(i), "runtime_slots": int(row["runtime_slots"])}
            for i, row in self.movies_df.iterrows()
        ]

        self.valid_starts: dict[int, list[int]] = {
            m: [t for t in range(self.num_slots) if t + self.movie_meta[m]["runtime_slots"] <= self.num_slots]
            for m in range(self.num_movies)
        }

        self.revenue_lookup = np.zeros((self.num_movies, self.num_halls, self.num_slots))
        for m in range(self.num_movies):
            for h in range(self.num_halls):
                for t in self.valid_starts[m]:
                    self.revenue_lookup[m, h, t] = calculate_screening_revenue(
                        movie=self.movie_meta[m],
                        hall=self.hall_dicts[h],
                        start_slot=t,
                        demand_matrix=self.demand_matrix,
                    )

        self.max_screening_revenue = float(self.revenue_lookup.max())
        self.penalty_min_shows = max(self.max_screening_revenue * 50.0, 1e-4)

        self._movie_id_to_idx = {str(self.movies_df.at[i, "id"]): i for i in range(self.num_movies)}
        self._hall_id_to_idx = {self.halls[i].id: i for i in range(self.num_halls)}

    @classmethod
    def from_files(cls, movies_csv: Path, config_json: Path, day_type: str) -> GAProblem:
        movies_df = pd.read_csv(movies_csv)
        config = json.loads(config_json.read_text(encoding="utf-8"))
        return cls(movies_df, config, day_type)

    def slot_to_wall_clock(self, slot: int) -> str:
        total = self.opening_min + slot * self.slot_duration
        total %= 24 * 60
        h, m = divmod(total, 60)
        return f"{h:02d}:{m:02d}"

    def blocked_interval(self, movie_idx: int, start_slot: int) -> tuple[int, int]:
        return (
            start_slot,
            start_slot + self.movie_meta[movie_idx]["runtime_slots"] + self.cleaning_buffer,
        )

    def screenings_by_hall(self, chromosome: Chromosome) -> dict[int, list[Screening]]:
        by_hall: dict[int, list[Screening]] = {h: [] for h in range(self.num_halls)}
        for s in chromosome:
            by_hall[s[1]].append(s)
        return by_hall

    def movie_show_counts(self, chromosome: Chromosome) -> dict[int, int]:
        counts: dict[int, int] = {m: 0 for m in range(self.num_movies)}
        for m, _h, _t in chromosome:
            counts[m] += 1
        return counts

    def _can_place(
        self,
        m: int,
        h: int,
        t: int,
        hall_occ: dict[int, np.ndarray],
        starts: np.ndarray,
    ) -> bool:
        lo, hi = self.blocked_interval(m, t)
        if hi > self.num_slots:
            return False
        if hall_occ[h][lo:hi].any():
            return False
        if starts[t] >= self.lobby_max_starts:
            return False
        return True

    def _place(
        self,
        m: int,
        h: int,
        t: int,
        hall_occ: dict[int, np.ndarray],
        starts: np.ndarray,
    ) -> None:
        lo, hi = self.blocked_interval(m, t)
        hall_occ[h][lo : min(hi, self.num_slots)] = True
        starts[t] += 1

    def repair(self, chromosome: Chromosome) -> Chromosome:
        hall_occ = {h: np.zeros(self.num_slots, dtype=bool) for h in range(self.num_halls)}
        starts = np.zeros(self.num_slots, dtype=int)
        repaired: Chromosome = []

        scored = sorted(
            chromosome,
            key=lambda s: self.revenue_lookup[s[0], s[1], s[2]],
            reverse=True,
        )

        for m, h, t in scored:
            if self._can_place(m, h, t, hall_occ, starts):
                self._place(m, h, t, hall_occ, starts)
                repaired.append((m, h, t))

        counts = self.movie_show_counts(repaired)
        for m in range(self.num_movies):
            deficit = int(self.movies_df.at[m, "min_shows"]) - counts[m]
            if deficit <= 0:
                continue
            candidates = [
                (h, t)
                for h in range(self.num_halls)
                for t in self.valid_starts[m]
            ]
            candidates.sort(key=lambda ht: self.revenue_lookup[m, ht[0], ht[1]], reverse=True)
            for h, t in candidates:
                if deficit <= 0:
                    break
                if self._can_place(m, h, t, hall_occ, starts):
                    self._place(m, h, t, hall_occ, starts)
                    repaired.append((m, h, t))
                    deficit -= 1

        return repaired

    def _random_individual(self) -> Chromosome:
        hall_occ = {h: np.zeros(self.num_slots, dtype=bool) for h in range(self.num_halls)}
        starts = np.zeros(self.num_slots, dtype=int)
        chrom: Chromosome = []

        movie_order = list(range(self.num_movies))
        random.shuffle(movie_order)
        for m in movie_order:
            needed = int(self.movies_df.at[m, "min_shows"])
            placed = 0
            attempts = 0
            while placed < needed and attempts < 300:
                attempts += 1
                h = random.randint(0, self.num_halls - 1)
                if not self.valid_starts[m]:
                    break
                t = random.choice(self.valid_starts[m])
                if self._can_place(m, h, t, hall_occ, starts):
                    self._place(m, h, t, hall_occ, starts)
                    chrom.append((m, h, t))
                    placed += 1

        stalls = 0
        while stalls < 80:
            m = random.randint(0, self.num_movies - 1)
            h = random.randint(0, self.num_halls - 1)
            if not self.valid_starts[m]:
                stalls += 1
                continue
            t = random.choice(self.valid_starts[m])
            if self._can_place(m, h, t, hall_occ, starts):
                self._place(m, h, t, hall_occ, starts)
                chrom.append((m, h, t))
                stalls = 0
            else:
                stalls += 1
        return chrom

    def init_population(self, size: int) -> list[Chromosome]:
        return [self._random_individual() for _ in range(size)]

    def evaluate(self, chromosome: Chromosome) -> tuple[float, float, int]:
        total_rev = sum(self.revenue_lookup[m, h, t] for m, h, t in chromosome)
        violations = 0
        counts = self.movie_show_counts(chromosome)
        for m in range(self.num_movies):
            deficit = int(self.movies_df.at[m, "min_shows"]) - counts[m]
            if deficit > 0:
                violations += deficit
        fitness = total_rev - violations * self.penalty_min_shows
        return fitness, total_rev, violations

    def tournament_select(
        self,
        pop: list[Chromosome],
        fits: list[tuple[float, float, int]],
        tournament_size: int,
    ) -> Chromosome:
        idxs = random.sample(range(len(pop)), tournament_size)
        winner = max(idxs, key=lambda i: fits[i][0])
        return copy.deepcopy(pop[winner])

    def crossover(self, p1: Chromosome, p2: Chromosome, crossover_rate: float) -> tuple[Chromosome, Chromosome]:
        if random.random() > crossover_rate:
            return copy.deepcopy(p1), copy.deepcopy(p2)

        bh1, bh2 = self.screenings_by_hall(p1), self.screenings_by_hall(p2)
        c1: Chromosome = []
        c2: Chromosome = []
        for h in range(self.num_halls):
            if random.random() < 0.5:
                c1.extend(bh1[h])
                c2.extend(bh2[h])
            else:
                c1.extend(bh2[h])
                c2.extend(bh1[h])
        return c1, c2

    def mutate(self, chromosome: Chromosome, mutation_rate: float) -> Chromosome:
        if random.random() > mutation_rate:
            return chromosome
        chromosome = list(chromosome)
        op = random.choice(["add", "remove", "shift", "swap_hall"])

        if op == "add":
            m = random.randint(0, self.num_movies - 1)
            h = random.randint(0, self.num_halls - 1)
            if self.valid_starts[m]:
                t = random.choice(self.valid_starts[m])
                chromosome.append((m, h, t))

        elif op == "remove" and chromosome:
            chromosome.pop(random.randint(0, len(chromosome) - 1))

        elif op == "shift" and chromosome:
            idx = random.randint(0, len(chromosome) - 1)
            m, h, _t = chromosome[idx]
            if self.valid_starts[m]:
                chromosome[idx] = (m, h, random.choice(self.valid_starts[m]))

        elif op == "swap_hall" and chromosome and self.num_halls > 1:
            idx = random.randint(0, len(chromosome) - 1)
            m, old_h, t = chromosome[idx]
            new_h = random.choice([hh for hh in range(self.num_halls) if hh != old_h])
            chromosome[idx] = (m, new_h, t)

        return chromosome

    def solve(
        self,
        *,
        pop_size: int = 150,
        generations: int = 500,
        mutation_rate: float = 0.4,
        crossover_rate: float = 0.85,
        tournament_size: int = 5,
        elitism_count: int = 2,
        seed: int | None = 42,
        verbose: bool = False,
        trace_memory: bool = False,
    ) -> tuple[dict[str, Any], dict[str, list], dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if trace_memory:
            tracemalloc.start()

        t_wall0 = time.perf_counter()
        rusage_before = resource.getrusage(resource.RUSAGE_SELF)

        population = self.init_population(pop_size)
        population = [self.repair(ind) for ind in population]

        best_chrom: Chromosome | None = None
        best_fit = float("-inf")
        best_rev = 0.0
        best_viol = 0
        last_improve_gen = 0

        history: dict[str, list] = {
            "gen": [],
            "best_fit": [],
            "avg_fit": [],
            "best_rev": [],
        }

        for gen in range(generations):
            fits = [self.evaluate(ind) for ind in population]

            for i, (f, r, v) in enumerate(fits):
                if f > best_fit:
                    best_fit, best_rev, best_viol = f, r, v
                    best_chrom = copy.deepcopy(population[i])
                    last_improve_gen = gen

            avg_f = float(np.mean([f[0] for f in fits]))
            history["gen"].append(gen)
            history["best_fit"].append(best_fit)
            history["avg_fit"].append(avg_f)
            history["best_rev"].append(best_rev)

            if verbose and (gen % 50 == 0 or gen == generations - 1):
                print(
                    f"Gen {gen:4d}  |  best fitness {best_fit:.6e}  |  "
                    f"revenue {best_rev:.6e}  |  violations {best_viol}  |  "
                    f"avg fitness {avg_f:.6e}"
                )

            ranked = sorted(range(len(population)), key=lambda i: fits[i][0], reverse=True)
            new_pop: list[Chromosome] = [
                copy.deepcopy(population[ranked[j]]) for j in range(elitism_count)
            ]

            while len(new_pop) < pop_size:
                p1 = self.tournament_select(population, fits, tournament_size)
                p2 = self.tournament_select(population, fits, tournament_size)
                c1, c2 = self.crossover(p1, p2, crossover_rate)
                c1 = self.repair(self.mutate(c1, mutation_rate))
                c2 = self.repair(self.mutate(c2, mutation_rate))
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)

            population = new_pop

        elapsed = time.perf_counter() - t_wall0
        rusage_after = resource.getrusage(resource.RUSAGE_SELF)

        tracemalloc_peak: int | None = None
        if trace_memory:
            _, tracemalloc_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        max_rss_delta = rusage_after.ru_maxrss - rusage_before.ru_maxrss
        if sys.platform == "darwin":
            peak_rss_bytes = int(rusage_after.ru_maxrss)
        else:
            peak_rss_bytes = int(rusage_after.ru_maxrss * 1024)

        assert best_chrom is not None

        schedule = []
        for m, h, t in best_chrom:
            rs = self.movie_meta[m]["runtime_slots"]
            end_slot = t + rs
            schedule.append(
                {
                    "hall_id": self.halls[h].id,
                    "movie_id": str(self.movies_df.at[m, "id"]),
                    "movie_title": str(self.movies_df.at[m, "title"]),
                    "weeks_since_release": int(self.movies_df.at[m, "weeks_since_release"]),
                    "start_slot": int(t),
                    "start_time": self.slot_to_wall_clock(t),
                    "end_slot": int(end_slot),
                    "end_time": self.slot_to_wall_clock(end_slot),
                    "expected_revenue": float(self.revenue_lookup[m, h, t]),
                }
            )
        schedule.sort(key=lambda r: (r["hall_id"], r["start_slot"]))

        result = {
            "metadata": {
                "algorithm": "Genetic Algorithm",
                "solver_status": "COMPLETED",
                "execution_time_seconds": elapsed,
                "total_revenue": float(sum(s["expected_revenue"] for s in schedule)),
                "fitness_score": float(best_fit),
                "constraints_violated": int(best_viol),
                "day_type": self.day_type,
                "num_slots": self.num_slots,
                "num_halls": self.num_halls,
                "num_movies": self.num_movies,
                "ga_parameters": {
                    "population_size": pop_size,
                    "num_generations": generations,
                    "tournament_size": tournament_size,
                    "crossover_rate": crossover_rate,
                    "mutation_rate": mutation_rate,
                    "elitism_count": elitism_count,
                    "random_seed": seed,
                },
            },
            "schedule": schedule,
        }

        diagnostics = {
            "wall_time_seconds": elapsed,
            "last_improvement_generation": last_improve_gen,
            "generations_without_improvement_after_last": generations - 1 - last_improve_gen,
            "peak_rss_bytes": peak_rss_bytes,
            "max_rss_delta_rusage_units": max_rss_delta,
            "tracemalloc_peak_bytes": tracemalloc_peak,
        }

        return result, history, diagnostics

    def expected_revenue_for_screening(self, movie_id: str, hall_id: str, start_slot: int) -> float | None:
        mid = self._movie_id_to_idx.get(str(movie_id))
        hid = self._hall_id_to_idx.get(hall_id)
        if mid is None or hid is None:
            return None
        if start_slot < 0 or start_slot >= self.num_slots:
            return None
        if start_slot not in self.valid_starts[mid]:
            return None
        return float(self.revenue_lookup[mid, hid, start_slot])
