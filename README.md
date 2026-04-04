# Multi-Screen Automated Movie Scheduling

This document describes the optimisation problem for **one multiplex** with multiple screens (halls). The model is **not** intended to schedule all multiplexes nationwide; it applies **per multiplex**.

Scheduling is more than choosing a time slot from a list of free slots. The formulation must balance several competing goals.

## Design considerations

1. **Capacity vs. demand** — High-blockbuster titles should be placed in larger halls during peak periods when demand is strongest.
2. **Turnaround time** — Cleaning and maintenance must be respected between consecutive shows in the same hall.
3. **Operational constraints** — Overlapping start times should not exceed what lobby and concession staff can handle.
4. **Revenue maximisation** — Prefer screenings that yield higher expected revenue (e.g. high-margin slots and demand-weighted attendance).

## Demand and time discretisation

- **Demand** can be modelled using the **genre** of each movie and may **decay or reduce over time** within the day (e.g. late shows vs. prime time).
- The **day is discretised into 15-minute windows** (slots). All start times and durations are expressed in these slots.

## Decision variables

Let $x_{m,h,t}$ be a **binary** variable:

- $x_{m,h,t} = 1$ if movie $m$ **starts** in hall $h$ at time slot $t$,
- $x_{m,h,t} = 0$ otherwise.

## Objective function (maximise total expected revenue)

Maximise

$$
Z = \sum_{m} \sum_{h} \sum_{t} \bigl( x_{m,h,t} \cdot \mathrm{Capacity}_h \cdot \mathrm{AvgTicketPrice}_h \cdot \mathrm{DemandWeight}_{m,t} \bigr).
$$

Interpretation: for each feasible start $(m,h,t)$, expected revenue is proportional to hall capacity, average ticket price for that hall, and a demand weight for movie $m$ at slot $t$ (which can encode genre and time-of-day effects).

## Constraints

1. **No overlap in a hall** — Hall $h$ cannot run two movies in the same slot, and a new show cannot start until the previous one has **ended**, accounting for required buffers (see constraint 4).
2. **End-by deadline** — All screenings must **finish** by the last allowed slot $T$ (e.g. **1:00 AM** in the problem statement), expressed in the same 15-minute indexing as $t$.
3. **Lobby / concurrent starts** — In any single 15-minute window, the **number of movie starts** is limited (e.g. **at most 2** simultaneous starts) so lobby and ticket-scanning load stays manageable.
4. **Cleaning and scanning buffer** — Enforce **one slot (15 minutes)** of buffer between the end of one show and the start of the next in the same hall (cleaning, ticket scanning, turnover), consistent with the no-overlap rules.
5. **Movie assignment / contracts** — Each movie must be shown **at least once**, or a **contract-specific** minimum number of screenings, depending on business rules.

## Output format

A run (e.g. genetic algorithm or exact solver) can report results as JSON with two top-level keys:

- **`metadata`** — Run summary: algorithm name, wall-clock time, aggregate revenue, fitness (if used), and how many constraints were violated.
- **`schedule`** — Ordered list of screenings; each entry ties a hall to one movie with **slot indices** (15-minute units, aligned with the model) and human-readable times, plus **expected revenue** for that screening.

Example:

```json
{
  "metadata": {
    "algorithm": "Genetic Algorithm",
    "execution_time_seconds": 12.4,
    "total_revenue": 14500.50,
    "fitness_score": 0.98,
    "constraints_violated": 0
  },
  "schedule": [
    {
      "hall_id": "Hall_01",
      "movie_id": "M_102",
      "movie_title": "Interstellar",
      "start_slot": 4,
      "start_time": "11:00 AM",
      "end_slot": 13,
      "end_time": "01:15 PM",
      "expected_revenue": 450.00
    },
    {
      "hall_id": "Hall_01",
      "movie_id": "M_205",
      "movie_title": "The Batman",
      "start_slot": 15,
      "start_time": "01:45 PM",
      "end_slot": 27,
      "end_time": "04:45 PM",
      "expected_revenue": 620.00
    }
  ]
}
```

---

*This README states the mathematical problem; implementation (exact solver vs. heuristic, data sources for $\mathrm{DemandWeight}_{m,t}$, and concrete values for $T$ and hall parameters) lives in the codebase and configuration.*
