# Multi-Screen Automated Movie Scheduling

This document describes the optimisation problem for **one multiplex** with multiple screens (halls). The model is **not** intended to schedule all multiplexes nationwide; it applies **per multiplex**.

Scheduling is more than choosing a time slot from a list of free slots. The formulation must balance several competing goals.

## Design considerations

1. **Capacity vs. demand** — High-blockbuster titles should be placed in larger halls during peak periods when demand is strongest.
2. **Turnaround time** — Cleaning and maintenance must be respected between consecutive shows in the same hall.
3. **Operational constraints** — Overlapping start times should not exceed what lobby and concession staff can handle.
4. **Revenue maximisation** — Prioritise high-margin screenings and demand-weighted expected attendance.
5. **Genre–time alignment** — A single Gaussian peak (e.g. “8:00 PM for everything”) is too coarse. In practice, animated/family titles often peak around late morning or mid-afternoon, horror/R-rated titles later in the evening, and blockbusters often have a broader peak.
6. **Movie lifecycle** — A title that premiered recently is typically worth more in schedule terms than one that has already been on screen for several weeks (reflected in base popularity, release age, or similar inputs).

## Demand and time discretisation

- **Demand** can be modelled using the **genre** of each movie and may **decay or reduce over time** (within the day and/or over weeks since release).
- The **day is discretised into 15-minute windows** (slots). All start times and durations are expressed in these slots.

## Decision variables

Let $x_{m,h,t}$ be a **binary** variable:

- $x_{m,h,t} = 1$ if movie $m$ **starts** in hall $h$ at time slot $t$,
- $x_{m,h,t} = 0$ otherwise.

## Objective function (maximise total expected revenue)

Maximise

$$
Z = \sum_{m} \sum_{h} \sum_{t} \bigl( x_{m,h,t} \cdot \mathrm{Capacity}_h \cdot P_h \cdot W_{m,h,t} \bigr).
$$

- $P_h$ — average ticket price for hall $h$ (hall type / pricing tier).
- $W_{m,h,t}$ — **demand weight** for movie $m$ (and hall $h$ if you model hall-specific demand) at slot $t$, scaled **between 0 and 1**.

A convenient factorisation for the time- and genre-driven part is

$$
W_{m,t} = \mathrm{BasePopularity}_m \times \mathrm{Decay}_t \times \mathrm{GenreTimeFit}_{g,t} \times \mathrm{DayMulti}_{\mathrm{day}} \times \mathrm{TimeMulti}_{\mathrm{slot}}.
$$

Here $g$ is the genre of movie $m$, and $\mathrm{BasePopularity}_m$ can encode lifecycle (e.g. freshness vs. weeks since release). When demand does not depend on hall, use $W_{m,h,t} = W_{m,t}$; otherwise multiply by a hall-specific factor inside $W_{m,h,t}$.

Exponential decay (example for a weight component over slot index $t$):

$$
\mathrm{Decay} = W_0 \cdot e^{-\lambda t}.
$$

(Implementations may apply decay to the **within-day** slot index, to **weeks since release**, or both—depending on how $\mathrm{BasePopularity}_m$ and $\mathrm{Decay}_t$ are defined in your data pipeline.)

## Constraints

1. **No overlap in a hall** — Hall $h$ cannot run two movies in the same slot, and a new show cannot start until the previous one has **ended**, accounting for required buffers (see constraint 4).
2. **End-by deadline** — All screenings must **finish** by the last allowed slot $T$ (e.g. **1:00 AM** in the problem statement), expressed in the same 15-minute indexing as $t$.
3. **Lobby / concurrent starts** — In any single 15-minute window, the **number of movie starts** is limited (e.g. **at most 2** simultaneous starts) so lobby and ticket-scanning load stays manageable.
4. **Cleaning and scanning buffer** — Enforce **one slot (15 minutes)** of buffer between the end of one show and the start of the next in the same hall (cleaning, ticket scanning, turnover), consistent with the no-overlap rules.
5. **Movie assignment / contracts** — Each movie must be shown **at least once**, or a **contract-specific** minimum number of screenings, depending on business rules.

## Data required

| Dataset component | Why we need it | Where to find |
| --- | --- | --- |
| **Movie CSV** (e.g. Kaggle / TMDB / IMDb / MovieLens) | Durations, genres, **release date** or **weeks since release** as raw input for lifecycle and grouping. | TMDB, IMDb, MovieLens, Kaggle, etc. |
| **Cinema JSON** | Hall types, capacities, **average ticket price per hall type**. | Synthetic (config file) |
| **Enhanced demand matrix** | Time-of-day + genre fit + decay; applies decay (and multipliers) to build demand weights. | Synthetic / derived |
| **Historical demand** (optional calibration) | Average ticket price by context, occupancy by genre, weekend vs. weekday multipliers. | Kaggle / box-office style datasets |
| **Staffing constraints** | Max starts per window (lobby cap). | Hardcode in notebooks or config |
| **Movie contracts** | Minimum number of showings per title. | Synthetic |

## Output format (each method)

Every method (exact, heuristic, etc.) should be able to emit the same JSON shape: **`metadata`** (run summary) and **`schedule`** (list of screenings with slot indices and human-readable times).

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

## Constraint validator

A small **`validator.py`** (or equivalent) can read this JSON and check feasibility, for example:

- **Overlaps (per hall)** — For the same `hall_id`, does the interval `[start_slot, end_slot)` of one screening overlap another (after accounting for buffer rules if you model them explicitly in slots)?
- **Lobby congestion** — Count screenings by `start_slot`; if any slot has **more than 2** starts, the schedule is invalid (per the lobby constraint).
- **1 AM rule** — Does any `end_slot` exceed the maximum allowed end slot for the operating day?

## Future work

- **Screen-type compatibility** — 3D, IMAX, standard, etc.
- **Finer pricing** — Price varying by seat tier within the same hall, not only by time/hall.
- **Genre–hall fit** — Preferring certain genres or formats in specific hall types.

---

*Problem statement, data sources, and output contract live here; solvers, notebooks, and `validator.py` live in the repository implementation.*
