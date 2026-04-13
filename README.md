# Multi-Screen Automated Movie Scheduling

This document describes the optimisation problem for **one multiplex** with multiple screens (halls). The model is **not** intended to schedule all multiplexes nationwide; it applies **per multiplex** for **a single operating day**. In a report, note that while we optimise **per day**, a real cinema chain would typically run this **seven times** (once per day) to build a **weekly roster**.

## Python environment (notebooks / collaborators)

Dependencies for notebooks (e.g. `heuristic(GA)/genetic_algorithm_optimizer.ipynb`) are listed in **`requirements.txt`**. Use a **project virtual environment** so installs work the same on every machine (and avoid macOS Homebrew “externally managed environment” / PEP 668 errors).

**Easiest — run once from the repo root:**

| OS            | Command                                          |
| ------------- | ------------------------------------------------ |
| macOS / Linux | `bash setup_env.sh`                              |
| Windows       | `setup_env.bat` (double-click or run from `cmd`) |

**Then in VS Code:** Command Palette → **Python: Select Interpreter** → choose the interpreter under **`.venv`** (or **Enter interpreter path**: macOS/Linux `.venv/bin/python`, Windows `.venv\Scripts\python.exe`).

**Manual (same effect):** `python3 -m venv .venv` → activate (macOS/Linux: `source .venv/bin/activate`; Windows: `.venv\Scripts\activate`) → `python -m pip install -r requirements.txt`.

## Design considerations

1. **Capacity vs. demand** — High-blockbuster titles need **larger halls during peak hours** when demand is strongest.
2. **Turnaround time** — **Cleaning and maintenance** between shows must be respected in the same hall.
3. **Operational constraints** — **Overlapping start times** must not overwhelm lobby and concession staff (modelled via a cap on simultaneous starts per window).
4. **Revenue maximisation** — Prioritise **high-margin** screenings via capacity, price, and demand-weighted weights.
5. **Genre–time alignment** — A simple Gaussian curve that says “8:00 PM peaks for everything” is too coarse. In practice, **animated/family** titles often peak around **11:00 AM or 2:00 PM**, **horror / R-rated** titles peak **after 9:00 PM**, and **blockbusters** tend to have a **broader** peak.
6. **Movie lifecycle** — A movie that **premiered yesterday** is typically worth more in schedule terms than one that has been out for **several weeks** (e.g. via decay on weeks since release).

## Demand and time discretisation

- The **day is discretised into 15-minute windows** (slots). All start times and durations are expressed in these slots; **durations are rounded up to the nearest 15 minutes** so staffing and slot indexing stay clean.

## Decision variables

Let $x_{m,h,t}$ be a **binary** variable:

- $x_{m,h,t} = 1$ if movie $m$ **starts** in hall $h$ at time slot $t$,
- $x_{m,h,t} = 0$ otherwise.

## Objective function (maximise total expected revenue)

Maximise total expected revenue:

$$
Z = \sum_{m} \sum_{h} \sum_{t} \bigl( x_{m,h,t} \cdot \mathrm{Capacity}_h \cdot P_h \cdot W_{m,t} \bigr).
$$

- $P_h$ — **average ticket price** for hall $h$ (hall type / pricing tier).
- $W_{m,t}$ — **demand weight** for movie $m$ at start slot $t$, a value **between 0 and 1** (interpreted as an expected fill / demand multiplier in the revenue term).
- $k_m$ — **weeks since release** for movie $m$ (used inside decay).

## Constraints

1. **No overlap in a hall** — Hall $h$ cannot run two movies in the same slot, and a new show cannot start until the previous one has **ended**, accounting for required buffers (see constraint 4).
2. **End-by deadline** — All screenings must **finish** by the last allowed slot $T$ (e.g. **1:00 AM**), expressed in the same 15-minute indexing as $t$.
3. **Lobby / concurrent starts** — In any single 15-minute window, the **number of movie starts** is limited (e.g. **at most 2** simultaneous starts) so lobby and ticket-scanning load stays manageable.
4. **Cleaning and scanning buffer** — Enforce **one slot (15 minutes)** of buffer between the end of one show and the start of the next in the same hall (cleaning, ticket scanning, turnover), consistent with the no-overlap rules.
5. **Movie assignment / contracts** — Each movie should be shown **a target number of times** as specified in its contract (`min_shows` / contractual obligation).

## Data required

While movie metadata was sourced from the **TMDB**-style Kaggle dataset, we **engineer a multi-factor demand matrix** to approximate real-world behaviour: **genre-specific temporal peaks**, **release-date decay**, and time-of-day multipliers. That way the optimiser prioritises **high-value** placements rather than naive packing.

The **primary dataset** is the Kaggle TMDB movies release (**`TMDB_movie_dataset_v11.csv`**): [TMDB Movies Dataset 2023 (930k+ movies) on Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies?select=TMDB_movie_dataset_v11.csv).

**Cinema config by dataset size** — Config files differ for **small / medium / large** tiers:

| **Dataset size**        | **Configuration (halls)**         | **Inventory (movies)** | **Purpose**                                                                                                              |
| ----------------------- | --------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Small** (debugger)    | **2 halls** (e.g. 100 & 50 seats) | **4–6 movies**         | Prove the formulation; ILP should reach a global optimum in **under ~1 second**.                                         |
| **Medium** (multiplex)  | **5–8 halls** (mixed capacities)  | **12–18 movies**       | “Real world” scale; ILP may take **~10–60 s**; GA should get **close** to ILP revenue.                                   |
| **Large** (mega-cinema) | **15–20 halls**                   | **30–50 movies**       | **Scalability**; ILP may **time out** or take **10+ minutes**; GA should finish in **seconds** for industrial-style use. |

When splitting the parent CSV into **small / medium / large**, select the **most popular** movies, add **`weeks_since_release`** from the release date, **`min_shows`** (e.g. random integer **1–3**), and **`genre_peak_slot`** / **`genre_sigma`** for demand generation by mapping the movie’s **primary genre** to the table below.

| **Genre**              | **genre_peak_slot** | **genre_sigma** | **Reasoning**                                       |
| ---------------------- | ------------------- | --------------- | --------------------------------------------------- |
| **Animation / Family** | **12** (1:00 PM)    | **4**           | Kids’ movies peak earlier; narrow “bedtime” cutoff. |
| **Comedy / Romance**   | **32** (6:00 PM)    | **6**           | Pre-dinner / early evening.                         |
| **Action / Adventure** | **40** (8:00 PM)    | **8**           | Prime-time blockbuster; broader spread.             |
| **Horror / Thriller**  | **48** (10:00 PM)   | **5**           | Late peak; low morning demand.                      |
| **Documentary**        | **20** (3:00 PM)    | **10**          | Broad, lower-intensity daytime spread.              |

While historical metadata gives runtimes and genres, it **lacks operational demand parameters**. The latent features **`genre_peak_slot`** and **`genre_sigma`** are **derived from behavioural attendance patterns** so the optimiser can place **family** content in **morning/afternoon** slots and **horror** later, instead of treating all genres as uniform over the day.

For each dataset tier, build a demand matrix of shape **(number of movies) × (number of time slots)**. From **10:00 AM to 1:00 AM** there are **60** fifteen-minute slots. Every slot column is filled because **start times are decision variables**—the matrix is a **lookup table of weights** for “if movie $m$ starts at $t$.”

We need to round up movie durations to the nearest 15 minutes so staffing and slot constraints stay consistent.

| Dataset component                                     | Why we need it                                                                                                          | Where to find                                                                                                                                                                                               |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Movie CSV** (e.g. Kaggle / TMDB / IMDb / MovieLens) | Durations, genres, **release date** / **weeks since release**; engineered fields for contracts and demand.              | Primary: [Kaggle — `TMDB_movie_dataset_v11.csv`](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies?select=TMDB_movie_dataset_v11.csv). Alternates: TMDB, IMDb, MovieLens, etc. |
| **Cinema JSON**                                       | Hall types, capacities, average ticket price per hall type, max starts per window, window size, last movie end time.    | Synthetic (config file per tier: small / medium / large)                                                                                                                                                    |
| **Enhanced demand matrix**                            | Time-of-day + genre–time fit + decay; **value landscape** for every $(m,t)$ start possibility.                          | Synthetic / generated by `DemandGenerator`                                                                                                                                                                  |
| **Movie contracts**                                   | Target number of showings per title (`min_shows`); may be enforced strictly or via **penalty** (soft) under congestion. | Synthetic fields added when splitting into small, medium, and large datasets.                                                                                                                               |

## Output format for easy comparison (each method)

Every method (exact, heuristic, etc.) should be able to emit the same JSON shape: **`metadata`** (run summary) and **`schedule`** (list of screenings with slot indices and human-readable times).

Example:

```json
{
	"metadata": {
		"algorithm": "Genetic Algorithm",
		"execution_time_seconds": 12.4,
		"total_revenue": 14500.5,
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
			"expected_revenue": 450.0
		},
		{
			"hall_id": "Hall_01",
			"movie_id": "M_205",
			"movie_title": "The Batman",
			"start_slot": 15,
			"start_time": "01:45 PM",
			"end_slot": 27,
			"end_time": "04:45 PM",
			"expected_revenue": 620.0
		}
	]
}
```

## Future work

- **Screen-type compatibility** — 3D, IMAX, standard, etc.
- **Finer pricing** — Price varying by **seat tier** and **time** within the same hall, not only by hall type.
- **Genre–hall fit** — e.g. placing certain titles (e.g. “event” / Nolan-style releases) in **IMAX** or premium screens.
- **Parental guide rating** — Incorporate ratings when modelling demand or pairings.
- **Language and country** — Movie language and cineplex country can shift demand.
- **Synthetic `min_shows`** — Tie contractual min/max show counts to **release date** and **popularity** instead of purely random draws.
- **Ads and trailers** — Add fixed or distributional trailer/ad time to **effective** runtime.
- **Lobby cap vs. hall scale** — Currently a **fixed** max simultaneous starts; large sites might allow **more** concurrent starts.
- **Ticket price as multipliers** — Model hall prices as **multiples of a base price** (e.g. IMAX = $2\times$ base).

---

_Problem statement, data sources, and output contract live here; solvers, notebooks, and `validator.py` live in the repository implementation._

## GA vs ILP scalability benchmark (reproducible)

To test and report scalability without changing solver internals, run:

`python common/benchmark_scalability.py --tiers small,medium,large --day-type weekday --ilp-time-limits-seconds 60,180,600 --ga-seeds 11,22,33,44,55`

What it does:

- Runs ILP (`exact_method/exact_scheduler.py`) for each tier and each ILP time limit.
- Runs GA (`common/ga_solve.py`) for each tier and each GA seed.
- Validates every generated schedule using `validator.py`.
- Writes per-run JSONs under `benchmark_outputs/<tier>/`.
- Writes one analysis-ready table: `benchmark_outputs/benchmark_summary.csv`.

Suggested assignment evidence from `benchmark_summary.csv`:

- Runtime growth vs dataset tier (`execution_time_seconds`).
- ILP status degradation (`solver_status`) as size grows.
- Feasibility/validity stability (`valid_schedule`, `constraints_violated`).
- Revenue competitiveness under fixed time budgets (`total_revenue`).
