from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.stats import norm

_MINUTES_PER_DAY = 24 * 60
_CINEPLEX_CONFIG_PATH = Path(__file__).resolve().parent / "cineplex_config.json"


def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"expected 'HH:MM', got {value!r}")
    h, m = int(parts[0], 10), int(parts[1], 10)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"out-of-range time {value!r}")
    return h, m


def _load_scheduling() -> tuple[int, int, int, bool]:
    with _CINEPLEX_CONFIG_PATH.open(encoding="utf-8") as f:
        constraints = json.load(f)["constraints"]
    duration = int(constraints["slot_duration_minutes"])
    zh, zm = _parse_hhmm(constraints["opening_time"])
    closing_next_day = bool(constraints.get("closing_time_is_next_calendar_day", True))
    return duration, zh, zm, closing_next_day


_SLOT_DURATION_MINUTES, _SLOT_ZERO_HOUR, _SLOT_ZERO_MINUTE, _CLOSING_NEXT_CALENDAR_DAY = (
    _load_scheduling()
)


def _slot_zero_minutes_from_midnight() -> int:
    return _SLOT_ZERO_HOUR * 60 + _SLOT_ZERO_MINUTE


def slot_to_time(slot_index: int) -> str:
    """Map slot index to wall-clock ``HH:MM`` (grid from ``opening_time`` in ``cineplex_config.json``)."""
    start = _slot_zero_minutes_from_midnight()
    total = start + int(slot_index) * _SLOT_DURATION_MINUTES
    m = total % _MINUTES_PER_DAY
    h, mn = divmod(m, 60)
    return f"{h:02d}:{mn:02d}"


def time_to_slot(time_string: str) -> int:
    """
    Map ``HH:MM`` to the smallest non-negative slot index consistent with :func:`slot_to_time`.

    When ``closing_time_is_next_calendar_day`` is true in ``cineplex_config.json``, clock times
    before ``opening_time`` (e.g. ``00:45``, ``01:00``) are interpreted as the next calendar day.

    Raises:
        ValueError: if the string is malformed or the time is not on the slot grid.
    """
    hour, minute = _parse_hhmm(time_string)
    target = hour * 60 + minute
    z = _slot_zero_minutes_from_midnight()
    if (target - z) % _SLOT_DURATION_MINUTES != 0:
        raise ValueError(
            f"time {time_string!r} is not aligned to {_SLOT_DURATION_MINUTES}-minute slots"
        )

    if target >= z:
        n = 0
    elif _CLOSING_NEXT_CALENDAR_DAY:
        n = 1
    else:
        opening = f"{_SLOT_ZERO_HOUR:02d}:{_SLOT_ZERO_MINUTE:02d}"
        raise ValueError(
            f"time {time_string!r} is before opening ({opening}) "
            "and overnight sessions are disabled in cineplex_config.json"
        )
    minutes_offset = target + n * _MINUTES_PER_DAY - z
    return minutes_offset // _SLOT_DURATION_MINUTES



def generate_robust_demand_matrix(movies, day_type, num_slots=60):
    """
    Implements: W_mt = BasePop * Decay * GenreTimeFit * DayMulti * TimeMulti
    """
    # 1. DayMulti (Scalar based on the dataset/day being simulated)
    # e.g., Weekday = 1.0, Weekend = 1.3
    day_multi = 1.3 if day_type == "weekend" else 1.0
    
    # 2. TimeMulti (Vector of 60 slots - general 'prime time' pulse)
    # Even regardless of genre, 7-9 PM is generally busier
    time_slots = np.arange(num_slots)
    time_multi = norm.pdf(time_slots, 40, 10) # General peak at slot 40
    time_multi = time_multi / time_multi.max() 

    matrix = np.zeros((len(movies), num_slots))

    for i, m in enumerate(movies):
        # A. BasePopularity (from CSV)
        base_pop = m['base_popularity']
        
        # B. Decay (Exponential formula: W0 * e^-lambda*t)
        # Here t = weeks_since_release
        lambda_val = 0.15 
        decay = np.exp(-lambda_val * m['weeks_since_release'])
        
        # C. GenreTimeFit (The Gaussian specific to the genre)
        # Animation peaks early, Horror late, etc.
        mu = m['genre_peak_slot']
        sigma = m['genre_sigma']
        genre_fit = norm.pdf(time_slots, mu, sigma)
        genre_fit = genre_fit / genre_fit.max()

        # D. Final calculation per slot
        # W_mt = BasePop * Decay * GenreTimeFit * DayMulti * TimeMulti
        matrix[i] = base_pop * decay * genre_fit * day_multi * time_multi

    return matrix

def calculate_screening_revenue(movie, hall, start_slot, demand_matrix):
    """
    Standardized revenue math for ONE screening.
    Both ILP and GA should use this logic.
    """
    duration = movie['runtime_slots'] # You'll need to calculate this from 'runtime'
    end_slot = start_slot + duration
    
    # Calculate average demand over the movie duration
    # (Handling potential index out of bounds)
    avg_demand = np.mean(demand_matrix[movie['index'], start_slot:end_slot])
    
    return hall['capacity'] * hall['average_ticket_price'] * avg_demand
