from __future__ import annotations

import json
from pathlib import Path

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
