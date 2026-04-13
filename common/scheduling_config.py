from __future__ import annotations

from typing import Any

from common.utils import parse_hhmm


def _minutes_from_hhmm(hhmm: str) -> int:
    h, m = parse_hhmm(hhmm)
    return h * 60 + m


def opening_minutes_from_constraints(constraints: dict[str, Any]) -> int:
    return _minutes_from_hhmm(str(constraints["opening_time"]))


def num_slots_from_config(config: dict[str, Any]) -> int:
    constraints = config["constraints"]
    slot_duration = int(constraints["slot_duration_minutes"])
    opening = _minutes_from_hhmm(str(constraints["opening_time"]))
    closing = _minutes_from_hhmm(str(constraints["closing_time"]))
    closing_next_day = bool(constraints.get("closing_time_is_next_calendar_day", True))

    if closing_next_day and closing <= opening:
        closing += 24 * 60
    if closing <= opening:
        raise ValueError("closing time must be after opening time")

    span = closing - opening
    if span % slot_duration != 0:
        raise ValueError("operating window is not divisible by slot duration")
    return span // slot_duration


def slot_to_wall_clock(slot_index: int, opening_minutes: int, slot_duration_minutes: int) -> str:
    total = int(opening_minutes) + int(slot_index) * int(slot_duration_minutes)
    total %= 24 * 60
    h, m = divmod(total, 60)
    return f"{h:02d}:{m:02d}"
