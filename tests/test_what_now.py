"""``what_now``: the supervisor's next step from the time, the day's schedule and its state.

Table-driven: each row is a New York time, a plan and a state, and the step
expected. The defaults of ``live.toml`` apply: a 20-minute start margin, a
prepare poll every 15 minutes, start retries every 5 minutes, a node start
no earlier than 07:15 and settling up to an hour after the close.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quantlab_live.live.config import LiveConfig
from quantlab_live.live.supervisor import (
    Attempt,
    DaySchedule,
    DayState,
    Done,
    NotTraded,
    PostClose,
    Prepare,
    Start,
    AfterOpen,
    Value,
    Wait,
    what_now,
)
from tests.helpers import ny

DAY = "2024-01-03"  # a full session, 09:30 to 16:00
EARLY = "2024-07-03"  # an early close at 13:00
HOLIDAY = "2024-07-04"
BAR = pd.Timestamp("2024-01-02")


def schedule(day: str = DAY, at: str = "open_auction") -> DaySchedule:
    return DaySchedule.of(day, LiveConfig(run="run", at=at))


def state(day: str = DAY, **fields) -> DayState:
    return replace(DayState(date=day), **fields)


def ready(day: str = DAY, **fields) -> DayState:
    """Prepared and valued, its after-open mail sent unless ``briefed=False``."""
    fields.setdefault("briefed", True)
    return state(day, bar=BAR, valued=True, **fields)


def attempt(day: str, hh_mm: str, *, mode="trade", running=False, started=False, finished=False, outcomes=False):
    at = ny(day, hh_mm)
    if running:
        return Attempt(mode=mode, at=at, handle="node")
    return Attempt(
        mode=mode, at=at, handle=None, ended=at + pd.Timedelta(minutes=1),
        started=started, finished=finished, outcomes=outcomes,
    )


def test_the_schedule_follows_the_plan_and_the_calendar():
    # The open auction's orders go by 09:20; the node must be up 20 minutes earlier.
    assert schedule().latest_start == ny(DAY, "09:00")
    # The close auction's orders go by 15:45, moved with an early close.
    assert schedule(at="close_auction").latest_start == ny(DAY, "15:25")
    assert schedule(EARLY, at="close_auction").latest_start == ny(EARLY, "12:25")
    assert schedule(EARLY).close == ny(EARLY, "13:00")
    assert schedule(at="10:30").latest_start == ny(DAY, "10:10")
    assert schedule(HOLIDAY).session is None


CASES = {
    # Non-trading days and finished days.
    "holiday": (HOLIDAY, "open_auction", "08:00", state(HOLIDAY), Done),
    "weekend": ("2024-01-06", "open_auction", "08:00", state("2024-01-06"), Done),
    "day already done": (DAY, "open_auction", "08:00", state(done="traded"), Done),
    # Data: polled every 15 minutes until the cut-off (the latest start).
    "first prepare": (DAY, "open_auction", "06:00", state(), Prepare),
    "data late, polled again 15 minutes on": (
        DAY, "open_auction", "06:15", state(prepared_at=ny(DAY, "06:00"), pending="running"), Prepare),
    "data late, waiting for the next poll": (
        DAY, "open_auction", "06:10", state(prepared_at=ny(DAY, "06:00"), pending="running"), Wait),
    "data still missing at the cut-off": (
        DAY, "open_auction", "09:00", state(prepared_at=ny(DAY, "08:45"), pending="running"), NotTraded),
    "data late for the open, in time for the close": (
        DAY, "close_auction", "11:00", state(prepared_at=ny(DAY, "10:45"), pending="running"), Prepare),
    "data late on an early close": (
        EARLY, "close_auction", "12:25", state(EARLY, prepared_at=ny(EARLY, "12:10"), pending="running"),
        NotTraded),
    "prepared, not valued": (DAY, "open_auction", "06:30", state(bar=BAR), Value),
    # Starts: no earlier than 07:15, retried every 5 minutes until the latest start.
    "ready before the earliest start": (DAY, "open_auction", "06:30", ready(), Wait),
    "first start": (DAY, "open_auction", "07:15", ready(), Start),
    "failed start, retry not due": (DAY, "open_auction", "07:18", ready(attempts=(attempt(DAY, "07:15"),)), Wait),
    "failed start, retried": (DAY, "open_auction", "07:20", ready(attempts=(attempt(DAY, "07:15"),)), Start),
    "failed starts until the latest start": (
        DAY, "open_auction", "09:00", ready(attempts=(attempt(DAY, "08:55"),)), NotTraded),
    "never started, past the latest start": (DAY, "open_auction", "09:01", ready(), NotTraded),
    "timed plan, start still in time": (DAY, "10:30", "10:05", ready(), Start),
    "node running": (DAY, "open_auction", "11:00", ready(attempts=(attempt(DAY, "07:15", running=True),)), Wait),
    # The after-open mail, 15 minutes after the open, in place of a wait.
    "node running, after the open": (
        DAY, "open_auction", "09:45",
        ready(briefed=False, attempts=(attempt(DAY, "07:15", running=True),)), AfterOpen),
    "node running, before the after-open mail": (
        DAY, "open_auction", "09:44",
        ready(briefed=False, attempts=(attempt(DAY, "07:15", running=True),)), Wait),
    "a crash restarts before the after-open mail": (
        DAY, "open_auction", "11:00",
        ready(briefed=False, attempts=(attempt(DAY, "07:15", started=True),)), Start),
    # Crashes: restarted at once until the close, whatever the latest start.
    "crash before the close": (
        DAY, "open_auction", "11:00", ready(attempts=(attempt(DAY, "07:15", started=True),)), Start),
    "failed restart after a crash, retried": (
        DAY, "open_auction", "11:10",
        ready(attempts=(attempt(DAY, "07:15", started=True), attempt(DAY, "11:00"))), Start),
    "failed restart after a crash, not due": (
        DAY, "open_auction", "11:02",
        ready(attempts=(attempt(DAY, "07:15", started=True), attempt(DAY, "11:00"))), Wait),
    "crash after an early close settles": (
        EARLY, "open_auction", "13:05", ready(EARLY, attempts=(attempt(EARLY, "07:15", started=True),)), Start),
    # After the close: settle while outcomes are missing, then the post-close mail.
    "finished with outcomes": (
        DAY, "open_auction", "16:05",
        ready(attempts=(attempt(DAY, "07:15", started=True, finished=True, outcomes=True),)), PostClose),
    "finished without outcomes": (
        DAY, "open_auction", "16:05", ready(attempts=(attempt(DAY, "07:15", started=True, finished=True),)), Start),
    "settled": (
        DAY, "open_auction", "16:20",
        ready(attempts=(
            attempt(DAY, "07:15", started=True, finished=True),
            attempt(DAY, "16:05", mode="settle", started=True, finished=True, outcomes=True),
        )), PostClose),
    "settle failed, retried": (
        DAY, "open_auction", "16:20",
        ready(attempts=(attempt(DAY, "07:15", started=True), attempt(DAY, "16:05", mode="settle"))), Start),
    "settle given up an hour after the close": (
        DAY, "open_auction", "17:00",
        ready(attempts=(attempt(DAY, "07:15", started=True), attempt(DAY, "16:55", mode="settle"))), PostClose),
}


@pytest.mark.parametrize("day, at, hh_mm, day_state, expected", CASES.values(), ids=CASES.keys())
def test_what_now(day, at, hh_mm, day_state, expected):
    step = what_now(ny(day, hh_mm), schedule(day, at), day_state)
    assert isinstance(step, expected), step


def test_restarts_and_settles_name_their_mode():
    crashed = ready(attempts=(attempt(DAY, "07:15", started=True),))
    assert what_now(ny(DAY, "11:00"), schedule(), crashed) == Start("trade")
    assert what_now(ny(DAY, "16:01"), schedule(), crashed) == Start("settle")


def test_a_wait_ends_at_the_next_due_step():
    late = state(prepared_at=ny(DAY, "08:50"), pending="running")
    assert what_now(ny(DAY, "08:55"), schedule(), late).until == ny(DAY, "09:00")
    assert what_now(ny(DAY, "06:30"), schedule(), ready()).until == ny(DAY, "07:15")
