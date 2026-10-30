"""ScheduledExecution's action: a pure function of now, the plan and the day's session."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab_live.trading.scheduled_execution import Action, ExecutionPlan, Session, action
from tests.helpers import ny

NY = "America/New_York"
REGULAR = Session.of(pd.Timestamp("2024-01-03 08:00", tz=NY))
EARLY_CLOSE = Session.of(pd.Timestamp("2024-07-03 08:00", tz=NY))


def test_the_session_is_the_first_whose_close_is_after_now():
    assert (REGULAR.open, REGULAR.close) == (ny("2024-01-03", "09:30"), ny("2024-01-03", "16:00"))
    # At a session's close the next session is the day's.
    assert Session.of(ny("2024-01-03", "16:00")).open == ny("2024-01-04", "09:30")
    # Over a weekend and a holiday.
    assert Session.of(ny("2024-07-03", "16:00")).open == ny("2024-07-05", "09:30")
    assert EARLY_CLOSE.close == ny("2024-07-03", "13:00")


@pytest.mark.parametrize(
    ("plan", "session", "now", "expected"),
    [
        # The opening auction, by its 09:20 cut-off; after it, the fallback.
        ("open_auction", REGULAR, ("2024-01-02", "16:00"), Action("open_auction", ny("2024-01-03", "09:30"))),
        ("open_auction", REGULAR, ("2024-01-03", "09:20"), Action("open_auction", ny("2024-01-03", "09:30"))),
        ("open_auction", REGULAR, ("2024-01-03", "09:21"), Action("fallback", ny("2024-01-03", "09:35"))),
        ("open_auction", REGULAR, ("2024-01-03", "11:00"), Action("fallback", ny("2024-01-03", "11:00"))),
        # The closing auction, by its 15:45 cut-off, moved with an early close; no fallback.
        ("close_auction", REGULAR, ("2024-01-03", "15:45"), Action("close_auction", ny("2024-01-03", "16:00"))),
        ("close_auction", EARLY_CLOSE, ("2024-07-03", "08:00"), Action("close_auction", ny("2024-07-03", "13:00"))),
        ("close_auction", EARLY_CLOSE, ("2024-07-03", "12:46"), Action("none", None, "missed the closing auction's cut-off")),
        # A time: on time, late before its deadline (30 minutes), or past it.
        ("10:30", REGULAR, ("2024-01-03", "08:00"), Action("day", ny("2024-01-03", "10:30"))),
        ("10:30", REGULAR, ("2024-01-03", "10:59"), Action("day", ny("2024-01-03", "10:59"))),
        ("10:30", REGULAR, ("2024-01-03", "11:00"), Action("none", None, "past the 10:30 plan's deadline")),
        # A time after an early close never comes.
        ("14:00", EARLY_CLOSE, ("2024-07-03", "08:00"), Action("none", None, "past the 14:00 plan's deadline")),
    ],
)
def test_the_action_follows_from_now_the_plan_and_the_session(plan, session, now, expected):
    assert action(ny(*now), ExecutionPlan(plan), session) == expected


def test_a_missed_opening_auction_falls_back_and_a_missed_closing_auction_does_not():
    at_open = ny("2024-01-03", "09:30")
    assert action(at_open, ExecutionPlan("open_auction"), REGULAR, auction_missed=True) == Action(
        "fallback", ny("2024-01-03", "09:35")
    )
    closed = action(ny("2024-01-03", "16:00"), ExecutionPlan("close_auction"), REGULAR, auction_missed=True)
    assert closed == Action("none", None, "a closing auction has no fallback")


def test_a_restart_gets_the_same_action():
    # The algorithm keeps nothing: a fresh plan and session, built again after
    # a restart, give the action the first process had.
    now = ny("2024-01-03", "09:25")
    first = action(now, ExecutionPlan("open_auction"), Session.of(now))
    assert action(now, ExecutionPlan("open_auction"), Session.of(now)) == first


def test_a_plan_is_checked_when_built():
    with pytest.raises(ValueError, match="at must be"):
        ExecutionPlan("noon")
    with pytest.raises(ValueError, match="band_bps"):
        ExecutionPlan("open_auction", order="limit")
