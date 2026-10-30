"""replay with restarts: a restarted node resumes its decision and never duplicates an order."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab_live import records
from quantlab_live.replay import replay
from quantlab_live.trading.scheduled_execution import ExecutionPlan
from tests.quantlab_run_fixture import build_quantlab_run

NY = "America/New_York"
BARS = pd.bdate_range("2024-01-02", periods=3)
# Bar 0 buys 500, 150 and 40 shares at its closes; the later bars keep them.
OPEN = {10001: [10.0, 10.5, 11.5], 10002: [20.0, 20.5, 21.0], 10003: [50.0, 48.0, 41.0]}
CLOSE = {10001: [10.0, 11.0, 12.0], 10002: [20.0, 20.0, 25.0], 10003: [50.0, 40.0, 40.0]}
KEEP = float("nan")
WEIGHTS = {10001: [0.5, KEEP, KEEP], 10002: [0.3, KEEP, KEEP], 10003: [0.2, KEEP, KEEP]}


def ny(day: str, time: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {time}", tz=NY).tz_convert("UTC")


def fills(out_dir) -> list[tuple]:
    frame = pd.read_parquet(out_dir / "reports" / "fills.parquet")
    return sorted(
        (pd.to_datetime(r.ts_event, utc=True), r.instrument_id, float(r.last_qty), float(r.last_px))
        for r in frame.itertuples()
    )


def orders(out_dir) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "reports" / "orders.parquet")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return build_quantlab_run(tmp_path_factory.mktemp("resume"), BARS, OPEN, CLOSE, WEIGHTS)


AT_12_45 = [
    (ny("2024-01-03", "12:45"), "10001.SIM", 500.0, 10.75),
    (ny("2024-01-03", "12:45"), "10002.SIM", 150.0, 20.25),
    (ny("2024-01-03", "12:45"), "10003.SIM", 40.0, 44.0),
]


def test_a_restart_before_the_send_cancels_the_kept_orders_and_sends_them_once(run_dir, tmp_path):
    out_dir = replay(run_dir, tmp_path, loop="open", plan="12:45", restarts=["2024-01-03 10:00"])
    assert fills(out_dir) == AT_12_45
    # The primaries kept for 12:45 were lost with the crash; the restart
    # cancels them and sends the decision's remainder, the whole order.
    primaries = orders(out_dir)
    primaries = primaries[primaries["exec_spawn_id"].isna() | (primaries["exec_spawn_id"] == primaries.index)]
    assert sorted(primaries["status"]) == ["CANCELED"] * 3 + ["INITIALIZED"] * 3
    assert records.read_outcomes(out_dir, BARS[0])["status"].eq("filled").all()


def test_repeated_restarts_never_duplicate_an_order(run_dir, tmp_path):
    restarts = ["2024-01-03 09:00", "2024-01-03 10:00", "2024-01-03 12:50", "2024-01-03 15:00", "2024-01-04 10:00"]
    out_dir = replay(run_dir, tmp_path, loop="open", plan="12:45", restarts=restarts)
    assert fills(out_dir) == AT_12_45
    outcomes = records.read_outcomes(out_dir, BARS[0])
    assert outcomes["filled_quantity"].to_dict() == {10001: 500.0, 10002: 150.0, 10003: 40.0}
    assert outcomes["status"].eq("filled").all()


def test_a_restart_counts_the_orders_working_at_the_venue(run_dir, tmp_path):
    # Buy limits at bar 0's closes from 10:00: only 10003 trades below its
    # limit then; the other two rest at the venue when the node restarts at
    # 11:00, and 10002 fills at its limit at the close.
    plan = ExecutionPlan("10:00", order="limit", band_bps=0)
    out_dir = replay(run_dir, tmp_path, loop="open", plan=plan, restarts=["2024-01-03 11:00"])
    # Nothing is ordered again: three primaries and their three children.
    assert len(orders(out_dir)) == 6
    assert [(t, i, q) for t, i, q, _ in fills(out_dir)] == [
        (ny("2024-01-03", "10:00"), "10003.SIM", 40.0),
        (ny("2024-01-03", "16:00"), "10002.SIM", 150.0),
    ]
    outcomes = records.read_outcomes(out_dir, BARS[0])
    assert outcomes["status"].to_dict() == {10001: "venue_cancelled", 10002: "filled", 10003: "filled"}
