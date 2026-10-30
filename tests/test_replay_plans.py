"""replay: each execution plan fills at its print or time, and the fallback chain."""

from __future__ import annotations

import pandas as pd
import pytest
from nautilus_trader.execution import FillModel
from nautilus_trader.model import BookType, OrderBook

from quantlab_live import records
from quantlab_live.replay import replay
from quantlab_live.trading.scheduled_execution import ExecutionPlan
from tests.quantlab_run_fixture import build_quantlab_run
from tests.helpers import ny

BARS = pd.bdate_range("2024-01-02", periods=3)
# Bar 0 buys 500, 150 and 40 shares at its closes; the later bars keep them.
OPEN = {10001: [10.0, 10.5, 11.5], 10002: [20.0, 20.5, 21.0], 10003: [50.0, 48.0, 41.0]}
CLOSE = {10001: [10.0, 11.0, 12.0], 10002: [20.0, 20.0, 25.0], 10003: [50.0, 40.0, 40.0]}
KEEP = float("nan")
WEIGHTS = {10001: [0.5, KEEP, KEEP], 10002: [0.3, KEEP, KEEP], 10003: [0.2, KEEP, KEEP]}
BOUGHT = {"10001.SIM": 500.0, "10002.SIM": 150.0, "10003.SIM": 40.0}
# Around the 4th of July 2024: the 3rd closes at 13:00.
EARLY_BARS = pd.DatetimeIndex(["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05"])
EARLY_OPEN = {s: [p[0], p[0], p[1], p[2]] for s, p in OPEN.items()}
EARLY_CLOSE = {s: [p[0], p[0], p[1], p[2]] for s, p in CLOSE.items()}
EARLY_WEIGHTS = {s: [KEEP, w[0], KEEP, KEEP] for s, w in WEIGHTS.items()}


class NeverFillsAuctions(FillModel):
    """Fills nothing an auction child asks for: its simulated book is empty."""

    def get_orderbook_for_fill_simulation(self, instrument, order, best_bid, best_ask):
        if any(tag.endswith("_auction") for tag in order.tags or []):
            return OrderBook(instrument.id, BookType.L2_MBP)
        return None


def fills(out_dir) -> dict[str, tuple[pd.Timestamp, float, float]]:
    """Each instrument's one fill: when, how many and at what price."""
    frame = pd.read_parquet(out_dir / "reports" / "fills.parquet")
    got = {}
    for row in frame.itertuples():
        assert row.instrument_id not in got, f"{row.instrument_id} filled twice"
        got[row.instrument_id] = (pd.to_datetime(row.ts_event, utc=True), float(row.last_qty), float(row.last_px))
    return got


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return build_quantlab_run(tmp_path_factory.mktemp("plans"), BARS, OPEN, CLOSE, WEIGHTS)


@pytest.fixture(scope="module")
def early_run_dir(tmp_path_factory):
    return build_quantlab_run(
        tmp_path_factory.mktemp("early"), EARLY_BARS, EARLY_OPEN, EARLY_CLOSE, EARLY_WEIGHTS
    )


def test_an_opening_auction_plan_fills_at_the_next_opening_print(run_dir, tmp_path):
    got = fills(replay(run_dir, tmp_path, loop="open", plan="open_auction"))
    at = ny("2024-01-03", "09:30")
    assert got == {
        "10001.SIM": (at, 500.0, 10.5),
        "10002.SIM": (at, 150.0, 20.5),
        "10003.SIM": (at, 40.0, 48.0),
    }


def test_a_closing_auction_plan_fills_at_the_next_closing_print(run_dir, tmp_path):
    out_dir = replay(run_dir, tmp_path, loop="open", plan="close_auction")
    at = ny("2024-01-03", "16:00")
    assert fills(out_dir) == {
        "10001.SIM": (at, 500.0, 11.0),
        "10002.SIM": (at, 150.0, 20.0),
        "10003.SIM": (at, 40.0, 40.0),
    }
    # Bar 1 decides on its close after the fills, so it keeps them.
    assert records.read_decision(out_dir, BARS[1])["order_quantity"].eq(0).all()


def test_a_timed_plan_fills_at_that_time_between_the_open_and_the_close(run_dir, tmp_path):
    # 12:45 is halfway through the session: the print is halfway from open to close.
    got = fills(replay(run_dir, tmp_path, loop="open", plan="12:45"))
    at = ny("2024-01-03", "12:45")
    assert got == {
        "10001.SIM": (at, 500.0, 10.75),
        "10002.SIM": (at, 150.0, 20.25),
        "10003.SIM": (at, 40.0, 44.0),
    }


def test_an_early_close_moves_the_closing_auction(early_run_dir, tmp_path):
    got = fills(replay(early_run_dir, tmp_path, loop="open", plan="close_auction"))
    assert {i: (t, px) for i, (t, _, px) in got.items()} == {
        "10001.SIM": (ny("2024-07-03", "13:00"), 11.0),
        "10002.SIM": (ny("2024-07-03", "13:00"), 20.0),
        "10003.SIM": (ny("2024-07-03", "13:00"), 40.0),
    }


def test_a_timed_plan_after_an_early_close_sends_nothing_and_is_recorded(early_run_dir, tmp_path):
    out_dir = replay(early_run_dir, tmp_path, loop="open", plan="14:00")
    assert pd.read_parquet(out_dir / "reports" / "fills.parquet").empty
    outcomes = records.read_outcomes(out_dir, EARLY_BARS[1])
    assert outcomes["status"].eq("not_sent").all()
    assert outcomes["order_quantity"].to_dict() == {10001: 500.0, 10002: 150.0, 10003: 40.0}
    orders = pd.read_parquet(out_dir / "reports" / "orders.parquet")
    assert orders["status"].eq("DENIED").all()


def test_a_limit_plan_fills_only_inside_its_band_and_falls_back_for_the_rest(run_dir, tmp_path):
    # Buy limits 3% above bar 0's closes: 10.30, 20.60 and 51.50. Only 10001's
    # open (10.50) is beyond its band, so its opening-auction child expires
    # and the fallback buys at 09:35, 5/390 of the way from open to close.
    plan = ExecutionPlan("open_auction", order="limit", band_bps=300)
    out_dir = replay(run_dir, tmp_path, loop="open", plan=plan)
    got = fills(out_dir)
    assert got["10002.SIM"] == (ny("2024-01-03", "09:30"), 150.0, 20.5)
    assert got["10003.SIM"] == (ny("2024-01-03", "09:30"), 40.0, 48.0)
    assert got["10001.SIM"] == (ny("2024-01-03", "09:35"), 500.0, 10.5064)
    outcomes = records.read_outcomes(out_dir, BARS[0])
    assert outcomes["status"].to_dict() == {
        10001: "venue_cancelled_fallback", 10002: "filled", 10003: "filled",
    }
    assert outcomes["order_quantity"].to_dict() == {10001: 500.0, 10002: 150.0, 10003: 40.0}
    assert outcomes["filled_quantity"].to_dict() == {10001: 500.0, 10002: 150.0, 10003: 40.0}


def test_a_missed_opening_auction_falls_back_to_a_day_market_order(run_dir, tmp_path):
    out_dir = replay(run_dir, tmp_path, loop="open", plan="open_auction", fill_model=NeverFillsAuctions())
    at = ny("2024-01-03", "09:35")
    assert fills(out_dir) == {
        "10001.SIM": (at, 500.0, 10.5064),
        "10002.SIM": (at, 150.0, 20.4936),
        "10003.SIM": (at, 40.0, 47.8974),
    }
    orders = pd.read_parquet(out_dir / "reports" / "orders.parquet")
    children = orders[orders["exec_spawn_id"].notna() & (orders["exec_spawn_id"] != orders.index)]
    assert sorted(children["status"]) == ["EXPIRED"] * 3 + ["FILLED"] * 3
    outcomes = records.read_outcomes(out_dir, BARS[0])
    assert outcomes["status"].eq("venue_cancelled_fallback").all()
    assert outcomes["avg_px"].to_dict() == {10001: 10.5064, 10002: 20.4936, 10003: 47.8974}


def test_a_missed_closing_auction_is_recorded_as_cancelled_by_the_venue(run_dir, tmp_path):
    out_dir = replay(run_dir, tmp_path, loop="open", plan="close_auction", fill_model=NeverFillsAuctions())
    assert pd.read_parquet(out_dir / "reports" / "fills.parquet").empty
    outcomes = records.read_outcomes(out_dir, BARS[0])
    assert outcomes["status"].eq("venue_cancelled").all()
    assert outcomes["filled_quantity"].eq(0).all()
