"""replay: a quantlab run's rebalance table executed open loop on BacktestEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab_live import records
from quantlab_live.replay import replay
from quantlab_live.trading.decision_bar import NEW_YORK
from tests.quantlab_run_fixture import build_quantlab_run

BARS = pd.bdate_range("2024-01-02", periods=4)
# Raw prices; the fixture's adjusted group is half of them, so a raw/adjusted
# mix-up changes every number below.
OPEN = {10001: [10.0, 10.5, 11.5, 12.5], 10002: [20.0, 20.5, 21.0, 24.0], 10003: [50.0, 48.0, 41.0, 45.0]}
CLOSE = {10001: [10.0, 11.0, 12.0, 13.0], 10002: [20.0, 20.0, 25.0, 25.0], 10003: [50.0, 40.0, 40.0, 50.0]}
KEEP = np.nan
WEIGHTS = {
    10001: [0.5, KEEP, 0.0, KEEP],
    10002: [0.3, KEEP, 0.5, KEEP],
    10003: [0.2, KEEP, 0.5, KEEP],
}
NEW_YORK_OPEN = pd.Timedelta(hours=9, minutes=30)


def _open_print(bar: pd.Timestamp) -> pd.Timestamp:
    return (bar.tz_localize(NEW_YORK) + NEW_YORK_OPEN).tz_convert("UTC")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return build_quantlab_run(tmp_path_factory.mktemp("open"), BARS, OPEN, CLOSE, WEIGHTS, init_cash=10_000.0)


@pytest.fixture(scope="module")
def replayed(run_dir, tmp_path_factory):
    return replay(run_dir, tmp_path_factory.mktemp("replayed") / "replay", loop="open")


def test_each_bars_orders_fill_at_the_next_opening_print(replayed):
    fills = pd.read_parquet(replayed / "reports" / "fills.parquet")
    got = sorted(
        (
            pd.to_datetime(row.ts_event, utc=True),
            row.instrument_id,
            row.order_side,
            float(row.last_qty),
            float(row.last_px),
        )
        for row in fills.itertuples()
    )
    # Bar 0 sizes 10,000 at its closes and fills at bar 1's open; bar 1 keeps;
    # bar 2 sizes its equity of 11,100.50 (cash -249.50, holdings 11,350) in whole
    # shares and fills at bar 3's open; the last bar has no next open.
    assert got == sorted(
        [
            (_open_print(BARS[1]), "10001.SIM", "BUY", 500.0, 10.5),
            (_open_print(BARS[1]), "10002.SIM", "BUY", 150.0, 20.5),
            (_open_print(BARS[1]), "10003.SIM", "BUY", 40.0, 48.0),
            (_open_print(BARS[3]), "10001.SIM", "SELL", 500.0, 12.5),
            (_open_print(BARS[3]), "10002.SIM", "BUY", 72.0, 24.0),
            (_open_print(BARS[3]), "10003.SIM", "BUY", 98.0, 45.0),
        ]
    )


def test_nautilus_trader_reports_are_exported_to_the_replay_directory(replayed):
    for name in ("orders", "order_fills", "fills", "positions", "account"):
        assert len(pd.read_parquet(replayed / "reports" / f"{name}.parquet"))


def test_equity_is_valued_at_each_bars_raw_close(replayed):
    equity = records.read_equity(replayed)
    assert list(equity.index) == list(BARS)
    # Bar 1: cash 10,000 - 10,245 of fills - 4.50 of IBKR fees (2.50 on 500
    # shares, the 1.00 minimum on 150 and on 40), holdings 500 x 11 + 150 x 20 + 40 x 40.
    assert equity["cash"].tolist() == [10_000.0, -249.5, -249.5, -142.0]
    assert equity["equity"].tolist() == [10_000.0, 9_850.5, 11_100.5, 12_308.0]


def test_snapshots_hold_each_bars_positions_in_raw_shares(replayed):
    assert records.read_snapshot(replayed, BARS[0]).positions.empty
    second = records.read_snapshot(replayed, BARS[2])
    assert second.positions["quantity"].to_dict() == {10001: 500.0, 10002: 150.0, 10003: 40.0}
    last = records.read_snapshot(replayed, BARS[3])
    assert (last.bar, last.cash) == (BARS[3], -142.0)
    assert last.positions["quantity"].to_dict() == {10002: 222.0, 10003: 138.0}


def test_a_decision_records_the_bars_targets_at_its_raw_closes(replayed):
    decision = records.read_decision(replayed, BARS[2])
    assert decision["close"].to_dict() == {10001: 12.0, 10002: 25.0, 10003: 40.0}
    assert decision["target_quantity"].to_dict() == {10001: 0.0, 10002: 222.0, 10003: 138.0}
    assert decision["order_quantity"].to_dict() == {10001: -500.0, 10002: 72.0, 10003: 98.0}
    kept = records.read_decision(replayed, BARS[1])
    assert kept["order_quantity"].eq(0).all()


def test_outcomes_give_each_ordered_symbol_its_fills(replayed):
    outcomes = records.read_outcomes(replayed, BARS[2])
    assert outcomes[["order_quantity", "filled_quantity", "avg_px"]].to_dict("index") == {
        10001: {"order_quantity": -500.0, "filled_quantity": -500.0, "avg_px": 12.5},
        10002: {"order_quantity": 72.0, "filled_quantity": 72.0, "avg_px": 24.0},
        10003: {"order_quantity": 98.0, "filled_quantity": 98.0, "avg_px": 45.0},
    }
    assert outcomes["status"].eq("filled").all()


def test_records_are_typed_and_keyed_by_bar_timestamp(replayed):
    for path in (replayed / "decisions").glob("*.parquet"):
        frame = pd.read_parquet(path)
        assert frame["bar"].dtype == "datetime64[ns]" and frame["symbol"].dtype == "int64"
        assert frame["bar"].nunique() == 1
        assert path.stem == frame["bar"].iloc[0].date().isoformat()


def _fills(out_dir) -> list[tuple]:
    frame = pd.read_parquet(out_dir / "reports" / "fills.parquet")
    return sorted(
        (row.ts_event, row.instrument_id, row.order_side, float(row.last_qty), float(row.last_px))
        for row in frame.itertuples()
    )


def test_a_second_replay_into_a_decided_directory_resumes_its_decisions(run_dir, replayed, tmp_path):
    out_dir = tmp_path / "replay"
    first = replay(run_dir, out_dir, loop="open")
    decided = {p.name: p.read_bytes() for p in (first / "decisions").iterdir()}
    again = replay(run_dir, out_dir, loop="open")
    # Nothing is decided again, and every order is sent once.
    assert {p.name: p.read_bytes() for p in (again / "decisions").iterdir()} == decided
    assert _fills(again) == _fills(replayed)
    assert records.read_equity(again)["equity"].tolist() == [10_000.0, 9_850.5, 11_100.5, 12_308.0]


def test_a_replay_stopped_part_way_resumes_where_it_stopped(run_dir, replayed, tmp_path):
    out_dir = tmp_path / "replay"
    stopped = replay(run_dir, out_dir, loop="open", end=BARS[2])
    # The stop left bar 2's orders unsent; the second run sends them.
    assert records.read_outcomes(stopped, BARS[2])["status"].eq("not_sent").all()
    resumed = replay(run_dir, out_dir, loop="open")
    assert _fills(resumed) == _fills(replayed)
    assert records.read_outcomes(resumed, BARS[2])["status"].eq("filled").all()
    assert list(records.read_equity(resumed).index) == list(BARS)


def test_no_order_is_sent_for_a_decision_that_could_not_be_written(tmp_path):
    run_dir = build_quantlab_run(tmp_path, BARS, OPEN, CLOSE, WEIGHTS)
    out_dir = tmp_path / "replay"
    out_dir.mkdir()
    (out_dir / "decisions").write_text("not a directory")
    with pytest.raises(RuntimeError, match="the strategy failed on bar 2024-01-02"):
        replay(run_dir, out_dir, loop="open")
    assert pd.read_parquet(out_dir / "reports" / "orders.parquet").empty


def test_an_order_without_an_opening_print_is_not_filled(tmp_path):
    open_ = {symbol: list(prices) for symbol, prices in OPEN.items()}
    open_[10003][1] = np.nan  # no print on the bar bar 0's orders fill at
    run_dir = build_quantlab_run(tmp_path, BARS, open_, CLOSE, WEIGHTS)
    out_dir = replay(run_dir, tmp_path / "replay", loop="open")
    outcomes = records.read_outcomes(out_dir, BARS[0])
    assert outcomes.loc[10003, "filled_quantity"] == 0.0
    assert outcomes.loc[10003, "status"] == "not_sent"
    assert outcomes.loc[10001, "avg_px"] == 10.5
