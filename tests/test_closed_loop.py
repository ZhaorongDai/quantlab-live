"""replay, closed loop: quantlab's construction decides on the holdings the replay has.

The rule is qlib's TopkDropout (top 2, one replacement per bar): it keeps a
held name at its current weight, so every decision shows the holdings the
replay's fills left, and an order that did not fill is bought again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab_live import records
from quantlab_live.replay import replay
from quantlab_live.trading.decision_bar import NEW_YORK
from tests.quantlab_run_fixture import RET_5, build_constructor_run, build_quantlab_run, topk_dropout
from tests.helpers import replay_fills

A, B, C = 10001, 10002, 10003
BARS = pd.bdate_range("2024-01-02", periods=4)
OPEN = {A: [10.0, 10.5, 11.5, 12.5], B: [20.0, 20.5, 21.0, 24.0], C: [50.0, 48.0, 41.0, 45.0]}
CLOSE = {A: [10.0, 11.0, 12.0, 13.0], B: [20.0, 20.0, 25.0, 25.0], C: [50.0, 40.0, 40.0, 50.0]}
# A ranks first and B second on every bar.
PREDICTIONS = {"ret_5": {A: [0.03] * 4, B: [0.02] * 4, C: [0.01] * 4}}
LABELS = RET_5


def _rule():
    return topk_dropout()


def _closed_run(root, open_=OPEN):
    run_dir, _ = build_constructor_run(root, BARS, open_, CLOSE, PREDICTIONS, _rule(), LABELS)
    return run_dir


def _open_print(bar: pd.Timestamp) -> pd.Timestamp:
    return (bar.tz_localize(NEW_YORK) + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")


def _fills(out_dir) -> list[tuple]:
    return replay_fills(out_dir, "ts", "instrument_id", "side", "qty")


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    root = tmp_path_factory.mktemp("closed")
    return replay(_closed_run(root), root / "replay")


def test_replay_defaults_to_the_closed_loop(replayed):
    # The first bar buys the top two with half the capital each, at bar 1's open.
    assert _fills(replayed) == [
        (_open_print(BARS[1]), "10001.SIM", "BUY", 500.0),
        (_open_print(BARS[1]), "10002.SIM", "BUY", 250.0),
    ]


def test_the_next_decision_keeps_the_holdings_at_the_weight_the_fills_left(replayed):
    # Filled at 10.5 and 20.5 with 3.75 of IBKR fees: cash -378.75; at bar 1's
    # closes A is worth 5,500 and B 5,000 of an equity of 10,121.25, not the
    # half each the first bar targeted.
    decision = records.read_decision(replayed, BARS[1])
    assert decision.loc[A, "weight"] == pytest.approx(5_500 / 10_121.25)
    assert decision.loc[B, "weight"] == pytest.approx(5_000 / 10_121.25)
    assert decision["order_quantity"].eq(0).all()


def test_an_order_that_did_not_fill_is_decided_again_on_the_next_bar(tmp_path):
    open_ = {symbol: list(prices) for symbol, prices in OPEN.items()}
    open_[B][1] = np.nan  # B has no print where the first bar's orders fill
    out_dir = replay(_closed_run(tmp_path, open_), tmp_path / "replay")
    assert records.read_outcomes(out_dir, BARS[0]).loc[B, "status"] == "not_sent"
    # Holding only A on bar 1 (cash 4,747.50 after its fee, A worth 5,500), the
    # rule keeps A and fills the book with the rest. B cannot trade there
    # (quantlab's tradability: no fill price), so it buys C: 4,750 / 40 = 118.
    first = records.read_decision(out_dir, BARS[1])
    assert first["order_quantity"].to_dict() == {A: 0.0, B: 0.0, C: 118.0}
    # C filled at 41 (cash -91.50); on bar 2 A is worth 6,000 and C 4,720. B,
    # tradable again, replaces C with its sold weight: 4,720 / 25 = 188.
    second = records.read_decision(out_dir, BARS[2])
    assert second["order_quantity"].to_dict() == {A: 0.0, B: 188.0, C: -118.0}
    assert (_open_print(BARS[3]), "10002.SIM", "BUY", 188.0) in _fills(out_dir)


def test_the_last_bar_of_a_replay_is_not_decided(replayed):
    assert records.read_decision(replayed, BARS[-1])["order_quantity"].eq(0).all()


def test_a_closed_loop_window_trades_only_its_bars(tmp_path):
    out_dir = replay(_closed_run(tmp_path), tmp_path / "replay", start=BARS[1], end=BARS[3])
    assert list(records.read_equity(out_dir).index) == list(BARS[1:])
    # Bar 1 decides on an empty book and fills at bar 2's open; bar 3 ends the window.
    assert _fills(out_dir) == [
        (_open_print(BARS[2]), "10001.SIM", "BUY", 454.0),
        (_open_print(BARS[2]), "10002.SIM", "BUY", 250.0),
    ]


def test_an_open_loop_window_trades_only_its_bars(tmp_path):
    weights = {A: [0.5, np.nan, 0.0, np.nan], B: [0.3, np.nan, 0.5, np.nan], C: [0.2, np.nan, 0.5, np.nan]}
    run_dir = build_quantlab_run(tmp_path, BARS, OPEN, CLOSE, weights)
    out_dir = replay(run_dir, tmp_path / "replay", loop="open", start=BARS[0], end=BARS[2])
    assert list(records.read_equity(out_dir).index) == list(BARS[:3])
    # Bar 2's orders would fill at bar 3's open, after the window.
    assert {fill[0] for fill in _fills(out_dir)} == {_open_print(BARS[1])}
    assert records.read_outcomes(out_dir, BARS[2])["status"].eq("not_sent").all()


def test_a_strategy_with_spare_cash_in_its_account_trades_as_if_alone_with_its_initial_capital(tmp_path):
    run_dir = _closed_run(tmp_path / "run")
    alone = replay(run_dir, tmp_path / "alone", initial_cash=10_000.0)
    spare = replay(run_dir, tmp_path / "spare", initial_cash=25_000.0, initial_capital=10_000.0)

    # It sizes on its initial capital plus its P&L, so the spare cash buys nothing ...
    assert _fills(spare) == _fills(alone)
    for bar in records.decision_bars(alone):
        assert records.decision_capital(spare, bar) == pytest.approx(records.decision_capital(alone, bar))
    # ... and its recorded equity, and so its metrics, are its own: 10,000 plus its P&L.
    pd.testing.assert_frame_equal(records.read_equity(spare), records.read_equity(alone))
