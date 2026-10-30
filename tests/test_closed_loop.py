"""replay, closed loop: quantlab's construction decides on the holdings the replay has.

The rule is qlib's TopkDropout (top 2, one replacement per bar): it keeps a
held name at its current weight, so every decision shows the holdings the
replay's fills left, and an order that did not fill is bought again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from quantlab.portfolio.config import TopkDropoutConfig
from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
from quantlab.runs.prediction_panel import LabelSpec

from quantlab_live import records
from quantlab_live.replay import replay
from quantlab_live.trading.decision_bar import NEW_YORK
from tests.quantlab_run_fixture import build_constructor_run, build_quantlab_run

A, B, C = 10001, 10002, 10003
BARS = pd.bdate_range("2024-01-02", periods=4)
OPEN = {A: [10.0, 10.5, 11.5, 12.5], B: [20.0, 20.5, 21.0, 24.0], C: [50.0, 48.0, 41.0, 45.0]}
CLOSE = {A: [10.0, 11.0, 12.0, 13.0], B: [20.0, 20.0, 25.0, 25.0], C: [50.0, 40.0, 40.0, 50.0]}
# A ranks first and B second on every bar.
PREDICTIONS = {"ret_5": {A: [0.03] * 4, B: [0.02] * 4, C: [0.01] * 4}}
LABELS = [LabelSpec("ret_5", "raw", 1, 5)]


def _rule():
    return TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=2, n_drop=1))


def _closed_run(root, open_=OPEN):
    run_dir, _ = build_constructor_run(root, BARS, open_, CLOSE, PREDICTIONS, _rule(), LABELS)
    return run_dir


def _open_print(bar: pd.Timestamp) -> pd.Timestamp:
    return (bar.tz_localize(NEW_YORK) + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")


def _fills(out_dir) -> list[tuple]:
    fills = pd.read_parquet(out_dir / "reports" / "fills.parquet")
    return sorted(
        (pd.to_datetime(r.ts_event, utc=True), r.instrument_id, r.order_side, float(r.last_qty))
        for r in fills.itertuples()
    )


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
    # Filled at 10.5 and 20.5: cash -375; at bar 1's closes A is worth 5,500 and
    # B 5,000 of an equity of 10,125, not the half each the first bar targeted.
    decision = records.read_decision(replayed, BARS[1])
    assert decision.loc[A, "weight"] == pytest.approx(5_500 / 10_125)
    assert decision.loc[B, "weight"] == pytest.approx(5_000 / 10_125)
    assert decision["order_quantity"].eq(0).all()


def test_an_order_that_did_not_fill_is_decided_again_on_the_next_bar(tmp_path):
    open_ = {symbol: list(prices) for symbol, prices in OPEN.items()}
    open_[B][1] = np.nan  # B has no print where the first bar's orders fill
    out_dir = replay(_closed_run(tmp_path, open_), tmp_path / "replay")
    assert records.read_outcomes(out_dir, BARS[0]).loc[B, "status"] == "not_sent"
    # Holding only A on bar 1 (cash 4,750, A worth 5,500 at its close), the
    # rule keeps A and fills the book with the rest. B cannot trade there
    # (quantlab's tradability: no fill price), so it buys C: 4,750 / 40 = 118.
    first = records.read_decision(out_dir, BARS[1])
    assert first["order_quantity"].to_dict() == {A: 0.0, B: 0.0, C: 118.0}
    # C filled at 41 (cash -88); on bar 2 A is worth 6,000 and C 4,720. B,
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
