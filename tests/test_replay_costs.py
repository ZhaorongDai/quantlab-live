"""replay: fees, slippage, splits, dividends and delistings, in raw-unit records."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab_live import records
from quantlab_live.replay import ProportionalSlippage, replay
from tests.quantlab_run_fixture import build_quantlab_run

KEEP = np.nan
BARS = pd.bdate_range("2024-01-02", periods=6)
NO_SPLIT = [1.0] * 6
NO_DIVIDEND = [0.0] * 6


def _fills(out_dir) -> list[tuple]:
    """The replay's fills: (instrument, side, raw-or-unit quantity, price, commission)."""
    fills = pd.read_parquet(out_dir / "reports" / "fills.parquet")
    return sorted(
        (
            row.instrument_id,
            row.order_side,
            float(row.last_qty),
            float(row.last_px),
            float(str(row.commission).split()[0]),
        )
        for row in fills.itertuples()
    )


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    """A two-for-one split of 10001 on bar 2 and a one-for-five reverse split of 10002 on bar 3.

    Prices are constant in value, so equity is constant while the positions
    are held; both positions are sold on bar 4, after the splits.
    """
    root = tmp_path_factory.mktemp("split")
    prices = {10001: [10.0, 10.0, 5.0, 5.0, 5.0, 5.0], 10002: [20.0, 20.0, 20.0, 100.0, 100.0, 100.0]}
    weights = {10001: [0.5, KEEP, KEEP, KEEP, 0.0, KEEP], 10002: [0.5, KEEP, KEEP, KEEP, 0.0, KEEP]}
    split_factor = {10001: [1.0, 1.0, 2.0, 1.0, 1.0, 1.0], 10002: [1.0, 1.0, 1.0, 0.2, 1.0, 1.0]}
    run_dir = build_quantlab_run(
        root, BARS, prices, prices, weights, init_cash=100_000.0, variables={"splitFactor": split_factor}
    )
    return replay(run_dir, root / "replay", loop="open")


def test_splits_leave_equity_continuous(split):
    equity = records.read_equity(split)["equity"]
    # Bar 1 pays 37.50 of fees on its buys; bar 5 pays 52.50 on its sells.
    assert equity.tolist() == [100_000.0, 99_962.5, 99_962.5, 99_962.5, 99_962.5, 99_910.0]


def test_per_share_fees_are_charged_on_raw_shares(split):
    # The engine trades split-adjusted units (the shares at the first bar):
    # 5,000 units of 10001 are 10,000 raw shares after its split, 2,500 units
    # of 10002 are 500 raw shares after its reverse split.
    assert _fills(split) == [
        ("10001.SIM", "BUY", 5_000.0, 10.0, 25.0),
        ("10001.SIM", "SELL", 5_000.0, 10.0, 50.0),
        ("10002.SIM", "BUY", 2_500.0, 20.0, 12.5),
        ("10002.SIM", "SELL", 2_500.0, 20.0, 2.5),
    ]


def test_split_replay_records_are_in_raw_units(split):
    assert records.read_snapshot(split, BARS[1]).positions["quantity"].to_dict() == {10001: 5_000.0, 10002: 2_500.0}
    assert records.read_snapshot(split, BARS[2]).positions["quantity"].to_dict() == {10001: 10_000.0, 10002: 2_500.0}
    assert records.read_snapshot(split, BARS[3]).positions["quantity"].to_dict() == {10001: 10_000.0, 10002: 500.0}
    decision = records.read_decision(split, BARS[4])
    assert decision["close"].to_dict() == {10001: 5.0, 10002: 100.0}
    assert decision["position_quantity"].to_dict() == {10001: 10_000.0, 10002: 500.0}
    assert decision["order_quantity"].to_dict() == {10001: -10_000.0, 10002: -500.0}
    outcomes = records.read_outcomes(split, BARS[4])
    assert outcomes[["order_quantity", "filled_quantity", "avg_px"]].to_dict("index") == {
        10001: {"order_quantity": -10_000.0, "filled_quantity": -10_000.0, "avg_px": 5.0},
        10002: {"order_quantity": -500.0, "filled_quantity": -500.0, "avg_px": 100.0},
    }
    assert records.read_snapshot(split, BARS[5]).cash == 99_910.0


def test_a_dividend_is_credited_once_on_the_prior_closes_position_and_a_short_pays(tmp_path):
    bars = BARS[:4]
    prices = {10001: [10.0] * 4, 10002: [20.0] * 4}
    # 10001 is bought on bar 0's decision and sold at its ex-date's open;
    # 10002 is sold short and held.
    weights = {10001: [0.5, 0.0, KEEP, KEEP], 10002: [-0.25, KEEP, KEEP, KEEP]}
    dividend = {10001: [0.0, 0.0, 0.5, 0.0], 10002: [0.0, 0.0, 1.0, 0.0]}
    run_dir = build_quantlab_run(tmp_path, bars, prices, prices, weights, variables={"divCash": dividend})
    out_dir = replay(run_dir, tmp_path / "replay", loop="open")
    equity = records.read_equity(out_dir)["equity"]
    # Bar 1 pays 3.50 of fees on 500 and -125 shares. Before bar 2's open the
    # 500 long shares receive 250 and the 125 short shares pay 125; bar 2's
    # sale of the long pays 2.50.
    assert equity.tolist() == [10_000.0, 9_996.5, 10_119.0, 10_119.0]


def test_a_delisting_closes_the_position_at_the_last_valuation_without_a_fee(tmp_path):
    bars = BARS[:4]
    open_ = {10001: [10.0] * 4, 10003: [40.0, 32.0, 31.0, np.nan]}
    close = {10001: [10.0] * 4, 10003: [40.0, 32.0, 30.0, np.nan]}  # last valued on bar 2
    weights = {10001: [0.5, KEEP, KEEP, KEEP], 10003: [0.5, KEEP, KEEP, KEEP]}
    run_dir = build_quantlab_run(tmp_path, bars, open_, close, weights)
    out_dir = replay(run_dir, tmp_path / "replay", loop="open")
    assert _fills(out_dir) == [
        ("10001.SIM", "BUY", 500.0, 10.0, 2.5),
        ("10003.SIM", "BUY", 125.0, 32.0, 1.0),
        ("10003.SIM", "SELL", 125.0, 30.0, 0.0),
    ]
    assert records.read_snapshot(out_dir, bars[2]).positions["quantity"].to_dict() == {10001: 500.0}
    # Bar 2: cash 10,000 - 5,000 - 4,000 - 3.50 of fees + 125 x 30, holdings 500 x 10.
    assert records.read_equity(out_dir)["equity"].tolist() == [10_000.0, 9_996.5, 9_746.5, 9_746.5]


def test_proportional_slippage_fills_each_order_worse_by_its_rate(tmp_path):
    bars = BARS[:3]
    prices = {10001: [10.0, 20.0, 20.0]}
    weights = {10001: [0.5, 0.0, KEEP]}
    run_dir = build_quantlab_run(tmp_path, bars, prices, prices, weights)
    out_dir = replay(run_dir, tmp_path / "replay", loop="open", fill_model=ProportionalSlippage(0.01))
    assert [fill[:4] for fill in _fills(out_dir)] == [
        ("10001.SIM", "BUY", 500.0, 20.2),
        ("10001.SIM", "SELL", 500.0, 19.8),
    ]
