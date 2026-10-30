"""Reports: ``report.html`` and ``metrics.json`` from the records, by quantlab's functions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantlab_live import records
from quantlab_live.quantlab_gate import open_run
from quantlab_live.replay import replay
from tests.quantlab_run_fixture import build_quantlab_run
from tests.test_replay import BARS, CLOSE, OPEN, WEIGHTS

KEEP = np.nan


def _metrics(directory) -> dict:
    return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return build_quantlab_run(tmp_path_factory.mktemp("open"), BARS, OPEN, CLOSE, WEIGHTS, init_cash=10_000.0)


@pytest.fixture(scope="module")
def replayed(run_dir, tmp_path_factory):
    return replay(run_dir, tmp_path_factory.mktemp("replayed") / "replay", loop="open")


def test_a_replay_writes_metrics_from_its_records(replayed):
    whole = _metrics(replayed)["whole"]
    # The equity record: 10,000, 9,850.50, 11,100.50, 12,308 (tests/test_replay.py).
    assert whole["Start Value"] == 10_000.0
    assert whole["End Value"] == 12_308.0
    assert whole["Total Return [%]"] == pytest.approx(23.08)
    assert whole["Max Drawdown [%]"] == pytest.approx(1.495)
    # Six fills: 4.50 of IBKR fees on bar 1's three, 2.50 + 1.00 + 1.00 on bar 3's.
    assert whole["Total Orders"] == 6
    assert whole["Total Fees Paid"] == pytest.approx(9.0)
    # 10001 is bought at 10.50 and sold at 12.50; 10002 and 10003 are still held.
    assert (whole["Total Closed Trades"], whole["Total Open Trades"]) == (1, 2)
    assert whole["Best Trade [%]"] == pytest.approx((500 * 12.5 - 500 * 10.5 - 2.5 - 2.5) / (500 * 10.5) * 100)


def test_a_replay_writes_quantlabs_report(replayed):
    page = (replayed / "report.html").read_text(encoding="utf-8")
    assert f"<h1>{replayed.name}</h1>" in page


def test_the_metrics_are_read_from_the_records_alone(run_dir, replayed, tmp_path):
    # A directory holding only the records, with no NautilusTrader report: its
    # equity record is the replay's with the last bar's value doubled.
    for kind in (records.DECISIONS, records.OUTCOMES, records.SNAPSHOTS):
        for path in (replayed / kind).glob("*.parquet"):
            (tmp_path / kind).mkdir(exist_ok=True)
            (tmp_path / kind / path.name).write_bytes(path.read_bytes())
    equity = records.read_equity(replayed)
    for bar, row in equity.iterrows():
        cash = row["cash"] + (row["equity"] if bar == equity.index[-1] else 0.0)
        records.append_equity(tmp_path, bar, cash, row["holdings"])
    metrics = open_run(run_dir).report(tmp_path)
    assert metrics["whole"]["End Value"] == 2 * 12_308.0
    assert _metrics(tmp_path)["whole"]["End Value"] == 2 * 12_308.0
    assert (tmp_path / "report.html").exists()


def test_round_trips_and_fees_hold_across_splits(tmp_path):
    bars = pd.bdate_range("2024-01-02", periods=6)
    prices = {10001: [10.0, 10.0, 5.0, 5.0, 5.0, 5.0], 10002: [20.0, 20.0, 20.0, 100.0, 100.0, 100.0]}
    weights = {10001: [0.5, KEEP, KEEP, KEEP, 0.0, KEEP], 10002: [0.5, KEEP, KEEP, KEEP, 0.0, KEEP]}
    split_factor = {10001: [1.0, 1.0, 2.0, 1.0, 1.0, 1.0], 10002: [1.0, 1.0, 1.0, 0.2, 1.0, 1.0]}
    run_dir = build_quantlab_run(
        tmp_path, bars, prices, prices, weights, init_cash=100_000.0, variables={"splitFactor": split_factor}
    )
    whole = _metrics(replay(run_dir, tmp_path / "replay", loop="open"))["whole"]
    # Each position is one round trip, bought before its split and sold after
    # it at the same value; the fees (25 + 12.50 in, 50 + 2.50 out) are the loss.
    assert (whole["Total Closed Trades"], whole["Total Open Trades"]) == (2, 0)
    assert whole["Total Fees Paid"] == pytest.approx(90.0)
    assert whole["Best Trade [%]"] == pytest.approx(-15 / 50_000 * 100)
    assert whole["Worst Trade [%]"] == pytest.approx(-75 / 50_000 * 100)


def test_a_delisting_is_a_settlement_outside_turnover(tmp_path):
    bars = pd.bdate_range("2024-01-02", periods=4)
    open_ = {10001: [10.0] * 4, 10003: [40.0, 32.0, 31.0, np.nan]}
    close = {10001: [10.0] * 4, 10003: [40.0, 32.0, 30.0, np.nan]}  # last valued on bar 2
    weights = {10001: [0.5, KEEP, KEEP, KEEP], 10003: [0.5, KEEP, KEEP, KEEP]}
    run_dir = build_quantlab_run(tmp_path, bars, open_, close, weights)
    whole = _metrics(replay(run_dir, tmp_path / "replay", loop="open"))["whole"]
    # 10003 is settled at its last valuation of 30 on bar 2, without a fee.
    assert (whole["Total Closed Trades"], whole["Total Open Trades"]) == (1, 1)
    assert whole["Best Trade [%]"] == pytest.approx((125 * 30 - 125 * 32 - 1.0) / (125 * 32) * 100)
    assert whole["Total Fees Paid"] == pytest.approx(3.5)
    # Only bar 1's buys of 9,000 turn over the 10,000 of capital.
    assert whole["Total Turnover [%]"] == pytest.approx(90.0)
