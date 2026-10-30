"""``live.value``: the latest snapshot valued at quantlab's close, before the node starts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab_live import records
from quantlab_live.live import value
from tests.quantlab_run_fixture import build_quantlab_run

BARS = pd.bdate_range("2024-01-02", periods=4)
OPEN = {10001: [10.0, 11.0, 12.0, 13.0], 10002: [20.0, 20.0, np.nan, 21.0]}
# 10002 has no close on the snapshot's bar: quantlab's close mark carries its last one.
CLOSE = {10001: [10.0, 11.0, 12.0, 13.0], 10002: [19.0, 20.0, np.nan, 21.0]}
WEIGHTS = {10001: [0.5, np.nan, np.nan, np.nan], 10002: [0.5, np.nan, np.nan, np.nan]}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return build_quantlab_run(tmp_path_factory.mktemp("value"), BARS, OPEN, CLOSE, WEIGHTS)


@pytest.fixture
def live_dir(tmp_path, run_dir):
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    (live_dir / "live.toml").write_text(f'run = "{run_dir}"\n')
    return live_dir


def test_the_latest_snapshot_is_valued_at_quantlabs_close_once(live_dir):
    positions = pd.DataFrame({"quantity": [100.0, 50.0]}, index=pd.Index([10001, 10002], name="symbol"))
    records.write_snapshot(live_dir, records.Snapshot(BARS[1], positions * 0 + 1, cash=0.0))
    records.write_snapshot(live_dir, records.Snapshot(BARS[2], positions, cash=500.0))

    assert value(live_dir, BARS[3]) == BARS[2]
    equity = records.read_equity(live_dir)
    assert equity.index.tolist() == [BARS[2]]
    assert equity.loc[BARS[2]].tolist() == [500.0, 100 * 12.0 + 50 * 20.0, 2_700.0]

    assert value(live_dir, BARS[3]) is None  # already valued
    assert len(records.read_equity(live_dir)) == 1


def test_without_a_snapshot_there_is_nothing_to_value(live_dir):
    assert value(live_dir, BARS[3]) is None


def test_with_an_initial_capital_the_equity_is_the_strategys_own(live_dir, run_dir):
    (live_dir / "live.toml").write_text(f'run = "{run_dir}"\ninitial_capital = 1_000.0\n')
    # The first decision, on BARS[1], held 10 of 10001 at its close of 11; it bought 40 more at 11.5 with 1.00 fees.
    decision = pd.DataFrame(
        {"weight": [0.5], "close": [11.0], "position_quantity": [10.0], "target_quantity": [50.0],
         "order_quantity": [40.0], "skip": [""]},
        index=pd.Index([10001], name="symbol"),
    )
    records.write_decision(live_dir, BARS[1], decision, capital=1_000.0)
    outcomes = pd.DataFrame(
        {"order_quantity": [40.0], "filled_quantity": [40.0], "avg_px": [11.5], "fees": [1.0], "status": ["filled"]},
        index=pd.Index([10001], name="symbol"),
    )
    records.write_outcomes(live_dir, BARS[1], outcomes)
    # The account holds far more cash than the strategy: it is not the strategy's.
    held = pd.DataFrame({"quantity": [50.0]}, index=pd.Index([10001], name="symbol"))
    records.write_snapshot(live_dir, records.Snapshot(BARS[2], held, cash=90_000.0))

    assert value(live_dir, BARS[3]) == BARS[2]

    cash = 1_000.0 - 10 * 11.0 - 40 * 11.5 - 1.0
    assert records.read_equity(live_dir).loc[BARS[2]].tolist() == [cash, 50 * 12.0, cash + 50 * 12.0]
