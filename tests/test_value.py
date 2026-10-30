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
