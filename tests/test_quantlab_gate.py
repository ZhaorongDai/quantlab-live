"""quantlab_gate: decision bars and target inputs read from a tiny real quantlab run."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab.runs.prediction_panel import LabelSpec

from quantlab_live import quantlab_gate
from quantlab_live.quantlab_gate import QuantlabNotReady, open_run
from quantlab_live.trading.decision_bar import DecisionBar
from tests.quantlab_run_fixture import ADJUSTED_SCALE, build_constructor_run, build_quantlab_run

PERMNOS = (10001, 10002, 10003)
LABELS = [LabelSpec("ret_5", "raw", 1, 5)]
BARS = pd.bdate_range("2024-01-02", periods=6)
PACKAGE = Path(quantlab_gate.__file__).parent


def _market(n_bars=len(BARS)):
    rng = np.random.default_rng(1)
    close, open_ = {}, {}
    for k, permno in enumerate(PERMNOS):
        path = (20.0 + 10 * k) * np.exp(np.cumsum(rng.normal(0, 0.02, n_bars)))
        close[permno] = list(np.round(path, 2))
        open_[permno] = list(np.round(path * 1.001, 2))
    return open_, close


def _predictions(seed=2):
    rng = np.random.default_rng(seed)
    return {"ret_5": {p: list(rng.normal(0, 0.01, len(BARS))) for p in PERMNOS}}


@pytest.fixture(scope="module")
def closed_run(tmp_path_factory):
    open_, close = _market()
    run_dir, _ = build_constructor_run(
        tmp_path_factory.mktemp("closed"), BARS, open_, close, _predictions(),
        TopNConstructor(TopNConfig(direction="long_only", top_n=2)), LABELS,
    )
    return run_dir, close


def test_bars_carry_the_raw_close_and_the_predictions_of_each_bar(closed_run):
    run_dir, close = closed_run
    bars = list(open_run(run_dir).bars())
    assert [b.timestamp for b in bars] == list(BARS)
    expected = _predictions()["ret_5"]
    for i, bar in enumerate(bars):
        marks = bar.marks()
        assert marks["close"].to_dict() == {p: close[p][i] for p in PERMNOS}
        assert not marks["delisted"].any()
        predictions = bar.predictions()["ret_5"]
        assert np.allclose(predictions.sel(symbol=list(PERMNOS)).values, [expected[p][i] for p in PERMNOS])


def test_bars_survive_the_custom_data_round_trip(closed_run):
    run_dir, _ = closed_run
    bar = next(open_run(run_dir).bars())
    again = DecisionBar.from_dict(bar.to_dict())
    assert (again.bar, again.ts_event, again.ts_init) == (bar.bar, bar.ts_event, bar.ts_init)
    pd.testing.assert_frame_equal(again.frame(), bar.frame())
    xr.testing.assert_identical(again.predictions(), bar.predictions())


def test_a_bar_after_the_data_is_not_ready(closed_run):
    run_dir, _ = closed_run
    with pytest.raises(QuantlabNotReady):
        list(open_run(run_dir).bars(end=BARS[-1] + pd.offsets.BDay(3)))


def test_a_delisting_bar_marks_the_last_valuation_and_carries_it_forward(tmp_path):
    open_, close = _market()
    gone = PERMNOS[0]
    open_[gone][3:] = [np.nan] * 3
    close[gone][3:] = [np.nan] * 3
    # The delisting row: no raw price, an adjusted close carrying a -10% delisting return.
    adj_close = {p: [c * ADJUSTED_SCALE for c in close[p]] for p in PERMNOS}
    adj_close[gone][3] = close[gone][2] * ADJUSTED_SCALE * 0.9
    weights = {p: [1 / len(PERMNOS)] + [np.nan] * (len(BARS) - 1) for p in PERMNOS}
    run_dir = build_quantlab_run(
        tmp_path, BARS, open_, close, weights, variables={"adjClose": adj_close}
    )
    bars = list(open_run(run_dir).bars())
    assert bars[3].marks().loc[gone, "delisted"]
    assert bars[3].marks().loc[gone, "close"] == pytest.approx(round(close[gone][2] * 0.9, 4))
    assert bars[4].marks().loc[gone, "close"] == bars[3].marks().loc[gone, "close"]


def test_targets_are_the_rebalance_table_open_loop_and_the_decision_inputs_closed_loop(closed_run, tmp_path):
    run_dir, _ = closed_run
    run = open_run(run_dir)
    assert set(run.targets("open").dims) == {"timestamp", "symbol"}
    assert run.targets("closed").rebalances(BARS[0])
    open_, close = _market()
    weights_only = build_quantlab_run(tmp_path, BARS, open_, close, {p: [0.3] * len(BARS) for p in PERMNOS})
    run = open_run(weights_only)
    assert not list(run.bars())[0].predictions().data_vars
    with pytest.raises(ValueError, match="no prediction panel"):
        run.targets("closed")


def test_only_the_gate_imports_quantlab_and_only_its_allowlist():
    found: dict[Path, set[tuple[str, str]]] = {}
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "quantlab":
                found.setdefault(path, set()).update((node.module, a.name) for a in node.names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "quantlab":
                        found.setdefault(path, set()).add((alias.name, "*"))
    assert set(found) == {Path(quantlab_gate.__file__)}
    allowed = {(module, name) for module, names in quantlab_gate.QUANTLAB_IMPORTS.items() for name in names}
    assert found[Path(quantlab_gate.__file__)] == allowed
