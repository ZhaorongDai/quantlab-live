"""Synthetic, model-free quantlab runs built through quantlab's public API.

``build_quantlab_run`` writes a small CRSP-shaped price store (the
``CrspStockDataset`` panel layout: integer PERMNOs on ``symbol``, raw
``open``/``high``/``low``/``close``, the adjusted ``adj*`` group, ``divCash``,
``splitFactor``, ``cumfacpr``, ``cumfacshr``) and backtests a given rebalance
table on it with quantlab's ``WeightsVectorBt.run_weights``, which writes a
real run directory (read through quantlab's ``BacktestRun``). Nothing here
imports quantlab's ``tests/``. Test-only: the allowlist of
``quantlab_live.quantlab_gate`` covers the package, not the tests.

``build_constructor_run`` builds a closed-loop run the way ADR 0007 has
the parity fixtures built: a model-free ``PredictionPanel``, quantlab's
``DecisionInputs.weights`` over it (the inputs quantlab's cross-section
backtester assembles: tradability, the rebalance schedule, the rule's
``history_bars`` price window, the delisting marks), ``run_weights`` of a
``CrossSectionBacktestConfig``
carrying the rule (so the run's recipe records it), and the panel written
where a run with a model keeps its prediction panel (``PREDICTION_PANEL``).
What the rule read while deciding (its factors, a risk model's exposures and
estimate) is recorded under the backtester's component paths and added to the
run's ``data_fingerprint``, as quantlab's ``run()`` records it.

Later tickets extend the fixture by passing their own raw prices and
variables (``variables=``: a split, a dividend, a halt, a delisting) and
their own table (``weights=``); ``adjusted_scale`` keeps the adjusted group a
fixed multiple of the raw one so a raw/adjusted mix-up shows in the numbers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.predefined.weights import WeightsVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig, WeightsBacktestConfig
from quantlab.dataset.config import CrspDatasetConfig
from quantlab.portfolio.base import PortfolioConstructor
from quantlab.runs.prediction_panel import LabelSpec, PredictionPanel
from quantlab.tracking.base import Tracker
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.core.component import walk_components
from quantlab.runs.record import DataRecorder

#: Where a quantlab run with a model keeps its prediction panel; the fixture
#: writes one into a model-free run (the one place outside quantlab that names
#: a run file, for a test stand-in only).
PREDICTION_PANEL = "predictions.zarr"

#: The adjusted group is the raw group times this, so a raw/adjusted mix-up
#: changes every number a test checks.
ADJUSTED_SCALE = 0.5


def write_crsp_store(
    path: Path,
    bars: pd.DatetimeIndex,
    open_: Mapping[int, Sequence[float]],
    close: Mapping[int, Sequence[float]],
    *,
    variables: Mapping[str, Mapping[int, Sequence[float]]] | None = None,
    drop_variables: Sequence[str] = (),
    adjusted_scale: float = ADJUSTED_SCALE,
) -> None:
    """Write a CRSP-shaped daily panel to the Zarr store at ``path``.

    ``open_`` and ``close`` are the raw prices per PERMNO (NaN where the
    market had none). ``high``/``low`` are the max/min of the two, the
    adjusted group is the raw one times ``adjusted_scale``, corporate-action
    fields are neutral (``splitFactor`` 1, ``divCash`` 0, factors 1) unless
    ``variables`` overrides them, and ``drop_variables`` leaves names out.
    """
    permnos = sorted(close)
    raw_open = np.array([open_[p] for p in permnos], dtype=float).T
    raw_close = np.array([close[p] for p in permnos], dtype=float).T
    priced = np.isfinite(raw_close).astype(float)
    priced[priced == 0] = np.nan
    panel = {
        "open": raw_open,
        "high": np.fmax(raw_open, raw_close),
        "low": np.fmin(raw_open, raw_close),
        "close": raw_close,
        "volume": 1e6 * priced,
        "divCash": 0.0 * priced,
        "splitFactor": 1.0 * priced,
        "cumfacpr": 1.0 * priced,
        "cumfacshr": 1.0 * priced,
    }
    for name in ("open", "high", "low", "close"):
        panel["adj" + name.capitalize()] = panel[name] * adjusted_scale
    panel["adjVolume"] = panel["volume"]
    for name, values in (variables or {}).items():
        panel[name] = np.array([values[p] for p in permnos], dtype=float).T
    dataset = xr.Dataset(
        {
            name: (("timestamp", "symbol"), values)
            for name, values in panel.items()
            if name not in drop_variables
        },
        coords={"timestamp": bars, "symbol": np.array(permnos, dtype=np.int64)},
    )
    dataset.to_zarr(path, mode="w")


def build_quantlab_run(
    root: Path,
    bars: pd.DatetimeIndex,
    open_: Mapping[int, Sequence[float]],
    close: Mapping[int, Sequence[float]],
    weights: Mapping[int, Sequence[float]],
    *,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
    slippage: float = 0.0,
    fill_price_column: str = "adjOpen",
    valuation_price_column: str = "adjClose",
    variables: Mapping[str, Mapping[int, Sequence[float]]] | None = None,
    drop_variables: Sequence[str] = (),
    benchmark: tuple[Mapping[int, Sequence[float]], Mapping[int, Sequence[float]]] | None = None,
    tracker: Tracker | None = None,
) -> Path:
    """Write a CRSP-shaped store and a quantlab weights run on it; return the run dir.

    ``weights`` is the rebalance table per PERMNO over ``bars`` (NaN keeps
    the holding); every PERMNO of ``close`` must be in it. ``benchmark`` is
    the raw ``(open, close)`` of one PERMNO, written to its own store and
    bought and held by the run; ``tracker`` is the run's tracker.
    """
    root = Path(root)
    dataset = _crsp_dataset(
        root, bars, open_, close, variables=variables, drop_variables=drop_variables
    )
    start, end = (b.strftime("%Y-%m-%d") for b in (bars[0], bars[-1]))
    permnos = sorted(close)
    table = xr.DataArray(
        np.array([weights[p] for p in permnos], dtype=float).T,
        dims=("timestamp", "symbol"),
        coords={"timestamp": bars, "symbol": np.array(permnos, dtype=np.int64)},
    )
    backtester = WeightsVectorBt(
        WeightsBacktestConfig(
            price_dataset=dataset,
            start_date=start,
            end_date=end,
            output_dir=str(root / "runs"),
            rebalance_periods=1,
            fees=fees,
            slippage=slippage,
            init_cash=init_cash,
            fill_price_column=fill_price_column,
            valuation_price_column=valuation_price_column,
            trading_days_per_year=252,
            session_minutes_per_day=390,
            benchmark_dataset=None
            if benchmark is None
            else _crsp_dataset(root / "benchmark", bars, *benchmark),
            **({} if tracker is None else {"tracker": tracker}),
        )
    )
    return Path(backtester.run_weights(table).run_dir)


def _crsp_dataset(root: Path, bars, open_, close, **store) -> CrspStockDataset:
    """Write the store under ``root`` and return the ``CrspStockDataset`` reading it."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "crsp.zarr"
    write_crsp_store(path, bars, open_, close, **store)
    return CrspStockDataset(
        CrspDatasetConfig(
            zarr_file_path=str(path),
            raw_data_dir_path=str(root / "raw"),
            reference_dir=str(root / "reference"),
            start_date=bars[0].strftime("%Y-%m-%d"),
            end_date=bars[-1].strftime("%Y-%m-%d"),
        )
    )


def build_constructor_run(
    root: Path,
    bars: pd.DatetimeIndex,
    open_: Mapping[int, Sequence[float]],
    close: Mapping[int, Sequence[float]],
    predictions: Mapping[str, Mapping[int, Sequence[float]]],
    rule: PortfolioConstructor | Callable[[CrspStockDataset], PortfolioConstructor],
    labels: Sequence[LabelSpec],
    *,
    first_bar: int = 0,
    rebalance_periods: int = 1,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
    slippage: float = 0.0,
    variables: Mapping[str, Mapping[int, Sequence[float]]] | None = None,
    benchmark: tuple[Mapping[int, Sequence[float]], Mapping[int, Sequence[float]]] | None = None,
) -> tuple[Path, list[str]]:
    """Write a CRSP-shaped store and a closed-loop-ready quantlab run; return it.

    The store holds every bar of ``bars``; the run's window starts at
    ``bars[first_bar]`` (the bars before it are the rule's warm-up).
    ``predictions`` gives each label's prediction per PERMNO over the
    window's bars. ``rule`` may be a function of the price dataset, for a
    rule built on it (a factor risk model's stores). ``benchmark`` is the
    raw ``(open, close)`` of one PERMNO over ``bars``, written to its own
    store as the run's benchmark (``build_quantlab_run``'s). Returns the run
    directory and the ISO timestamps of the bars ``DecisionInputs.weights``
    held after a failure.
    """
    root = Path(root)
    dataset = _crsp_dataset(root, bars, open_, close, variables=variables)
    if not isinstance(rule, PortfolioConstructor):
        rule = rule(dataset)
    window = bars[first_bar:]
    first, last = window[0], window[-1]
    prices = dataset.panel(first, last).load()
    symbols = prices["symbol"].values
    panel = xr.Dataset(
        {
            name: (
                ("timestamp", "symbol"),
                np.array([values[p] for p in symbols], dtype=float).T,
            )
            for name, values in predictions.items()
        },
        coords={"timestamp": window, "symbol": symbols},
    )
    rule.bind(labels)
    backtester = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=dataset,
            start_date=first.strftime("%Y-%m-%d"),
            end_date=last.strftime("%Y-%m-%d"),
            output_dir=str(root / "runs"),
            rebalance_periods=rebalance_periods,
            constructor=rule,
            fees=fees,
            slippage=slippage,
            init_cash=init_cash,
            benchmark_dataset=None
            if benchmark is None
            else _crsp_dataset(root / "benchmark", bars, *benchmark),
        )
    )
    # Keyed as quantlab's backtester keys the reads of run() (its components' paths).
    with DataRecorder(keys=[(item, path) for path, item in walk_components(backtester)]) as reads:
        weights = DecisionInputs(
            dataset,
            rule,
            fill_column="adjOpen",
            valuation_column="adjClose",
            rebalance_periods=rebalance_periods,
            anchor=first,
        ).weights(panel, delisted=dataset.delisting_bars(prices, "adjClose"))
    failed = list(weights.attrs.pop("failed_bars"))
    weights.attrs.clear()
    run_dir = Path(backtester.run_weights(weights).run_dir)
    _add_reads(run_dir, reads.records)
    # A model-free stand-in for a run with a model: the panel goes where
    # quantlab's BacktestRun.predictions reads a run's prediction panel.
    PredictionPanel(panel, labels).write(run_dir / PREDICTION_PANEL)
    return run_dir, failed


def _add_reads(run_dir: Path, records: Mapping[str, list[dict]]) -> None:
    """Add the rule's recorded reads to the run's ``data_fingerprint`` in ``run.json``.

    ``run_weights`` records only the window's price read; quantlab's ``run()``
    records the rule's reads too, in the same recorder. A request already
    recorded is kept once.
    """
    path = run_dir / "run.json"
    record = json.loads(path.read_text())
    fingerprint = record.setdefault("data_fingerprint", {})
    for key, entries in records.items():
        kept = fingerprint.setdefault(key, [])
        requests = [entry["request"] for entry in kept]
        kept.extend(entry for entry in entries if entry["request"] not in requests)
    path.write_text(json.dumps(record, indent=2))
