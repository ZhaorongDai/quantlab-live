"""The quantlab gate: the only module that imports quantlab.

Everything quantlab-live reads from quantlab passes through here, so
quantlab's surface stays small and visible. ``QUANTLAB_IMPORTS`` is the exact
allowlist; a test fails when any other module imports quantlab or when this
one imports anything else from it.

``open_run`` serves the node and replay with in-process reads of a quantlab
backtest run: its decision bars (``DecisionBar``), the raw inputs of its
targets and its symbols' listings.

``prepare_day`` is the process-level contract of a live day: it reads the
status file of quantlab's daily data update and runs quantlab's live
prediction job as a subprocess bound to NUMA node 1. Run as a module
(``python -m quantlab_live.quantlab_gate predict ...``), this file is that
job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantlab.config import get_data_root
from quantlab.dataset.base import MarketDataset
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.runs.backtest_run import BacktestRun, Market
from quantlab.runs.live_predictions import LivePredictionStore

from quantlab_live.trading.decision_bar import PREDICTION, DecisionBar

#: The exact quantlab imports of quantlab-live: module -> names.
QUANTLAB_IMPORTS: dict[str, tuple[str, ...]] = {
    # The prediction job only, imported in its own process.
    "quantlab.backtest.live": ("LivePredictionRefused", "predict_live_bar"),
    "quantlab.config": ("get_data_root",),
    "quantlab.dataset.base": ("MarketDataset",),
    "quantlab.portfolio.decision_inputs": ("DecisionInputs",),
    "quantlab.runs.backtest_run": ("BacktestRun", "Market"),
    "quantlab.runs.live_predictions": ("LivePredictionStore",),
}

#: Decimal places of a raw price mark (instrument price precision).
PRICE_PRECISION = 4

#: The cores of NUMA node 1 on the trading server, which every job is bound to.
NUMA_NODE_1_CPUS = "64-127"

#: quantlab's daily data update writes its outcome here, under the data root.
UPDATE_STATUS_FILE = "update_status.json"

#: Exit status of the prediction job per outcome (``python -m quantlab_live.quantlab_gate``).
JOB_APPENDED, JOB_FAILED, JOB_MISSING_DATA, JOB_ALREADY_PREDICTED = 0, 1, 2, 3


class QuantlabNotReady(LookupError):
    """quantlab has no data yet for a bar asked for.

    Examples
    --------
    >>> issubclass(QuantlabNotReady, LookupError)
    True
    """


@dataclass(frozen=True)
class Run:
    """A quantlab backtest run, read in-process.

    Attributes
    ----------
    run_dir : pathlib.Path
        The run directory, absolute.
    window : tuple of pandas.Timestamp
        The run's first and last bar.

    Examples
    --------
    ::

        run = open_run("runs/USEquityCrossectionSelectStockVectorBt_20261020")
        for bar in run.bars():
            predictions = bar.predictions()
    """

    run_dir: Path
    window: tuple[pd.Timestamp, pd.Timestamp]
    _backtest_run: BacktestRun
    _dataset: MarketDataset
    _market: Market

    def bars(self, start=None, end=None) -> Iterator[DecisionBar]:
        """Yield the decision bar of every dataset bar from ``start`` to ``end``.

        Each bar carries, by symbol of the run's universe, the raw close
        (carried forward over missing bars; on a delisting bar the security's
        last valuation in raw prices: its last raw close grown by the
        valuation column's return since), whether it delists at the bar, and
        the bar's predictions when the run has a prediction panel.

        Parameters
        ----------
        start, end : datetime-like, optional
            Default to the run's window.

        Raises
        ------
        QuantlabNotReady
            If the dataset, or the prediction panel of a run with one, has no
            data for a bar up to ``end``.

        Examples
        --------
        ::

            last = list(run.bars(end=run.window[1]))[-1]
        """
        start = self.window[0] if start is None else pd.Timestamp(start)
        end = self.window[1] if end is None else pd.Timestamp(end)
        calendar = pd.DatetimeIndex(self._dataset.calendar(start, end))
        if not len(calendar) or calendar[-1] < end.normalize():
            raise QuantlabNotReady(f"the price dataset has no bar at {end.date()} yet")
        predictions = None
        if self._backtest_run.has_predictions:
            predictions = self._backtest_run.predictions().predictions
            have = pd.DatetimeIndex(predictions.indexes["timestamp"])
            if have[-1] < calendar[-1]:
                raise QuantlabNotReady(f"the prediction panel has no bar at {calendar[-1].date()} yet")
            predictions = predictions.sel(timestamp=slice(calendar[0], calendar[-1])).load()
        valuation_column = self._market.valuation_price_column
        prices = self._dataset.panel(calendar[0], calendar[-1]).load()
        raw_close = prices["close"].transpose("timestamp", "symbol").to_pandas()
        delisted = self._dataset.delisting_bars(prices, valuation_column).to_pandas()
        last_value = _last_valuation(
            raw_close, prices[valuation_column].transpose("timestamp", "symbol").to_pandas()
        )
        close = raw_close.mask(delisted, last_value).ffill()
        for t in calendar:
            frame = pd.DataFrame({"close": close.loc[t], "delisted": delisted.loc[t].astype(bool)})
            if predictions is not None and t in predictions.indexes["timestamp"]:
                row = predictions.sel(timestamp=t, drop=True).to_dataframe()
                frame = frame.join(row.add_prefix(PREDICTION))
            yield DecisionBar.from_frame(t, frame.rename_axis("symbol"))

    @property
    def init_cash(self) -> float:
        """The run's initial cash.

        Examples
        --------
        ::

            capital = run.init_cash
        """
        return self._backtest_run.init_cash

    def opens(self, start=None, end=None) -> pd.DataFrame:
        """Return the raw open of every dataset bar from ``start`` to ``end``.

        A replay feeds these as the opening prints. NaN where the market had
        no open.

        Parameters
        ----------
        start, end : datetime-like, optional
            Default to the run's window.

        Returns
        -------
        pandas.DataFrame
            Indexed by bar, one column per symbol.

        Examples
        --------
        ::

            first_prints = run.opens().iloc[0]
        """
        start = self.window[0] if start is None else pd.Timestamp(start)
        end = self.window[1] if end is None else pd.Timestamp(end)
        prices = self._dataset.panel(start, end).load()
        frame = prices["open"].transpose("timestamp", "symbol").to_pandas()
        return frame.rename_axis(index=None, columns=None)

    def targets(self, loop: str) -> DecisionInputs | xr.DataArray:
        """Return the raw inputs of the run's targets.

        Parameters
        ----------
        loop : {"closed", "open"}
            ``"closed"``: quantlab's decision inputs (the run's rule bound to
            its label specs, schedule left open-ended as a live run's is);
            ``"open"``: the run's rebalance table, ``weight`` on
            ``(timestamp, symbol)``.

        Raises
        ------
        ValueError
            For another ``loop``, or ``"closed"`` on a run without a prediction
            panel or valued at raw prices.

        Examples
        --------
        ::

            table = run.targets("open")
            inputs = run.targets("closed")
        """
        if loop == "open":
            return self._backtest_run.weights()["weight"]
        if loop != "closed":
            raise ValueError(f"loop must be 'closed' or 'open', not {loop!r}")
        if not self._backtest_run.has_predictions:
            raise ValueError(
                f"quantlab run {self.run_dir} has no prediction panel; replay it open loop"
            )
        if not self._market.valuation_price_column.startswith("adj"):
            raise ValueError(
                f"quantlab run {self.run_dir} is valued at raw prices; a closed loop "
                f"decides on adjusted closes"
            )
        return DecisionInputs.from_run(self._backtest_run, end=None, price_dataset=self._dataset)

    def listings(self, date, symbols=None) -> pd.DataFrame:
        """Return the ticker and primary exchange of each symbol as of ``date``.

        Read from the run's price dataset (a Sharadar store's ticker sidecar,
        refreshed by every data update). The exchange is the vendor's name of
        it (``NYSE``, ``NASDAQ``, ``NYSEMKT``, ``NYSEARCA``, ``BATS``). Either
        is ``None`` when quantlab does not know it: a symbol it cannot place,
        or a ticker the symbol had before its current one, whose exchange
        the vendor does not keep.

        Parameters
        ----------
        date : datetime-like
            The day the listings hold on.
        symbols : sequence, optional
            The symbols to list; every symbol of the price dataset's store by
            default.

        Returns
        -------
        pandas.DataFrame
            Indexed by symbol, columns ``ticker`` and ``exchange``.

        Raises
        ------
        ValueError
            If the price dataset names no tickers.
        NotImplementedError
            If its ticker lookup records no listings (a CRSP store).
        FileNotFoundError
            If its ticker sidecar is missing.

        Examples
        --------
        ::

            run.listings("2026-10-29").loc[194817].tolist()  # ['META', 'NASDAQ']
        """
        lookup = self._dataset.ticker_lookup()
        if lookup is None:
            raise ValueError(
                f"quantlab run {self.run_dir}: its price dataset "
                f"{type(self._dataset).__name__} names no tickers"
            )
        symbols = list(self._dataset.stored_symbols() if symbols is None else symbols)
        found = lookup.listings(symbols, pd.Timestamp(date).date())
        return pd.DataFrame(
            {"ticker": [x.ticker for x in found], "exchange": [x.exchange for x in found]},
            index=pd.Index(symbols, name="symbol"),
            dtype=object,
        )


def open_run(run_dir: str | Path) -> Run:
    """Open the quantlab backtest run at ``run_dir``.

    Raises
    ------
    ValueError
        If ``run_dir`` is not a quantlab run directory, or its price dataset
        is not a ``MarketDataset``.

    Examples
    --------
    >>> open_run("no/such/run")
    Traceback (most recent call last):
    ...
    ValueError: ... is not a quantlab run directory: ...
    """
    run_dir = Path(run_dir).resolve()
    try:
        backtest_run = BacktestRun.open(run_dir)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{run_dir} is not a quantlab run directory: {error}") from error
    dataset = backtest_run.rebuild("price_dataset")
    if not isinstance(dataset, MarketDataset):
        raise ValueError(
            f"quantlab run {run_dir}: its price dataset {type(dataset).__name__} "
            f"is not a MarketDataset"
        )
    window = tuple(pd.Timestamp(bar) for bar in backtest_run.window)
    return Run(run_dir, window, backtest_run, dataset, backtest_run.market)


def _last_valuation(raw_close: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    """Return each bar's last valuation in raw prices, at the price precision.

    The last raw close times the valuation's return since that close; the last
    raw close itself where the valuation cannot say. A total-loss delisting
    (valuation 0) is worth 0.
    """
    paired = raw_close.notna() & valuation.notna()
    last_close = raw_close.ffill()
    grown = raw_close.where(paired).ffill() * valuation / valuation.where(paired).ffill()
    value = grown.where(np.isfinite(grown) & (grown >= 0), last_close)
    return value.round(PRICE_PRECISION)


@dataclass(frozen=True)
class Ready:
    """The day's bar is predicted: the live prediction store holds it.

    Attributes
    ----------
    bar : pandas.Timestamp
        The bar predicted, the last bar of the run's price dataset.

    Examples
    --------
    >>> Ready(pd.Timestamp("2026-10-28"))
    Ready(bar=Timestamp('2026-10-28 00:00:00'))
    """

    bar: pd.Timestamp


@dataclass(frozen=True)
class Pending:
    """The day cannot be traded yet; ask again later.

    Attributes
    ----------
    reason : str
        Why, for the day's log and mail.

    Examples
    --------
    >>> Pending("the data update is running").reason
    'the data update is running'
    """

    reason: str


class PredictionFailed(RuntimeError):
    """quantlab's prediction job failed in a way polling again will not cure.

    Examples
    --------
    >>> issubclass(PredictionFailed, RuntimeError)
    True
    """


def prepare_day(
    run_dir: str | Path,
    date,
    *,
    store: str | Path,
    data_root: str | Path | None = None,
    may_lag: Iterable[str | Path] = (),
    cpus: str = NUMA_NODE_1_CPUS,
) -> Ready | Pending:
    """Make the run's prediction of the day's bar, once quantlab's data is up to date.

    ``Pending`` until quantlab's daily data update has written ``done`` for
    ``date`` in ``<data_root>/update_status.json``. Then the bar is the last
    bar of the run's price dataset: if the live prediction store already
    holds it the day is ``Ready`` at once, and otherwise quantlab's
    prediction job (``quantlab.backtest.live.predict_live_bar``) runs as a
    subprocess under ``taskset -c <cpus>`` and appends it. The job is run
    only while the bar is missing, so polling is safe and never predicts a
    bar twice. Thread environment variables are passed through as they are.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        The quantlab backtest run, with a model.
    date : datetime-like
        The New York trading day the update ran for.
    store : str or pathlib.Path
        The run's live prediction store; the first job creates it.
    data_root : str or pathlib.Path, optional
        quantlab's data root; quantlab's own (``QUANTLAB_DATA_DIR``) by
        default. The job gets it as ``QUANTLAB_DATA_DIR``.
    may_lag : iterable of path, optional
        Stores the run reads that may end before the bar (a rate published
        the next business day).
    cpus : str, default ``NUMA_NODE_1_CPUS``
        The ``taskset`` CPU list of the job.

    Returns
    -------
    Ready or Pending
        ``Pending`` also when the job finds an input lacking the bar.

    Raises
    ------
    ValueError
        If the store holds another run's or checkpoint's predictions, or the
        run has no model.
    PredictionFailed
        If the job fails for another reason (its output is in the message).

    Examples
    --------
    ::

        while isinstance(day := prepare_day(run_dir, today, store=store), Pending):
            log(day.reason)
            sleep(15 * 60)
    """
    data_root = Path(get_data_root() if data_root is None else data_root)
    status_path = data_root / UPDATE_STATUS_FILE
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return Pending(f"the data update's status {status_path} is unreadable: {error}")
    day = pd.Timestamp(date).date().isoformat()
    if status.get("date") != day or status.get("state") != "done":
        return Pending(
            f"the data update is not done for {day}: {status_path} says "
            f"{status.get('state')!r} for {status.get('date')!r}"
        )
    run = open_run(run_dir)
    bar = run._dataset.last_bar()
    if bar is None:
        return Pending(f"quantlab run {run.run_dir}: its price store holds no bar")
    live = LivePredictionStore(store)
    live.check_owner(run._backtest_run)
    if live.has(bar):
        return Ready(bar)
    command = [
        "taskset", "-c", cpus,
        sys.executable, "-m", "quantlab_live.quantlab_gate", "predict", str(run.run_dir),
        "--store", str(Path(store).absolute()),
        *(arg for path in may_lag for arg in ("--may-lag", str(path))),
    ]
    env = {**os.environ, "QUANTLAB_DATA_DIR": str(data_root)}
    done = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    output = (done.stderr or done.stdout).strip()[-2000:]
    if done.returncode == JOB_MISSING_DATA:
        return Pending(f"the prediction job found data missing for {bar.date()}: {output}")
    if done.returncode in (JOB_APPENDED, JOB_ALREADY_PREDICTED) and live.has(bar):
        return Ready(bar)
    raise PredictionFailed(
        f"the prediction job of {run.run_dir} for {bar.date()} exited {done.returncode} "
        f"without the bar in {store}: {output}"
    )


def _predict(argv: list[str] | None = None) -> int:
    """Run quantlab's live prediction of a run's last bar; return the job's exit status."""
    from quantlab.backtest.live import LivePredictionRefused, predict_live_bar

    parser = argparse.ArgumentParser(prog="python -m quantlab_live.quantlab_gate predict")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--may-lag", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        done = predict_live_bar(args.run_dir, args.store, may_lag=args.may_lag)
    except LivePredictionRefused as refused:
        print(f"refused ({refused.reason}): {refused}", file=sys.stderr)
        return {"already_predicted": JOB_ALREADY_PREDICTED, "missing_data": JOB_MISSING_DATA}.get(
            refused.reason, JOB_FAILED
        )
    print(f"predicted {done.timestamp.date()} into {done.store.path}")
    return JOB_APPENDED


if __name__ == "__main__":
    if sys.argv[1:2] != ["predict"]:
        sys.exit("usage: python -m quantlab_live.quantlab_gate predict RUN_DIR --store STORE")
    sys.exit(_predict(sys.argv[2:]))
