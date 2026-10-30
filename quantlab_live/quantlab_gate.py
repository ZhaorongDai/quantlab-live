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

``Run.report`` writes a live or replay directory's ``metrics.json`` and
``report.html`` with quantlab's metric and report functions, from the
directory's records alone.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from quantlab.runs.backtest_report import ReportExtras, write_backtest_report
from quantlab.runs.backtest_stats import (
    exposure_stats,
    return_stats,
    round_trip_stats,
    round_trips,
    year_freq,
)
from quantlab.runs.live_predictions import LivePredictionStore
from quantlab.runs.run_metrics import AccountRecord, run_metrics
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.jsonable import to_jsonable

from quantlab_live import records
from quantlab_live.trading.decision_bar import PREDICTION, DecisionBar
from quantlab_live.trading.outcomes import REJECTED, VENUE_CANCELLED

#: The exact quantlab imports of quantlab-live: module -> names.
QUANTLAB_IMPORTS: dict[str, tuple[str, ...]] = {
    # The prediction job only, imported in its own process.
    "quantlab.backtest.live": ("LivePredictionRefused", "predict_live_bar"),
    "quantlab.config": ("get_data_root",),
    "quantlab.dataset.base": ("MarketDataset",),
    "quantlab.portfolio.decision_inputs": ("DecisionInputs",),
    "quantlab.runs.backtest_report": ("ReportExtras", "write_backtest_report"),
    "quantlab.runs.backtest_run": ("BacktestRun", "Market"),
    "quantlab.runs.backtest_stats": (
        "exposure_stats", "return_stats", "round_trip_stats", "round_trips", "year_freq",
    ),
    "quantlab.runs.live_predictions": ("LivePredictionStore",),
    "quantlab.runs.run_metrics": ("AccountRecord", "run_metrics"),
    "quantlab.utils.atomic": ("write_json_atomically",),
    "quantlab.utils.jsonable": ("to_jsonable",),
}

#: What a directory's metrics and report are written to.
METRICS_FILE, REPORT_FILE = "metrics.json", "report.html"

#: A record's bars are daily.
BAR_INTERVAL = "1D"

#: The outcomes whose order found no fill: the report's rejected orders.
_NO_FILL = (REJECTED, VENUE_CANCELLED)

#: The notes of every metrics.json and report.html written from records.
REPORT_NOTES = [
    "Computed from quantlab-live's records: the strategy's equity and holdings "
    "valued at quantlab's close, the fills of the order outcomes on the bar after "
    "their decision, and the snapshots' positions.",
    "A position change no fill explains (a delisting, a position taken over) is "
    "settled at the bar's close mark without a fee and left out of turnover.",
    "The trade metrics are the position level view: one entry to flat round trip "
    "per symbol, split adjusted, with dividends in its profit.",
]

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
        close, delisted = self._marks(calendar[0], calendar[-1])
        for t in calendar:
            frame = pd.DataFrame({"close": close.loc[t], "delisted": delisted.loc[t].astype(bool)})
            if predictions is not None and t in predictions.indexes["timestamp"]:
                row = predictions.sel(timestamp=t, drop=True).to_dataframe()
                frame = frame.join(row.add_prefix(PREDICTION))
            yield DecisionBar.from_frame(t, frame.rename_axis("symbol"))

    def marks(self, bar, *, lookback=pd.Timedelta(days=366)) -> pd.Series:
        """Return quantlab's close marks at ``bar``, by symbol: what holdings are valued at.

        The marks of the bar's ``DecisionBar``: the raw close, carried
        forward over the ``lookback`` before the bar, and on a delisting bar
        the security's last valuation. Unlike ``bars``, it needs no
        prediction of the bar.

        Raises
        ------
        QuantlabNotReady
            If the dataset has no bar at ``bar``.

        Examples
        --------
        ::

            holdings = (snapshot.positions["quantity"] * run.marks(snapshot.bar)).sum()
        """
        bar = pd.Timestamp(bar).normalize()
        calendar = pd.DatetimeIndex(self._dataset.calendar(bar - lookback, bar))
        if not len(calendar) or calendar[-1] != bar:
            raise QuantlabNotReady(f"the price dataset has no bar at {bar.date()}")
        close, _ = self._marks(calendar[0], bar)
        return close.loc[bar].rename_axis("symbol")

    def _marks(self, start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The raw close marks and the delisting flags of every bar from ``start`` to ``end``."""
        valuation_column = self._market.valuation_price_column
        prices = self._dataset.panel(start, end).load()
        raw_close = prices["close"].transpose("timestamp", "symbol").to_pandas()
        delisted = self._dataset.delisting_bars(prices, valuation_column).to_pandas()
        last_value = _last_valuation(
            raw_close, prices[valuation_column].transpose("timestamp", "symbol").to_pandas()
        )
        return raw_close.mask(delisted, last_value).ffill(), delisted

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
        return self._raw("open", start, end)

    def closes(self, start=None, end=None) -> pd.DataFrame:
        """Return the raw close of every dataset bar from ``start`` to ``end``.

        A replay feeds these as the closing prints. NaN where the market had
        no close; unlike a decision bar's marks, nothing is carried forward.

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

            first_closes = run.closes().iloc[0]
        """
        return self._raw("close", start, end)

    def _raw(self, name: str, start, end) -> pd.DataFrame:
        start = self.window[0] if start is None else pd.Timestamp(start)
        end = self.window[1] if end is None else pd.Timestamp(end)
        prices = self._dataset.panel(start, end).load()
        frame = prices[name].transpose("timestamp", "symbol").to_pandas()
        return frame.rename_axis(index=None, columns=None)

    def corporate_actions(self, start=None, end=None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the split factor and the cash dividend of every dataset bar.

        Both are quantlab's shared panel variables, on their ex-dates: the
        ``splitFactor`` (new shares per old share, 2 for a two-for-one split,
        1 where none) and the ``divCash`` per share (0 where none).

        Parameters
        ----------
        start, end : datetime-like, optional
            Default to the run's window.

        Returns
        -------
        split_factor, dividend : pandas.DataFrame
            Indexed by bar, one column per symbol; neutral where the
            dataset has no value.

        Examples
        --------
        ::

            split_factor, dividend = run.corporate_actions()
        """
        start = self.window[0] if start is None else pd.Timestamp(start)
        end = self.window[1] if end is None else pd.Timestamp(end)
        prices = self._dataset.panel(start, end, variables=["splitFactor", "divCash"]).load()

        def frame(name: str, neutral: float) -> pd.DataFrame:
            values = prices[name].transpose("timestamp", "symbol").to_pandas()
            return values.fillna(neutral).rename_axis(index=None, columns=None)

        return frame("splitFactor", 1.0), frame("divCash", 0.0)

    def benchmark(self, start, end) -> tuple[str, pd.Series] | None:
        """Return the run's benchmark: its ticker and its valuation prices from ``start`` to ``end``.

        The benchmark is the one the run was made with, rebuilt from the
        run's recorded recipe (a dataset the run held in memory is read from
        the copy beneath the run directory), so it never drifts with a
        later configuration. Its ticker is its ticker lookup's label as of
        ``end``, or the symbol's id when the dataset names no tickers. The
        prices are the market's valuation column (adjusted closes), so
        their ratio is the benchmark's return.

        Returns
        -------
        tuple of (str, pandas.Series) or None
            The ticker and the prices by bar; ``None`` for a run without a
            benchmark.

        Examples
        --------
        ::

            ticker, prices = run.benchmark("2026-10-01", "2026-10-28")
            print(f"{ticker}: {prices.iloc[-1] / prices.iloc[0] - 1:+.2%}")
        """
        dataset = self._backtest_run.rebuild("benchmark_dataset")
        if dataset is None:
            return None
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        column = self._market.valuation_price_column
        prices = dataset.panel(start, end, variables=[column]).load()[column]
        frame = prices.transpose("timestamp", "symbol").to_pandas()
        (symbol,) = frame.columns
        lookup = dataset.ticker_lookup()
        ticker = str(symbol) if lookup is None else str(lookup.label([symbol], end.date())[0])
        return ticker, frame[symbol].rename(None).rename_axis(None)

    def targets(self, loop: str, *, end=None) -> DecisionInputs | xr.DataArray:
        """Return the raw inputs of the run's targets.

        Parameters
        ----------
        loop : {"closed", "open"}
            ``"closed"``: quantlab's decision inputs (the run's rule bound to
            its label specs); ``"open"``: the run's rebalance table,
            ``weight`` on ``(timestamp, symbol)``.
        end : datetime-like, optional
            Closed loop: the last bar of a replay, which is not decided (its
            orders would have no next bar to fill on). Without it the
            schedule is open-ended, as a live run's is.

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
        return DecisionInputs.from_run(self._backtest_run, end=end, price_dataset=self._dataset)

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

    def report(self, directory, *, title: str | None = None) -> dict:
        """Write ``metrics.json`` and ``report.html`` of a live or replay directory.

        Both are quantlab's (``run_metrics`` and ``write_backtest_report``),
        computed from the directory's records alone (``records.read_book``),
        so live, replay and quantlab's research backtests share one
        convention. NautilusTrader's own reports, mark-to-market and
        portfolio statistics are never read. The account is the strategy's:

        - its value is the equity record (cash plus holdings at quantlab's
          close), and the value before the first bar is the first bar's;
        - its fills are the outcomes' fills, on the bar after their decision
          (a decision on the last bar has none yet); a position change no
          fill explains (a delisting, a position held at the first bar) is a
          settlement at the bar's close mark, without a fee;
        - shares and prices are split adjusted to the first bar with the
          run's split factors, and each dividend is a cash flow of the
          position held into its ex-date;
        - the rebalances are the bars whose decision sets a weight.

        The ``whole`` block opens with the rows vectorbt's statistics give a
        research run, computed from these fills and values with quantlab's
        statistics.

        Parameters
        ----------
        directory : path-like
            The live or replay directory.
        title : str, optional
            The report's title; the directory's name by default.

        Returns
        -------
        dict
            The metrics, as ``metrics.json`` holds them.

        Raises
        ------
        FileNotFoundError
            If the directory records no equity.

        Examples
        --------
        ::

            metrics = open_run(run_dir).report(replay_dir)
            metrics["whole"]["Total Return [%]"]
        """
        directory = Path(directory)
        book = records.read_book(directory)
        bars = book.equity.index
        split_factor, dividend = self.corporate_actions(bars[0], bars[-1])
        symbols = book.positions.columns
        factor = split_factor.reindex(index=bars, columns=symbols).fillna(1.0)
        factor.iloc[0] = 1.0  # a split before the first bar is in its raw records
        ratio = factor.cumprod()  # raw shares per unit of the first bar
        held = book.positions.ffill().fillna(0.0) / ratio
        marks = book.closes.ffill() * ratio
        fills, settlements = _fills(book, ratio, held, marks)
        flows = _dividends(dividend.reindex(index=bars, columns=symbols).fillna(0.0), ratio, held)
        annualization = self._backtest_run.annualization
        account = _account(book, fills, settlements, marks, year_freq(
            BAR_INTERVAL, annualization.trading_days_per_year, annualization.session_minutes_per_day
        ))
        close = _on_bars(marks)
        trips = round_trips(account.orders, close, cash_flows=flows)
        account = dataclasses.replace(account, trades=trips)
        metrics = run_metrics(account, engine_rows=_engine_rows(account, close, book.equity["cash"]))
        metrics["notes"] = list(REPORT_NOTES)
        metrics = to_jsonable(metrics)
        write_json_atomically(directory / METRICS_FILE, metrics, indent=2)
        write_backtest_report(
            account,
            directory / REPORT_FILE,
            title=directory.name if title is None else title,
            metrics=metrics,
            weights=_on_bars(book.weights),
            config=self._backtest_run.recipe(),
            extras=ReportExtras(
                tables={"Order outcomes": book.outcomes["status"].value_counts().sort_index().to_dict()},
                setup={"Fees": "the broker's fills, from the order outcomes", "Run": self.run_dir.name},
                prices=close,
            ),
        )
        return metrics


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


def _on_bars(frame: pd.DataFrame) -> xr.DataArray:
    """A frame by bar and symbol as a ``(timestamp, symbol)`` array."""
    return xr.DataArray(
        frame.to_numpy(dtype=float),
        dims=("timestamp", "symbol"),
        coords={"timestamp": pd.DatetimeIndex(frame.index).values, "symbol": frame.columns.to_numpy()},
    )


def _orders(rows: list[dict]) -> xr.Dataset:
    """Fill rows as quantlab's orders dataset, on an ``order`` dimension."""
    if not rows:
        return xr.Dataset()
    frame = pd.DataFrame(rows)
    return xr.Dataset(
        {name: ("order", frame[name].to_numpy()) for name in ("timestamp", "symbol", "size", "price", "fees")}
    )


def _fills(book, ratio: pd.DataFrame, held: pd.DataFrame, marks: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """The fills of a book in first-bar units, settlements included, and the settlements.

    An outcome's fills are on the bar after its decision. A held change the
    fills of a bar do not explain is settled at that bar's mark.
    """
    bars = book.equity.index
    outcomes = book.outcomes[book.outcomes["filled_quantity"] != 0]
    by_bar: dict[pd.Timestamp, list[dict]] = {}
    for row in outcomes.itertuples():
        at = bars.searchsorted(pd.Timestamp(row.bar), "right")
        if at == len(bars):
            continue  # its fill bar is not valued yet
        bar, units = bars[at], float(ratio.at[bars[at], row.symbol])
        by_bar.setdefault(bar, []).append(
            {"timestamp": bar, "symbol": int(row.symbol), "size": row.filled_quantity / units,
             "price": row.avg_px * units, "fees": float(row.fees)}
        )
    fills, settlements = [], []
    before = pd.Series(0.0, index=held.columns)
    for bar in bars:
        mine = by_bar.get(bar, [])
        traded = pd.Series({f["symbol"]: 0.0 for f in mine}, dtype=float)
        for fill in mine:
            traded[fill["symbol"]] += fill["size"]
        gap = held.loc[bar] - before.add(traded, fill_value=0.0).reindex(held.columns, fill_value=0.0)
        fills.extend(mine)
        for symbol, size in gap[~np.isclose(gap, 0.0, atol=1e-6)].items():
            fills.append({"timestamp": bar, "symbol": int(symbol), "size": float(size),
                          "price": float(marks.at[bar, symbol]), "fees": 0.0})
            settlements.append({"settlement_timestamp": bar, "axis_symbol": int(symbol)})
        before = held.loc[bar]
    return fills, settlements


def _dividends(dividend: pd.DataFrame, ratio: pd.DataFrame, held: pd.DataFrame) -> xr.Dataset | None:
    """Each dividend paid to the units held into its ex-date, as round-trip cash flows."""
    paid = dividend.iloc[1:] * ratio.iloc[1:] * held.shift(1).iloc[1:]
    rows = [
        (bar, int(symbol), float(amount))
        for (bar, symbol), amount in paid.stack().items()
        if amount != 0 and np.isfinite(amount)
    ]
    if not rows:
        return None
    bars, symbols, amounts = zip(*rows, strict=True)
    return xr.Dataset({
        "timestamp": ("flow", pd.DatetimeIndex(bars).values),
        "symbol": ("flow", np.array(symbols)),
        "amount": ("flow", np.array(amounts)),
    })


def _account(book, fills: list[dict], settlements: list[dict], marks: pd.DataFrame, year: pd.Timedelta) -> AccountRecord:
    """The strategy's account record from its book and fills."""
    bars = book.equity.index
    equity = book.equity["equity"]
    init_cash = float(equity.iloc[0])
    returns = equity / equity.shift(1).fillna(init_cash) - 1.0
    on_bars = dict(dims="timestamp", coords={"timestamp": bars.values})
    decided = book.weights.notna().any(axis=1).to_numpy()
    worth = book.positions.ffill().fillna(0.0) * book.closes.ffill()
    holdings = worth.div(equity, axis=0).fillna(0.0)
    deviation = []
    for at in np.flatnonzero(decided[:-1]):
        target = book.weights.iloc[at]
        target = target[target.notna()]
        deviation.append(float((holdings.iloc[at + 1].reindex(target.index, fill_value=0.0) - target).abs().max()))
    rejected = [
        {"fill_timestamp": _next_bar(bars, row.bar), "symbol": int(row.symbol), "status": row.status}
        for row in book.outcomes[book.outcomes["status"].isin(_NO_FILL)].itertuples()
    ]
    return AccountRecord(
        value=xr.DataArray(equity.to_numpy(dtype=float), **on_bars),
        returns=xr.DataArray(returns.to_numpy(dtype=float), **on_bars),
        orders=_orders(fills),
        rebalances=xr.Dataset(
            {"scheduled": ("timestamp", decided), "decided": ("timestamp", decided)},
            coords={"timestamp": bars.values},
        ),
        init_cash=init_cash,
        bar_interval=BAR_INTERVAL,
        year_freq=year,
        settlements=settlements,
        rejected_orders=rejected,
        max_target_deviation=max(deviation) if deviation else None,
        holdings=_on_bars(holdings),
    )


def _next_bar(bars: pd.DatetimeIndex, bar) -> pd.Timestamp:
    """The bar after ``bar``, or ``bar`` itself when it is the last."""
    at = bars.searchsorted(pd.Timestamp(bar), "right")
    return bars[min(at, len(bars) - 1)]


def _engine_rows(account: AccountRecord, close: xr.DataArray, cash: pd.Series) -> dict:
    """The rows vectorbt's statistics open a research run's ``whole`` block with, from fills and values."""
    stats = return_stats(account.returns, bar_interval=account.bar_interval, year_freq=account.year_freq)
    orders = account.orders
    fees = float(orders["fees"].sum()) if orders.sizes.get("order", 0) else 0.0
    cash_on_bars = xr.DataArray(cash.to_numpy(dtype=float), dims="timestamp", coords={"timestamp": close.timestamp})
    exposure_fills = orders if orders.sizes.get("order", 0) else xr.Dataset()
    rows = {name: stats[name] for name in ("Start", "End", "Period")}
    rows["Start Value"] = account.init_cash
    rows["End Value"] = float(account.value[-1])
    rows["Total Return [%]"] = stats["Total Return [%]"]
    rows.update(exposure_stats(exposure_fills, close, cash_on_bars))
    rows["Total Fees Paid"] = fees
    rows["Max Drawdown [%]"] = stats["Max Drawdown [%]"]
    rows["Max Drawdown Duration"] = stats["Max Drawdown Duration"]
    rows.update(round_trip_stats(account.trades, bar_interval=account.bar_interval))
    for name in ("Sharpe Ratio", "Calmar Ratio", "Omega Ratio", "Sortino Ratio"):
        rows[name] = stats[name]
    return rows


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
