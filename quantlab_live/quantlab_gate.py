"""The quantlab gate: the only module that imports quantlab.

Everything quantlab-live reads from quantlab passes through here, so
quantlab's surface stays small and visible. ``QUANTLAB_IMPORTS`` is the exact
allowlist; a test fails when any other module imports quantlab or when this
one imports anything else from it.

``open_run`` serves the node and replay with in-process reads of a quantlab
backtest run: its decision bars (``DecisionBar``) and the raw inputs of its
targets.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantlab.dataset.base import MarketDataset
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.runs.backtest_run import BacktestRun, Market

from quantlab_live.trading.decision_bar import PREDICTION, DecisionBar

#: The exact quantlab imports of quantlab-live: module -> names.
QUANTLAB_IMPORTS: dict[str, tuple[str, ...]] = {
    "quantlab.dataset.base": ("MarketDataset",),
    "quantlab.portfolio.decision_inputs": ("DecisionInputs",),
    "quantlab.runs.backtest_run": ("BacktestRun", "Market"),
}

#: Decimal places of a raw price mark (instrument price precision).
PRICE_PRECISION = 4


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
        return DecisionInputs.from_run(self.run_dir, end=end)


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
