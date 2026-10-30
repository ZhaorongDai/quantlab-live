"""Target sources: what the strategy holds after a decision bar.

A target source turns a decision bar into target weights by symbol. The
strategy sizes them; it does not know where they come from.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr

from quantlab_live.trading.decision_bar import DecisionBar


class RebalanceTableTargets:
    """The targets of a finished rebalance table (the open loop).

    Parameters
    ----------
    table : xarray.DataArray
        Target weights on ``(timestamp, symbol)``; NaN keeps a symbol's
        holding.

    Examples
    --------
    >>> table = xr.DataArray(
    ...     [[0.6, float("nan")]],
    ...     dims=("timestamp", "symbol"),
    ...     coords={"timestamp": [pd.Timestamp("2024-01-02")], "symbol": [10001, 10002]},
    ... )
    >>> bar = DecisionBar(bar=pd.Timestamp("2024-01-02").value)
    >>> RebalanceTableTargets(table).weights(bar, pd.Series(dtype=float)).to_dict()
    {10001: 0.6, 10002: nan}
    """

    def __init__(self, table: xr.DataArray):
        self._table = table.transpose("timestamp", "symbol").to_pandas()

    def weights(self, bar: DecisionBar, current: pd.Series) -> pd.Series:
        """Return the target weight of every symbol at ``bar``, NaN where kept.

        A bar the table does not have keeps every holding. The holdings
        (``current``) never change the table: fills do not feed back.

        Examples
        --------
        >>> table = xr.DataArray(
        ...     [[0.6]], dims=("timestamp", "symbol"),
        ...     coords={"timestamp": [pd.Timestamp("2024-01-02")], "symbol": [10001]},
        ... )
        >>> bar = DecisionBar(bar=pd.Timestamp("2024-01-03").value)
        >>> RebalanceTableTargets(table).weights(bar, pd.Series(dtype=float)).to_dict()
        {10001: nan}
        """
        t = bar.timestamp
        if t not in self._table.index:
            return pd.Series(float("nan"), index=self._table.columns)
        return self._table.loc[t].rename(None).rename_axis(None)


class ConstructorTargets:
    """quantlab's portfolio construction on the holdings (the closed loop, and live).

    On every rebalance bar of the run's schedule, the rule decides on the
    context quantlab's ``DecisionInputs.context`` builds from the bar's
    predictions and the weights the strategy actually holds, so each fill
    feeds the next decision. Any other bar, and a bar the rule holds, keeps
    every holding.

    Parameters
    ----------
    inputs : quantlab DecisionInputs
        The run's decision inputs, from ``Run.targets("closed")``: its
        ``rebalances``, ``context`` and ``constructor``. A replay preloads
        them over its window.

    Examples
    --------
    ::

        targets = ConstructorTargets(run.targets("closed"))
        weights = targets.weights(bar, current)
    """

    def __init__(self, inputs):
        self._inputs = inputs

    def weights(self, bar: DecisionBar, current: pd.Series) -> pd.Series:
        """Return the rule's target weight of every symbol at ``bar``, NaN where kept.

        Parameters
        ----------
        bar : DecisionBar
            The bar decided on, with its predictions.
        current : pandas.Series
            The weight of each held symbol.

        Examples
        --------
        ::

            weights = targets.weights(bar, pd.Series({10001: 0.55}))
        """
        t = bar.timestamp
        if not self._inputs.rebalances(t):
            return pd.Series(dtype=float)
        held = xr.DataArray(
            current.to_numpy(dtype=float), dims="symbol", coords={"symbol": current.index.to_numpy()}
        )
        context = self._inputs.context(t, bar.predictions(), held)
        decision = self._inputs.constructor.decide(context)
        return decision.weights.to_series().rename(None).rename_axis(None)
