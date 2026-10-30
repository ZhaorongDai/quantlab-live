"""Target sources: what the strategy holds after a decision bar.

A target source turns a decision bar into target weights by symbol. The
strategy sizes them; it does not know where they come from.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr


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
    >>> RebalanceTableTargets(table).weights(pd.Timestamp("2024-01-02")).to_dict()
    {10001: 0.6, 10002: nan}
    """

    def __init__(self, table: xr.DataArray):
        self._table = table.transpose("timestamp", "symbol").to_pandas()

    def weights(self, bar: pd.Timestamp) -> pd.Series:
        """Return the target weight of every symbol at ``bar``, NaN where kept.

        A bar the table does not have keeps every holding.

        Examples
        --------
        >>> table = xr.DataArray(
        ...     [[0.6]], dims=("timestamp", "symbol"),
        ...     coords={"timestamp": [pd.Timestamp("2024-01-02")], "symbol": [10001]},
        ... )
        >>> RebalanceTableTargets(table).weights(pd.Timestamp("2024-01-03")).to_dict()
        {10001: nan}
        """
        if bar not in self._table.index:
            return pd.Series(float("nan"), index=self._table.columns)
        return self._table.loc[bar].rename(None).rename_axis(None)
