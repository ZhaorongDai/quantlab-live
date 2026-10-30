"""``value``: the latest snapshot valued at quantlab's close, appended to the strategy equity."""

from __future__ import annotations

import pandas as pd

from quantlab_live import records
from quantlab_live.live.config import load_config
from quantlab_live.quantlab_gate import open_run


def value(live_dir, date) -> pd.Timestamp | None:
    """Value the latest snapshot before ``date`` at quantlab's close and append the strategy equity.

    The supervisor runs it before the node starts, so the day's decision
    sizes on an up-to-date strategy capital. The snapshot's positions (raw
    shares by symbol) are valued at quantlab's close marks of the snapshot's
    bar (``quantlab_gate.Run.marks``), and the equity row is the snapshot's
    cash plus those holdings, at that bar. A snapshot already valued is
    left alone, so the step can run again after a restart.

    Parameters
    ----------
    live_dir : path-like
        The live directory, with its ``live.toml``.
    date : datetime-like
        The New York trading day about to trade.

    Returns
    -------
    pandas.Timestamp or None
        The bar valued; ``None`` when there was nothing to value.

    Raises
    ------
    ValueError
        If a held symbol has no close mark.
    QuantlabNotReady
        If quantlab has no bar at the snapshot's bar.

    Examples
    --------
    ::

        value(live_dir, "2026-10-29")  # values the snapshot of 2026-10-28
    """
    snapshot = records.latest_snapshot(live_dir, date)
    if snapshot is None:
        return None
    try:
        equity = records.read_equity(live_dir)
    except FileNotFoundError:
        equity = None
    if equity is not None and len(equity) and equity.index[-1] >= snapshot.bar:
        return None
    marks = open_run(load_config(live_dir).run_dir).marks(snapshot.bar)
    quantity = snapshot.positions["quantity"]
    close = marks.reindex(quantity.index)
    missing = close[close.isna()].index.tolist()
    if missing:
        raise ValueError(f"no close mark at {snapshot.bar.date()} for held symbols {missing}")
    records.append_equity(live_dir, snapshot.bar, snapshot.cash, float((quantity * close).sum()))
    return snapshot.bar
