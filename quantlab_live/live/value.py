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
    shares by symbol, the strategy's own) are valued at quantlab's close
    marks of the snapshot's bar (``quantlab_gate.Run.marks``). The equity
    row is the strategy's cash plus those holdings, at that bar:

    - with an ``initial_capital``, the strategy's cash is its own
      (``strategy_cash``), so the equity is ``initial_capital`` plus the
      strategy's cumulative P&L at quantlab's close;
    - without one, it is the snapshot's (the account's) cash, and the
      equity is the account's.

    A snapshot already valued is left alone, so the step can run again
    after a restart.

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
    >>> import tempfile
    >>> from pathlib import Path
    >>> live_dir = Path(tempfile.mkdtemp())
    >>> _ = (live_dir / "live.toml").write_text('run = "run"\\n')
    >>> value(live_dir, "2024-01-08") is None  # no snapshot yet: nothing to value
    True
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
    config = load_config(live_dir)
    marks = open_run(config.run_dir).marks(snapshot.bar)
    quantity = snapshot.positions["quantity"]
    close = marks.reindex(quantity.index)
    missing = close[close.isna()].index.tolist()
    if missing:
        raise ValueError(f"no close mark at {snapshot.bar.date()} for held symbols {missing}")
    holdings = float((quantity * close).sum())
    if config.initial_capital is None:
        cash = snapshot.cash
    else:
        cash = strategy_cash(live_dir, snapshot.bar, config.initial_capital, holdings)
    records.append_equity(live_dir, snapshot.bar, cash, holdings)
    return snapshot.bar


def strategy_cash(live_dir, bar: pd.Timestamp, initial_capital: float, holdings: float) -> float:
    """Return the strategy's own cash at ``bar``'s close: its cash before, less what its fills cost since.

    The cash before is the latest equity row's before ``bar``; before any
    row, it is ``initial_capital`` less the holdings the first decision
    found at its bar's closes (the strategy's P&L is zero there). Each
    outcome of a decision since then costs its filled shares at their
    average price, plus its fees. With neither a row nor a decision, the
    P&L is zero at ``bar``: the cash is ``initial_capital`` less
    ``holdings``.

    Dividends, delistings and other cash changes in the account are not
    the strategy's cash.

    Examples
    --------
    >>> import tempfile
    >>> live_dir = tempfile.mkdtemp()
    >>> bar, next_bar = pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")
    >>> decision = pd.DataFrame(
    ...     {"weight": [0.5], "close": [10.0], "position_quantity": [0.0], "target_quantity": [500.0],
    ...      "order_quantity": [500.0], "skip": [""]}, index=pd.Index([10001], name="symbol"))
    >>> _ = records.write_decision(live_dir, bar, decision, capital=10_000.0)
    >>> outcomes = pd.DataFrame(
    ...     {"order_quantity": [500.0], "filled_quantity": [500.0], "avg_px": [10.2], "fees": [2.5],
    ...      "status": ["filled"]}, index=pd.Index([10001], name="symbol"))
    >>> _ = records.write_outcomes(live_dir, bar, outcomes)
    >>> strategy_cash(live_dir, next_bar, 10_000.0, holdings=500 * 10.4)
    4897.5
    """
    bar = pd.Timestamp(bar)
    try:
        equity = records.read_equity(live_dir)
        rows = equity[equity.index < bar]
    except FileNotFoundError:
        rows = pd.DataFrame()
    if len(rows):
        since, cash = rows.index[-1], float(rows["cash"].iloc[-1])
    else:
        decided = [b for b in records.decision_bars(live_dir) if b < bar]
        if not decided:
            return float(initial_capital) - float(holdings)
        since = decided[0]
        first = records.read_decision(live_dir, since)
        cash = float(initial_capital) - float((first["position_quantity"] * first["close"]).sum())
    for traded in records.outcome_bars(live_dir):
        if since <= traded < bar:
            outcomes = records.read_outcomes(live_dir, traded)
            cost = (outcomes["filled_quantity"] * outcomes["avg_px"]).fillna(0.0).sum()
            cash -= float(cost) + float(outcomes["fees"].fillna(0.0).sum())
    return cash
