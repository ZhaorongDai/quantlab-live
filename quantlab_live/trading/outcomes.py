"""Order outcomes: what each ordered symbol of a decision came to.

``order_outcomes`` is a pure function of a bar's decision, its orders, their
fills and, optionally, the positions once the orders are done. It gives every
symbol the decision orders exactly one outcome:

- ``filled``: the order quantity filled;
- ``partially_filled``: part of it filled, and the rest ended;
- ``cancelled``: we cancelled it, and nothing filled;
- ``venue_cancelled``: the venue cancelled or expired it without our request
  (a missed closing auction), and nothing filled;
- ``venue_cancelled_fallback``: the venue cancelled or expired an auction
  order without our request, and a fallback was sent for the remainder (its
  fills are in the filled quantity);
- ``rejected``: the venue rejected it, and nothing filled;
- ``unmapped`` and ``foreign``: skipped before submission (no instrument, or
  a position the strategy does not own on its instrument);
- ``not_sent``: nothing reached the venue (an order kept until the end, or
  denied);
- ``unverified``: its state is not known to be final: an order still working
  or in flight, more filled than ordered, or a position its fills do not
  explain.

Fill prices come only from fills, never from an order or a position.

``order_frames`` reads the frames ``order_outcomes`` takes from NautilusTrader
orders.
"""

from __future__ import annotations

import pandas as pd
from nautilus_trader.model import OrderFilled, OrderPendingCancel, OrderSide

from quantlab_live.trading.scheduled_execution import AUCTIONS, CHILD_TAG

#: The prefix of the order tag naming an order's decision bar.
BAR_TAG = "bar="

FILLED = "filled"
PARTIALLY_FILLED = "partially_filled"
CANCELLED = "cancelled"
VENUE_CANCELLED = "venue_cancelled"
VENUE_CANCELLED_FALLBACK = "venue_cancelled_fallback"
REJECTED = "rejected"
UNMAPPED = "unmapped"
FOREIGN = "foreign"
NOT_SENT = "not_sent"
UNVERIFIED = "unverified"
#: The skip reasons a decision records, which are also outcomes.
SKIPS = (UNMAPPED, FOREIGN)

#: The columns of an orders frame, indexed by client order id: the order's
#: symbol, its primary's client order id (its own for a primary), its child
#: kind (``None`` for a primary), its NautilusTrader status name, its
#: unsigned leaves quantity and whether we asked to cancel it.
ORDER_COLUMNS = ["symbol", "primary", "child", "status", "leaves", "cancel_requested"]
#: The columns of a fills frame: the order filled, its symbol, the signed
#: quantity and the price. An optional ``fees`` column holds each fill's
#: commission in the account's currency.
FILL_COLUMNS = ["client_order_id", "symbol", "quantity", "price"]
#: A child in one of these states may still change.
_UNSETTLED = {
    "SUBMITTED", "ACCEPTED", "TRIGGERED", "PENDING_UPDATE", "PENDING_CANCEL",
    "PARTIALLY_FILLED", "EMULATED", "RELEASED", "VOIDED",
}
_ENDED = {"CANCELED", "EXPIRED"}


def order_outcomes(
    decision: pd.DataFrame, orders: pd.DataFrame, fills: pd.DataFrame, positions: pd.Series | None
) -> pd.DataFrame:
    """Return the outcome of every symbol a bar's decision orders.

    Parameters
    ----------
    decision : pandas.DataFrame
        The bar's decision by ``symbol`` (``records.write_decision``): its
        signed ``order_quantity``, ``position_quantity`` and ``skip``.
    orders : pandas.DataFrame
        The bar's orders, primaries and children, by client order id, with
        the ``ORDER_COLUMNS``.
    fills : pandas.DataFrame
        Their fills, with the ``FILL_COLUMNS`` and optionally ``fees``.
    positions : pandas.Series or None
        The signed position by symbol once the orders are done; when given,
        a symbol whose position is not its decision's position plus its
        fills is unverified.

    Returns
    -------
    pandas.DataFrame
        By ``symbol``: the signed ``order_quantity`` and ``filled_quantity``,
        the fills' ``avg_px`` (NaN without a fill), their summed ``fees`` (0
        when ``fills`` has no ``fees``) and the ``status``.

    Examples
    --------
    >>> decision = pd.DataFrame(
    ...     {"position_quantity": [0.0, 0.0], "order_quantity": [100.0, 50.0], "skip": ["", "unmapped"]},
    ...     index=pd.Index([10001, 10002], name="symbol"),
    ... )
    >>> orders = pd.DataFrame(
    ...     {"symbol": [10001, 10001], "primary": ["P", "P"], "child": [None, "open_auction"],
    ...      "status": ["INITIALIZED", "FILLED"], "leaves": [0.0, 0.0], "cancel_requested": [False, False]},
    ...     index=["P", "C"],
    ... )
    >>> fills = pd.DataFrame({"client_order_id": ["C"], "symbol": [10001], "quantity": [100.0], "price": [10.5]})
    >>> outcomes = order_outcomes(decision, orders, fills, positions=None)
    >>> outcomes["status"].to_dict(), float(outcomes.loc[10001, "avg_px"])
    ({10001: 'filled', 10002: 'unmapped'}, 10.5)
    """
    ordered = decision[decision["order_quantity"] != 0]
    rows = {}
    for symbol, row in ordered.iterrows():
        mine = orders[orders["symbol"] == symbol]
        done = fills[fills["symbol"] == symbol]
        filled = float(done["quantity"].sum())
        traded = float(done["quantity"].abs().sum())
        avg_px = float((done["quantity"].abs() * done["price"]).sum() / traded) if traded else float("nan")
        fees = float(done["fees"].sum()) if "fees" in done else 0.0
        position = None if positions is None else float(positions.get(symbol, 0.0))
        rows[symbol] = {
            "order_quantity": float(row["order_quantity"]),
            "filled_quantity": filled,
            "avg_px": avg_px,
            "fees": fees,
            "status": _status(row, mine, filled, position),
        }
    frame = pd.DataFrame.from_dict(
        rows, orient="index", columns=["order_quantity", "filled_quantity", "avg_px", "fees", "status"]
    )
    return frame.rename_axis("symbol")


def redecided(outcomes: pd.DataFrame, decision: pd.DataFrame, orders: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """Return ``outcomes`` with each ``unverified`` row decided again from the bar's orders as they are now.

    A later node, once its start-up reconciliation has brought the orders
    up to date, decides again what was not known at the bar's stop. The
    positions have moved on since, so they are not compared; an order still
    working stays unverified, and the other rows are kept as they are.

    Parameters
    ----------
    outcomes : pandas.DataFrame
        The bar's outcomes (``order_outcomes``).
    decision, orders, fills : pandas.DataFrame
        As ``order_outcomes`` takes them: the bar's decision, all its orders
        and their fills.

    Examples
    --------
    >>> decision = pd.DataFrame(
    ...     {"position_quantity": [0.0], "order_quantity": [100.0], "skip": [""]}, index=pd.Index([10001], name="symbol"),
    ... )
    >>> before = pd.DataFrame(
    ...     {"order_quantity": [100.0], "filled_quantity": [0.0], "avg_px": [float("nan")], "fees": [0.0],
    ...      "status": ["unverified"]}, index=pd.Index([10001], name="symbol"),
    ... )
    >>> orders = pd.DataFrame(
    ...     {"symbol": [10001, 10001], "primary": ["P", "P"], "child": [None, "day"],
    ...      "status": ["INITIALIZED", "FILLED"], "leaves": [0.0, 0.0], "cancel_requested": [False, False]},
    ...     index=["P", "C"],
    ... )
    >>> fills = pd.DataFrame({"client_order_id": ["C"], "symbol": [10001], "quantity": [100.0], "price": [10.5]})
    >>> redecided(before, decision, orders, fills)[["filled_quantity", "status"]].to_dict("records")
    [{'filled_quantity': 100.0, 'status': 'filled'}]
    """
    unverified = outcomes.index[outcomes["status"] == UNVERIFIED]
    if unverified.empty:
        return outcomes
    again = order_outcomes(decision.loc[decision.index.intersection(unverified)], orders, fills, None)
    result = outcomes.copy()
    result.loc[again.index, again.columns] = again
    return result


def _status(row: pd.Series, orders: pd.DataFrame, filled: float, position: float | None) -> str:
    skip = row.get("skip") or ""
    if skip:
        return skip
    quantity = float(row["order_quantity"])
    children = orders[orders["child"].notna()]
    sent = children[children["status"] != "DENIED"]
    if sent["status"].isin(_UNSETTLED).any() or abs(filled) > abs(quantity):
        return UNVERIFIED
    if position is not None and position != float(row["position_quantity"]) + filled:
        return UNVERIFIED
    ended = sent[sent["status"].isin(_ENDED) & (sent["leaves"] > 0)]
    venue_ended = ended[~ended["cancel_requested"].astype(bool)]
    if venue_ended["child"].isin(AUCTIONS).any() and (sent["child"] == "fallback").any():
        return VENUE_CANCELLED_FALLBACK
    if filled == quantity:
        return FILLED
    if filled:
        return PARTIALLY_FILLED
    if sent.empty:
        return NOT_SENT
    if (sent["status"] == "REJECTED").any():
        return REJECTED
    if len(venue_ended):
        return VENUE_CANCELLED
    if len(ended):
        return CANCELLED
    return UNVERIFIED


def order_bar(tags) -> pd.Timestamp | None:
    """Return the decision bar an order's ``bar=`` tag names; ``None`` without one.

    Examples
    --------
    >>> order_bar(["bar=2024-01-02", "close=10.0"]), order_bar([])
    (Timestamp('2024-01-02 00:00:00'), None)
    """
    named = [t for t in tags or [] if t.startswith(BAR_TAG)]
    return pd.Timestamp(named[0].removeprefix(BAR_TAG)) if named else None


def order_frames(orders, symbol_of) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the orders and fills frames of NautilusTrader ``orders``.

    Parameters
    ----------
    orders : iterable of nautilus_trader orders
        Primaries and the children an execution algorithm spawned from them
        (a child names its primary by ``exec_spawn_id``).
    symbol_of : callable
        The symbol of an instrument id (``SecurityMap.to_symbol``).

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        The orders (``ORDER_COLUMNS`` by client order id, plus their ``tags``
        and ``ts_init``)
        and their fills (``FILL_COLUMNS`` plus ``fees`` and ``ts_event``), in
        the orders' units.

    Examples
    --------
    ::

        frame, fills = order_frames(cache.orders(strategy_id=strategy_id), security_map.to_symbol)
    """
    rows, fill_rows = {}, []
    for order in orders:
        own = order.client_order_id.value
        spawn = order.exec_spawn_id
        kinds = [t.removeprefix(CHILD_TAG) for t in order.tags or [] if t.startswith(CHILD_TAG)]
        symbol = symbol_of(order.instrument_id)
        events = order.events()
        rows[own] = {
            "symbol": symbol,
            "primary": own if spawn is None else spawn.value,
            "child": kinds[0] if kinds else None,
            "status": order.status.name,
            "leaves": float(order.leaves_qty),
            "cancel_requested": any(isinstance(e, OrderPendingCancel) for e in events),
            "tags": list(order.tags or []),
            "ts_init": order.ts_init,
        }
        sign = 1.0 if order.side == OrderSide.BUY else -1.0
        for event in events:
            if isinstance(event, OrderFilled):
                fill_rows.append(
                    {
                        "client_order_id": own,
                        "symbol": symbol,
                        "quantity": sign * float(event.last_qty),
                        "price": float(event.last_px),
                        "fees": float(event.commission),
                        "ts_event": event.ts_event,
                    }
                )
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=[*ORDER_COLUMNS, "tags", "ts_init"])
    return frame, pd.DataFrame(fill_rows, columns=[*FILL_COLUMNS, "fees", "ts_event"])
