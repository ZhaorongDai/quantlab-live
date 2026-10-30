"""``ScheduledExecution``: sends a decision's orders at the planned time.

Orders reach it through their ``exec_algorithm_id`` only; the plan lives in
the algorithm's own configuration, an ``ExecutionPlan``. It is stateless:
what it does with an order follows from now, the plan and the exchange's
session (``action``), and the orders themselves are read back from the cache
when a time alert names them. A restart therefore needs no saved state.

The plans:

- ``open_auction``: the opening auction, if the order is ready by the
  opening cut-off (10 minutes before the open). An auction child that the
  venue cancels or expires without our request, or an order ready only after
  the cut-off, falls back to a ``DAY`` market order ``fallback_minutes``
  after the open for the remainder.
- ``close_auction``: the closing auction, if the order is ready by the
  closing cut-off (15 minutes before the close). It has no fallback.
- ``HH:MM`` (New York time): a ``DAY`` order at that time; a late order is
  sent at once while it is still before the deadline (``deadline_minutes``
  after the time, and before the close), and not at all after it.

Sessions, early closes included, come from the ``exchange_calendars`` XNYS
calendar. An order is either a market order or a limit order whose price is
a band of ``band_bps`` around the decision's last close (above it for a buy,
below it for a sell), which the strategy tags the order with.

Live, auction children carry the auction time in force (market or limit on
open or close). NautilusTrader's simulator rejects those and has no
auctions, so in a backtest an auction child is a limit order spawned at the
auction's print, priced at the print for a market plan (so it fills at the
print) or at the band for a limit plan, which expires one second after the
print. ``DAY`` children of a backtest are market orders at the print, or
limit orders that expire at the close. A child whose print is missing (a
halt, or no data) is not sent, so nothing ever fills at a stale price.
NautilusTrader's sandbox (``Environment.SANDBOX``) rejects auction time in
force too: there an auction child is a ``DAY`` order sent at the auction's
time.

A live child waits for its instrument's first quote or trade in the cache
(at most ``PRICE_WAIT_SECONDS``): the risk engine denies a market order it
cannot price.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass

import exchange_calendars
import pandas as pd
from nautilus_trader.common import Environment
from nautilus_trader.model import (
    ClientOrderId,
    ExecAlgorithmId,
    OrderPendingCancel,
    OrderSide,
    Price,
    Quantity,
    TimeInForce,
)
from nautilus_trader.trading import ExecutionAlgorithm, ExecutionAlgorithmConfig

NEW_YORK = "America/New_York"
#: The auction plans; any other plan is a time ``HH:MM``.
AUCTIONS = ("open_auction", "close_auction")
ORDERS = ("market", "limit")
#: The prefix of a child's tag naming what it is (``Action.child``).
CHILD_TAG = "child="
#: The prefix of an order's tag carrying the decision's last close.
MARK_TAG = "close="
#: How long a live child waits for its instrument's first price, in seconds.
PRICE_WAIT_SECONDS = 60
#: How long a backtest auction child lives after its print.
AUCTION_LIFE = pd.Timedelta(seconds=1)
_TIME = re.compile(r"([01]\d|2[0-3]):[0-5]\d")


@dataclass(frozen=True)
class ExecutionPlan:
    """When and how a decision's orders trade.

    Parameters
    ----------
    at : str
        ``"open_auction"``, ``"close_auction"`` or a New York time ``"HH:MM"``.
    order : {"market", "limit"}
        The order type of the children.
    band_bps : float, optional
        A limit order's band around the last close, in basis points; required
        for ``order="limit"``.
    open_cutoff_minutes, close_cutoff_minutes : int
        How long before the open or the close an auction order must be sent.
    fallback_minutes : int
        How long after the open a missed opening auction's fallback trades.
    deadline_minutes : int
        How long after its time a late ``HH:MM`` plan may still send.

    Raises
    ------
    ValueError
        For an unknown ``at`` or ``order``, or a limit plan without a band.

    Examples
    --------
    >>> ExecutionPlan("15:30", order="limit", band_bps=50.0).at
    '15:30'
    >>> ExecutionPlan("noon")
    Traceback (most recent call last):
    ...
    ValueError: at must be 'open_auction', 'close_auction' or 'HH:MM', not 'noon'
    """

    at: str = "open_auction"
    order: str = "market"
    band_bps: float | None = None
    open_cutoff_minutes: int = 10
    close_cutoff_minutes: int = 15
    fallback_minutes: int = 5
    deadline_minutes: int = 30

    def __post_init__(self):
        if self.at not in AUCTIONS and not _TIME.fullmatch(self.at):
            raise ValueError(f"at must be 'open_auction', 'close_auction' or 'HH:MM', not {self.at!r}")
        if self.order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, not {self.order!r}")
        if self.order == "limit" and (self.band_bps is None or self.band_bps < 0):
            raise ValueError("a limit plan needs a band_bps of 0 or more")

    def limit_price(self, side: OrderSide, last_close: float) -> float:
        """Return a limit child's price: the band beyond ``last_close``, against the side.

        Examples
        --------
        >>> round(ExecutionPlan("open_auction", order="limit", band_bps=100).limit_price(OrderSide.BUY, 20.0), 6)
        20.2
        """
        sign = 1.0 if side == OrderSide.BUY else -1.0
        return last_close * (1.0 + sign * self.band_bps / 10_000)


@functools.cache
def _xnys():
    return exchange_calendars.get_calendar("XNYS")


@dataclass(frozen=True)
class Session:
    """One XNYS session's open and close, as UTC timestamps.

    Examples
    --------
    >>> Session.on("2024-07-03").close
    Timestamp('2024-07-03 17:00:00+0000', tz='UTC')
    """

    open: pd.Timestamp
    close: pd.Timestamp

    @classmethod
    def on(cls, day) -> Session:
        """Return the session on the trading day ``day``.

        Raises
        ------
        ValueError
            If ``day`` is not an XNYS session.

        Examples
        --------
        >>> Session.on("2024-01-03").open
        Timestamp('2024-01-03 14:30:00+0000', tz='UTC')
        """
        calendar = _xnys()
        label = pd.Timestamp(day).normalize().tz_localize(None)
        if not calendar.is_session(label):
            raise ValueError(f"{label.date()} is not an XNYS session")
        return cls(calendar.session_open(label), calendar.session_close(label))

    @classmethod
    def of(cls, now) -> Session:
        """Return the day's session at ``now``: the first whose close is after ``now``.

        Examples
        --------
        >>> Session.of(pd.Timestamp("2024-01-05 16:00", tz="America/New_York")).open
        Timestamp('2024-01-08 14:30:00+0000', tz='UTC')
        """
        calendar = _xnys()
        close = calendar.next_close(pd.Timestamp(now).tz_convert("UTC"))
        return cls.on(calendar.minute_to_session(close, direction="previous"))

    def at(self, hh_mm: str) -> pd.Timestamp:
        """Return the time ``hh_mm`` (New York) on the session's day, in UTC.

        Examples
        --------
        >>> Session.on("2024-01-03").at("10:30")
        Timestamp('2024-01-03 15:30:00+0000', tz='UTC')
        """
        day = self.open.tz_convert(NEW_YORK).normalize()
        hours, minutes = map(int, hh_mm.split(":"))
        return (day + pd.Timedelta(hours=hours, minutes=minutes)).tz_convert("UTC")


@dataclass(frozen=True)
class Action:
    """What to do with an order.

    Attributes
    ----------
    child : {"open_auction", "close_auction", "day", "fallback", "none"}
        The child to send: one in an auction, a ``DAY`` order of the plan's
        type, a ``DAY`` market order falling back from the opening auction,
        or nothing.
    at : pandas.Timestamp or None
        When the child trades (an auction's print, or a time), UTC.
    reason : str
        Why nothing is sent.
    """

    child: str
    at: pd.Timestamp | None
    reason: str = ""


def action(now, plan: ExecutionPlan, session: Session, *, auction_missed: bool = False) -> Action:
    """Return what to do with an order at ``now``: a pure function of its inputs.

    Parameters
    ----------
    now : pandas.Timestamp
        The time, tz-aware.
    plan : ExecutionPlan
        The plan.
    session : Session
        The day's session (``Session.of(now)``).
    auction_missed : bool
        Whether the order's auction child ended unfilled without our request.

    Returns
    -------
    Action

    Examples
    --------
    >>> session = Session.on("2024-01-03")
    >>> action(pd.Timestamp("2024-01-03 09:25", tz="America/New_York"), ExecutionPlan("open_auction"), session)
    Action(child='fallback', at=Timestamp('2024-01-03 14:35:00+0000', tz='UTC'), reason='')
    """
    now = pd.Timestamp(now).tz_convert("UTC")
    if plan.at == "open_auction":
        cutoff = session.open - pd.Timedelta(minutes=plan.open_cutoff_minutes)
        if now <= cutoff and not auction_missed:
            return Action("open_auction", session.open)
        at = max(now, session.open + pd.Timedelta(minutes=plan.fallback_minutes))
        if at >= session.close:
            return Action("none", None, "past the close; the opening auction's fallback is not sent")
        return Action("fallback", at)
    if plan.at == "close_auction":
        if auction_missed:
            return Action("none", None, "a closing auction has no fallback")
        if now <= session.close - pd.Timedelta(minutes=plan.close_cutoff_minutes):
            return Action("close_auction", session.close)
        return Action("none", None, "missed the closing auction's cut-off")
    at = session.at(plan.at)
    deadline = min(at + pd.Timedelta(minutes=plan.deadline_minutes), session.close)
    if now >= deadline or at >= session.close:
        return Action("none", None, f"past the {plan.at} plan's deadline")
    return Action("day", max(now, at))


class ScheduledExecution(ExecutionAlgorithm):
    """Send each order as its plan says.

    Parameters
    ----------
    exec_algorithm_id : ExecAlgorithmId
        The id orders name to reach this algorithm.
    plan : ExecutionPlan
        When and how the orders trade.
    environment : nautilus_trader.common.Environment
        The kernel's environment. A Python ``ExecutionAlgorithm`` of
        NautilusTrader 2.0.0rc8 cannot read it, so whoever builds the kernel
        passes it.
    session : Session, optional
        The one session the orders trade in (a live node's day); by default
        the session at each order's time (``Session.of``), as a replay of
        many days needs.
    alert : callable, optional
        ``alert(text)``, called at once when an order falls back from the
        opening auction, a closing-auction order is missed or ended by the
        venue, or a timed plan's order is past its deadline (a live node
        mails it); such events are only logged without it.

    Examples
    --------
    >>> algorithm = ScheduledExecution(ExecAlgorithmId("SCHEDULED"), ExecutionPlan("close_auction"), Environment.BACKTEST)
    >>> algorithm.exec_algorithm_id
    ExecAlgorithmId('SCHEDULED')
    """

    def __new__(cls, *args, **kwargs):
        # The native base takes only a configuration; __init__ passes it.
        return super().__new__(cls)

    def __init__(
        self, exec_algorithm_id: ExecAlgorithmId, plan: ExecutionPlan, environment: Environment,
        session: Session | None = None,
        alert: Callable[[str], None] | None = None,
    ):
        super().__init__(ExecutionAlgorithmConfig(exec_algorithm_id=exec_algorithm_id))
        self._plan = plan
        self._session = session
        self._alert = alert
        self._backtest = environment == Environment.BACKTEST
        # NautilusTrader's sandbox, like its simulator, rejects auction time in force.
        self._sandbox = environment == Environment.SANDBOX

    def on_order(self, order) -> None:
        now = self._now()
        todo = action(now, self._plan, self._session_at(now))
        if todo.child == "none":
            self._warn(f"{order.instrument_id}: {order.side.name} {order.quantity} is not sent: {todo.reason}")
            self.deny_order(order, todo.reason)
            return
        if todo.child == "fallback":
            self._warn(
                f"{order.instrument_id}: {order.side.name} {order.quantity} was ready after the opening auction's "
                f"cut-off; it falls back to a DAY market order at {_ny(todo.at)} New York"
            )
        self._schedule(order, todo, order.quantity)

    def on_order_canceled(self, event) -> None:
        self._on_child_ended(event)

    def on_order_expired(self, event) -> None:
        self._on_child_ended(event)

    def _on_child_ended(self, event) -> None:
        """Fall back from an auction child the venue ended without our request."""
        child = self.cache.order(event.client_order_id)
        kind = _child_kind(child)
        if kind not in AUCTIONS or child.leaves_qty.as_double() == 0:
            return
        if any(isinstance(e, OrderPendingCancel) for e in child.events()):
            return  # we asked for it
        now = self._now()
        todo = action(now, self._plan, self._session_at(now), auction_missed=True)
        primary = self.cache.order(child.exec_spawn_id)
        what = f"{child.instrument_id}: the venue ended the {kind} order {child.client_order_id} with {child.leaves_qty} unfilled"
        if todo.child == "none":
            self._warn(f"{what} ({todo.reason}); it is venue_cancelled")
            return
        self._warn(f"{what}; the remainder falls back to a DAY market order at {_ny(todo.at)} New York")
        self._schedule(primary, todo, child.leaves_qty)

    def _schedule(self, primary, todo: Action, quantity: Quantity) -> None:
        if not self._backtest and not self._sandbox and todo.child in AUCTIONS:
            self._send_live(primary, todo.child, quantity, 0)
            return
        name = f"{todo.child}:{primary.client_order_id.value}:{quantity}"
        self.clock.set_time_alert_ns(name, todo.at.value, self._on_alert, allow_past=True)

    def _send_live(self, primary, kind: str, quantity: Quantity, waited: int) -> None:
        """Submit a live child once its instrument has a price, waiting at most ``PRICE_WAIT_SECONDS``.

        NautilusTrader's risk engine denies a market order on an instrument
        without a quote or trade in the cache, which a node just started, or
        an instrument just loaded, may not have yet.
        """
        instrument_id = primary.instrument_id
        priced = self.cache.quote(instrument_id) is not None or self.cache.trade(instrument_id) is not None
        if priced or waited >= PRICE_WAIT_SECONDS:
            if not priced:
                self.log.warning(f"{instrument_id} has no price after {waited} s; sending {kind} anyway")
            self._submit(primary, kind, quantity, last_print=None)
            return
        name = f"{kind}:{primary.client_order_id.value}:{quantity}:{waited + 1}"
        self.clock.set_time_alert_ns(name, self.clock.timestamp_ns() + 1_000_000_000, self._on_alert)

    def _on_alert(self, event) -> None:
        kind, primary_id, quantity, *waited = event.name.split(":")
        primary = self.cache.order(ClientOrderId(primary_id))
        if not self._backtest:
            self._send_live(primary, kind, Quantity.from_str(quantity), int(waited[0]) if waited else 0)
            return
        last_print = None
        if self._backtest:
            last = self.cache.trade(primary.instrument_id)
            if last is None or last.ts_event != event.ts_event:
                reason = f"{primary.instrument_id} has no print at {pd.Timestamp(event.ts_event, tz='UTC')}"
                if kind == "fallback":
                    self.log.warning(f"{reason}; the fallback of {primary_id} is not sent")
                else:
                    self.deny_order(primary, reason)
                return
            last_print = last.price
        self._submit(primary, kind, Quantity.from_str(quantity), last_print)

    def _submit(self, primary, kind: str, quantity: Quantity, last_print: Price | None) -> None:
        """Spawn and submit a ``kind`` child; ``last_print`` is a backtest's print."""
        tags = [*(primary.tags or []), CHILD_TAG + kind]
        reduce = kind != "fallback"  # a fallback takes the remainder of a child
        limit = self._plan.order == "limit" and kind != "fallback"
        price = self._limit_price(primary) if limit else None
        if kind in AUCTIONS and not self._sandbox:
            if self._backtest:
                expire = pd.Timestamp(self.clock.timestamp_ns(), tz="UTC") + AUCTION_LIFE
                child = self.spawn_limit(
                    primary, quantity, price or last_print, TimeInForce.GTD,
                    expire_time=expire.to_pydatetime(), tags=tags, reduce_primary=reduce,
                )
            else:
                tif = TimeInForce.AT_THE_OPEN if kind == "open_auction" else TimeInForce.AT_THE_CLOSE
                child = (
                    self.spawn_limit(primary, quantity, price, tif, tags=tags, reduce_primary=reduce)
                    if limit
                    else self.spawn_market(primary, quantity, tif, tags=tags, reduce_primary=reduce)
                )
        elif limit:
            if self._backtest:
                # The simulator never expires a DAY order; this one expires at the close.
                close = self._session_at(self._now()).close
                child = self.spawn_limit(
                    primary, quantity, price, TimeInForce.GTD, expire_time=close.to_pydatetime(),
                    tags=tags, reduce_primary=reduce,
                )
            else:
                child = self.spawn_limit(primary, quantity, price, TimeInForce.DAY, tags=tags, reduce_primary=reduce)
        else:
            child = self.spawn_market(primary, quantity, TimeInForce.DAY, tags=tags, reduce_primary=reduce)
        self.submit_order(child)

    def _limit_price(self, primary) -> Price:
        (mark,) = (t for t in primary.tags or [] if t.startswith(MARK_TAG))
        precision = self.cache.instrument(primary.instrument_id).price_precision
        value = self._plan.limit_price(primary.side, float(mark.removeprefix(MARK_TAG)))
        return Price(round(value, precision), precision)

    def _now(self) -> pd.Timestamp:
        return pd.Timestamp(self.clock.timestamp_ns(), tz="UTC")

    def _warn(self, text: str) -> None:
        self.log.warning(text)
        if self._alert is not None:
            self._alert(text)

    def _session_at(self, now: pd.Timestamp) -> Session:
        return self._session if self._session is not None else Session.of(now)


def _ny(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).tz_convert(NEW_YORK).strftime("%H:%M")


def _child_kind(order) -> str | None:
    """The kind of child an order is (its ``child=`` tag); ``None`` for a primary."""
    tags = [t for t in order.tags or [] if t.startswith(CHILD_TAG)]
    return tags[0].removeprefix(CHILD_TAG) if tags else None
