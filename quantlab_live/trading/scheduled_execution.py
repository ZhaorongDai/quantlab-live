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

import dataclasses
import enum
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


class Child(enum.StrEnum):
    """The kinds of child an order spawns; an order's ``child=`` tag names its kind.

    Examples
    --------
    >>> Child("fallback"), Child.OPEN_AUCTION in AUCTION_CHILDREN
    ('fallback', True)
    """

    OPEN_AUCTION = "open_auction"
    CLOSE_AUCTION = "close_auction"
    DAY = "day"
    FALLBACK = "fallback"
    NONE = "none"

    __repr__ = str.__repr__


#: The children that trade in an auction.
AUCTION_CHILDREN = (Child.OPEN_AUCTION, Child.CLOSE_AUCTION)


@dataclass(frozen=True)
class Action:
    """What to do with an order.

    Attributes
    ----------
    child : Child
        The child to send: one in an auction, a ``DAY`` order of the plan's
        type, a ``DAY`` market order falling back from the opening auction,
        or nothing (``Child.NONE``).
    at : pandas.Timestamp or None
        When the child trades (an auction's print, or a time), UTC.
    reason : str
        Why nothing is sent.
    """

    child: Child
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
            return Action(Child.OPEN_AUCTION, session.open)
        at = max(now, session.open + pd.Timedelta(minutes=plan.fallback_minutes))
        if at >= session.close:
            return Action(Child.NONE, None, "past the close; the opening auction's fallback is not sent")
        return Action(Child.FALLBACK, at)
    if plan.at == "close_auction":
        if auction_missed:
            return Action(Child.NONE, None, "a closing auction has no fallback")
        if now <= session.close - pd.Timedelta(minutes=plan.close_cutoff_minutes):
            return Action(Child.CLOSE_AUCTION, session.close)
        return Action(Child.NONE, None, "missed the closing auction's cut-off")
    at = session.at(plan.at)
    deadline = min(at + pd.Timedelta(minutes=plan.deadline_minutes), session.close)
    if now >= deadline or at >= session.close:
        return Action(Child.NONE, None, f"past the {plan.at} plan's deadline")
    return Action(Child.DAY, max(now, at))


@dataclass(frozen=True)
class _Venue:
    """How the children of one environment are sent.

    ``auction_tif``: an auction child carries the auction time in force and
    goes to the venue at once (live). ``simulated``: every child waits for
    its print and is priced and expired as NautilusTrader's simulator needs
    (backtest). The sandbox is neither: it rejects auction time in force, so
    an auction child there is a ``DAY`` order sent at the auction's time.
    """

    auction_tif: bool
    simulated: bool


_VENUES = {
    Environment.LIVE: _Venue(auction_tif=True, simulated=False),
    Environment.SANDBOX: _Venue(auction_tif=False, simulated=False),
    Environment.BACKTEST: _Venue(auction_tif=False, simulated=True),
}


@dataclass(frozen=True)
class _Send:
    """A child to send for a primary, as a time alert names it.

    ``waited`` counts the seconds a live child has waited for a first price.
    """

    child: Child
    primary: str
    quantity: str
    waited: int = 0

    @property
    def name(self) -> str:
        return f"{self.child}:{self.primary}:{self.quantity}:{self.waited}"

    @classmethod
    def named(cls, name: str) -> _Send:
        child, primary, quantity, waited = name.split(":")
        return cls(Child(child), primary, quantity, int(waited))


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
        self._venue = _VENUES[environment]

    def on_order(self, order) -> None:
        now = self._now()
        todo = action(now, self._plan, self._session_at(now))
        if todo.child == Child.NONE:
            self._warn(f"{order.instrument_id}: {order.side.name} {order.quantity} is not sent: {todo.reason}")
            self.deny_order(order, todo.reason)
            return
        if todo.child == Child.FALLBACK:
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
        if kind not in AUCTION_CHILDREN or child.leaves_qty.as_double() == 0:
            return
        if any(isinstance(e, OrderPendingCancel) for e in child.events()):
            return  # we asked for it
        now = self._now()
        todo = action(now, self._plan, self._session_at(now), auction_missed=True)
        primary = self.cache.order(child.exec_spawn_id)
        what = f"{child.instrument_id}: the venue ended the {kind} order {child.client_order_id} with {child.leaves_qty} unfilled"
        if todo.child == Child.NONE:
            self._warn(f"{what} ({todo.reason}); it is venue_cancelled")
            return
        self._warn(f"{what}; the remainder falls back to a DAY market order at {_ny(todo.at)} New York")
        self._schedule(primary, todo, child.leaves_qty)

    def _schedule(self, primary, todo: Action, quantity: Quantity) -> None:
        send = _Send(todo.child, primary.client_order_id.value, str(quantity))
        if self._venue.auction_tif and todo.child in AUCTION_CHILDREN:
            self._send_live(send)
            return
        self.clock.set_time_alert_ns(send.name, todo.at.value, self._on_alert, allow_past=True)

    def _send_live(self, send: _Send) -> None:
        """Submit a live child once its instrument has a price, waiting at most ``PRICE_WAIT_SECONDS``.

        NautilusTrader's risk engine denies a market order on an instrument
        without a quote or trade in the cache, which a node just started, or
        an instrument just loaded, may not have yet.
        """
        primary = self.cache.order(ClientOrderId(send.primary))
        instrument_id = primary.instrument_id
        priced = self.cache.quote(instrument_id) is not None or self.cache.trade(instrument_id) is not None
        if priced or send.waited >= PRICE_WAIT_SECONDS:
            if not priced:
                self.log.warning(f"{instrument_id} has no price after {send.waited} s; sending {send.child} anyway")
            self._submit(primary, send.child, Quantity.from_str(send.quantity), last_print=None)
            return
        later = dataclasses.replace(send, waited=send.waited + 1)
        self.clock.set_time_alert_ns(later.name, self.clock.timestamp_ns() + 1_000_000_000, self._on_alert)

    def _on_alert(self, event) -> None:
        send = _Send.named(event.name)
        if not self._venue.simulated:
            self._send_live(send)
            return
        primary = self.cache.order(ClientOrderId(send.primary))
        last = self.cache.trade(primary.instrument_id)
        if last is None or last.ts_event != event.ts_event:
            reason = f"{primary.instrument_id} has no print at {pd.Timestamp(event.ts_event, tz='UTC')}"
            if send.child == Child.FALLBACK:
                self.log.warning(f"{reason}; the fallback of {send.primary} is not sent")
            else:
                self.deny_order(primary, reason)
            return
        self._submit(primary, send.child, Quantity.from_str(send.quantity), last.price)

    def _submit(self, primary, kind: Child, quantity: Quantity, last_print: Price | None) -> None:
        """Spawn and submit a ``kind`` child; ``last_print`` is a backtest's print."""
        tags = [*(primary.tags or []), CHILD_TAG + kind]
        reduce = kind != Child.FALLBACK  # a fallback takes the remainder of a child
        limit = self._plan.order == "limit" and kind != Child.FALLBACK
        price = self._limit_price(primary) if limit else None
        auction = kind in AUCTION_CHILDREN
        if auction and self._venue.simulated:
            expire = self._now() + AUCTION_LIFE
            child = self.spawn_limit(
                primary, quantity, price or last_print, TimeInForce.GTD,
                expire_time=expire.to_pydatetime(), tags=tags, reduce_primary=reduce,
            )
        elif auction and self._venue.auction_tif:
            tif = TimeInForce.AT_THE_OPEN if kind == Child.OPEN_AUCTION else TimeInForce.AT_THE_CLOSE
            child = (
                self.spawn_limit(primary, quantity, price, tif, tags=tags, reduce_primary=reduce)
                if limit
                else self.spawn_market(primary, quantity, tif, tags=tags, reduce_primary=reduce)
            )
        elif limit and self._venue.simulated:
            # The simulator never expires a DAY order; this one expires at the close.
            close = self._session_at(self._now()).close
            child = self.spawn_limit(
                primary, quantity, price, TimeInForce.GTD, expire_time=close.to_pydatetime(),
                tags=tags, reduce_primary=reduce,
            )
        elif limit:
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


def _child_kind(order) -> Child | None:
    """The kind of child an order is (its ``child=`` tag); ``None`` for a primary."""
    tags = [t for t in order.tags or [] if t.startswith(CHILD_TAG)]
    return Child(tags[0].removeprefix(CHILD_TAG)) if tags else None
