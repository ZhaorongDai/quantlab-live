"""``replay``: a quantlab run traded on NautilusTrader's ``BacktestEngine``.

The replay runs the live strategy and execution algorithm against one
synthetic ``SIM`` venue with one MARGIN account. Each symbol is the
instrument ``<symbol>.SIM`` (the identity security map), the prints the
execution plan trades at are fed as ``TradeTick`` data and the run's
decision bars are added to the engine as ``DecisionBar`` data, so the
strategy receives them exactly as it does live.

A daily quantlab run has an open and a close per bar, so the prints are:

- the opening print at the session's open, at the open;
- the closing print at the session's close (early closes included), at the
  close;
- a print at any other time of the session (an ``HH:MM`` plan, the opening
  auction's fallback), on the straight line from the open to the close by
  the time elapsed in the session;
- after an auction print, the same price again one second later: the
  simulator expires an order only on its instrument's next trade, and an
  unfilled auction child must expire there, before anything else trades.

Only the prints the plan can trade at are fed.

Costs and corporate actions:

- the broker's fee model charges every fill, and an optional ``FillModel``
  (``ProportionalSlippage``) moves fill prices;
- one ``SplitTable`` holds the split factors. NautilusTrader cannot change a
  position's quantity from Python, so the engine trades in split-adjusted
  units (one unit is the raw shares of one share at the replay's first
  bar): the prints and marks it is fed are split adjusted, the fee model
  converts units to raw shares and the ``DividendModule`` pays dividends
  per unit;
- a delisting is an ``InstrumentClose`` (``CONTRACT_EXPIRED``) at the
  security's last valuation, just before the decision on its last bar,
  which closes the position there without a fee.

Replay converts back to raw units when it writes its records. NautilusTrader's
own reports under ``reports/`` stay in the engine's units.
"""

from __future__ import annotations

import contextlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig, SimulationModule
from nautilus_trader.common import Environment, LoggerConfig, LogLevel
from nautilus_trader.execution import FillModel
from nautilus_trader.model import (
    AccountType,
    AggressorSide,
    BookOrder,
    BookType,
    ClientId,
    Currency,
    CustomData,
    DataType,
    Equity,
    ExecAlgorithmId,
    InstrumentClose,
    InstrumentCloseType,
    InstrumentId,
    Money,
    OmsType,
    OrderBook,
    OrderSide,
    Price,
    Quantity,
    Symbol,
    TradeId,
    TradeTick,
    Venue,
)

from quantlab_live import brokers, records
from quantlab_live.quantlab_gate import PRICE_PRECISION, open_run
from quantlab_live.trading.decision_bar import NEW_YORK, DecisionBar, decision_ns
from quantlab_live.trading.scheduled_execution import (
    AUCTION_LIFE,
    ExecutionPlan,
    ScheduledExecution,
    Session,
)
from quantlab_live.trading.strategy import BAR_TAG, QuantlabStrategy
from quantlab_live.trading.targets import ConstructorTargets, RebalanceTableTargets

SIM = Venue("SIM")
USD = Currency.from_str("USD")
#: A print's size: replay takes the whole order at the print.
PRINT_SIZE = Quantity(10**9, 0)
#: The NautilusTrader reports a replay exports, by file name.
REPORTS = ("orders", "order_fills", "fills", "positions", "account")


class IdentityMap:
    """The replay's security map: symbol ``s`` is the instrument ``s.SIM``.

    Examples
    --------
    >>> IdentityMap().to_id(10001), IdentityMap().to_symbol(InstrumentId.from_str("10001.SIM"))
    (InstrumentId('10001.SIM'), 10001)
    """

    def to_id(self, symbol: int) -> InstrumentId:
        """Return the instrument of ``symbol``: ``<symbol>.SIM``."""
        return InstrumentId.from_str(f"{int(symbol)}.{SIM}")

    def to_symbol(self, instrument_id: InstrumentId) -> int | None:
        """Return the symbol of a ``SIM`` instrument; ``None`` for another venue."""
        if instrument_id.venue != SIM:
            return None
        return int(instrument_id.symbol.value)


class SplitTable:
    """The raw shares in one engine unit of each symbol, by bar.

    A unit is the raw shares of one share at the replay's first bar, so the
    table is the cumulative product of the split factors after it. An
    ex-date's factor applies from its session's open.

    Parameters
    ----------
    split_factor : pandas.DataFrame
        New shares per old share on each ex-date (1 where none), by bar and
        symbol.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-02", periods=3)
    >>> splits = SplitTable(pd.DataFrame({7: [3.0, 2.0, 0.25]}, index=bars))
    >>> splits.ratio(bars[2]).to_dict()
    {7: 0.5}
    >>> monday_open = pd.Timestamp("2024-01-03 09:30", tz=NEW_YORK).value
    >>> splits.shares_per_unit(InstrumentId.from_str("7.SIM"), monday_open)
    2.0
    """

    def __init__(self, split_factor: pd.DataFrame):
        factor = split_factor.astype(float).copy()
        factor.iloc[0] = 1.0  # a split before the first bar is in its raw prices
        self._ratio = factor.cumprod()
        self._days = pd.DatetimeIndex(self._ratio.index).normalize()

    def to_units(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return raw ``prices`` (by bar and symbol) per engine unit."""
        return prices * self._ratio.reindex(index=prices.index, columns=prices.columns)

    def ratio(self, bar: pd.Timestamp) -> pd.Series:
        """Return the raw shares per unit of every symbol at ``bar``."""
        return self._ratio.loc[pd.Timestamp(bar)]

    def at(self, ts_ns: int) -> pd.Series:
        """Return the raw shares per unit of every symbol in the session of ``ts_ns``.

        Before an ex-date's session the previous bar's ratio holds.
        """
        day = pd.Timestamp(ts_ns, tz="UTC").tz_convert(NEW_YORK).tz_localize(None).normalize()
        return self._ratio.iloc[max(int(self._days.searchsorted(day, "right")) - 1, 0)]

    def shares_per_unit(self, instrument_id: InstrumentId, ts_ns: int) -> float:
        """Return the raw shares in one unit of a ``SIM`` instrument at ``ts_ns``."""
        return float(self.at(ts_ns)[int(instrument_id.symbol.value)])


class DividendModule(SimulationModule):
    """Credit cash dividends to the account before each ex-date's open.

    A dividend is paid on the position held at the prior bar's close: the
    module runs after the decision on that bar, before the next session's
    orders reach the venue. A short position pays it.

    Parameters
    ----------
    payments : list of (int, InstrumentId, float)
        When to pay (UNIX nanoseconds), the instrument and the dividend per
        engine unit, in time order.

    Examples
    --------
    ::

        engine.add_venue(..., modules=[DividendModule(payments)])
    """

    def __new__(cls, *args, **kwargs):
        # The native base takes no arguments; __init__ keeps them.
        return super().__new__(cls)

    def __init__(self, payments: list[tuple[int, InstrumentId, float]]):
        super().__init__()
        self._payments = list(payments)
        #: The payments made: (UNIX nanoseconds, instrument, units held, amount).
        self.paid: list[tuple[int, InstrumentId, float, float]] = []

    def process(self, ts_now: int, context) -> list[Money] | None:
        due = [p for p in self._payments if p[0] <= ts_now]
        if not due:
            return None
        self._payments = self._payments[len(due) :]
        held = {p.instrument_id: float(p.signed_qty) for p in context.positions}
        credits = []
        for _, instrument_id, per_unit in due:
            units = held.get(instrument_id, 0.0)
            amount = round(units * per_unit, 2)
            if amount:
                credits.append(Money(amount, USD))
                self.paid.append((ts_now, instrument_id, units, amount))
        return credits

    def acknowledge(self, outcomes) -> None:
        for outcome in outcomes:
            if not outcome.applied:
                raise RuntimeError(f"a dividend was not credited: {outcome.error}")


class ProportionalSlippage(FillModel):
    """Fill every order a fixed fraction worse than the market.

    A buy fills at the best ask times ``1 + rate``, a sell at the best bid
    times ``1 - rate``, at the instrument's price precision and in full.

    Parameters
    ----------
    rate : float
        The slippage, a fraction of the price (0.001 is 10 basis points).

    Examples
    --------
    ::

        replay(run_dir, fill_model=ProportionalSlippage(0.001))
    """

    def __new__(cls, *args, **kwargs):
        # The native base takes no arguments; __init__ keeps them.
        return super().__new__(cls)

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def is_limit_filled(self) -> bool:
        return True

    def is_slipped(self) -> bool:
        return False

    def get_orderbook_for_fill_simulation(self, instrument, order, best_bid=None, best_ask=None):
        if best_bid is None or best_ask is None:
            return None
        book = OrderBook(instrument.id, BookType.L2_MBP)
        size = Quantity(10**9, instrument.size_precision)
        bid = instrument.make_price(float(best_bid) * (1 - self.rate))
        ask = instrument.make_price(float(best_ask) * (1 + self.rate))
        book.add(BookOrder(OrderSide.BUY, bid, size, 1), 0, 0, 0)
        book.add(BookOrder(OrderSide.SELL, ask, size, 2), 0, 0, 0)
        return book


def replay(
    run_dir,
    out_dir=None,
    *,
    loop: str = "closed",
    plan: str | ExecutionPlan = "open_auction",
    start=None,
    end=None,
    broker: str = "ibkr",
    fill_model: FillModel | None = None,
) -> Path:
    """Replay the quantlab run at ``run_dir`` and write the replay directory.

    Parameters
    ----------
    run_dir : path-like
        The quantlab run.
    out_dir : path-like, optional
        The replay directory; defaults to ``replay-<loop>`` inside ``run_dir``.
    loop : {"closed", "open"}
        ``"closed"`` decides every rebalance bar with quantlab's portfolio
        construction on the holdings the replay has, so fills feed back
        into the next decision; ``"open"`` executes the run's finished
        rebalance table.
    plan : str or ExecutionPlan
        When and how the orders trade: ``"open_auction"``,
        ``"close_auction"``, a New York time ``"HH:MM"`` (market orders), or
        a full ``ExecutionPlan``.
    start, end : datetime-like, optional
        The replay's window, defaulting to the run's. Only its bars are
        decided and traded: the orders of its last bar are not (closed
        loop: not decided; open loop: left unsent, with no print after it).
    broker : {"ibkr"}
        Whose fee model charges the fills.
    fill_model : nautilus_trader.execution.FillModel, optional
        How orders fill, such as ``ProportionalSlippage``; by default at the
        print.

    Returns
    -------
    pathlib.Path
        The replay directory, holding NautilusTrader's reports under
        ``reports/``.

    Raises
    ------
    ValueError
        For an unknown loop, a closed loop on a run quantlab cannot decide
        again (no prediction panel, or valued at raw prices), or a broker
        without a fee model.
    RuntimeError
        If the strategy failed on a bar (``out_dir`` then holds the records
        up to that bar).

    Examples
    --------
    ::

        out_dir = replay("runs/USEquityCrossectionSelectStockVectorBt_20261020")
        january = replay(run_dir, loop="open", start="2024-01-02", end="2024-01-31")
    """
    if loop not in ("closed", "open"):
        raise ValueError(f"loop must be 'closed' or 'open', not {loop!r}")
    plan = plan if isinstance(plan, ExecutionPlan) else ExecutionPlan(plan)
    run = open_run(run_dir)
    start = run.window[0] if start is None else pd.Timestamp(start)
    end = run.window[1] if end is None else pd.Timestamp(end)
    out_dir = Path(run.run_dir / f"replay-{loop}" if out_dir is None else out_dir)
    security_map = IdentityMap()
    split_factor, dividend = run.corporate_actions(start, end)
    splits = SplitTable(split_factor)
    bars = [_in_units(bar, splits) for bar in run.bars(start, end)]

    engine = BacktestEngine(BacktestEngineConfig(logging=LoggerConfig(stdout_level=LogLevel.WARNING)))
    engine.add_venue(
        SIM,
        OmsType.NETTING,
        AccountType.MARGIN,
        [Money(run.init_cash, USD)],
        base_currency=USD,
        default_leverage=Decimal(1),
        fill_model=fill_model,
        fee_model=brokers.fee_model(broker, splits.shares_per_unit),
        modules=[DividendModule(_dividend_payments(dividend, splits, security_map))],
    )
    opens, closes = splits.to_units(run.opens(start, end)), splits.to_units(run.closes(start, end))
    for symbol in opens.columns:
        engine.add_instrument(_equity(security_map.to_id(symbol)))
    engine.add_data(_prints(opens, closes, plan, security_map))
    delistings = _delistings(bars, security_map)
    if delistings:  # the engine refuses empty data
        engine.add_data(delistings)
    engine.add_data(
        [CustomData(DataType(DecisionBar.__name__), bar) for bar in bars],
        client_id=ClientId("QUANTLAB"),
    )
    algorithm_id = ExecAlgorithmId("SCHEDULED")
    engine.add_exec_algorithm(ScheduledExecution(algorithm_id, plan, Environment.BACKTEST))
    if loop == "closed":
        inputs = run.targets("closed", end=end)
        targets, preload = ConstructorTargets(inputs), inputs.preloaded(start, end)
    else:
        targets, preload = RebalanceTableTargets(run.targets("open")), contextlib.nullcontext()
    strategy = QuantlabStrategy(
        targets,
        security_map,
        algorithm_id,
        on_decision=lambda bar, decision, equity: _record_bar(
            out_dir, bar, decision, equity, splits.ratio(bar)
        ),
    )
    engine.add_strategy(strategy)
    with preload:
        engine.run()
    if strategy.failure is not None:
        bar, error = strategy.failure
        engine.dispose()
        raise RuntimeError(f"the strategy failed on bar {bar.date()}: {error}") from error
    for bar, outcomes in _outcomes(engine.cache.orders(), security_map, splits).items():
        records.write_outcomes(out_dir, bar, outcomes)
    _export_reports(engine, out_dir / "reports")
    engine.dispose()
    return out_dir


def _record_bar(
    out_dir: Path, bar: pd.Timestamp, decision: pd.DataFrame, equity: float, ratio: pd.Series
) -> None:
    """Record a bar's decision, and its close's snapshot and equity, from the decision.

    The decision holds the positions at the bar's close and the equity it
    sized on, valued at the bar's closes; it is in engine units, and is
    recorded in raw units (``ratio`` raw shares per unit).
    """
    ratio = ratio.reindex(decision.index)
    decision = decision.copy()
    decision["close"] = decision["close"] / ratio
    for column in ("position_quantity", "target_quantity", "order_quantity"):
        decision[column] = decision[column] * ratio
    records.write_decision(out_dir, bar, decision)
    position = decision["position_quantity"]
    holdings = float((position * decision["close"]).sum())
    cash = equity - holdings
    positions = decision.loc[position != 0, ["position_quantity"]].rename(
        columns={"position_quantity": "quantity"}
    )
    records.write_snapshot(out_dir, records.Snapshot(bar, positions, cash))
    records.append_equity(out_dir, bar, cash, holdings)


def _outcomes(orders, security_map: IdentityMap, splits: SplitTable) -> dict[pd.Timestamp, pd.DataFrame]:
    """What each bar's orders did, by bar and symbol, in raw units.

    A strategy order is the primary; the execution algorithm's children
    carry its client order id as their ``exec_spawn_id`` and hold the fills.
    An order is converted at the session its children were sent in, an
    order never sent at its decision bar.
    """
    # The venue's own orders (a delisting's EXPIRATION-*) belong to no decision.
    primaries = [o for o in orders if _is_primary(o) and not brokers.is_expiration(o)]
    children: dict[str, list] = {}
    for order in orders:
        if not _is_primary(order):
            children.setdefault(order.exec_spawn_id.value, []).append(order)
    rows: dict[pd.Timestamp, dict[int, dict]] = {}
    for primary in primaries:
        spawned = children.get(primary.client_order_id.value, [])
        sign = 1.0 if primary.side == OrderSide.BUY else -1.0
        quantity = float(primary.events()[0].quantity)  # as initialized, before spawning
        filled = sum(float(c.filled_qty) for c in spawned)
        cost = sum(float(c.filled_qty) * float(c.avg_px) for c in spawned if float(c.filled_qty))
        if not spawned:
            status = "not_sent"
        elif filled == quantity:
            status = "filled"
        else:
            status = "partially_filled" if filled else "unfilled"
        bar = _bar_of(primary)
        symbol = security_map.to_symbol(primary.instrument_id)
        sent = spawned[0].ts_init if spawned else decision_ns(bar)
        ratio = float(splits.at(sent)[symbol])
        rows.setdefault(bar, {})[symbol] = {
            "order_quantity": sign * quantity * ratio,
            "filled_quantity": sign * filled * ratio,
            "avg_px": cost / filled / ratio if filled else float("nan"),
            "status": status,
        }
    return {
        bar: pd.DataFrame.from_dict(by_symbol, orient="index").rename_axis("symbol")
        for bar, by_symbol in rows.items()
    }


def _is_primary(order) -> bool:
    return order.exec_spawn_id is None or order.exec_spawn_id == order.client_order_id


def _bar_of(order) -> pd.Timestamp:
    """The decision bar named by an order's ``bar_tag``."""
    (tag,) = (t for t in order.tags or [] if t.startswith(BAR_TAG))
    return pd.Timestamp(tag.removeprefix(BAR_TAG))


def _in_units(bar: DecisionBar, splits: SplitTable) -> DecisionBar:
    """``bar`` with its closes per engine unit."""
    frame = bar.frame()
    frame["close"] = frame["close"] * splits.ratio(bar.timestamp).reindex(frame.index)
    return DecisionBar.from_frame(bar.timestamp, frame, ts_event=bar.ts_event, ts_init=bar.ts_init)


def _delistings(bars: list[DecisionBar], security_map: IdentityMap) -> list[InstrumentClose]:
    """An expiring ``InstrumentClose`` at each delisting's last valuation.

    It comes just before the decision on the security's last bar, so that
    decision finds the position closed into cash.
    """
    closes = []
    for bar in bars:
        marks = bar.marks()
        ts = bar.ts_event - 1
        for symbol, close in marks.loc[marks["delisted"], "close"].items():
            closes.append(
                InstrumentClose(
                    security_map.to_id(symbol),
                    Price(float(close), PRICE_PRECISION),
                    InstrumentCloseType.CONTRACT_EXPIRED,
                    ts,
                    ts,
                )
            )
    return closes


def _dividend_payments(
    dividend: pd.DataFrame, splits: SplitTable, security_map: IdentityMap
) -> list[tuple[int, InstrumentId, float]]:
    """Each dividend's payment: after the decision on the bar before its ex-date.

    A dividend per raw share on the ex-date is paid per engine unit at that
    date's ratio. A dividend on the first bar has no prior close and is not
    paid.
    """
    payments = []
    for previous, ex_date in zip(dividend.index[:-1], dividend.index[1:], strict=True):
        per_unit = dividend.loc[ex_date] * splits.ratio(ex_date)
        for symbol, amount in per_unit[per_unit != 0].items():
            payments.append((decision_ns(previous), security_map.to_id(symbol), float(amount)))
    return sorted(payments, key=lambda payment: payment[0])


def _equity(instrument_id: InstrumentId) -> Equity:
    return Equity(
        instrument_id,
        Symbol(instrument_id.symbol.value),
        USD,
        PRICE_PRECISION,
        Price(10**-PRICE_PRECISION, PRICE_PRECISION),
        0,
        0,
    )


def _prints(
    opens: pd.DataFrame, closes: pd.DataFrame, plan: ExecutionPlan, security_map: IdentityMap
) -> list[TradeTick]:
    """The prints ``plan`` can trade at, per bar and symbol with prices for them."""
    ticks = []
    for bar in opens.index:
        session = Session.on(bar)
        for name, at, priced_at in _print_times(plan, session):
            if priced_at == session.open:
                prices = opens.loc[bar]
            elif priced_at == session.close:
                prices = closes.loc[bar]
            else:
                fraction = (priced_at - session.open) / (session.close - session.open)
                prices = opens.loc[bar] + (closes.loc[bar] - opens.loc[bar]) * fraction
            for symbol, price in prices.round(PRICE_PRECISION).items():
                if not np.isfinite(price):
                    continue
                ticks.append(
                    TradeTick(
                        security_map.to_id(symbol),
                        Price(float(price), PRICE_PRECISION),
                        PRINT_SIZE,
                        AggressorSide.NO_AGGRESSOR,
                        TradeId(f"{name}-{symbol}-{pd.Timestamp(bar).date()}"),
                        at.value,
                        at.value,
                    )
                )
    return sorted(ticks, key=lambda tick: tick.ts_event)


def _print_times(plan: ExecutionPlan, session: Session) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """A session's prints that ``plan`` can trade at: name, time and the time priced at."""
    opening = [("open", session.open, session.open), ("open-end", session.open + AUCTION_LIFE, session.open)]
    closing = [("close", session.close, session.close), ("close-end", session.close + AUCTION_LIFE, session.close)]
    if plan.at == "open_auction":
        fallback = session.open + pd.Timedelta(minutes=plan.fallback_minutes)
        return opening + [("fallback", fallback, fallback)]
    if plan.at == "close_auction":
        return closing
    at = session.at(plan.at)
    timed = [("timed", at, at)] if session.open <= at < session.close else []
    # A timed limit child rests until it expires at the close.
    return timed + (closing if plan.order == "limit" else [])


def _export_reports(engine: BacktestEngine, reports_dir: Path) -> None:
    """Write NautilusTrader's reports as parquet; nested values become JSON text."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name in REPORTS:
        report = getattr(engine, f"generate_{name}_report")
        frame = report(SIM) if name == "account" else report()
        for column in frame.columns:
            if frame[column].map(lambda v: isinstance(v, (dict, list, tuple))).any():
                frame[column] = frame[column].map(lambda v: json.dumps(v, default=str))
        frame.to_parquet(reports_dir / f"{name}.parquet")
