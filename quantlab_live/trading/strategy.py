"""``QuantlabStrategy``: one strategy for live trading and both replays.

The strategy receives each ``DecisionBar`` in ``on_data`` only; it never
knows whether it runs live or in a replay. On a bar it values its holdings at
the bar's marks, asks its target source for the bar's weights given those
holdings, sizes them in whole shares of the strategy capital, writes the
decision durably and only then submits the difference to its execution
algorithm, which decides when the orders reach the venue. A bar already
decided is resumed, never decided again. Each order is tagged with its bar
and with the bar's close, which a limit plan's band is around.
"""

from __future__ import annotations

import math
from typing import Protocol

import pandas as pd
from nautilus_trader.model import (
    ClientId,
    CustomData,
    DataType,
    ExecAlgorithmId,
    InstrumentId,
    OrderSide,
    OrderStatus,
    Quantity,
    StrategyId,
    TimeInForce,
)
from nautilus_trader.trading import Strategy, StrategyConfig

from quantlab_live import records
from quantlab_live.trading import outcomes
from quantlab_live.trading.outcomes import BAR_TAG
from quantlab_live.trading.decision_bar import DecisionBar
from quantlab_live.trading.scheduled_execution import MARK_TAG


class TargetSource(Protocol):
    """Where a decision bar's target weights come from.

    ``current`` is the weight of each symbol held, valued at the bar's
    closes on the capital sized on; a source that does not decide on the
    holdings (a finished rebalance table) ignores it.
    """

    def weights(self, bar: DecisionBar, current: pd.Series) -> pd.Series: ...


class SecurityMap(Protocol):
    """Symbols to instruments and back; ``None`` where unmapped."""

    def to_id(self, symbol: int) -> InstrumentId | None: ...

    def to_symbol(self, instrument_id: InstrumentId) -> int | None: ...


class DecisionRecords(Protocol):
    """Where the strategy keeps its decisions, one per bar.

    ``write`` returns only once the decision is durable; the strategy sends
    no order of a bar before it has. It is given the capital sized on and
    the strategy's equity at the bar's closes. ``read`` returns ``None`` for
    a bar not yet decided. Quantities and prices are in the strategy's
    units.
    """

    def read(self, bar: pd.Timestamp) -> pd.DataFrame | None: ...

    def write(self, bar: pd.Timestamp, decision: pd.DataFrame, capital: float, equity: float) -> None: ...


class StrategyPnl(Protocol):
    """The strategy's cumulative profit and loss at a bar's closes, since its ``initial_capital``.

    ``holdings`` is the value of the strategy's positions and
    ``account_equity`` the account's equity, both at the bar's closes.
    """

    def pnl(self, bar: pd.Timestamp, account_equity: float, holdings: float) -> float: ...


class RecordedPnl:
    """A live directory's strategy P&L: its strategy equity as ``live.value`` records it.

    The strategy's cash is the latest equity row's at or before the bar (its
    P&L until then, valued at quantlab's close), and its holdings are valued
    at the bar's closes. Before any row the P&L is zero (the takeover).

    Parameters
    ----------
    directory : path-like
        The live directory.
    initial_capital : float
        The strategy's initial capital.

    Examples
    --------
    >>> import tempfile
    >>> directory = tempfile.mkdtemp()
    >>> RecordedPnl(directory, 10_000.0).pnl(pd.Timestamp("2024-01-02"), 50_000.0, 9_000.0)
    0.0
    >>> _ = records.append_equity(directory, pd.Timestamp("2024-01-02"), cash=1_500.0, holdings=9_000.0)
    >>> RecordedPnl(directory, 10_000.0).pnl(pd.Timestamp("2024-01-02"), 50_000.0, 9_200.0)
    700.0
    """

    def __init__(self, directory, initial_capital: float):
        self._directory, self._initial = directory, float(initial_capital)

    def pnl(self, bar: pd.Timestamp, account_equity: float, holdings: float) -> float:
        try:
            equity = records.read_equity(self._directory)
        except FileNotFoundError:
            return 0.0
        rows = equity[equity.index <= pd.Timestamp(bar)]
        if rows.empty:
            return 0.0
        return float(rows["cash"].iloc[-1]) + float(holdings) - self._initial


class AccountPnl:
    """The P&L of a strategy alone in its account (a replay): the account's equity less its starting cash.

    Examples
    --------
    >>> AccountPnl(20_000.0).pnl(pd.Timestamp("2024-01-02"), 21_000.0, 9_000.0)
    1000.0
    """

    def __init__(self, starting_cash: float):
        self._start = float(starting_cash)

    def pnl(self, bar: pd.Timestamp, account_equity: float, holdings: float) -> float:
        return float(account_equity) - self._start


class DirectoryDecisions:
    """The decisions of a live directory, in ``records`` (a unit is a raw share).

    Parameters
    ----------
    directory : path-like
        The live directory.

    Examples
    --------
    >>> import tempfile
    >>> decisions = DirectoryDecisions(tempfile.mkdtemp())
    >>> decisions.read(pd.Timestamp("2024-01-02")) is None
    True
    """

    def __init__(self, directory):
        self._directory = directory

    def read(self, bar: pd.Timestamp) -> pd.DataFrame | None:
        """Return the decision on ``bar``, or ``None`` if it has none."""
        try:
            return records.read_decision(self._directory, bar)
        except FileNotFoundError:
            return None

    def write(self, bar: pd.Timestamp, decision: pd.DataFrame, capital: float, equity: float) -> None:
        """Write the decision on ``bar`` and the capital it sized on durably (tmp, fsync and rename).

        The strategy equity is recorded by ``live.value``, not here.
        """
        records.write_decision(self._directory, bar, decision, capital)


#: The strategy's id: one strategy per trader.
STRATEGY_ID = StrategyId("QUANTLAB-001")
#: How long an order waits for its instrument, requested at runtime, in seconds.
INSTRUMENT_WAIT_SECONDS = 300
INSTRUMENT_TIMER = "instruments"


def _config(external_order_claims=()) -> StrategyConfig:
    return StrategyConfig(strategy_id=STRATEGY_ID, external_order_claims=list(external_order_claims))




def bar_tag(bar: pd.Timestamp) -> str:
    """Return the order tag naming the decision bar an order belongs to.

    Examples
    --------
    >>> bar_tag(pd.Timestamp("2024-01-02"))
    'bar=2024-01-02'
    """
    return BAR_TAG + pd.Timestamp(bar).date().isoformat()


class QuantlabStrategy(Strategy):
    """Trade a quantlab run's targets, one decision per ``DecisionBar``.

    On a bar not yet decided, the strategy decides, writes the decision and
    only then submits its orders. On a bar already decided (a restarted node
    receiving the day's bar again, or a second replay into a decided
    directory) it does not decide again: it cancels the primaries the
    execution algorithm kept locally and never sent, and submits what is
    left of the decision, ``remaining_orders``.

    Its holdings are its own positions. A position it does not own is
    foreign: never traded, outside the strategy capital, and a target on its
    instrument is skipped. A target without an instrument is skipped as
    unmapped. A skipped symbol stays in cash; the decision records why.

    Parameters
    ----------
    targets : TargetSource
        The bar's target weights by symbol (NaN keeps a holding): a finished
        rebalance table (open loop) or quantlab's construction on the
        holdings (closed loop and live). Nothing else differs between them.
    security_map : SecurityMap
        The symbol of each instrument and back.
    exec_algorithm_id : ExecAlgorithmId
        The execution algorithm every order is sent to.
    decisions : DecisionRecords
        Where the decisions are kept.
    external_order_claims : sequence of InstrumentId, optional
        The instruments whose orders and positions the strategy claims from
        start-up reconciliation (live): its holdings.
    instrument_client : ClientId, optional
        The data client to request an instrument from when an order's
        instrument is not loaded yet (live). Each order is submitted once
        its instrument arrives; an order whose instrument has not arrived
        ``INSTRUMENT_WAIT_SECONDS`` after the request is left unsent.
    dry_run : bool, default False
        Decide and write the decision, but submit nothing: the orders that
        would be submitted are kept in ``planned``.
    initial_capital : float, optional
        The strategy capital's start: orders size on it plus the strategy's
        P&L (``pnl``), capped by the account's equity (``strategy_capital``).
        Unset, they size on the account's equity.
    pnl : StrategyPnl, optional
        The strategy's P&L; required with ``initial_capital``.

    Examples
    --------
    >>> import tempfile
    >>> from quantlab_live.replay import IdentityMap
    >>> targets = None  # a TargetSource: RebalanceTableTargets or ConstructorTargets (``trading.parts``)
    >>> strategy = QuantlabStrategy(
    ...     targets, IdentityMap(), ExecAlgorithmId("SCHEDULED"), DirectoryDecisions(tempfile.mkdtemp()),
    ...     initial_capital=10_000.0, pnl=AccountPnl(10_000.0),
    ... )
    >>> strategy.strategy_id
    StrategyId('QUANTLAB-001')
    """

    def __new__(cls, *args, external_order_claims=(), **kwargs):
        # The native base reads its configuration in __new__, not __init__.
        return super().__new__(cls, _config(external_order_claims))

    def __init__(
        self,
        targets: TargetSource,
        security_map: SecurityMap,
        exec_algorithm_id: ExecAlgorithmId,
        decisions: DecisionRecords,
        *,
        external_order_claims=(),
        instrument_client: ClientId | None = None,
        dry_run: bool = False,
        initial_capital: float | None = None,
        pnl: StrategyPnl | None = None,
    ):
        super().__init__(_config(external_order_claims))
        if initial_capital is not None and pnl is None:
            raise ValueError("an initial_capital needs the strategy's P&L (pnl)")
        self._initial_capital = initial_capital
        self._pnl = pnl
        self._targets = targets
        self._map = security_map
        self._exec_algorithm_id = exec_algorithm_id
        self._decisions = decisions
        self._instrument_client = instrument_client
        self._dry_run = dry_run
        #: Orders waiting for their instrument: (bar, symbol, quantity, close).
        self._waiting: list[tuple[pd.Timestamp, int, float, float]] = []
        self._wait_until = 0
        #: A dry run's orders: bar, symbol, instrument id and signed quantity.
        self.planned: list[dict] = []
        #: The first error a decision raised, with its bar; the strategy decides
        #: nothing after it. NautilusTrader logs a handler's error and carries
        #: on, so whoever runs the strategy reads this.
        self.failure: tuple[pd.Timestamp, Exception] | None = None

    def on_start(self) -> None:
        self.subscribe_data(DataType(DecisionBar.__name__))

    def on_data(self, data) -> None:
        record = data.data if isinstance(data, CustomData) else data
        if not isinstance(record, DecisionBar) or self.failure is not None:
            return
        try:
            decision = self._decisions.read(record.timestamp)
            if decision is None:
                self._decide(record)
            else:
                self._resume(record.timestamp, decision)
        except Exception as error:
            self.failure = (record.timestamp, error)
            raise

    def _decide(self, record: DecisionBar) -> None:
        bar = record.timestamp
        close = record.marks()["close"]
        held = self._holdings()
        value = held["quantity"] * close.reindex(held.index)
        holdings = float(value.sum())
        account_equity = self._cash(close) + holdings
        if self._initial_capital is None:
            pnl, equity = 0.0, account_equity
        else:
            pnl = self._pnl.pnl(bar, account_equity, holdings)
            equity = self._initial_capital + pnl
        capital = strategy_capital(account_equity, self._initial_capital, pnl)
        current = value / capital if capital else value * 0.0
        decision = size_orders(self._targets.weights(record, current), close, held, capital)
        decision["skip"] = self._skips(decision)
        self._decisions.write(bar, decision, capital, equity)
        ordered = decision[(decision["order_quantity"] != 0) & (decision["skip"] == "")]
        self._submit(bar, ordered["order_quantity"], decision["close"])

    def _resume(self, bar: pd.Timestamp, decision: pd.DataFrame) -> None:
        """Carry on with a decided bar: cancel the kept primaries, send the rest."""
        for order in self.cache.orders(strategy_id=self.strategy_id):
            if _is_primary(order) and order.status == OrderStatus.INITIALIZED and order.leaves_qty.as_double() > 0:
                self.cancel_order(order.client_order_id)
        position = self._holdings()["quantity"]
        working = {}
        open_orders = self.cache.orders_open(strategy_id=self.strategy_id)
        for order in [*open_orders, *self.cache.orders_inflight(strategy_id=self.strategy_id)]:
            symbol = self._map.to_symbol(order.instrument_id)
            if _is_primary(order) or symbol is None:
                continue
            sign = 1.0 if order.side == OrderSide.BUY else -1.0
            working[symbol] = working.get(symbol, 0.0) + sign * order.leaves_qty.as_double()
        remaining = remaining_orders(decision, position, pd.Series(working, dtype=float))
        self.log.info(f"resuming the decision on {bar.date()}: {len(remaining)} orders remain")
        self._submit(bar, remaining, decision["close"])

    def _submit(self, bar: pd.Timestamp, quantities: pd.Series, close: pd.Series) -> None:
        missing = []
        for symbol, quantity in quantities.items():
            instrument_id = self._map.to_id(symbol)
            if self._dry_run:
                self.planned.append(
                    {"bar": bar, "symbol": int(symbol), "instrument_id": str(instrument_id), "quantity": float(quantity)}
                )
            elif self._instrument_client is not None and self.cache.instrument(instrument_id) is None:
                self._waiting.append((bar, symbol, float(quantity), float(close[symbol])))
                missing.append(instrument_id)
            else:
                self._send(bar, symbol, float(quantity), float(close[symbol]))
        if missing:
            self.log.info(f"requesting {len(missing)} instruments not loaded yet")
            self.request_instrument(missing, client_id=self._instrument_client, callback=self._on_instruments)
            # The request's callback runs on its first response only, so the
            # waiting orders are also looked at every second until they leave.
            self._wait_until = self.clock.timestamp_ns() + INSTRUMENT_WAIT_SECONDS * 1_000_000_000
            if INSTRUMENT_TIMER not in self.clock.timer_names():
                self.clock.set_timer_ns(INSTRUMENT_TIMER, 1_000_000_000, callback=self._on_instruments)

    def _on_instruments(self, event) -> None:
        """Submit the waiting orders whose instrument has arrived; give up on the rest at the deadline."""
        waiting, self._waiting = self._waiting, []
        for bar, symbol, quantity, close in waiting:
            if self.cache.instrument(self._map.to_id(symbol)) is None:
                self._waiting.append((bar, symbol, quantity, close))
            else:
                self._send(bar, symbol, quantity, close)
        if self._waiting and self.clock.timestamp_ns() >= self._wait_until:
            for _, symbol, _, _ in self._waiting:
                self.log.error(f"{self._map.to_id(symbol)} could not be loaded; symbol {symbol} is not sent")
            self._waiting = []
        if not self._waiting and INSTRUMENT_TIMER in self.clock.timer_names():
            self.clock.cancel_timer(INSTRUMENT_TIMER)

    def _send(self, bar: pd.Timestamp, symbol: int, quantity: float, close: float) -> None:
        order = self.order_factory.market(
            self._map.to_id(symbol),
            OrderSide.BUY if quantity > 0 else OrderSide.SELL,
            Quantity(abs(quantity), 0),
            time_in_force=TimeInForce.DAY,
            exec_algorithm_id=self._exec_algorithm_id,
            tags=[bar_tag(bar), MARK_TAG + repr(close)],
        )
        self.submit_order(order)

    def _skips(self, decision: pd.DataFrame) -> pd.Series:
        """Why each ordered symbol is not traded: ``unmapped``, ``foreign`` or ``""``."""
        foreign = {p.instrument_id for p in self._foreign_positions()}
        skips = pd.Series("", index=decision.index, dtype=object)
        for symbol in decision.index[decision["order_quantity"] != 0]:
            instrument_id = self._map.to_id(symbol)
            if instrument_id is None:
                skips[symbol] = outcomes.UNMAPPED
            elif instrument_id in foreign:
                skips[symbol] = outcomes.FOREIGN
            else:
                continue
            self.log.warning(f"symbol {symbol} is {skips[symbol]}; it is not traded")
        return skips

    def _foreign_positions(self) -> list:
        """The open positions the strategy does not own."""
        return [p for p in self.cache.positions_open() if p.strategy_id != self.strategy_id]

    def _holdings(self) -> pd.DataFrame:
        """The signed quantity and average open price of each held symbol."""
        rows = {}
        for position in self.cache.positions_open(strategy_id=self.strategy_id):
            symbol = self._map.to_symbol(position.instrument_id)
            if symbol is not None:
                rows[symbol] = (float(position.signed_qty), float(position.avg_px_open))
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=["quantity", "avg_px"])
        return frame.astype(float)

    def _cash(self, close: pd.Series) -> float:
        """The account's cash (``account_cash``) on the venue the bar's instruments trade on.

        A position the strategy does not own, or cannot map, costs cash but
        adds no value to the strategy capital.
        """
        ids = (self._map.to_id(symbol) for symbol in close.index)
        venue = next(i for i in ids if i is not None).venue
        return account_cash(self.cache, self.cache.account_for_venue(venue))


def account_cash(cache, account) -> float:
    """Return an account's cash: its balance less the cost of its every open position.

    Parameters
    ----------
    cache : nautilus_trader.common.Cache
    account : nautilus_trader account
        A MARGIN account, whose balance does not pay for its positions.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> account = SimpleNamespace(id="SMART-001", balance_total=lambda: SimpleNamespace(as_decimal=lambda: 10_000))
    >>> held = [SimpleNamespace(signed_qty=100, avg_px_open=25.0), SimpleNamespace(signed_qty=-10, avg_px_open=50.0)]
    >>> cache = SimpleNamespace(positions_open=lambda account_id: held)
    >>> account_cash(cache, account)
    8000.0
    """
    balance = float(account.balance_total().as_decimal())
    positions = cache.positions_open(account_id=account.id)
    return balance - sum(float(p.signed_qty) * float(p.avg_px_open) for p in positions)


def _is_primary(order) -> bool:
    """Whether ``order`` is a strategy's order rather than a child spawned from one."""
    return order.exec_spawn_id is None or order.exec_spawn_id == order.client_order_id


def remaining_orders(decision: pd.DataFrame, position: pd.Series, working: pd.Series) -> pd.Series:
    """Return what is left to order of a decision: target less position less working orders.

    Only the symbols the decision orders and does not skip are traded, so a
    resumed decision never trades a kept holding or a skipped symbol. Sending
    the result again once it is working gives nothing, so resuming any
    number of times never duplicates an order.

    Parameters
    ----------
    decision : pandas.DataFrame
        By ``symbol``: ``target_quantity``, ``order_quantity`` and ``skip``.
    position : pandas.Series
        The signed position by symbol.
    working : pandas.Series
        The signed quantity still working at the venue by symbol.

    Returns
    -------
    pandas.Series
        The signed order quantity by symbol, without zeros.

    Examples
    --------
    >>> decision = pd.DataFrame(
    ...     {"target_quantity": [500.0, 40.0], "order_quantity": [500.0, 40.0], "skip": ["", ""]},
    ...     index=pd.Index([10001, 10002], name="symbol"),
    ... )
    >>> remaining_orders(decision, pd.Series({10001: 300.0}), pd.Series({10001: 100.0})).to_dict()
    {10001: 100.0, 10002: 40.0}
    """
    ordered = decision[(decision["order_quantity"] != 0) & (decision["skip"].fillna("") == "")]
    target = ordered["target_quantity"]
    left = target - position.reindex(target.index, fill_value=0.0) - working.reindex(target.index, fill_value=0.0)
    return left[left != 0].rename(None)


def strategy_capital(
    account_equity: float, initial_capital: float | None = None, pnl: float = 0.0
) -> float:
    """Return the capital a decision sizes its orders on.

    The strategy capital is ``initial_capital`` plus the strategy's
    cumulative profit and loss, valued at quantlab's close, capped by the
    account's equity, so money the strategy has not earned or been given is
    never used. Without an ``initial_capital`` the orders size on the
    account's equity.

    Parameters
    ----------
    account_equity : float
        The account's equity at the bar's closes.
    initial_capital : float, optional
        The capital given to the strategy.
    pnl : float, default 0.0
        The strategy's cumulative profit and loss at the bar's closes.

    Examples
    --------
    >>> strategy_capital(50_000.0, initial_capital=10_000.0, pnl=1_250.0)
    11250.0
    >>> strategy_capital(9_000.0, initial_capital=10_000.0, pnl=500.0)
    9000.0
    >>> strategy_capital(9_000.0)
    9000.0
    """
    if initial_capital is None:
        return float(account_equity)
    return float(min(initial_capital + pnl, account_equity))


def size_orders(
    weights: pd.Series, close: pd.Series, held: pd.DataFrame, capital: float
) -> pd.DataFrame:
    """Size a bar's target weights in whole shares of ``capital`` at the bar's closes.

    A target quantity is ``weight * capital / close`` truncated toward zero;
    a NaN weight, or a held symbol without a weight, keeps the holding.

    Parameters
    ----------
    weights : pandas.Series
        Target weight by symbol, NaN where kept.
    close : pandas.Series
        The bar's raw close by symbol.
    held : pandas.DataFrame
        ``quantity`` held by symbol.
    capital : float
        The capital sized on (``strategy_capital``).

    Returns
    -------
    pandas.DataFrame
        By symbol of ``weights`` and ``held``: ``weight``, ``close``,
        ``position_quantity``, ``target_quantity`` and ``order_quantity``.

    Examples
    --------
    >>> weights = pd.Series({1: 0.5, 2: float("nan")})
    >>> close = pd.Series({1: 10.0, 2: 20.0})
    >>> held = pd.DataFrame({"quantity": [5.0]}, index=[2])
    >>> size_orders(weights, close, held, capital=1_000.0)["order_quantity"].to_dict()
    {1: 50.0, 2: 0.0}
    """
    symbols = weights.index.union(held.index)
    position = held["quantity"].reindex(symbols, fill_value=0.0)
    marks = close.reindex(symbols)
    weight = weights.reindex(symbols)
    target = (weight * capital / marks).map(lambda q: float(math.trunc(q)) if math.isfinite(q) else q)
    target = target.where(weight.notna(), position)
    return pd.DataFrame(
        {
            "weight": weight,
            "close": marks,
            "position_quantity": position,
            "target_quantity": target,
            "order_quantity": target - position,
        }
    ).rename_axis("symbol")
