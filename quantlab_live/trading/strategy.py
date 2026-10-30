"""``QuantlabStrategy``: one strategy for live trading and both replays.

The strategy receives each ``DecisionBar`` in ``on_data`` only; it never
knows whether it runs live or in a replay. On a bar it values its holdings at
the bar's marks, asks its target source for the bar's weights, sizes them in
whole shares and submits the difference to its execution algorithm, which
decides when the orders reach the venue. Each order is tagged with its bar
and with the bar's close, which a limit plan's band is around.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import pandas as pd
from nautilus_trader.model import (
    CustomData,
    DataType,
    ExecAlgorithmId,
    InstrumentId,
    OrderSide,
    Quantity,
    StrategyId,
    TimeInForce,
)
from nautilus_trader.trading import Strategy, StrategyConfig

from quantlab_live.trading.decision_bar import DecisionBar
from quantlab_live.trading.scheduled_execution import MARK_TAG


class TargetSource(Protocol):
    """Where a decision bar's target weights come from."""

    def weights(self, bar: pd.Timestamp) -> pd.Series: ...


class SecurityMap(Protocol):
    """Symbols to instruments and back; ``None`` where unmapped."""

    def to_id(self, symbol: int) -> InstrumentId | None: ...

    def to_symbol(self, instrument_id: InstrumentId) -> int | None: ...


def _config() -> StrategyConfig:
    return StrategyConfig(strategy_id=StrategyId("QUANTLAB-001"))


#: The prefix of the order tag naming an order's decision bar.
BAR_TAG = "bar="


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

    Parameters
    ----------
    targets : TargetSource
        The bar's target weights by symbol (NaN keeps a holding).
    security_map : SecurityMap
        The symbol of each instrument and back.
    exec_algorithm_id : ExecAlgorithmId
        The execution algorithm every order is sent to.
    on_decision : callable, optional
        Called with the bar, its decision frame and the equity sized on,
        before the first order.

    Examples
    --------
    ::

        strategy = QuantlabStrategy(targets, security_map, ExecAlgorithmId("SCHEDULED"))
    """

    def __new__(cls, *args, **kwargs):
        # The native base reads its configuration in __new__, not __init__.
        return super().__new__(cls, _config())

    def __init__(
        self,
        targets: TargetSource,
        security_map: SecurityMap,
        exec_algorithm_id: ExecAlgorithmId,
        on_decision: Callable[[pd.Timestamp, pd.DataFrame, float], None] | None = None,
    ):
        super().__init__(_config())
        self._targets = targets
        self._map = security_map
        self._exec_algorithm_id = exec_algorithm_id
        self._on_decision = on_decision
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
            self._decide(record)
        except Exception as error:
            self.failure = (record.timestamp, error)
            raise

    def _decide(self, record: DecisionBar) -> None:
        bar = record.timestamp
        close = record.marks()["close"]
        held = self._holdings()
        decision, equity = size_orders(
            self._targets.weights(bar), close, held, self._cash(close, held)
        )
        if self._on_decision is not None:
            self._on_decision(bar, decision, equity)
        for symbol, quantity in decision["order_quantity"].items():
            instrument_id = self._map.to_id(symbol)
            if quantity == 0 or instrument_id is None:
                continue
            order = self.order_factory.market(
                instrument_id,
                OrderSide.BUY if quantity > 0 else OrderSide.SELL,
                Quantity(abs(quantity), 0),
                time_in_force=TimeInForce.DAY,
                exec_algorithm_id=self._exec_algorithm_id,
                tags=[bar_tag(bar), MARK_TAG + repr(float(close[symbol]))],
            )
            self.submit_order(order)

    def _holdings(self) -> pd.DataFrame:
        """The signed quantity and average open price of each held symbol."""
        rows = {}
        for position in self.cache.positions_open(strategy_id=self.strategy_id):
            symbol = self._map.to_symbol(position.instrument_id)
            if symbol is not None:
                rows[symbol] = (float(position.signed_qty), float(position.avg_px_open))
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=["quantity", "avg_px"])
        return frame.astype(float)

    def _cash(self, close: pd.Series, held: pd.DataFrame) -> float:
        """The account's cash: its balance less the cost of the open positions.

        The account is the one of the venue the bar's instruments trade on.
        """
        ids = (self._map.to_id(symbol) for symbol in close.index)
        venue = next(i for i in ids if i is not None).venue
        account = self.cache.account_for_venue(venue)
        balance = float(account.balance_total().as_decimal())
        return balance - float((held["quantity"] * held["avg_px"]).sum())


def size_orders(
    weights: pd.Series, close: pd.Series, held: pd.DataFrame, cash: float
) -> tuple[pd.DataFrame, float]:
    """Size a bar's target weights in whole shares, on equity at the bar's closes.

    Equity is ``cash`` plus the holdings valued at ``close``. A target
    quantity is ``weight * equity / close`` truncated toward zero; a NaN
    weight keeps the holding.

    Parameters
    ----------
    weights : pandas.Series
        Target weight by symbol, NaN where kept.
    close : pandas.Series
        The bar's raw close by symbol.
    held : pandas.DataFrame
        ``quantity`` held by symbol.
    cash : float
        The account's cash.

    Returns
    -------
    decision : pandas.DataFrame
        By symbol of ``weights`` and ``held``: ``weight``, ``close``,
        ``position_quantity``, ``target_quantity`` and ``order_quantity``.
    equity : float
        The equity sized on.

    Examples
    --------
    >>> weights = pd.Series({1: 0.5, 2: float("nan")})
    >>> close = pd.Series({1: 10.0, 2: 20.0})
    >>> held = pd.DataFrame({"quantity": [5.0]}, index=[2])
    >>> decision, equity = size_orders(weights, close, held, cash=900.0)
    >>> equity, decision["order_quantity"].to_dict()
    (1000.0, {1: 50.0, 2: 0.0})
    """
    symbols = weights.index.union(held.index)
    position = held["quantity"].reindex(symbols, fill_value=0.0)
    marks = close.reindex(symbols)
    equity = cash + float((position * marks).sum())
    weight = weights.reindex(symbols)
    target = (weight * equity / marks).map(lambda q: float(math.trunc(q)) if math.isfinite(q) else q)
    target = target.where(weight.notna(), position)
    decision = pd.DataFrame(
        {
            "weight": weight,
            "close": marks,
            "position_quantity": position,
            "target_quantity": target,
            "order_quantity": target - position,
        }
    ).rename_axis("symbol")
    return decision, equity
