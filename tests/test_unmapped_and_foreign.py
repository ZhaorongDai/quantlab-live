"""The strategy skips unmapped and foreign symbols before submission and records them.

The strategy runs on a ``BacktestEngine`` built here rather than through
``replay``: a replay maps every symbol and owns every position, while this
engine has a security map with a gap and a second strategy holding a
position the quantlab strategy does not own.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest
from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common import Environment, LoggerConfig, LogLevel
from nautilus_trader.model import (
    AccountType,
    AggressorSide,
    ClientId,
    CustomData,
    DataType,
    Equity,
    ExecAlgorithmId,
    InstrumentId,
    Money,
    OmsType,
    OrderSide,
    Price,
    Quantity,
    StrategyId,
    Symbol,
    TradeId,
    TradeTick,
)
from nautilus_trader.trading import Strategy, StrategyConfig

from quantlab_live import brokers, records
from quantlab_live.quantlab_gate import PRICE_PRECISION, open_run
from quantlab_live.replay import PRINT_SIZE, SIM, USD, IdentityMap
from quantlab_live.trading.decision_bar import DecisionBar
from quantlab_live.trading.outcomes import order_frames, order_outcomes
from quantlab_live.trading.scheduled_execution import AUCTION_LIFE, ExecutionPlan, ScheduledExecution, Session
from quantlab_live.trading.strategy import DirectoryDecisions, QuantlabStrategy
from quantlab_live.trading.targets import RebalanceTableTargets
from tests.quantlab_run_fixture import build_quantlab_run

A, B, C = 10001, 10002, 10003
BARS = pd.bdate_range("2024-01-02", periods=2)
OPEN = {A: [10.0, 10.5], B: [20.0, 20.5], C: [50.0, 48.0]}
CLOSE = {A: [10.0, 11.0], B: [20.0, 20.0], C: [50.0, 40.0]}
KEEP = float("nan")
WEIGHTS = {A: [0.5, KEEP], B: [0.3, KEEP], C: [0.2, KEEP]}
FOREIGN_SHARES = 10


class GappedMap(IdentityMap):
    """The identity map, without an instrument for ``B``."""

    def to_id(self, symbol):
        return None if symbol == B else super().to_id(symbol)


class Manual(Strategy):
    """Someone else's trade in the account: buys ``C`` at its first print."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, StrategyConfig(strategy_id=StrategyId("MANUAL-002"), order_id_tag="002"))

    def __init__(self):
        super().__init__(StrategyConfig(strategy_id=StrategyId("MANUAL-002"), order_id_tag="002"))
        self._bought = False

    def on_start(self):
        self.subscribe_trades(InstrumentId.from_str(f"{C}.SIM"))

    def on_trade(self, tick):
        if not self._bought:
            self._bought = True
            self.submit_order(self.order_factory.market(tick.instrument_id, OrderSide.BUY, Quantity(FOREIGN_SHARES, 0)))


def _prints(run) -> list[TradeTick]:
    """Each session's opening print, repeated as replay does, at the open."""
    ticks = []
    for bar, prices in run.opens().iterrows():
        session = Session.on(bar)
        for n, at in enumerate((session.open, session.open + AUCTION_LIFE)):
            for symbol, price in prices.items():
                ticks.append(
                    TradeTick(
                        InstrumentId.from_str(f"{symbol}.SIM"), Price(float(price), PRICE_PRECISION), PRINT_SIZE,
                        AggressorSide.NO_AGGRESSOR, TradeId(f"{n}-{symbol}-{bar.date()}"), at.value, at.value,
                    )
                )
    return sorted(ticks, key=lambda tick: tick.ts_event)


@pytest.fixture(scope="module")
def traded(tmp_path_factory):
    root = tmp_path_factory.mktemp("skips")
    run = open_run(build_quantlab_run(root, BARS, OPEN, CLOSE, WEIGHTS, init_cash=10_000.0))
    engine = BacktestEngine(BacktestEngineConfig(logging=LoggerConfig(stdout_level=LogLevel.WARNING)))
    engine.add_venue(
        SIM, OmsType.NETTING, AccountType.MARGIN, [Money(10_000, USD)], base_currency=USD,
        default_leverage=Decimal(1), fee_model=brokers.fee_model("ibkr"),
    )
    for symbol in OPEN:
        instrument_id = InstrumentId.from_str(f"{symbol}.SIM")
        engine.add_instrument(
            Equity(instrument_id, Symbol(str(symbol)), USD, PRICE_PRECISION,
                   Price(10**-PRICE_PRECISION, PRICE_PRECISION), 0, 0)
        )
    engine.add_data(_prints(run))
    engine.add_data(
        [CustomData(DataType(DecisionBar.__name__), bar) for bar in run.bars()], client_id=ClientId("QUANTLAB")
    )
    algorithm_id = ExecAlgorithmId("SCHEDULED")
    engine.add_exec_algorithm(ScheduledExecution(algorithm_id, ExecutionPlan("open_auction"), Environment.BACKTEST))
    decisions_dir = root / "live"
    strategy = QuantlabStrategy(
        RebalanceTableTargets(run.targets("open")), GappedMap(), algorithm_id, DirectoryDecisions(decisions_dir)
    )
    engine.add_strategy(Manual())
    engine.add_strategy(strategy)
    engine.run()
    assert strategy.failure is None
    orders, fills = order_frames(engine.cache.orders(strategy_id=strategy.strategy_id), GappedMap().to_symbol)
    foreign = [(str(p.instrument_id), float(p.signed_qty)) for p in engine.cache.positions_open()
               if p.strategy_id != strategy.strategy_id]
    engine.dispose()
    decision = records.read_decision(decisions_dir, BARS[0])
    return decision, order_outcomes(decision, orders, fills, positions=None), fills, foreign


def test_unmapped_and_foreign_targets_are_skipped_and_recorded_in_the_decision(traded):
    decision, _, fills, _ = traded
    assert decision["skip"].to_dict() == {A: "", B: "unmapped", C: "foreign"}
    assert set(fills["symbol"]) == {A}


def test_unmapped_and_foreign_targets_appear_in_the_outcomes(traded):
    _, outcomes, _, _ = traded
    assert outcomes["status"].to_dict() == {A: "filled", B: "unmapped", C: "foreign"}
    assert outcomes.loc[[B, C], "filled_quantity"].eq(0).all()


def test_a_foreign_position_is_never_traded_and_is_outside_the_strategy_capital(traded):
    decision, _, _, foreign = traded
    assert foreign == [(f"{C}.SIM", float(FOREIGN_SHARES))]
    # The capital is 10,000 less the 500 and the 1.00 fee paid for the
    # foreign shares; the skipped symbols' weights stay in cash.
    assert decision["target_quantity"].to_dict() == {A: 474.0, B: 142.0, C: 37.0}
