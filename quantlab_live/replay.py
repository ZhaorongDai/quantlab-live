"""``replay``: a quantlab run traded on NautilusTrader's ``BacktestEngine``.

The replay runs the live strategy and execution algorithm against one
synthetic ``SIM`` venue with one MARGIN account. Each symbol is the
instrument ``<symbol>.SIM`` (the identity security map), the prints the
execution plan trades at are fed as ``TradeTick`` data and the run's
decision bars are added to the engine as ``DecisionBar`` data, so the
strategy receives them exactly as it does live.

A daily quantlab run has an open and a close per bar, so the prints are:

- the opening print at the session's open, at the raw open;
- the closing print at the session's close (early closes included), at the
  raw close;
- a print at any other time of the session (an ``HH:MM`` plan, the opening
  auction's fallback), on the straight line from the open to the close by
  the time elapsed in the session;
- after an auction print, the same price again one second later: the
  simulator expires an order only on its instrument's next trade, and an
  unfilled auction child must expire there, before anything else trades.

Only the prints the plan can trade at are fed.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common import Environment, LoggerConfig, LogLevel
from nautilus_trader.execution import FixedFeeModel
from nautilus_trader.model import (
    AccountType,
    AggressorSide,
    ClientId,
    Currency,
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
    Symbol,
    TradeId,
    TradeTick,
    Venue,
)

from quantlab_live import records
from quantlab_live.quantlab_gate import PRICE_PRECISION, open_run
from quantlab_live.trading.decision_bar import NEW_YORK, DecisionBar
from quantlab_live.trading.scheduled_execution import (
    AUCTION_LIFE,
    ExecutionPlan,
    ScheduledExecution,
    Session,
)
from quantlab_live.trading.strategy import BAR_TAG, QuantlabStrategy
from quantlab_live.trading.targets import RebalanceTableTargets

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


def replay(
    run_dir,
    out_dir=None,
    *,
    loop: str = "open",
    plan: str | ExecutionPlan = "open_auction",
    fill_model=None,
) -> Path:
    """Replay the quantlab run at ``run_dir`` and write the replay directory.

    Parameters
    ----------
    run_dir : path-like
        The quantlab run.
    out_dir : path-like, optional
        The replay directory; defaults to ``replay-<loop>`` inside ``run_dir``.
    loop : {"open"}
        ``"open"`` executes the run's finished rebalance table.
    plan : str or ExecutionPlan
        When and how the orders trade: ``"open_auction"``,
        ``"close_auction"``, a New York time ``"HH:MM"`` (market orders), or
        a full ``ExecutionPlan``.
    fill_model : nautilus_trader.execution.FillModel, optional
        The simulated venue's fill model.

    Returns
    -------
    pathlib.Path
        The replay directory, holding NautilusTrader's reports under
        ``reports/``.

    Raises
    ------
    ValueError
        For a loop this version does not replay.
    RuntimeError
        If the strategy failed on a bar (``out_dir`` then holds the records
        up to that bar).

    Examples
    --------
    ::

        out_dir = replay("runs/USEquityCrossectionSelectStockVectorBt_20261020", loop="open")
    """
    if loop != "open":
        raise ValueError(f"loop must be 'open', not {loop!r}")
    plan = plan if isinstance(plan, ExecutionPlan) else ExecutionPlan(plan)
    run = open_run(run_dir)
    out_dir = Path(run.run_dir / f"replay-{loop}" if out_dir is None else out_dir)
    security_map = IdentityMap()

    engine = BacktestEngine(BacktestEngineConfig(logging=LoggerConfig(stdout_level=LogLevel.WARNING)))
    engine.add_venue(
        SIM,
        OmsType.NETTING,
        AccountType.MARGIN,
        [Money(run.init_cash, USD)],
        base_currency=USD,
        default_leverage=Decimal(1),
        fee_model=FixedFeeModel(Money(0, USD)),
        fill_model=fill_model,
    )
    opens, closes = run.opens(), run.closes()
    for symbol in opens.columns:
        engine.add_instrument(_equity(security_map.to_id(symbol)))
    engine.add_data(_prints(opens, closes, plan, security_map))
    engine.add_data(
        [CustomData(DataType(DecisionBar.__name__), bar) for bar in run.bars()],
        client_id=ClientId("QUANTLAB"),
    )
    algorithm_id = ExecAlgorithmId("SCHEDULED")
    engine.add_exec_algorithm(ScheduledExecution(algorithm_id, plan, Environment.BACKTEST))
    strategy = QuantlabStrategy(
        RebalanceTableTargets(run.targets("open")),
        security_map,
        algorithm_id,
        on_decision=lambda bar, decision, equity: _record_bar(out_dir, bar, decision, equity),
    )
    engine.add_strategy(strategy)
    engine.run()
    if strategy.failure is not None:
        bar, error = strategy.failure
        engine.dispose()
        raise RuntimeError(f"the strategy failed on bar {bar.date()}: {error}") from error
    for bar, outcomes in _outcomes(engine.cache.orders(), security_map).items():
        records.write_outcomes(out_dir, bar, outcomes)
    _export_reports(engine, out_dir / "reports")
    engine.dispose()
    return out_dir


def _record_bar(out_dir: Path, bar: pd.Timestamp, decision: pd.DataFrame, equity: float) -> None:
    """Record a bar's decision, and its close's snapshot and equity, from the decision.

    The decision holds the positions at the bar's close and the equity it
    sized on, valued at the bar's raw closes.
    """
    records.write_decision(out_dir, bar, decision)
    position = decision["position_quantity"]
    holdings = float((position * decision["close"]).sum())
    cash = equity - holdings
    positions = decision.loc[position != 0, ["position_quantity"]].rename(
        columns={"position_quantity": "quantity"}
    )
    records.write_snapshot(out_dir, records.Snapshot(bar, positions, cash))
    records.append_equity(out_dir, bar, cash, holdings)


def _outcomes(orders, security_map: IdentityMap) -> dict[pd.Timestamp, pd.DataFrame]:
    """What each bar's orders did, by bar and symbol.

    A strategy order is the primary; the execution algorithm's children
    carry its client order id as their ``exec_spawn_id`` and hold the fills.
    """
    primaries = [o for o in orders if _is_primary(o)]
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
        rows.setdefault(_bar_of(primary), {})[security_map.to_symbol(primary.instrument_id)] = {
            "order_quantity": sign * quantity,
            "filled_quantity": sign * filled,
            "avg_px": cost / filled if filled else float("nan"),
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
