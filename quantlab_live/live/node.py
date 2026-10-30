"""``run_node``: one trading day's node, a NautilusTrader ``LiveNode``.

A node process trades one day, or settles it (settle-only), and reports to
the supervisor through its step report only (``records.StepReport``):

1. the start guards run before the node is built (``guards.start_refusals``),
   with the real-money switches of ``live.toml`` and the arming file;
2. the security map is rebuilt on the day from the run's listings, on top of
   its history;
3. a ``LiveNode`` is built with the broker profile's clients, the risk
   engine's notional cap on every mapped instrument and its submit rate
   (``LiveConfig.risk_engine_config``), its cache in
   Redis under the live directory's fixed ``trader_id``, and the trading
   components of ``trading.parts``; the instrument provider loads only the
   held instruments, and the strategy requests a target's instrument at
   runtime;
4. once the node runs (after start-up reconciliation) the guards run again
   on the accounts the broker reported, the step report is written with
   ``started``, and the ``DecisionSource`` publishes the day's decision bar;
   the strategy decides (or resumes a decision already written) and its
   orders go through ``ScheduledExecution``;
5. the node stops a while after the close and writes the outcomes of the
   day's decision, the end-of-day snapshot and its final step report.

A settle-only node restores from Redis and reconciles as a trading node
does, trades nothing (its decision source publishes no bar), stops as soon
as its start check has passed (start-up reconciliation is done by then), at
the latest ``SETTLE_TIMEOUT_SECONDS`` after its start, and writes the
outcomes.
A dry run decides and writes the decision and the planned orders under
``dry-run/`` in the live directory, and submits nothing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import exchange_calendars
import nautilus_trader
import pandas as pd
from nautilus_trader.common import DataActor, DataActorConfig, Environment, LoggerConfig, LogLevel
from nautilus_trader.infrastructure import RedisCacheConfig
from nautilus_trader.live import LiveNode
from nautilus_trader.model import AccountId, ActorId, ClientId, TraderId

from quantlab_live import records, trading
from quantlab_live.brokers import BrokerProfile
from quantlab_live.live.config import load_config
from quantlab_live.live.guards import start_refusals
from quantlab_live.quantlab_gate import QuantlabNotReady, open_run
from quantlab_live.trading.outcomes import FOREIGN, UNMAPPED, order_bar, order_frames, order_outcomes
from quantlab_live.trading.scheduled_execution import Session
from quantlab_live.trading.strategy import STRATEGY_ID, DirectoryDecisions, account_cash

#: Where a dry run writes its decisions and planned orders, in the live directory.
DRY_RUN_DIR = "dry-run"
#: A dry run's planned orders, per date, under ``DRY_RUN_DIR``.
PLANNED_ORDERS = "orders"
#: How often the market-data actor looks for instruments to subscribe to, in nanoseconds.
SUBSCRIBE_EVERY_NS = 2_000_000_000
NS = 1_000_000_000
#: The longest a settle-only node waits for its start-up (reconciliation) to pass.
SETTLE_TIMEOUT_SECONDS = 15 * 60


def run_node(
    live_dir,
    date,
    *,
    settle: bool = False,
    dry_run: bool = False,
    report=None,
    broker: BrokerProfile | None = None,
    redis: bool = True,
    session: Session | None = None,
) -> records.StepReport:
    """Run the day's node of ``live_dir`` until after the close; return its step report.

    Parameters
    ----------
    live_dir : path-like
        The live directory, with its ``live.toml``.
    date : datetime-like
        The New York trading day traded; its decision bar is the session
        before it.
    settle : bool, default False
        Settle only: restore from Redis, trade nothing, write the outcomes.
    dry_run : bool, default False
        Decide and write the decision and the planned orders under
        ``dry-run/``, submit nothing.
    report : path-like, optional
        Where the step report is written: once start-up passes
        (``started``) and again when the node ends. By default
        ``steps/<date>/manual-<mode>.json``.
    broker : BrokerProfile, optional
        The broker profile; the configured one (``LiveConfig.profile``) by
        default. A test seam, as ``redis`` is.
    redis : bool, default True
        Keep the node's cache in Redis (``LiveConfig.redis_host``). Only
        tests turn it off: such a node restores nothing at a restart, and a
        real-money start is refused without it.
    session : Session, optional
        The session traded, which the execution plan and the stop time
        follow; the XNYS session of ``date`` by default. Tests give one
        around the wall clock, so their orders trade at once.

    Returns
    -------
    records.StepReport
        Also written to ``report``. A refused start, a bar quantlab has not
        made yet and any other failure are reported in ``error``.

    Examples
    --------
    ::

        report = run_node("live/us3000_paper", "2026-11-02")
        report.started, report.outcomes
    """
    live_dir = Path(live_dir)
    date = pd.Timestamp(date).normalize()
    mode = "settle" if settle else "trade"
    path = Path(report) if report is not None else live_dir / "steps" / date.date().isoformat() / f"manual-{mode}.json"
    state = {"report": records.StepReport(mode)}

    def write(**changes) -> records.StepReport:
        state["report"] = records.StepReport(**{**state["report"].__dict__, **changes})
        records.write_step_report(path, state["report"])
        return state["report"]

    try:
        return _run(live_dir, date, settle, dry_run, broker, redis, session, write, lambda: state["report"].started)
    except Exception as error:
        return write(error=f"{type(error).__name__}: {error}")


def _run(
    live_dir: Path, date: pd.Timestamp, settle: bool, dry_run: bool, broker, redis: bool, session, write,
    state_started,
) -> records.StepReport:
    config = load_config(live_dir)
    profile = broker or config.profile()
    switches = dataclasses.replace(config.switches(), redis=redis)

    def refusals_of(reported=None) -> list[str]:
        return start_refusals(
            profile, account_id=config.ib_account, port=config.gateway_port, reported=reported,
            version=nautilus_trader.__version__, switches=switches,
        )

    refusals = refusals_of()
    if refusals:
        return write(error="the start is refused: " + "; ".join(refusals))
    session = session or Session.on(date)
    bar = decision_bar_of(date)
    run = open_run(config.run_dir)
    security_map = profile.build_map(run.listings(date), date, records.read_security_map(live_dir))
    records.write_security_map(live_dir, security_map)
    bars = None
    if not settle:
        try:
            # A bar past the run's end is predicted into the live prediction store.
            bars = list(run.bars(bar, bar, store=config.store_dir if bar > run.window[1] else None))
        except QuantlabNotReady as error:
            return write(error=f"quantlab is not ready: {error}")

    snapshot = records.latest_snapshot(live_dir, date)
    held = [] if snapshot is None else [security_map.to_id(s) for s in snapshot.positions.index]
    held = [i for i in held if i is not None]
    # At takeover (no snapshot yet) the strategy claims every mapped instrument.
    claims = held if snapshot is not None else security_map.ids()
    exec_clients = profile.exec_clients(held)
    data_clients = profile.data_clients(held)
    simulated = profile.simulated
    environment = Environment.SANDBOX if simulated else Environment.LIVE

    builder = (
        LiveNode.builder("QUANTLAB-LIVE", TraderId(config.node_trader_id), environment)
        .with_logging(LoggerConfig(stdout_level=LogLevel.INFO))
        .with_reconciliation(not simulated)
        .with_risk_engine_config(config.risk_engine_config(security_map.ids()))
    )
    if redis:
        builder = builder.with_cache_database_factory(RedisCacheConfig(host=config.redis_host, port=config.redis_port))
    for client in data_clients:
        builder = builder.add_data_client(client.name, client.factory, client.config)
    for client in exec_clients:
        if client.simulated:
            builder = builder.add_simulated_exec_client(client.name, client.factory, client.config)
        else:
            builder = builder.add_exec_client(client.name, client.factory, client.config)
    node = builder.build()
    handle = node.handle()
    accounts = [AccountId(c.account_id) for c in exec_clients if c.account_id is not None]

    def check(cache) -> str | None:
        reported = [str(a) for a in accounts if cache.account(a) is not None]
        refusals = refusals_of(reported)
        if refusals:
            return "the start is refused: " + "; ".join(refusals)
        write(started=True)
        if settle:  # start-up reconciliation is done: nothing is left to wait for
            handle.stop()
        return None

    def refuse(reason: str) -> None:
        write(error=reason)
        handle.stop()

    decisions = DirectoryDecisions(live_dir / DRY_RUN_DIR if dry_run else live_dir)
    data_client = ClientId(data_clients[0].name) if data_clients else None
    components = trading.parts(
        run, decisions, loop="closed", plan=config.plan, security_map=security_map,
        environment=environment, dry_run=dry_run, bars=bars if bars is not None else [],
        check=check, on_refused=refuse, external_order_claims=claims, instrument_client=data_client,
        session=session,
    )
    if settle:
        node.add_actor(components.decision_source)  # the start check; it publishes no bar
        node.add_actor(_StopAt(pd.Timestamp.now(tz="UTC").value, SETTLE_TIMEOUT_SECONDS * NS, handle.stop))
    else:
        components.add_to(node)
        stop_at = session.close + pd.Timedelta(minutes=config.stop_after_close_minutes)
        node.add_actor(_StopAt(stop_at.value, config.min_node_seconds * NS, handle.stop))
    if data_client is not None and not settle:
        node.add_actor(_MarketData(data_client))
    node.run()

    source = components.decision_source
    if source.refused is not None:
        node.dispose()
        return write(error=source.refused)
    if not state_started():
        node.dispose()
        return write(error="the node stopped before its start-up passed (see the node's log)")
    strategy = components.strategy
    if not settle and strategy.failure is not None:
        failed_bar, error = strategy.failure
        node.dispose()
        return write(finished=True, error=f"the strategy failed on bar {failed_bar.date()}: {error!r}")
    if dry_run:
        node.dispose()
        _write_planned(live_dir / DRY_RUN_DIR, date, strategy.planned)
        return write(finished=True)
    alerts = list(state_alerts(node.cache, security_map))
    outcomes = _write_outcomes(live_dir, bar, node.cache, security_map)
    _write_snapshot(live_dir, date, node.cache, security_map, accounts, exec_clients)
    node.dispose()
    if outcomes is not None:
        for symbol, status in outcomes["status"].items():
            if status in (UNMAPPED, FOREIGN):  # the post-close mail reports unverified outcomes
                alerts.append(f"symbol {symbol} is {status}: not traded on the decision of {bar.date()}")
    return write(finished=True, outcomes=outcomes is not None, alerts=tuple(alerts))


def decision_bar_of(date) -> pd.Timestamp:
    """Return the decision bar a trading day trades on: the XNYS session before it.

    Examples
    --------
    >>> decision_bar_of("2024-01-08")
    Timestamp('2024-01-05 00:00:00')
    """
    calendar = exchange_calendars.get_calendar("XNYS")
    return pd.Timestamp(calendar.previous_session(pd.Timestamp(date).normalize())).tz_localize(None)


def state_alerts(cache, security_map) -> list[str]:
    """Return an alert for each open position no symbol maps to.

    Examples
    --------
    ::

        alerts = state_alerts(node.cache, security_map)
    """
    alerts = []
    for position in cache.positions_open():
        if security_map.to_symbol(position.instrument_id) is None:
            alerts.append(f"held {position.signed_qty} of {position.instrument_id}, unmapped")
    return alerts


def _write_outcomes(live_dir: Path, bar: pd.Timestamp, cache, security_map) -> pd.DataFrame | None:
    """Judge the day's decision's orders and write their outcomes; ``None`` without a decision."""
    if not records.has_decision(live_dir, bar):
        return None
    decision = records.read_decision(live_dir, bar)
    frame, fills = order_frames(cache.orders(strategy_id=STRATEGY_ID), security_map.to_symbol)
    mine = frame[frame["tags"].map(order_bar) == bar]
    positions = _positions(cache, security_map)["quantity"]
    outcomes = order_outcomes(decision, mine, fills[fills["client_order_id"].isin(mine.index)], positions)
    records.write_outcomes(live_dir, bar, outcomes)
    return outcomes


def _positions(cache, security_map) -> pd.DataFrame:
    """The strategy's open positions by symbol, signed raw shares."""
    rows = {}
    for position in cache.positions_open(strategy_id=STRATEGY_ID):
        symbol = security_map.to_symbol(position.instrument_id)
        if symbol is not None:
            rows[symbol] = rows.get(symbol, 0.0) + float(position.signed_qty)
    frame = pd.DataFrame({"quantity": pd.Series(rows, dtype=float)})
    return frame.rename_axis("symbol")


def _write_snapshot(live_dir: Path, date: pd.Timestamp, cache, security_map, accounts, exec_clients) -> None:
    """Write the end-of-day snapshot: the strategy's positions by symbol, the account's cash, and
    the positions the strategy does not own (foreign) by instrument id."""
    account = next((cache.account(a) for a in accounts if cache.account(a) is not None), None)
    if account is None:
        venues = [getattr(c.config, "venue", None) for c in exec_clients]
        account = next((cache.account_for_venue(v) for v in venues if v is not None), None)
    cash = float("nan") if account is None else account_cash(cache, account)
    positions = _positions(cache, security_map)
    foreign = {}
    for position in cache.positions_open():
        if position.strategy_id != STRATEGY_ID:
            key = position.instrument_id.value
            foreign[key] = foreign.get(key, 0.0) + float(position.signed_qty)
    snapshot = records.Snapshot(date, positions[positions["quantity"] != 0], cash, foreign)
    records.write_snapshot(live_dir, snapshot)


def _write_planned(directory: Path, date: pd.Timestamp, planned: list[dict]) -> Path:
    """Write a dry run's planned orders: bar, symbol, instrument id and signed quantity."""
    frame = pd.DataFrame(planned, columns=["bar", "symbol", "instrument_id", "quantity"])
    path = directory / PLANNED_ORDERS / f"{date.date().isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


class _StopAt(DataActor):
    """Stop the node at ``at_ns``, and never before it has run ``least_ns``."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, at_ns: int, least_ns: int, stop: Callable[[], None]):
        super().__init__(DataActorConfig(actor_id=ActorId("STOP-AT-001")))
        self._at_ns, self._least_ns, self._stop = at_ns, least_ns, stop

    def on_start(self) -> None:
        at = max(self._at_ns, self.clock.timestamp_ns() + self._least_ns)
        self.clock.set_time_alert_ns("stop", at, lambda event: self._stop())


class _MarketData(DataActor):
    """Subscribe to the quotes and trades of every instrument loaded.

    The risk engine prices a market order from the cache, and a simulated
    venue fills on these quotes.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, client_id: ClientId):
        super().__init__(DataActorConfig(actor_id=ActorId("MARKET-DATA-001")))
        self._client_id = client_id
        self._subscribed: set = set()

    def on_start(self) -> None:
        self._subscribe(None)
        self.clock.set_timer_ns("subscribe", SUBSCRIBE_EVERY_NS, callback=self._subscribe)

    def _subscribe(self, event) -> None:
        for instrument_id in self.cache.instrument_ids():
            if instrument_id not in self._subscribed:
                self._subscribed.add(instrument_id)
                self.subscribe_quotes(instrument_id, client_id=self._client_id)
                self.subscribe_trades(instrument_id, client_id=self._client_id)
