"""A live directory's configuration: one small ``live.toml``, everything else defaulted.

The keys an operator sets are ``run`` (the quantlab run, relative to the live
directory or absolute), ``initial_capital``, ``at``, ``order`` (with
``band_bps`` for a limit plan), ``broker`` and ``real_money``, and for real
money the risk engine's ``max_notional_per_order`` and
``max_order_submit_rate``. Every other key has a default, and ``status``
prints them all, so a default never hides.
"""

from __future__ import annotations

import dataclasses
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from nautilus_trader.live import LiveRiskEngineConfig

from quantlab_live import brokers
from quantlab_live.brokers import BrokerProfile
from quantlab_live.live.guards import ARMING_FILE, Switches, arming_state
from quantlab_live.quantlab_gate import NUMA_NODE_1_CPUS
from quantlab_live.trading.scheduled_execution import ExecutionPlan

CONFIG_FILE = "live.toml"


@dataclass(frozen=True)
class LiveConfig:
    """The configuration of one live directory.

    Parameters
    ----------
    run : str or pathlib.Path
        The quantlab backtest run traded, with a model.
    initial_capital : float, optional
        The strategy capital's start; required for real money. When unset on
        paper, orders size on the account's equity.
    at, order, band_bps : optional
        The execution plan (``ExecutionPlan``): ``"open_auction"``,
        ``"close_auction"`` or ``"HH:MM"``; ``"market"`` or ``"limit"``.
    broker : str, default ``"ibkr_paper"``
        The broker profile (``brokers.profile``).
    real_money : bool, default False
        One of the two switches a real-money start needs; the other is the
        owner-only arming file ``real_money.armed`` in the live directory
        (``guards.start_refusals``).
    max_notional_per_order : float, optional
        The risk engine's cap on one order's notional, in the instrument's
        quote currency; it denies a larger order. Required for real money.
    max_order_submit_rate : str, optional
        The risk engine's order submit rate, such as ``"10/00:00:01"`` (ten
        a second); NautilusTrader's default (100 a second) when unset.
        Required for real money.
    account : str or pathlib.Path, optional
        The account history (``records.read_account_history``): one
        ``account.toml`` per account, shared by every live directory that
        trades it; ``account.toml`` in the live directory by default.
    after_open_minutes : int, default 15
        How long after the open the after-open mail (the decision, the
        orders and the close briefing, ``brief``) goes out.
    open_cutoff_minutes, close_cutoff_minutes, fallback_minutes, deadline_minutes : int
        The execution plan's cut-offs (``ExecutionPlan``).
    start_after : str, default ``"07:15"``
        The earliest New York time the node starts, after the broker
        gateway's daily restart.
    start_margin_minutes : int, default 20
        How long before the plan's execution time the node must be up: the
        latest start is the execution time minus this margin.
    prepare_every_minutes : int, default 15
        How often quantlab's data and prediction are polled, until the
        latest start.
    retry_minutes : int, default 5
        How long after a failed start (or settle) the next one is tried.
    settle_minutes : int, default 60
        How long after the close a missing outcome is still settled.
    heartbeat_stale_minutes : int, default 10
        How old the supervisor's heartbeat may be before it alerts.
    store : str or pathlib.Path, optional
        The run's live prediction store; ``live_predictions`` in the live
        directory by default.
    may_lag : list of path, optional
        Stores the run reads that may end before the bar
        (``quantlab_gate.prepare_day``).
    cpus : str
        The CPU list every job of the day is bound to (NUMA node 1).
    redis_host, redis_port, redis_db :
        The node's cache database. NautilusTrader 2.0.0rc8 cannot select a
        Redis database, so ``redis_db`` must stay 0: paper and real money
        use separate Redis instances.
    redis : bool, default True
        Whether the node keeps its cache in Redis; a node without it
        restores nothing at a restart (tests only).
    gateway_host : str, default ``"127.0.0.1"``
        The IB Gateway's API host.
    gateway_port : int, optional
        The IB Gateway's API port; an IB profile requires it.
    client_id : int, default 1
        The IB API client id of the node's data client; its execution
        client takes the next one.
    account_id : str, optional
        The IB account; the ``TWS_ACCOUNT`` environment variable by default.
    starting_cash : float, optional
        The ``ib_sandbox`` account's starting cash.
    stop_after_close_minutes : int, default 10
        How long after the close the trading node stops.
    min_node_seconds : int, default 15
        The least a node runs after its start, so a settle-only node, or a
        node started after its stop time, still restores and reconciles.
    trader_id : str, optional
        The node's fixed trader id; derived from the live directory's name.

    Examples
    --------
    >>> config = LiveConfig(run="run", at="close_auction")
    >>> config.plan.at, config.start_margin_minutes
    ('close_auction', 20)
    >>> LiveConfig(run="run", at="noon")
    Traceback (most recent call last):
    ...
    ValueError: at must be 'open_auction', 'close_auction' or 'HH:MM', not 'noon'
    """

    run: str | Path
    initial_capital: float | None = None
    at: str = "open_auction"
    order: str = "market"
    band_bps: float | None = None
    broker: str = "ibkr_paper"
    real_money: bool = False
    max_notional_per_order: float | None = None
    max_order_submit_rate: str | None = None
    account: str | Path | None = None
    after_open_minutes: int = 15
    open_cutoff_minutes: int = 10
    close_cutoff_minutes: int = 15
    fallback_minutes: int = 5
    deadline_minutes: int = 30
    start_after: str = "07:15"
    start_margin_minutes: int = 20
    prepare_every_minutes: int = 15
    retry_minutes: int = 5
    settle_minutes: int = 60
    heartbeat_stale_minutes: int = 10
    store: str | Path | None = None
    may_lag: tuple[str, ...] = ()
    cpus: str = NUMA_NODE_1_CPUS
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    trader_id: str | None = None
    redis: bool = True
    gateway_host: str = "127.0.0.1"
    gateway_port: int | None = None
    client_id: int = 1
    account_id: str | None = None
    starting_cash: float | None = None
    stop_after_close_minutes: int = 10
    min_node_seconds: int = 15
    #: Not a key: the directory the paths are relative to.
    live_dir: Path = field(default=Path("."), repr=False)

    def __post_init__(self):
        self.plan  # noqa: B018 - validates the plan
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", self.start_after):
            raise ValueError(f"start_after must be 'HH:MM', not {self.start_after!r}")
        if self.redis_db != 0:
            raise ValueError("redis_db must be 0: NautilusTrader 2.0.0rc8 cannot select a Redis database")
        if self.max_notional_per_order is not None and not self.max_notional_per_order > 0:
            raise ValueError(f"max_notional_per_order must be positive, not {self.max_notional_per_order!r}")
        self.risk_engine_config()  # validates the submit rate

    @property
    def plan(self) -> ExecutionPlan:
        """The execution plan of ``ScheduledExecution``."""
        return ExecutionPlan(
            self.at, order=self.order, band_bps=self.band_bps,
            open_cutoff_minutes=self.open_cutoff_minutes, close_cutoff_minutes=self.close_cutoff_minutes,
            fallback_minutes=self.fallback_minutes, deadline_minutes=self.deadline_minutes,
        )

    @property
    def run_dir(self) -> Path:
        """The quantlab run, resolved against the live directory."""
        return (self.live_dir / self.run).absolute()

    @property
    def account_path(self) -> Path:
        """The account history, resolved against the live directory."""
        return (self.live_dir / (self.account or "account.toml")).absolute()

    @property
    def store_dir(self) -> Path:
        """The live prediction store, resolved against the live directory."""
        return (self.live_dir / (self.store or "live_predictions")).absolute()

    @property
    def node_trader_id(self) -> str:
        """The node's trader id: set, or ``QL-<NAME>`` from the live directory's name.

        Examples
        --------
        >>> LiveConfig(run="run", live_dir=Path("/live/us3000_sel0-paper")).node_trader_id
        'QL-US3000SEL0PAPER'
        """
        if self.trader_id:
            return self.trader_id
        name = re.sub(r"[^A-Za-z0-9]", "", self.live_dir.absolute().name).upper()
        return f"QL-{name or 'LIVE'}"

    @property
    def ib_account(self) -> str | None:
        """The IB account: ``account_id``, or the ``TWS_ACCOUNT`` environment variable."""
        return self.account_id or os.environ.get("TWS_ACCOUNT") or None

    def profile(self) -> BrokerProfile:
        """The broker profile, connected as configured (``brokers.profile``)."""
        return brokers.profile(
            self.broker, host=self.gateway_host, port=self.gateway_port, client_id=self.client_id,
            account_id=self.ib_account, starting_cash=self.starting_cash,
        )

    def switches(self) -> Switches:
        """The real-money switches (``guards.Switches``), with the arming file's state read now."""
        return Switches(
            real_money=self.real_money, arming=arming_state(self.live_dir / ARMING_FILE),
            initial_capital=self.initial_capital, max_notional_per_order=self.max_notional_per_order,
            max_order_submit_rate=self.max_order_submit_rate, redis=self.redis,
        )

    def risk_engine_config(self, instrument_ids=()) -> LiveRiskEngineConfig:
        """The node's risk engine: the submit rate, and the notional cap on each of ``instrument_ids``.

        NautilusTrader caps the notional per instrument only, so the cap is
        set on every instrument the node may trade (the security map's).

        Examples
        --------
        >>> config = LiveConfig(run="run", max_notional_per_order=2_000.0, max_order_submit_rate="10/00:00:01")
        >>> risk = config.risk_engine_config(["AAA=STK.SMART"])
        >>> risk.max_order_submit_rate, risk.max_notional_per_order
        ('10/00:00:01', {'AAA=STK.SMART': '2000'})
        >>> LiveConfig(run="run", max_order_submit_rate="often")
        Traceback (most recent call last):
        ...
        ValueError: max_order_submit_rate must be 'N/HH:MM:SS', not 'often'
        """
        if self.max_order_submit_rate is not None and not re.fullmatch(
            r"[1-9]\d*/\d{2}:[0-5]\d:[0-5]\d", self.max_order_submit_rate
        ):
            raise ValueError(f"max_order_submit_rate must be 'N/HH:MM:SS', not {self.max_order_submit_rate!r}")
        caps = {}
        if self.max_notional_per_order is not None:
            caps = {str(i): f"{self.max_notional_per_order:f}".rstrip("0").rstrip(".") for i in instrument_ids}
        return LiveRiskEngineConfig(max_order_submit_rate=self.max_order_submit_rate, max_notional_per_order=caps)

    def effective(self) -> dict:
        """Every key with its effective value, defaults included, paths resolved.

        Examples
        --------
        >>> effective = LiveConfig(run="run", live_dir=Path("/live/a")).effective()
        >>> effective["run"], effective["start_margin_minutes"], effective["trader_id"]
        ('/live/a/run', 20, 'QL-A')
        """
        values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "live_dir"}
        values.update(
            run=str(self.run_dir), store=str(self.store_dir), trader_id=self.node_trader_id,
            account=str(self.account_path),
            may_lag=[str((self.live_dir / p).absolute()) for p in self.may_lag],
        )
        return values


def load_config(live_dir) -> LiveConfig:
    """Read ``<live_dir>/live.toml``.

    Raises
    ------
    FileNotFoundError
        If the live directory has no ``live.toml``.
    ValueError
        For an unknown key, a missing ``run`` or an invalid plan.

    Examples
    --------
    >>> import tempfile
    >>> live_dir = Path(tempfile.mkdtemp())
    >>> _ = (live_dir / "live.toml").write_text('run = "../run"\\nat = "15:30"\\n')
    >>> load_config(live_dir).plan.at
    '15:30'
    >>> _ = (live_dir / "live.toml").write_text('run = "../run"\\nmax_capital = 1\\n')
    >>> load_config(live_dir)
    Traceback (most recent call last):
    ...
    ValueError: unknown keys in .../live.toml: max_capital
    """
    live_dir = Path(live_dir)
    path = live_dir / CONFIG_FILE
    with open(path, "rb") as file:
        values = tomllib.load(file)
    known = {f.name for f in dataclasses.fields(LiveConfig)} - {"live_dir"}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown keys in {path}: {', '.join(unknown)}")
    if "run" not in values:
        raise ValueError(f"{path} names no quantlab run (run = ...)")
    if "may_lag" in values:
        values["may_lag"] = tuple(values["may_lag"])
    return LiveConfig(**values, live_dir=live_dir)
