"""A live directory's configuration: one small ``live.toml``, everything else defaulted.

The keys an operator sets are ``run`` (the quantlab run, relative to the live
directory or absolute), ``initial_capital``, ``at``, ``order`` (with
``band_bps`` for a limit plan), ``broker`` and ``real_money``. Every other
key has a default, and ``status`` prints them all, so a default never hides.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

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
        One of the two switches a real-money start needs.
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
        The node's cache database.
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
    #: Not a key: the directory the paths are relative to.
    live_dir: Path = field(default=Path("."), repr=False)

    def __post_init__(self):
        self.plan  # noqa: B018 - validates the plan
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", self.start_after):
            raise ValueError(f"start_after must be 'HH:MM', not {self.start_after!r}")

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
