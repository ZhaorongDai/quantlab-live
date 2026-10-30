"""The supervisor of a live day: ``day(live_dir)``, the hourly cron entry.

A trading day runs these steps, in order:

1. prepare: wait for quantlab's data update and run the prediction job,
   polled every 15 minutes until the cut-off (the latest start);
2. value: the latest snapshot at quantlab's close, appended to the strategy
   equity before the node sizes on it;
3. start the node, retried in a fresh process every 5 minutes until the
   latest start (the plan's execution time minus the start margin); a
   node that crashes is restarted at once, as often as needed, until the
   close;
4. the node trades and stops after the close; once the open is
   ``after_open_minutes`` past, the after-open mail goes out (``brief``:
   the decision, its orders and the previous close's briefing);
5. settle, with a settle-only node, while the day's outcomes are missing;
6. the post-close mail, after ``metrics.json`` and ``report.html`` are
   written from the records, and the account history starts a strategy
   segment when the day's orders were the strategy's first to reach the
   broker (``records.record_segment``).

``what_now(now, schedule, state)`` is the pure function that picks the next
step; ``day`` takes the day lock, resumes the day state and runs the steps
through four ports (``Ports``): ``prepare``, ``launch_node``, ``mailer`` and
``clock``. Only the supervisor writes the day state, a heartbeat included;
the node reports through its own step report (``records.StepReport``) and
the records it writes. A mail that fails never fails a step.

Files of a live directory, per New York date: ``day/<date>.json`` (the day
state), ``day/<date>.lock`` (the day lock), ``steps/<date>/`` (the node's
step reports) and ``logs/<date>.log`` (the supervisor's and the node's log).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import traceback
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import pandas as pd

from quantlab_live import records
from quantlab_live.live.brief import briefing
from quantlab_live.live.config import LiveConfig, load_config
from quantlab_live.live.ports import Ports, default_ports
from quantlab_live.live.value import value
from quantlab_live.quantlab_gate import Ready, open_run
from quantlab_live.trading.scheduled_execution import NEW_YORK, Session

DAY_DIR = "day"
LOG_DIR = "logs"
#: How often the supervisor beats while it waits.
HEARTBEAT_EVERY = pd.Timedelta(minutes=1)
TRADE, SETTLE = "trade", "settle"


# ---------------------------------------------------------------- schedule


@dataclass(frozen=True)
class DaySchedule:
    """The times of one New York date's live day, in UTC.

    Attributes
    ----------
    date : pandas.Timestamp
        The New York date.
    session : Session or None
        The XNYS session; ``None`` on a day the exchange is closed.
    start_after : pandas.Timestamp
        The earliest node start.
    latest_start : pandas.Timestamp
        The plan's execution time minus the start margin; also the cut-off
        of the data and the prediction.
    after_open : pandas.Timestamp
        When the after-open mail goes out.
    close : pandas.Timestamp
        The session's close, early closes included; crashes are restarted
        until then.
    settle_until : pandas.Timestamp
        The last settle start.
    prepare_every, retry_every : pandas.Timedelta
        The prepare poll and the start retry intervals.

    Examples
    --------
    >>> schedule = DaySchedule.of("2024-07-03", LiveConfig(run="run", at="close_auction"))
    >>> schedule.latest_start.tz_convert("America/New_York").strftime("%H:%M")
    '12:25'
    """

    date: pd.Timestamp
    session: Session | None
    start_after: pd.Timestamp | None = None
    latest_start: pd.Timestamp | None = None
    after_open: pd.Timestamp | None = None
    close: pd.Timestamp | None = None
    settle_until: pd.Timestamp | None = None
    prepare_every: pd.Timedelta = pd.Timedelta(minutes=15)
    retry_every: pd.Timedelta = pd.Timedelta(minutes=5)

    @classmethod
    def of(cls, date, config: LiveConfig) -> DaySchedule:
        """Return the schedule of the New York date ``date`` under ``config``."""
        date = pd.Timestamp(date).normalize().tz_localize(None)
        try:
            session = Session.on(date)
        except ValueError:
            return cls(date, None)
        plan = config.plan
        if plan.at == "open_auction":
            execution = session.open - pd.Timedelta(minutes=plan.open_cutoff_minutes)
        elif plan.at == "close_auction":
            execution = session.close - pd.Timedelta(minutes=plan.close_cutoff_minutes)
        else:
            execution = session.at(plan.at)
        return cls(
            date,
            session,
            start_after=session.at(config.start_after),
            latest_start=execution - pd.Timedelta(minutes=config.start_margin_minutes),
            after_open=session.open + pd.Timedelta(minutes=config.after_open_minutes),
            close=session.close,
            settle_until=session.close + pd.Timedelta(minutes=config.settle_minutes),
            prepare_every=pd.Timedelta(minutes=config.prepare_every_minutes),
            retry_every=pd.Timedelta(minutes=config.retry_minutes),
        )


# ------------------------------------------------------------------- state


@dataclass(frozen=True)
class Attempt:
    """One node process of the day: a trading node or a settle-only node.

    ``handle`` is the launcher's handle while the process runs, ``None``
    once it ended; then ``started``, ``finished``, ``outcomes`` and
    ``error`` come from its step report.
    """

    mode: str
    at: pd.Timestamp
    handle: str | None = None
    ended: pd.Timestamp | None = None
    started: bool = False
    finished: bool = False
    outcomes: bool = False
    error: str | None = None

    @property
    def running(self) -> bool:
        return self.ended is None


@dataclass(frozen=True)
class DayState:
    """The state of one live day, written only by its supervisor.

    Attributes
    ----------
    date : str
        The New York date.
    heartbeat : pandas.Timestamp, optional
        The supervisor's last beat.
    prepared_at, pending :
        The last prepare poll and why it was pending.
    bar : pandas.Timestamp, optional
        The bar prepared (quantlab's last bar, decided on today).
    valued : bool
        The latest snapshot is valued.
    briefed : bool
        The after-open mail went out (or failed to).
    attempts : tuple of Attempt
        The node processes so far.
    alerted : tuple of str
        The alerts already mailed, so each goes out once.
    mails : tuple of dict
        Every mail tried: its subject and whether it went out.
    done : str, optional
        How the day ended.

    Examples
    --------
    >>> DayState("2024-01-03").done is None
    True
    """

    date: str
    heartbeat: pd.Timestamp | None = None
    prepared_at: pd.Timestamp | None = None
    pending: str | None = None
    bar: pd.Timestamp | None = None
    valued: bool = False
    briefed: bool = False
    attempts: tuple[Attempt, ...] = ()
    alerted: tuple[str, ...] = ()
    mails: tuple[dict, ...] = ()
    done: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=_iso, indent=1)

    @classmethod
    def from_json(cls, text: str) -> DayState:
        raw = json.loads(text)
        raw["attempts"] = tuple(Attempt(**_timestamps(a, ("at", "ended"))) for a in raw["attempts"])
        raw = _timestamps(raw, ("heartbeat", "prepared_at", "bar"))
        raw["alerted"], raw["mails"] = tuple(raw["alerted"]), tuple(raw["mails"])
        return cls(**{f.name: raw[f.name] for f in fields(cls) if f.name in raw})


def _iso(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _timestamps(raw: dict, names) -> dict:
    return {k: (pd.Timestamp(v) if k in names and v is not None else v) for k, v in raw.items()}


# ------------------------------------------------------------------- steps


@dataclass(frozen=True)
class Done:
    """Nothing more to do today."""

    reason: str


@dataclass(frozen=True)
class Wait:
    """Nothing to do before ``until``."""

    until: pd.Timestamp
    why: str = ""


@dataclass(frozen=True)
class Prepare:
    """Poll quantlab's data update and prediction (``prepare``)."""


@dataclass(frozen=True)
class Value:
    """Value the latest snapshot at quantlab's close."""


@dataclass(frozen=True)
class Start:
    """Start a node: ``"trade"`` or ``"settle"`` (settle-only)."""

    mode: str


@dataclass(frozen=True)
class AfterOpen:
    """Mail the decision, its orders and the previous close's briefing."""


@dataclass(frozen=True)
class NotTraded:
    """The day cannot trade; mail why and end it."""

    reason: str


@dataclass(frozen=True)
class PostClose:
    """Mail the day's fills and outcomes and end it."""


Step = Done | Wait | Prepare | Value | Start | AfterOpen | NotTraded | PostClose


def what_now(now: pd.Timestamp, schedule: DaySchedule, state: DayState) -> Step:
    """Return the supervisor's next step at ``now``.

    A pure function of the time, the day's schedule and its state; the
    supervisor runs the step and asks again. The after-open mail takes the
    place of a wait, once a node was started and the open is
    ``after_open_minutes`` past.

    Examples
    --------
    >>> config = LiveConfig(run="run")
    >>> now = pd.Timestamp("2024-01-03 06:00", tz="America/New_York")
    >>> what_now(now, DaySchedule.of("2024-01-03", config), DayState("2024-01-03"))
    Prepare()
    >>> what_now(now, DaySchedule.of("2024-07-04", config), DayState("2024-07-04"))
    Done(reason='2024-07-04 is not an XNYS trading day')
    """
    step = _next(now, schedule, state)
    if isinstance(step, Wait) and state.valued and not state.briefed and state.attempts and now >= schedule.after_open:
        # The after-open mail goes out while the supervisor would only wait.
        return AfterOpen()
    return step


def _next(now: pd.Timestamp, schedule: DaySchedule, state: DayState) -> Step:
    if schedule.session is None:
        return Done(f"{schedule.date.date()} is not an XNYS trading day")
    if state.done is not None:
        return Done(state.done)
    if state.attempts and state.attempts[-1].running:
        return Wait(now + HEARTBEAT_EVERY, f"the {state.attempts[-1].mode} node is running")
    if state.bar is None:
        if now >= schedule.latest_start:
            return NotTraded(
                f"quantlab's data or prediction was not ready by the cut-off {_ny(schedule.latest_start)}: "
                f"{state.pending or 'never polled'}"
            )
        if state.prepared_at is None or now >= state.prepared_at + schedule.prepare_every:
            return Prepare()
        return Wait(min(state.prepared_at + schedule.prepare_every, schedule.latest_start), "data pending")
    if not state.valued:
        return Value()

    trades = [a for a in state.attempts if a.mode == TRADE]
    settles = [a for a in state.attempts if a.mode == SETTLE]
    if not any(a.finished for a in trades) and now < schedule.close:
        if any(a.started for a in trades):
            # A crash: restart at once; a restart that failed to start waits its retry.
            last = trades[-1]
            due = last.ended if last.started else last.at + schedule.retry_every
            return Start(TRADE) if now >= due else Wait(min(due, schedule.close), "restart retry")
        if now >= schedule.latest_start:
            error = trades[-1].error if trades else "no start was tried"
            return NotTraded(f"the node did not start by the latest start {_ny(schedule.latest_start)}: {error}")
        if now < schedule.start_after:
            return Wait(schedule.start_after, "the earliest start")
        if trades and now < trades[-1].at + schedule.retry_every:
            return Wait(min(trades[-1].at + schedule.retry_every, schedule.latest_start), "start retry")
        return Start(TRADE)

    if any(a.outcomes for a in state.attempts):
        return PostClose()
    if now < schedule.close:
        return Wait(schedule.close, "settle after the close")
    if now >= schedule.settle_until:
        return PostClose()
    if settles and now < settles[-1].at + schedule.retry_every:
        return Wait(min(settles[-1].at + schedule.retry_every, schedule.settle_until), "settle retry")
    return Start(SETTLE)


def _ny(t: pd.Timestamp) -> str:
    return t.tz_convert(NEW_YORK).strftime("%H:%M")


# -------------------------------------------------------------- the lock


class SupervisorRunning(RuntimeError):
    """The day lock is held: a supervisor (or a manual command) runs for the day.

    Examples
    --------
    >>> issubclass(SupervisorRunning, RuntimeError)
    True
    """


class _DayLock:
    """One holder per live directory and day: an exclusive ``flock`` on ``day/<date>.lock``.

    The lock is released when its holder's process ends, however it ends.
    The file names the holder, for the message of a refused call.
    """

    def __init__(self, live_dir: Path, date: pd.Timestamp):
        self.path = live_dir / DAY_DIR / f"{date.date().isoformat()}.lock"
        self._fd: int | None = None

    def acquire(self, holder: str) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"holder": holder, "pid": os.getpid()}).encode())
        self._fd = fd
        return True

    def holder(self) -> str:
        try:
            raw = json.loads(self.path.read_text())
            return f"{raw['holder']} (pid {raw['pid']})"
        except (OSError, ValueError, KeyError):
            return "another process"

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


@contextlib.contextmanager
def manual(live_dir, date, command: str) -> Iterator[Path]:
    """Hold the day lock for a manual command (``run-node``, ``brief``, ``status``).

    Raises
    ------
    SupervisorRunning
        If a supervisor or another manual command holds the lock.

    Examples
    --------
    >>> import tempfile
    >>> live_dir = Path(tempfile.mkdtemp())
    >>> with manual(live_dir, "2024-01-03", "status"):
    ...     with manual(live_dir, "2024-01-03", "brief"):
    ...         pass
    Traceback (most recent call last):
    ...
    quantlab_live.live.supervisor.SupervisorRunning: brief refused: the day 2024-01-03 is held by status (pid ...)
    """
    live_dir, date = Path(live_dir), _day(date)
    lock = _DayLock(live_dir, date)
    if not lock.acquire(command):
        raise SupervisorRunning(f"{command} refused: the day {date.date()} is held by {lock.holder()}")
    try:
        yield live_dir
    finally:
        lock.release()


# ------------------------------------------------------------ the commands


def day(live_dir, ports: Ports | None = None) -> str:
    """Run today's live day under the day lock, resuming its state; the hourly cron entry.

    The day is the New York date of the clock's now. A call that finds the
    lock held exits at once, and mails when the holder's heartbeat is stale.
    A call that takes the lock resumes the day from its state file (mailing
    when the previous supervisor's heartbeat had gone stale) and runs the
    steps of ``what_now`` until the day is done.

    Parameters
    ----------
    live_dir : path-like
        The live directory, with its ``live.toml``.
    ports : Ports, optional
        The day's four ports; the production adapters by default.

    Returns
    -------
    str
        How the call ended: ``"traded"``, ``"not traded: ..."``, ``"locked
        by ..."`` or why the day has nothing to run.

    Examples
    --------
    ::

        # crontab: 0 * * * *  taskset -c 64-127 .venv/bin/python -m quantlab_live day LIVE_DIR
        day(live_dir)
    """
    live_dir = Path(live_dir)
    config = load_config(live_dir)
    ports = ports or default_ports(live_dir, config)
    now = ports.clock.now()
    date = _day(now.tz_convert(NEW_YORK).tz_localize(None))
    schedule = DaySchedule.of(date, config)
    if schedule.session is None:
        return f"{date.date()} is not an XNYS trading day"
    supervisor = _Supervisor(live_dir, config, schedule, ports)
    lock = _DayLock(live_dir, date)
    if not lock.acquire("supervisor"):
        state = supervisor.read_state()
        if state is not None and state.done is None and supervisor.stale(state, now):
            supervisor.mail_once(
                "the supervisor's heartbeat is stale",
                f"The day is held by {lock.holder()}, whose last heartbeat was at {_ny(state.heartbeat)} New York.",
            )
        return f"locked by {lock.holder()}"
    try:
        return supervisor.run()
    finally:
        lock.release()


def status(live_dir, date=None) -> str:
    """Return the day's state and the effective configuration, defaults included, as text.

    ``status`` takes the day lock and writes nothing.

    Raises
    ------
    SupervisorRunning
        While a supervisor runs.

    Examples
    --------
    >>> import tempfile
    >>> live_dir = Path(tempfile.mkdtemp())
    >>> _ = (live_dir / "live.toml").write_text('run = "run"\\n')
    >>> print(status(live_dir, "2024-01-03"))
    # day 2024-01-03: no state yet
    ...
    start_margin_minutes = 20
    ...
    """
    live_dir = Path(live_dir)
    date = _day(pd.Timestamp.now(tz=NEW_YORK).tz_localize(None) if date is None else date)
    with manual(live_dir, date, "status"):
        config = load_config(live_dir)
        path = _state_path(live_dir, date)
        lines = [f"# day {date.date()}: " + ("state" if path.exists() else "no state yet")]
        if path.exists():
            lines.append(path.read_text(encoding="utf-8"))
        lines.append(f"# effective configuration of {live_dir / 'live.toml'}")
        for key, item in config.effective().items():
            lines.append(f"# {key} is unset" if item is None else f"{key} = {json.dumps(item)}")
        return "\n".join(lines)


# --------------------------------------------------------- the supervisor


class _Supervisor:
    """Runs the steps of one day; the only writer of its state."""

    def __init__(self, live_dir: Path, config: LiveConfig, schedule: DaySchedule, ports: Ports):
        self.live_dir, self.config, self.schedule, self.ports = live_dir, config, schedule, ports
        self.date = schedule.date
        self.state_path = _state_path(live_dir, self.date)
        self.log_path = live_dir / LOG_DIR / f"{self.date.date().isoformat()}.log"
        self.state: DayState | None = None

    # -- state

    def read_state(self) -> DayState | None:
        if not self.state_path.exists():
            return None
        return DayState.from_json(self.state_path.read_text(encoding="utf-8"))

    def _write(self, state: DayState) -> None:
        self.state = state
        records._write_text(self.state_path, state.to_json())

    def stale(self, state: DayState, now: pd.Timestamp) -> bool:
        limit = pd.Timedelta(minutes=self.config.heartbeat_stale_minutes)
        return state.heartbeat is not None and now - state.heartbeat > limit

    # -- the loop

    def run(self) -> str:
        now = self.ports.clock.now()
        state = self.read_state()
        if state is None:
            state = DayState(self.date.date().isoformat())
            self.log(f"supervisor starts the day ({self.config.at}, latest start {_ny(self.schedule.latest_start)})")
        else:
            self.state = state
            if state.done is None and self.stale(state, now):
                state = self._mail(
                    state, "the supervisor was down, resuming",
                    f"The day's last heartbeat was at {_ny(state.heartbeat)} New York; resumed at {_ny(now)}.",
                )
            self.log("supervisor resumes the day")
        while True:
            now = self.ports.clock.now()
            state = self._poll_node(now, state)
            state = self._alert_skipped(state)
            step = what_now(now, self.schedule, state)
            state = replace(state, heartbeat=now)
            if isinstance(step, Done):
                self._write(state)
                return step.reason
            if isinstance(step, Wait):
                self._write(state)
                pause = min(step.until - now, HEARTBEAT_EVERY)
                self.ports.clock.sleep(max(pause, pd.Timedelta(seconds=1)).total_seconds())
                continue
            self.log(f"step: {type(step).__name__} {getattr(step, 'mode', '')}".rstrip())
            state = self._run_step(step, now, state)
            self._write(state)

    def _run_step(self, step: Step, now: pd.Timestamp, state: DayState) -> DayState:
        if isinstance(step, Prepare):
            try:
                outcome = self.ports.prepare(self.date)
            except Exception as error:  # noqa: BLE001 - prepare failing ends the day, with a mail
                return self._not_traded(state, f"preparing the day failed: {error!r}")
            if isinstance(outcome, Ready):
                self.log(f"prepared bar {outcome.bar.date()}")
                return replace(state, prepared_at=now, bar=pd.Timestamp(outcome.bar), pending=None)
            self.log(f"prepare pending: {outcome.reason}")
            return replace(state, prepared_at=now, pending=outcome.reason)
        if isinstance(step, Value):
            try:
                valued = value(self.live_dir, self.date)
            except Exception as error:  # noqa: BLE001
                return self._not_traded(state, f"valuing the latest snapshot failed: {error!r}")
            self.log("valued: nothing to value" if valued is None else f"valued the snapshot of {valued.date()}")
            return replace(state, valued=True)
        if isinstance(step, Start):
            number = len(state.attempts) + 1
            try:
                handle = self.ports.launch_node.start(self.date, step.mode, number)
            except Exception as error:  # noqa: BLE001 - a launch failing is a failed start
                self.log(f"{step.mode} node {number} failed to launch: {error!r}")
                failed = Attempt(step.mode, now, ended=now, error=f"the launch failed: {error!r}")
                return replace(state, attempts=state.attempts + (failed,))
            self.log(f"{step.mode} node {number} launched")
            return replace(state, attempts=state.attempts + (Attempt(step.mode, now, handle=handle),))
        if isinstance(step, AfterOpen):
            return self._after_open(state)
        if isinstance(step, NotTraded):
            return self._not_traded(state, step.reason)
        if isinstance(step, PostClose):
            return self._post_close(state)
        raise TypeError(f"unknown step {step!r}")

    # -- the node

    def _poll_node(self, now: pd.Timestamp, state: DayState) -> DayState:
        if not state.attempts or not state.attempts[-1].running:
            return state
        last = state.attempts[-1]
        report = self.ports.launch_node.poll(last.handle)
        if report is None:
            return state
        outcomes = report.outcomes or (state.bar is not None and records.has_outcomes(self.live_dir, state.bar))
        ended = replace(
            last, handle=None, ended=now, started=report.started, finished=report.finished,
            outcomes=bool(outcomes), error=report.error,
        )
        state = replace(state, attempts=state.attempts[:-1] + (ended,))
        self.log(
            f"{ended.mode} node {len(state.attempts)} ended: started={ended.started} "
            f"finished={ended.finished} outcomes={ended.outcomes} error={ended.error}"
        )
        for alert in report.alerts:
            state = self._mail(state, "the node alerts", alert, key=f"alert:{alert}")
        if ended.mode == TRADE and ended.started and not ended.finished:
            what = "restarting" if now < self.schedule.close else "settling"
            state = self._mail(
                state, f"the node crashed, {what}",
                f"The trading node crashed at {_ny(now)} New York: {ended.error}\nLog: {self.log_path}",
            )
        return state

    def _alert_skipped(self, state: DayState) -> DayState:
        if state.bar is None or "skipped" in state.alerted or not records.has_decision(self.live_dir, state.bar):
            return state
        decision = records.read_decision(self.live_dir, state.bar)
        skipped = decision[decision["skip"].fillna("") != ""]
        if skipped.empty:
            return replace(state, alerted=state.alerted + ("skipped",))
        lines = [f"{symbol}: {row.skip}, order {row.order_quantity:+g} not sent" for symbol, row in skipped.iterrows()]
        return self._mail(
            state, "unmapped or foreign securities",
            f"The decision on {state.bar.date()} skipped these symbols:\n" + "\n".join(lines), key="skipped",
        )

    def _after_open(self, state: DayState) -> DayState:
        try:
            composed = briefing(self.live_dir, self.date)
            subject, body = composed.subject(), composed.text()
        except Exception as error:  # noqa: BLE001 - the briefing never fails a step
            self.log(f"the briefing failed: {error!r}")
            subject, body = "after the open: NEEDS CHECKING", f"Needs checking: the briefing failed: {error!r}\n"
        return replace(self._mail(state, subject, body), briefed=True)

    # -- the end of the day

    def _not_traded(self, state: DayState, reason: str) -> DayState:
        self.log(f"not traded: {reason}")
        state = self._mail(state, "not traded", f"The day {self.date.date()} is not traded.\n{reason}")
        return replace(state, done=f"not traded: {reason}")

    def _post_close(self, state: DayState) -> DayState:
        bar = state.bar
        if bar is not None and records.has_outcomes(self.live_dir, bar):
            outcomes = records.read_outcomes(self.live_dir, bar)
            unverified = outcomes[outcomes["status"] == "unverified"]
            body = f"Outcomes of the decision on {bar.date()}:\n{outcomes.to_string()}"
            check = None if unverified.empty else f"{len(unverified)} outcomes are unverified"
            self._record_segment(bar, outcomes)
        else:
            body, check = "", f"the day's outcomes are missing after settling until {_ny(self.schedule.settle_until)}"
        self._report()
        subject = "after the close" + (": NEEDS CHECKING" if check else "")
        state = self._mail(state, subject, (f"Needs checking: {check}.\n\n" if check else "") + body)
        return replace(state, done="traded" + (f"; outcomes need checking: {check}" if check else ""))

    def _record_segment(self, bar: pd.Timestamp, outcomes) -> None:
        """Start a strategy segment in the account history if the day's orders start one."""
        segment = records.Segment(
            bar, str(self.live_dir.absolute()), str(self.config.run_dir), self.config.broker,
            self.config.initial_capital,
        )
        try:
            if records.record_segment(self.config.account_path, segment, outcomes):
                self.log(f"a strategy segment starts at {bar.date()} in {self.config.account_path}")
        except Exception as error:  # noqa: BLE001 - the account history never fails a step
            self.log(f"recording the strategy segment failed: {error!r}")

    def _report(self) -> None:
        """Write ``metrics.json`` and ``report.html`` from the records; a failure is only logged."""
        if not (self.live_dir / records.EQUITY).exists():
            return
        try:
            open_run(self.config.run_dir).report(self.live_dir)
            self.log("wrote metrics.json and report.html")
        except Exception as error:  # noqa: BLE001 - the report never fails a step
            self.log(f"the report failed: {error!r}")

    # -- mails and the log

    def _mail(self, state: DayState, subject: str, body: str, *, key: str | None = None) -> DayState:
        """Send a mail; a failure is logged and recorded, never raised."""
        if key is not None and key in state.alerted:
            return state
        subject = f"quantlab-live {self.date.date()}: {subject}"
        try:
            self.ports.mailer.send(subject, body)
            sent, error = True, None
        except Exception as failure:  # noqa: BLE001 - a mail never fails a step
            sent, error = False, repr(failure)
            self.log(f"mail {subject!r} failed: {error}")
        mail = {"subject": subject, "at": self.ports.clock.now().isoformat(), "sent": sent, "error": error}
        alerted = state.alerted + ((key,) if key is not None else ())
        return replace(state, mails=state.mails + (mail,), alerted=alerted)

    def mail_once(self, subject: str, body: str) -> None:
        """Mail from a call that does not hold the lock, and so writes no state."""
        subject = f"quantlab-live {self.date.date()}: {subject}"
        try:
            self.ports.mailer.send(subject, body)
        except Exception as failure:  # noqa: BLE001
            self.log(f"mail {subject!r} failed: {failure!r}")

    def log(self, line: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = self.ports.clock.now().tz_convert(NEW_YORK).strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(f"{stamp} supervisor[{os.getpid()}] {line}\n")


def _state_path(live_dir: Path, date: pd.Timestamp) -> Path:
    return Path(live_dir) / DAY_DIR / f"{date.date().isoformat()}.json"


def _day(date) -> pd.Timestamp:
    t = pd.Timestamp(date)
    return (t.tz_localize(None) if t.tz is not None else t).normalize()
