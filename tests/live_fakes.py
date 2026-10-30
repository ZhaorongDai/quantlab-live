"""Fake adapters of the live day's four ports: a fake clock, a recording mailer,
a scripted node launcher and a scripted ``prepare``.

The clock's ``sleep`` moves time on and runs the test's hooks, so a test can
act while a supervisor runs (call ``day`` again, run a manual command).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from quantlab_live import records
from quantlab_live.quantlab_gate import Pending, Ready


def ny(day: str, hh_mm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hh_mm}", tz="America/New_York").tz_convert("UTC")


class FakeClock:
    def __init__(self, start: pd.Timestamp, *, stop_at: pd.Timestamp | None = None):
        self.t = start
        self.stop_at = stop_at
        self.hooks: list[Callable[[pd.Timestamp], None]] = []

    def now(self) -> pd.Timestamp:
        return self.t

    def sleep(self, seconds: float) -> None:
        assert seconds > 0, seconds
        self.t = self.t + pd.Timedelta(seconds=seconds)
        if self.stop_at is not None and self.t >= self.stop_at:
            raise SupervisorKilled(self.t)
        for hook in list(self.hooks):
            hook(self.t)


class SupervisorKilled(Exception):
    """The fake clock stops the supervisor, as a kill or a reboot would."""


@dataclass
class RecordingMailer:
    failing: bool = False
    mails: list[tuple[str, str]] = field(default_factory=list)

    def send(self, subject: str, body: str) -> None:
        if self.failing:
            raise OSError("the mail server is down")
        self.mails.append((subject, body))

    @property
    def subjects(self) -> list[str]:
        return [subject for subject, _ in self.mails]


class ScriptedPrepare:
    """``Pending`` until ``ready_at`` (never when ``None``), then ``Ready(bar)``."""

    def __init__(self, clock: FakeClock, bar: pd.Timestamp, ready_at: pd.Timestamp | None):
        self.clock, self.bar, self.ready_at = clock, bar, ready_at
        self.calls: list[pd.Timestamp] = []

    def __call__(self, date) -> Ready | Pending:
        self.calls.append(self.clock.now())
        if self.ready_at is not None and self.clock.now() >= self.ready_at:
            return Ready(self.bar)
        return Pending("the data update is running")


# A node's behaviour: (launcher, attempt) -> the report at the current time, or None while it runs.
Behaviour = Callable[["ScriptedLauncher", "Started"], "records.StepReport | None"]


@dataclass
class Started:
    date: pd.Timestamp
    mode: str
    number: int
    at: pd.Timestamp
    behaviour: Behaviour


class ScriptedLauncher:
    """Each start takes the next behaviour of its mode's script; the last repeats."""

    def __init__(self, clock: FakeClock, live_dir, trade: list[Behaviour], settle: list[Behaviour] | None = None):
        self.clock, self.live_dir = clock, live_dir
        self.scripts = {"trade": list(trade), "settle": list(settle or [finishes_at(None)])}
        self.started: list[Started] = []

    def start(self, date, mode, attempt) -> str:
        script = self.scripts[mode]
        behaviour = script.pop(0) if len(script) > 1 else script[0]
        self.started.append(Started(pd.Timestamp(date), mode, attempt, self.clock.now(), behaviour))
        return str(len(self.started) - 1)

    def poll(self, handle: str):
        node = self.started[int(handle)]
        return node.behaviour(self, node)

    def modes(self) -> list[str]:
        return [s.mode for s in self.started]


def fails_to_start(error: str = "the broker gateway refused the connection") -> Behaviour:
    return lambda launcher, node: records.StepReport(node.mode, error=error)


def crashes_at(t: pd.Timestamp) -> Behaviour:
    def behaviour(launcher, node):
        if launcher.clock.now() < t:
            return None
        return records.StepReport(node.mode, started=True, error="Segmentation fault")

    return behaviour


def finishes_at(t: pd.Timestamp | None, *, bar: pd.Timestamp | None = None, outcomes: bool = True, on_start=None):
    """Runs until ``t`` (at once when ``None``), then ends; writes ``bar``'s outcomes when asked."""

    def behaviour(launcher, node):
        if on_start is not None and not getattr(node, "on_start_done", False):
            on_start(launcher.live_dir)
            node.on_start_done = True
        if t is not None and launcher.clock.now() < t:
            return None
        if outcomes and bar is not None:
            frame = pd.DataFrame(
                {"order_quantity": [10.0], "filled_quantity": [10.0], "avg_px": [20.0], "status": ["filled"]},
                index=pd.Index([10001], name="symbol"),
            )
            records.write_outcomes(launcher.live_dir, bar, frame)
        return records.StepReport(node.mode, started=True, finished=True, outcomes=outcomes)

    return behaviour
