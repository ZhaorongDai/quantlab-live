"""The four ports of a live day and their production adapters.

- ``prepare(date) -> Ready | Pending``: quantlab's data and prediction
  (``quantlab_gate.prepare_day``);
- ``launch_node``: ``start(date, mode, attempt) -> handle`` starts a node
  and ``poll(handle) -> StepReport | None`` says whether it ended
  (``SubprocessLauncher``: a fresh ``node`` process per attempt);
- ``mailer``: ``send(subject, body)`` (``SmtpMailer``);
- ``clock``: ``now()`` and ``sleep(seconds)`` (``WallClock``).

The tests' adapters are fakes of the same shape. There is no broker port
(NautilusTrader's client factories are the broker seam) and no calendar
port (the XNYS calendar is fixed).
"""

from __future__ import annotations

import json
import os
import smtplib
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol

import pandas as pd

from quantlab_live import records
from quantlab_live.live.config import LiveConfig
from quantlab_live.quantlab_gate import Pending, Ready, prepare_day

STEPS_DIR = "steps"
#: The node's process: ``python -m quantlab_live node LIVE_DIR DATE --mode M --report PATH``.
NODE_PROGRAM = (sys.executable, "-m", "quantlab_live", "node")


class NodeLauncher(Protocol):
    def start(self, date: pd.Timestamp, mode: str, attempt: int) -> str: ...

    def poll(self, handle: str) -> records.StepReport | None: ...


class Mailer(Protocol):
    def send(self, subject: str, body: str) -> None: ...


class Clock(Protocol):
    def now(self) -> pd.Timestamp: ...

    def sleep(self, seconds: float) -> None: ...


@dataclass
class Ports:
    """The four ports of ``day``.

    Examples
    --------
    ::

        day(live_dir, Ports(prepare=..., launch_node=..., mailer=..., clock=...))
    """

    prepare: Callable[[pd.Timestamp], Ready | Pending]
    launch_node: NodeLauncher
    mailer: Mailer
    clock: Clock


def default_ports(live_dir, config: LiveConfig) -> Ports:
    """The production adapters of a live directory's ports."""

    def prepare(date):
        return prepare_day(config.run_dir, date, store=config.store_dir, may_lag=config.may_lag, cpus=config.cpus)

    return Ports(
        prepare=prepare,
        launch_node=SubprocessLauncher(live_dir, cpus=config.cpus),
        mailer=SmtpMailer.from_env_file(),
        clock=WallClock(),
    )


class SubprocessLauncher:
    """Starts each node in a fresh process bound with ``taskset``; its output goes to the day's log.

    The node writes its step report to ``steps/<date>/<attempt>-<mode>.json``,
    a path it gets on its command line. The handle names the process and
    the report, so a supervisor that restarts after a crash polls the node
    an earlier supervisor started. A process that ends without a report
    did not start.

    Parameters
    ----------
    live_dir : path-like
        The live directory.
    cpus : str
        The ``taskset`` CPU list.
    program : sequence of str
        The node's command before its arguments (``NODE_PROGRAM``).

    Examples
    --------
    ::

        launcher = SubprocessLauncher(live_dir, cpus="64-127")
        handle = launcher.start(pd.Timestamp("2024-01-03"), "trade", 1)
        while (report := launcher.poll(handle)) is None:
            time.sleep(60)
    """

    def __init__(self, live_dir, cpus: str, program: Sequence[str] = NODE_PROGRAM):
        self.live_dir = Path(live_dir).absolute()
        self.cpus = cpus
        self.program = tuple(program)
        self._children: dict[int, subprocess.Popen] = {}

    def start(self, date: pd.Timestamp, mode: str, attempt: int) -> str:
        day = pd.Timestamp(date).date().isoformat()
        report = self.live_dir / STEPS_DIR / day / f"{attempt}-{mode}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        log = self.live_dir / "logs" / f"{day}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "taskset", "-c", self.cpus, *self.program,
            str(self.live_dir), day, "--mode", mode, "--report", str(report),
        ]
        with open(log, "ab") as output:
            child = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
        self._children[child.pid] = child
        return json.dumps({"pid": child.pid, "report": str(report), "mode": mode})

    def poll(self, handle: str) -> records.StepReport | None:
        node = json.loads(handle)
        pid, report = node["pid"], Path(node["report"])
        child = self._children.get(pid)
        if child is not None:
            code = child.poll()
            if code is None:
                return None
            del self._children[pid]
        else:
            # Started by an earlier supervisor: alive while the pid is.
            code = None
            try:
                os.kill(pid, 0)
                return None
            except ProcessLookupError:
                pass
            except PermissionError:
                return None
        if report.exists():
            return records.read_step_report(report)
        exit_text = "an unknown status" if code is None else f"status {code}"
        return records.StepReport(mode=node["mode"], error=f"the node exited with {exit_text} without a step report")


class SmtpMailer:
    """Sends each mail over SMTP with SSL.

    ``from_env_file`` reads ``SMTP_HOST``, ``SMTP_PORT`` (465 by default),
    ``SMTP_USER``, ``SMTP_PASSWORD``, ``MAIL_FROM`` (the user by default)
    and ``MAIL_TO`` from an owner-only ``KEY=VALUE`` file.

    Examples
    --------
    >>> import tempfile
    >>> env = Path(tempfile.mkdtemp()) / "mail.env"
    >>> _ = env.write_text("SMTP_HOST=smtp.example.com\\nSMTP_USER=me@example.com\\nSMTP_PASSWORD=x\\n"
    ...                    "MAIL_TO=me@example.com\\n")
    >>> mailer = SmtpMailer.from_env_file(env)
    >>> mailer.host, mailer.port, mailer.sender
    ('smtp.example.com', 465, 'me@example.com')
    """

    def __init__(self, host: str, port: int, user: str, password: str, sender: str, to: str):
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender, self.to = sender, to

    @classmethod
    def from_env_file(cls, path=None) -> SmtpMailer | UnconfiguredMailer:
        """Read the mail settings; a missing file gives a mailer whose every send fails.

        ``path`` defaults to ``mail.env`` in ``$XDG_CONFIG_HOME/quantlab``
        (``~/.config/quantlab``).
        """
        if path is None:
            path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "quantlab" / "mail.env"
        path = Path(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            return UnconfiguredMailer(f"no mail settings at {path}: {error}")
        env = {}
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, text = line.removeprefix("export ").partition("=")
                env[key.strip()] = text.strip().strip("'\"")
        try:
            return cls(
                env["SMTP_HOST"], int(env.get("SMTP_PORT", 465)), env["SMTP_USER"], env["SMTP_PASSWORD"],
                env.get("MAIL_FROM", env["SMTP_USER"]), env["MAIL_TO"],
            )
        except KeyError as missing:
            return UnconfiguredMailer(f"the mail settings at {path} lack {missing}")

    def send(self, subject: str, body: str) -> None:
        message = EmailMessage()
        message["Subject"], message["From"], message["To"] = subject, self.sender, self.to
        message.set_content(body)
        with smtplib.SMTP_SSL(self.host, self.port, timeout=60) as smtp:
            smtp.login(self.user, self.password)
            smtp.send_message(message)


class UnconfiguredMailer:
    """A mailer without settings: every send raises, which the supervisor logs.

    Examples
    --------
    >>> UnconfiguredMailer("no settings").send("subject", "body")
    Traceback (most recent call last):
    ...
    RuntimeError: no settings
    """

    def __init__(self, reason: str):
        self.reason = reason

    def send(self, subject: str, body: str) -> None:
        raise RuntimeError(self.reason)


class WallClock:
    """The real time, in UTC.

    Examples
    --------
    >>> str(WallClock().now().tz)
    'UTC'
    """

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))
