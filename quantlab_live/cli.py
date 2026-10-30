"""The command line: ``python -m quantlab_live COMMAND``.

- ``day LIVE_DIR``: the supervisor of today's live day (the hourly cron entry);
- ``status LIVE_DIR [--date D]``: the day's state and the effective configuration;
- ``run-node LIVE_DIR [--date D] [--settle] [--dry-run]``: a node by hand;
- ``brief LIVE_DIR [--date D]``: the after-open mail by hand;
- ``replay RUN_DIR [--out OUT] [--loop closed|open]``: replay a quantlab run.

``status``, ``run-node`` and ``brief`` take the day lock and are refused while
a supervisor runs. ``node`` is the supervisor's own node process: it runs
under the supervisor's lock and is not for use by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

#: Exit status of a command refused because the day lock is held.
REFUSED = 3


def main(argv: list[str] | None = None) -> int:
    """Run one command; return its exit status.

    Examples
    --------
    >>> import tempfile
    >>> live_dir = Path(tempfile.mkdtemp())
    >>> _ = (live_dir / "live.toml").write_text('run = "run"\\n')
    >>> main(["status", str(live_dir), "--date", "2024-01-03"])
    # day 2024-01-03: no state yet
    ...
    0
    """
    parser = argparse.ArgumentParser(prog="python -m quantlab_live")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("day", "status", "run-node", "brief", "node"):
        command = commands.add_parser(name)
        command.add_argument("live_dir", type=Path)
        if name == "node":
            command.add_argument("date")
            command.add_argument("--mode", choices=("trade", "settle"), required=True)
            command.add_argument("--report", type=Path, required=True)
        elif name != "day":
            command.add_argument("--date", default=None, help="the New York date; today by default")
        if name == "run-node":
            command.add_argument("--settle", action="store_true", help="settle only: trade nothing")
            command.add_argument("--dry-run", action="store_true", help="decide and write, submit nothing")
    replay_command = commands.add_parser("replay")
    replay_command.add_argument("run_dir", type=Path)
    replay_command.add_argument("--out", type=Path, default=None)
    replay_command.add_argument("--loop", choices=("closed", "open"), default="closed")
    args = parser.parse_args(argv)

    from quantlab_live import live

    if args.command == "day":
        print(live.day(args.live_dir))
        return 0
    if args.command == "replay":
        from quantlab_live.replay import replay

        print(replay(args.run_dir, args.out, loop=args.loop))
        return 0
    if args.command == "node":
        return _node(args.live_dir, args.date, args.mode, False, args.report)
    date = _today() if args.date is None else pd.Timestamp(args.date).normalize()
    try:
        if args.command == "status":
            print(live.status(args.live_dir, date))
            return 0
        with live.manual(args.live_dir, date, args.command):
            if args.command == "run-node":
                mode = "settle" if args.settle else "trade"
                report = args.live_dir / "steps" / date.date().isoformat() / f"manual-{mode}.json"
                return _node(args.live_dir, date, mode, args.dry_run, report)
            brief = _later("brief")
            brief(args.live_dir, date)
            return 0
    except live.SupervisorRunning as refused:
        print(refused, file=sys.stderr)
        return REFUSED


def _node(live_dir: Path, date, mode: str, dry_run: bool, report: Path) -> int:
    """Run one node process: ``live.run_node``, which writes its step report to ``report``."""
    run_node = _later("run_node")
    run_node(live_dir, pd.Timestamp(date), settle=mode == "settle", dry_run=dry_run, report=report)
    return 0


def _later(name: str):
    """A ``live`` function that a later part of the system provides."""
    from quantlab_live import live

    function = getattr(live, name, None)
    if function is None:
        raise SystemExit(f"live.{name} is not available in this version")
    return function


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
