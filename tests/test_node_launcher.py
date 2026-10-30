"""The node seam: ``SubprocessLauncher`` and the manual commands of the command line.

The node behind the seam is a stand-in program that writes a step report
the way a node does, so the launcher's protocol (a fresh process per
attempt, bound with ``taskset``, polled through a handle that outlives the
supervisor) is checked without a broker. ``taskset`` is a stand-in on
``PATH`` that records its arguments.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time

import pandas as pd
import pytest

from quantlab_live import records
from quantlab_live.cli import REFUSED, main
from quantlab_live.live import manual
from quantlab_live.live.ports import SubprocessLauncher

DAY = pd.Timestamp("2024-01-03")

NODE = """
import sys, time, json, pathlib
args = sys.argv[1:]
report = pathlib.Path(args[args.index("--report") + 1])
mode = args[args.index("--mode") + 1]
release = pathlib.Path(args[0]) / "release"
if release.read_text() == "refuse":
    sys.exit(4)  # a start failure: no report
report.write_text(json.dumps({"mode": mode, "started": True}))
while release.read_text() == "hold":
    time.sleep(0.02)
if release.read_text() == "crash":
    sys.exit(-9)
report.write_text(json.dumps({"mode": mode, "started": True, "finished": True, "outcomes": True}))
"""


@pytest.fixture
def taskset(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "taskset.calls"
    script = bin_dir / "taskset"
    script.write_text(f'#!/bin/sh\necho "$@" >> {calls}\nshift 2\nexec "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return calls


@pytest.fixture
def launcher(tmp_path, taskset):
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    node = tmp_path / "node.py"
    node.write_text(NODE)
    return SubprocessLauncher(live_dir, cpus="64-127", program=(sys.executable, str(node)))


def _until_ended(launcher, handle) -> records.StepReport:
    deadline = time.monotonic() + 30
    while (report := launcher.poll(handle)) is None:
        assert time.monotonic() < deadline, "the node never ended"
        time.sleep(0.02)
    return report


def test_a_node_runs_in_a_fresh_bound_process_and_reports_through_its_step_report(launcher, taskset):
    (launcher.live_dir / "release").write_text("hold")
    handle = launcher.start(DAY, "trade", 1)
    assert launcher.poll(handle) is None
    deadline = time.monotonic() + 30
    while (progress := launcher.progress(handle)) is None:  # what the running node reported so far
        assert time.monotonic() < deadline, "the node never reported"
        time.sleep(0.02)
    assert progress.started and not progress.finished
    (launcher.live_dir / "release").write_text("go")
    report = _until_ended(launcher, handle)
    assert (report.started, report.finished, report.outcomes) == (True, True, True)
    assert taskset.read_text().startswith("-c 64-127 ")
    assert "--mode trade" in taskset.read_text()


def test_a_later_supervisor_polls_the_node_an_earlier_one_started(launcher):
    (launcher.live_dir / "release").write_text("hold")
    handle = launcher.start(DAY, "trade", 1)
    later = SubprocessLauncher(launcher.live_dir, cpus="64-127", program=launcher.program)
    assert later.poll(handle) is None
    (launcher.live_dir / "release").write_text("crash")
    launcher._children[json.loads(handle)["pid"]].wait()  # reaped by its parent, as init would
    report = _until_ended(later, handle)
    assert report.started and not report.finished


def test_a_process_that_ends_without_a_report_did_not_start(launcher):
    (launcher.live_dir / "release").write_text("refuse")
    report = _until_ended(launcher, launcher.start(DAY, "trade", 1))
    assert not report.started and "status 4 without a step report" in report.error


@pytest.fixture
def live_dir(tmp_path):
    live_dir = tmp_path / "manual"
    live_dir.mkdir()
    (live_dir / "live.toml").write_text('run = "run"\n')
    return live_dir


@pytest.mark.parametrize("command", [["status"], ["run-node"], ["run-node", "--settle"], ["brief"]])
def test_manual_commands_are_refused_while_a_supervisor_runs(live_dir, command, capsys):
    with manual(live_dir, DAY, "supervisor"):
        assert main([command[0], str(live_dir), "--date", "2024-01-03", *command[1:]]) == REFUSED
    assert "refused" in capsys.readouterr().err
    assert not (live_dir / "day" / "2024-01-03.json").exists()
