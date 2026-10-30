"""quantlab_gate.prepare_day: the data update's status, then quantlab's prediction job once.

The job runs as a subprocess under ``taskset -c 64-127`` (NUMA node 1). The
tests put a stand-in ``taskset`` first on ``PATH`` that records its
arguments and runs the command it was given, so the binding is checked on a
machine without ``taskset`` (or with fewer cores) while the job itself is
quantlab's, on a tiny real run with a model.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab_live.quantlab_gate import Pending, Ready, prepare_day
from tests.quantlab_run_fixture import build_model_run

PERMNOS = (10001, 10002, 10003, 10004)
BARS = pd.bdate_range("2024-01-02", periods=34)
REPOSITORY = Path(__file__).resolve().parents[1]
#: The New York trading day after the store's last bar (the day traded on it).
TODAY = "2024-02-15"


def _market():
    rng = np.random.default_rng(3)
    close = {
        p: list(np.round((20.0 + 5 * k) * np.exp(np.cumsum(rng.normal(0, 0.02, len(BARS)))), 2))
        for k, p in enumerate(PERMNOS)
    }
    return {p: [round(c * 1.001, 2) for c in v] for p, v in close.items()}, close


@pytest.fixture
def model_run(tmp_path):
    open_, close = _market()
    return build_model_run(tmp_path / "quantlab", BARS, open_, close, held_back=2)


@pytest.fixture
def taskset(tmp_path, monkeypatch):
    """A ``taskset`` stand-in on ``PATH``; returns the file listing each call's arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "taskset.calls"
    script = bin_dir / "taskset"
    script.write_text(f'#!/bin/sh\necho "$@" >> {calls}\nshift 2\nexec "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    # The run's factor class lives in the test fixture, which the job imports.
    monkeypatch.setenv("PYTHONPATH", str(REPOSITORY))
    return calls


def _status(data_root: Path, date: str, state: str) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "update_status.json").write_text(
        json.dumps({"date": date, "t": "2024-02-14", "last_done_t": None, "state": state})
    )


def _calls(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def test_the_day_is_pending_until_the_data_update_is_done_for_the_date(model_run, taskset, tmp_path):
    data_root, store = tmp_path / "data", tmp_path / "live" / "live_predictions.zarr"

    def prepare():
        return prepare_day(model_run.run_dir, TODAY, store=store, data_root=data_root)

    assert isinstance(prepare(), Pending)  # no status file yet
    assert "update_status.json" in prepare().reason
    _status(data_root, "2024-02-14", "done")  # yesterday's update
    assert "2024-02-14" in prepare().reason
    _status(data_root, TODAY, "running")
    assert "running" in prepare().reason
    _status(data_root, TODAY, "no_new_bar")
    assert isinstance(prepare(), Pending)
    assert _calls(taskset) == []
    assert not store.exists()


def test_a_done_update_runs_the_job_once_on_numa_node_1_and_the_day_is_ready(model_run, taskset, tmp_path):
    data_root, store = tmp_path / "data", tmp_path / "live" / "live_predictions.zarr"
    t = model_run.vendor_update()
    _status(data_root, TODAY, "done")

    assert prepare_day(model_run.run_dir, TODAY, store=store, data_root=data_root) == Ready(t)
    calls = _calls(taskset)
    assert len(calls) == 1 and calls[0].startswith("-c 64-127 ")
    assert "quantlab_live.quantlab_gate" in calls[0]

    # Polled again: ready at once, the job not run again.
    assert prepare_day(model_run.run_dir, TODAY, store=store, data_root=data_root) == Ready(t)
    assert len(_calls(taskset)) == 1

    # The next day's update appends a bar: the job runs for it.
    later = model_run.vendor_update()
    _status(data_root, "2024-02-16", "done")
    assert prepare_day(model_run.run_dir, "2024-02-16", store=store, data_root=data_root) == Ready(later)
    assert len(_calls(taskset)) == 2


def test_a_job_refused_for_missing_data_leaves_the_day_pending(model_run, tmp_path, monkeypatch):
    # A taskset whose job reports quantlab's missing-data refusal (exit 2).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "taskset"
    script.write_text('#!/bin/sh\necho "refused (missing_data): a store lacks the bar" >&2\nexit 2\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    data_root, store = tmp_path / "data", tmp_path / "live" / "live_predictions.zarr"
    model_run.vendor_update()
    _status(data_root, TODAY, "done")

    pending = prepare_day(model_run.run_dir, TODAY, store=store, data_root=data_root)
    assert isinstance(pending, Pending)
    assert "a store lacks the bar" in pending.reason


def test_a_store_of_another_run_is_refused(model_run, taskset, tmp_path):
    data_root, store = tmp_path / "data", tmp_path / "live" / "live_predictions.zarr"
    _status(data_root, TODAY, "done")
    open_, close = _market()
    other = build_model_run(tmp_path / "other", BARS, open_, close, held_back=2)
    assert prepare_day(other.run_dir, TODAY, store=store, data_root=data_root) == Ready(BARS[-3])

    with pytest.raises(ValueError, match="another run"):
        prepare_day(model_run.run_dir, TODAY, store=store, data_root=data_root)
    assert len(_calls(taskset)) == 1
