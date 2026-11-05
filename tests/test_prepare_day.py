"""quantlab_gate.prepare_day: the data update's status, then quantlab's prediction job once.

The job is quantlab's, on a tiny real run with a model. The tests record
each job command at the process boundary (``subprocess.run``) and run it,
so they can check that repeated polls run the job once.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab_live.live.config import load_config
from quantlab_live.quantlab_gate import Pending, QuantlabNotReady, Ready, open_run, prepare_day
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
def jobs(monkeypatch):
    """Each prediction job command, recorded as it runs; the job itself runs as it would."""
    calls, run = [], subprocess.run

    def recorded(command, *args, **kwargs):
        if "quantlab_live.quantlab_gate" in command:  # not quantlab's own git calls
            calls.append(" ".join(command))
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recorded)
    # The run's factor class lives in the test fixture, which the job imports.
    monkeypatch.setenv("PYTHONPATH", str(REPOSITORY))
    return calls


def _status(data_root: Path, date: str, state: str) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "update_status.json").write_text(
        json.dumps({"date": date, "t": "2024-02-14", "last_done_t": None, "state": state})
    )


def test_the_day_is_pending_until_the_data_update_is_done_for_the_date(model_run, jobs, tmp_path):
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
    assert jobs == []
    assert not store.exists()


def test_a_live_directory_without_a_store_setting_prepares_its_day(model_run, jobs, tmp_path):
    live_dir, data_root = tmp_path / "live", tmp_path / "data"
    live_dir.mkdir()
    (live_dir / "live.toml").write_text(f'run = "{model_run.run_dir}"\n')
    config = load_config(live_dir)
    t = model_run.vendor_update()
    _status(data_root, TODAY, "done")

    assert prepare_day(config.run_dir, TODAY, store=config.store_dir, data_root=data_root) == Ready(t)
    assert prepare_day(config.run_dir, TODAY, store=config.store_dir, data_root=data_root) == Ready(t)
    assert len(jobs) == 1


def test_a_done_update_runs_the_job_once_and_the_day_is_ready(model_run, jobs, tmp_path):
    data_root, store = tmp_path / "data", tmp_path / "live" / "live_predictions.zarr"
    t = model_run.vendor_update()
    _status(data_root, TODAY, "done")

    assert prepare_day(model_run.run_dir, TODAY, store=store, data_root=data_root) == Ready(t)
    assert len(jobs) == 1 and "quantlab_live.quantlab_gate predict" in jobs[0]

    # Polled again: ready at once, the job not run again.
    assert prepare_day(model_run.run_dir, TODAY, store=store, data_root=data_root) == Ready(t)
    assert len(jobs) == 1

    # The next day's update appends a bar: the job runs for it.
    later = model_run.vendor_update()
    _status(data_root, "2024-02-16", "done")
    assert prepare_day(model_run.run_dir, "2024-02-16", store=store, data_root=data_root) == Ready(later)
    assert len(jobs) == 2


def test_a_job_refused_for_missing_data_leaves_the_day_pending(model_run, tmp_path, monkeypatch):
    # A job that reports quantlab's missing-data refusal (exit 2).
    run = subprocess.run

    def refused(command, *args, **kwargs):
        if "quantlab_live.quantlab_gate" not in command:
            return run(command, *args, **kwargs)
        return subprocess.CompletedProcess(command, 2, "", "refused (missing_data): a store lacks the bar")

    monkeypatch.setattr(subprocess, "run", refused)
    data_root, store = tmp_path / "data", tmp_path / "live" / "live_predictions.zarr"
    model_run.vendor_update()
    _status(data_root, TODAY, "done")

    pending = prepare_day(model_run.run_dir, TODAY, store=store, data_root=data_root)
    assert isinstance(pending, Pending)
    assert "a store lacks the bar" in pending.reason


def test_a_store_of_another_run_is_refused(model_run, jobs, tmp_path):
    data_root, store = tmp_path / "data", tmp_path / "live" / "live_predictions.zarr"
    _status(data_root, TODAY, "done")
    open_, close = _market()
    other = build_model_run(tmp_path / "other", BARS, open_, close, held_back=2)
    assert prepare_day(other.run_dir, TODAY, store=store, data_root=data_root) == Ready(BARS[-3])

    with pytest.raises(ValueError, match="another run"):
        prepare_day(model_run.run_dir, TODAY, store=store, data_root=data_root)
    assert len(jobs) == 1


def test_the_day_bar_reads_its_predictions_from_the_live_store(model_run, jobs, tmp_path):
    data_root, store = tmp_path / "data", tmp_path / "live" / "live_predictions.zarr"
    t = model_run.vendor_update()
    run = open_run(model_run.run_dir)
    with pytest.raises(QuantlabNotReady, match="live prediction store"):
        list(run.bars(t, t, store=store))
    _status(data_root, TODAY, "done")
    assert prepare_day(model_run.run_dir, TODAY, store=store, data_root=data_root) == Ready(t)

    (bar,) = open_run(model_run.run_dir).bars(t, t, store=store)
    predictions = bar.predictions()
    assert bar.timestamp == t and len(predictions.data_vars) >= 1
    name = next(iter(predictions.data_vars))
    assert np.isfinite(predictions[name].values).any()
