"""The ``ib_sandbox`` smoke test: a node against a real IB Gateway and Redis.

Run on the trading server, not in CI, during US market hours, with the paper
Gateway and Redis up::

    QUANTLAB_LIVE_IB_GATEWAY=<host>:<port> QUANTLAB_LIVE_REDIS=<host>:<port> \\
        uv run pytest tests/test_ib_sandbox.py -v -s

and, to start ``ibkr_paper`` in a dry run on the paper account (its positions
held overnight, upstream NautilusTrader issue 5428), also
``QUANTLAB_LIVE_IB_ACCOUNT=<the paper account id>``; to check the snapshot's
cash on that account, also ``QUANTLAB_LIVE_IB_CASH=<its cash in TWS>``.
Without ``QUANTLAB_LIVE_IB_GATEWAY`` every test here is skipped.

The quantlab run is the tiny fixture run, listed as three real stocks; the
sandbox fills its orders on IB's quotes. It checks:

- the ``SMART`` venue loads (a held instrument at the start, the targets at
  runtime), IB's quotes stream and the sandbox fills on them, charging the
  IBKR fee model;
- the risk engine denies an order above the notional cap, and lets one
  below it through;
- a node killed mid-day restarts from Redis, finds its working orders and
  submits nothing again;
- a start on an account of the other mode is refused before any order;
- an ``ibkr_paper`` start with positions held overnight reconciles (dry run);
- the snapshot's cash on the paper account is IB's cash: ``account_cash``
  subtracts the open positions' cost from the account's balance, which holds
  for the sandbox's MARGIN account, but would count the positions twice if
  IB's balance is already net of them. Open until this test has run.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from quantlab.portfolio.config import TopkDropoutConfig
from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
from quantlab.runs.prediction_panel import LabelSpec

from quantlab_live import records
from quantlab_live.live import run_node
from quantlab_live.live.node import decision_bar_of
from quantlab_live.trading.scheduled_execution import NEW_YORK, Session
from tests.quantlab_run_fixture import build_constructor_run

GATEWAY = os.environ.get("QUANTLAB_LIVE_IB_GATEWAY")
REDIS = os.environ.get("QUANTLAB_LIVE_REDIS")
ACCOUNT = os.environ.get("QUANTLAB_LIVE_IB_ACCOUNT")
CASH = os.environ.get("QUANTLAB_LIVE_IB_CASH")
pytestmark = pytest.mark.skipif(not GATEWAY, reason="QUANTLAB_LIVE_IB_GATEWAY names no IB Gateway")

A, B, C = 10001, 10002, 10003
LISTINGS = {A: [(None, "AAPL", "NASDAQ")], B: [(None, "MSFT", "NASDAQ")], C: [(None, "KO", "NYSE")]}
REPOSITORY = Path(__file__).resolve().parents[1]


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NEW_YORK).tz_localize(None).normalize()


def _run(root: Path, close: dict) -> Path:
    """A closed-loop run whose last bar is today's decision bar: it buys AAPL and MSFT."""
    bars = pd.bdate_range(end=decision_bar_of(_today()), periods=4)
    closes = {symbol: [price] * 4 for symbol, price in close.items()}
    predictions = {"ret_5": {A: [0.03] * 4, B: [0.02] * 4, C: [0.01] * 4}}
    rule = TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=2, n_drop=1))
    run_dir, _ = build_constructor_run(
        root, bars, closes, closes, predictions, rule, [LabelSpec("ret_5", "raw", 1, 5)], listings=LISTINGS,
    )
    return run_dir


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    # Closes far above the market: the orders' sizes stay well inside the sandbox's cash.
    return _run(tmp_path_factory.mktemp("quantlab"), {A: 400.0, B: 800.0, C: 120.0})


@pytest.fixture(scope="module")
def low_run_dir(tmp_path_factory):
    # Closes far below the market: buy limits at them rest unfilled.
    return _run(tmp_path_factory.mktemp("quantlab-low"), {A: 20.0, B: 40.0, C: 10.0})


def _live_dir(tmp_path: Path, run_dir: Path, name: str, **keys) -> Path:
    """A live directory on the Gateway (and Redis) of the environment; ``keys`` override ``live.toml``.

    Its node stops ``min_node_seconds`` after its start: the close is hours away.
    """
    host, port = GATEWAY.rsplit(":", 1)
    values = {
        "run": str(run_dir), "broker": "ib_sandbox", "starting_cash": 100_000.0,
        "gateway_host": host, "gateway_port": int(port), "client_id": 71,
        "stop_after_close_minutes": -24 * 60, "min_node_seconds": 120,
    }
    if REDIS is not None:
        values["redis_host"], values["redis_port"] = REDIS.rsplit(":", 1)[0], int(REDIS.rsplit(":", 1)[1])
    values.update(keys)
    live_dir = tmp_path / name
    live_dir.mkdir()
    lines = [f"{k} = {_toml(v)}" for k, v in values.items()]
    (live_dir / "live.toml").write_text("\n".join(lines) + "\n")
    return live_dir


def _toml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return f'"{value}"' if isinstance(value, str) else repr(value)


def _soon() -> str:
    """A New York time two minutes from now, ``HH:MM``, inside today's session."""
    now = pd.Timestamp.now(tz=NEW_YORK)
    session = Session.of(now)
    if not session.open <= now.tz_convert("UTC") < session.close - pd.Timedelta(minutes=10):
        pytest.skip("run the smoke test during US market hours")
    return (now + pd.Timedelta(minutes=2)).strftime("%H:%M")


def test_the_smart_venue_loads_streams_and_fills_in_the_sandbox(tmp_path, run_dir):
    # The orders' notionals are about 50,000 each: inside the cap.
    live_dir = _live_dir(
        tmp_path, run_dir, "smoke", at=_soon(), min_node_seconds=240,
        max_notional_per_order=1_000_000.0, max_order_submit_rate="10/00:00:01",
    )
    today = _today()
    # Yesterday's snapshot holds KO: its instrument loads at the start; AAPL and MSFT at runtime.
    held = pd.DataFrame({"quantity": [5.0]}, index=pd.Index([C], name="symbol"))
    records.write_snapshot(live_dir, records.Snapshot(decision_bar_of(today), held, 100_000.0))

    report = run_node(live_dir, today, redis=REDIS is not None)

    assert report.started and report.finished and report.error is None, report
    outcomes = records.read_outcomes(live_dir, decision_bar_of(today))
    assert set(outcomes.index) == {A, B}
    assert (outcomes["status"] == "filled").all(), outcomes
    assert (outcomes["avg_px"] > 0).all() and (outcomes["fees"] >= 1.0).all()  # IBKR's minimum per order


def test_the_risk_engine_denies_an_order_above_the_notional_cap(tmp_path, run_dir):
    live_dir = _live_dir(tmp_path, run_dir, "capped", at=_soon(), min_node_seconds=240, max_notional_per_order=1_000.0)
    today = _today()

    report = run_node(live_dir, today, redis=REDIS is not None)

    assert report.started and report.finished and report.error is None, report
    outcomes = records.read_outcomes(live_dir, decision_bar_of(today))
    assert (outcomes["status"] == "not_sent").all(), outcomes
    assert (outcomes["filled_quantity"] == 0).all()


@pytest.mark.skipif(REDIS is None, reason="QUANTLAB_LIVE_REDIS names no Redis")
def test_a_node_killed_mid_day_restores_from_redis_and_submits_nothing_twice(tmp_path, low_run_dir):
    # Buy limits at the run's closes, far below the market: they rest, working, when the node dies.
    live_dir = _live_dir(tmp_path, low_run_dir, "crash", at=_soon(), order="limit", band_bps=0.0, min_node_seconds=60)
    today = _today()
    first = _node_process(live_dir, today, "first")
    report = live_dir / "first.json"
    deadline = time.time() + 120
    while not (report.exists() and records.read_step_report(report).started) and time.time() < deadline:
        time.sleep(1)
    time.sleep(150)  # past the plan's time: the children are working
    first.send_signal(signal.SIGKILL)
    first.wait()
    decision = live_dir / records.DECISIONS / f"{decision_bar_of(today).date()}.parquet"
    written = decision.read_bytes()

    second = _node_process(live_dir, today, "second")
    log, _ = second.communicate(timeout=300)

    assert decision.read_bytes() == written  # resumed, not decided again
    assert "0 orders remain" in log, log[-4000:]
    assert records.read_step_report(live_dir / "second.json").finished


def _node_process(live_dir: Path, date: pd.Timestamp, name: str) -> subprocess.Popen:
    command = [
        sys.executable, "-m", "quantlab_live", "node", str(live_dir), date.date().isoformat(),
        "--mode", "trade", "--report", str(live_dir / f"{name}.json"),
    ]
    env = {**os.environ, "PYTHONPATH": str(REPOSITORY)}
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)


def test_a_start_on_a_live_account_is_refused_before_any_order(tmp_path, run_dir):
    live_dir = _live_dir(tmp_path, run_dir, "refused", broker="ibkr_paper", account_id="U0000000")

    report = run_node(live_dir, _today())

    assert not report.started and "is live" in report.error
    assert not (live_dir / records.DECISIONS).exists()


@pytest.mark.skipif(ACCOUNT is None, reason="QUANTLAB_LIVE_IB_ACCOUNT names no paper account")
def test_an_ibkr_paper_start_with_positions_held_overnight_reconciles(tmp_path, run_dir):
    live_dir = _live_dir(tmp_path, run_dir, "paper", broker="ibkr_paper", account_id=ACCOUNT, min_node_seconds=60)

    report = run_node(live_dir, _today(), dry_run=True, redis=REDIS is not None)

    assert report.started and report.finished and report.error is None, report


@pytest.mark.skipif(ACCOUNT is None or CASH is None, reason="QUANTLAB_LIVE_IB_ACCOUNT and QUANTLAB_LIVE_IB_CASH are not both set")
def test_the_snapshot_cash_of_the_paper_account_is_ibs_cash(tmp_path, run_dir):
    # A settle-only node trades nothing and writes the end-of-day snapshot from the broker.
    live_dir = _live_dir(tmp_path, run_dir, "cash", broker="ibkr_paper", account_id=ACCOUNT)

    report = run_node(live_dir, _today(), settle=True, redis=REDIS is not None)

    assert report.started and report.error is None, report
    snapshot = records.read_snapshot(live_dir, _today())
    # Each position counted once: within a dollar per position of TWS's cash, not off by the portfolio's cost.
    assert snapshot.cash == pytest.approx(float(CASH), abs=1.0 + len(snapshot.positions) + len(snapshot.foreign))
