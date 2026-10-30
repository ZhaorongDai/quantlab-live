"""live.run_node: a day's node on a real ``LiveNode``, in-process, with a fake broker.

The broker is ``tests.fake_broker``: a Python data client serving instruments
and quotes, and NautilusTrader's sandbox execution client on ``SMART``, as the
``ib_sandbox`` profile has. The quantlab run is a tiny real closed-loop run
(qlib's TopkDropout, top 2) on a Sharadar-style store with listings. The
nodes run without Redis here; restoring from Redis is the server smoke
test's (``tests/test_ib_sandbox.py``).
"""

from __future__ import annotations

import pandas as pd
import pytest
from quantlab.portfolio.config import TopkDropoutConfig
from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
from quantlab.runs.prediction_panel import LabelSpec

from quantlab_live import records
from quantlab_live.cli import main
from quantlab_live.live import run_node
from quantlab_live.live.node import DRY_RUN_DIR
from tests.fake_broker import fake_profile
from tests.quantlab_run_fixture import build_constructor_run

A, B, C = 10001, 10002, 10003
BARS = pd.bdate_range("2024-01-02", periods=4)
OPEN = {A: [10.0, 10.5, 11.5, 12.5], B: [20.0, 20.5, 21.0, 24.0], C: [50.0, 48.0, 41.0, 45.0]}
CLOSE = {A: [10.0, 11.0, 12.0, 13.0], B: [20.0, 20.0, 25.0, 25.0], C: [50.0, 40.0, 40.0, 50.0]}
PREDICTIONS = {"ret_5": {A: [0.03] * 4, B: [0.02] * 4, C: [0.01] * 4}}
LISTINGS = {A: [(None, "AAA", "NYSE")], B: [(None, "BBB", "NASDAQ")], C: [(None, "CCC", "NYSE")]}
#: The day traded: the decision bar is the run's last bar, Friday 2024-01-05.
DATE = pd.Timestamp("2024-01-08")
BAR = BARS[-1]
PRICES = {"AAA=STK.SMART": 13.0, "BBB=STK.SMART": 25.0, "CCC=STK.SMART": 50.0}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    rule = TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=2, n_drop=1))
    run_dir, _ = build_constructor_run(
        tmp_path_factory.mktemp("quantlab"), BARS, OPEN, CLOSE, PREDICTIONS, rule,
        [LabelSpec("ret_5", "raw", 1, 5)], listings=LISTINGS,
    )
    return run_dir


@pytest.fixture
def live_dir(tmp_path, run_dir):
    live_dir = tmp_path / "paper"
    live_dir.mkdir()
    (live_dir / "live.toml").write_text(
        f'run = "{run_dir}"\nbroker = "ib_sandbox"\nredis = false\nmin_node_seconds = 3\n'
    )
    return live_dir


def test_a_node_decides_after_start_up_trades_and_writes_outcomes_and_the_snapshot(live_dir):
    profile, requested = fake_profile(PRICES)
    report = run_node(live_dir, DATE, broker=profile)

    assert report == records.StepReport("trade", started=True, finished=True, outcomes=True, alerts=report.alerts)
    assert report.error is None
    decision = records.read_decision(live_dir, BAR)
    ordered = decision[decision["order_quantity"] != 0]
    assert set(ordered.index) == {A, B}  # the top two
    # Nothing was held, so every target's instrument was requested at runtime.
    assert {i.value for i in requested} == {"AAA=STK.SMART", "BBB=STK.SMART"}
    outcomes = records.read_outcomes(live_dir, BAR)
    assert set(outcomes.index) == {A, B}
    snapshot = records.read_snapshot(live_dir, DATE)
    filled = outcomes["filled_quantity"]
    assert snapshot.positions["quantity"].to_dict() == filled[filled != 0].to_dict()
    assert snapshot.foreign == {} and snapshot.cash == 10_000.0
    assert records.decision_capital(live_dir, BAR) == 10_000.0  # sized on the account's equity
    assert (live_dir / "steps" / DATE.date().isoformat() / "manual-trade.json").exists()
    assert records.read_security_map(live_dir).to_id(A).value == "AAA=STK.SMART"


def test_a_dry_run_writes_the_decision_and_the_planned_orders_and_submits_nothing(live_dir):
    profile, requested = fake_profile(PRICES)
    report = run_node(live_dir, DATE, dry_run=True, broker=profile)

    assert (report.started, report.finished, report.error) == (True, True, None)
    assert not records.has_decision(live_dir, BAR)  # the live records are untouched
    decision = records.read_decision(live_dir / DRY_RUN_DIR, BAR)
    planned = pd.read_parquet(live_dir / DRY_RUN_DIR / "orders" / f"{DATE.date()}.parquet")
    ordered = decision[decision["order_quantity"] != 0]["order_quantity"]
    assert planned.set_index("symbol")["quantity"].to_dict() == ordered.to_dict()
    assert set(planned["instrument_id"]) == {"AAA=STK.SMART", "BBB=STK.SMART"}
    assert requested == []  # nothing was submitted, so nothing was loaded
    assert not records.has_outcomes(live_dir, BAR)


def test_a_start_on_an_account_of_the_other_mode_is_refused_before_the_node_is_built(live_dir, run_dir):
    (live_dir / "live.toml").write_text(
        f'run = "{run_dir}"\nbroker = "ibkr_paper"\naccount_id = "U1234567"\ngateway_port = 4002\nredis = false\n'
    )
    report = run_node(live_dir, DATE)

    assert not report.started
    assert "account U1234567 is live" in report.error
    assert not records.has_decision(live_dir, BAR)
    assert not (live_dir / records.SECURITY_MAP).exists()


def test_a_start_whose_account_the_broker_does_not_report_is_refused_before_any_order(live_dir):
    (live_dir / "live.toml").write_text((live_dir / "live.toml").read_text() + 'account_id = "DU1234567"\n')
    profile, requested = fake_profile(PRICES, account_id="IB-DU1234567")
    report = run_node(live_dir, DATE, broker=profile)

    assert not report.started
    assert "the broker reported accounts none, not DU1234567" in report.error
    assert not records.has_decision(live_dir, BAR)
    assert requested == []


def test_a_restarted_node_resumes_the_decision_it_finds_without_deciding_again(live_dir):
    run_node(live_dir, DATE, broker=fake_profile(PRICES)[0])
    path = live_dir / records.DECISIONS / f"{BAR.date()}.parquet"
    written = path.read_bytes()

    report = run_node(live_dir, DATE, broker=fake_profile(PRICES)[0])

    assert (report.started, report.finished, report.error) == (True, True, None)
    assert path.read_bytes() == written


def test_a_settle_only_node_trades_nothing_and_writes_the_outcomes(live_dir):
    run_node(live_dir, DATE, dry_run=False, broker=fake_profile(PRICES)[0])
    (live_dir / records.OUTCOMES / f"{BAR.date()}.parquet").unlink()
    profile, requested = fake_profile(PRICES)

    report = run_node(live_dir, DATE, settle=True, broker=profile)

    assert report == records.StepReport("settle", started=True, finished=True, outcomes=True, alerts=report.alerts)
    assert requested == []
    # Without Redis the settling node finds none of the day's orders: nothing it can verify.
    outcomes = records.read_outcomes(live_dir, BAR)
    assert set(outcomes.index) == {A, B}


def test_a_day_quantlab_has_not_made_yet_is_reported(live_dir):
    report = run_node(live_dir, "2024-01-09", broker=fake_profile(PRICES)[0])

    assert not report.started and "quantlab is not ready" in report.error


def test_the_command_line_runs_the_node_and_it_writes_its_step_report(live_dir, run_dir):
    (live_dir / "live.toml").write_text(
        f'run = "{run_dir}"\nbroker = "ibkr_paper"\naccount_id = "U1234567"\ngateway_port = 4002\nredis = false\n'
    )
    assert main(["run-node", str(live_dir), "--date", DATE.date().isoformat()]) == 0

    report = records.read_step_report(live_dir / "steps" / DATE.date().isoformat() / "manual-trade.json")
    assert not report.started and "is live" in report.error
