"""live.run_node: a day's node on a real ``LiveNode``, in-process, with a fake broker.

The broker is ``tests.fake_broker``: a Python data client serving instruments
and quotes, and NautilusTrader's sandbox execution client on ``SMART``, as the
``ib_sandbox`` profile has. The quantlab run is a tiny real closed-loop run
(qlib's TopkDropout, top 2) on a Sharadar-style store with listings. The
nodes run without Redis here; restoring from Redis is the server smoke
test's (``tests/test_ib_sandbox.py``).
"""

from __future__ import annotations

import time

import nautilus_trader
import pandas as pd
import pytest
from quantlab.portfolio.config import TopkDropoutConfig
from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
from quantlab.runs.prediction_panel import LabelSpec

from quantlab_live import records
from quantlab_live.cli import main
from quantlab_live.live import run_node
from quantlab_live.live.guards import ARMING_FILE
from quantlab_live.live.node import ARCHIVE_DIR, DRY_RUN_DIR, claimed_instruments
from quantlab_live.trading.scheduled_execution import Session
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
        f'run = "{run_dir}"\nbroker = "ib_sandbox"\nmin_node_seconds = 10\nstop_after_close_minutes = -60\n'
    )
    return live_dir


def around_now() -> Session:
    """A session that opened an hour ago and closes in an hour: the opening auction's fallback trades at once.

    The node stops ``min_node_seconds`` after its start (``stop_after_close_minutes = -60``).
    """
    now = pd.Timestamp.now(tz="UTC")
    return Session(now - pd.Timedelta(hours=1), now + pd.Timedelta(hours=1))


def test_a_node_decides_after_start_up_trades_and_writes_outcomes_and_the_snapshot(live_dir):
    profile, requested = fake_profile(PRICES)
    report = run_node(live_dir, DATE, broker=profile, redis=False, session=around_now())

    assert report == records.StepReport("trade", started=True, finished=True, outcomes=True, alerts=report.alerts)
    assert report.error is None
    decision = records.read_decision(live_dir, BAR)
    ordered = decision[decision["order_quantity"] != 0]
    assert set(ordered.index) == {A, B}  # the top two
    # Nothing was held, so every target's instrument was requested at runtime.
    assert {i.value for i in requested} == {"AAA=STK.SMART", "BBB=STK.SMART"}
    outcomes = records.read_outcomes(live_dir, BAR)
    # Both orders filled in the sandbox, at the fake broker's ask, with IBKR's fees.
    assert outcomes["status"].to_dict() == {A: "filled", B: "filled"}
    assert outcomes["filled_quantity"].to_dict() == ordered["order_quantity"].to_dict()
    assert outcomes["avg_px"].to_dict() == {A: 13.01, B: 25.01}
    assert (outcomes["fees"] >= 1.0).all()
    snapshot = records.read_snapshot(live_dir, DATE)
    assert snapshot.positions["quantity"].to_dict() == outcomes["filled_quantity"].to_dict()
    spent = float((outcomes["filled_quantity"] * outcomes["avg_px"]).sum() + outcomes["fees"].sum())
    assert snapshot.foreign == {} and snapshot.cash == pytest.approx(10_000.0 - spent)
    assert records.decision_capital(live_dir, BAR) == 10_000.0  # sized on the account's equity
    assert (live_dir / "steps" / DATE.date().isoformat() / "manual-trade.json").exists()
    # The streaming archive keeps the node's events, the orders' among them.
    archived = {p.parent.name for p in (live_dir / ARCHIVE_DIR).rglob("*.feather")}
    assert {"order_filled", "account_state"} <= archived, archived
    assert records.read_security_map(live_dir).to_id(A).value == "AAA=STK.SMART"


def test_a_dry_run_writes_the_decision_and_the_planned_orders_and_submits_nothing(live_dir):
    profile, requested = fake_profile(PRICES)
    report = run_node(live_dir, DATE, dry_run=True, broker=profile, redis=False)

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
        f'run = "{run_dir}"\nbroker = "ibkr_paper"\naccount_id = "U1234567"\ngateway_port = 4002\n'
    )
    report = run_node(live_dir, DATE)

    assert not report.started
    assert "account U1234567 is live" in report.error
    assert not records.has_decision(live_dir, BAR)
    assert not (live_dir / records.SECURITY_MAP).exists()


def test_a_start_whose_account_the_broker_does_not_report_is_refused_before_any_order(live_dir):
    (live_dir / "live.toml").write_text((live_dir / "live.toml").read_text() + 'account_id = "DU1234567"\n')
    profile, requested = fake_profile(PRICES, account_id="IB-DU1234567")
    report = run_node(live_dir, DATE, broker=profile, redis=False)

    assert not report.started
    assert "the broker reported accounts none, not DU1234567" in report.error
    assert not records.has_decision(live_dir, BAR)
    assert requested == []


def test_a_restarted_node_resumes_the_decision_it_finds_without_deciding_again(live_dir):
    run_node(live_dir, DATE, broker=fake_profile(PRICES)[0], redis=False, session=around_now())
    path = live_dir / records.DECISIONS / f"{BAR.date()}.parquet"
    written = path.read_bytes()

    report = run_node(live_dir, DATE, broker=fake_profile(PRICES)[0], redis=False, session=around_now())

    assert (report.started, report.finished, report.error) == (True, True, None)
    assert path.read_bytes() == written


def test_a_settle_only_node_trades_nothing_stops_once_reconciled_and_writes_the_outcomes(live_dir):
    run_node(live_dir, DATE, dry_run=False, broker=fake_profile(PRICES)[0], redis=False, session=around_now())
    (live_dir / records.OUTCOMES / f"{BAR.date()}.parquet").unlink()
    profile, requested = fake_profile(PRICES)
    (live_dir / "live.toml").write_text((live_dir / "live.toml").read_text().replace("min_node_seconds = 10", "min_node_seconds = 120"))

    started = time.monotonic()
    report = run_node(live_dir, DATE, settle=True, broker=profile, redis=False)

    assert time.monotonic() - started < 60  # it did not wait for min_node_seconds

    assert report == records.StepReport("settle", started=True, finished=True, outcomes=True, alerts=report.alerts)
    assert requested == []
    # Without Redis the settling node finds none of the day's orders: nothing it can verify.
    outcomes = records.read_outcomes(live_dir, BAR)
    assert set(outcomes.index) == {A, B}


def test_a_day_quantlab_has_not_made_yet_is_reported(live_dir):
    report = run_node(live_dir, "2024-01-09", broker=fake_profile(PRICES)[0], redis=False, session=around_now())

    assert not report.started and "quantlab is not ready" in report.error


def test_the_command_line_runs_the_node_and_it_writes_its_step_report(live_dir, run_dir):
    (live_dir / "live.toml").write_text(
        f'run = "{run_dir}"\nbroker = "ibkr_paper"\naccount_id = "U1234567"\ngateway_port = 4002\n'
    )
    assert main(["run-node", str(live_dir), "--date", DATE.date().isoformat()]) == 0

    report = records.read_step_report(live_dir / "steps" / DATE.date().isoformat() / "manual-trade.json")
    assert not report.started and "is live" in report.error


@pytest.mark.skipif(nautilus_trader.__version__.replace(".", "").isdigit(), reason="NautilusTrader is a final release")
def test_a_real_money_start_on_a_release_candidate_is_refused_before_the_node_is_built(live_dir, run_dir):
    (live_dir / "live.toml").write_text(
        f'run = "{run_dir}"\nbroker = "ibkr_real"\naccount_id = "U1234567"\ngateway_port = 4001\n'
        'real_money = true\ninitial_capital = 10000\n'
        'max_notional_per_order = 2000\nmax_order_submit_rate = "10/00:00:01"\n'
    )
    (live_dir / ARMING_FILE).write_text("")
    (live_dir / ARMING_FILE).chmod(0o600)
    report = run_node(live_dir, DATE)

    assert not report.started
    assert report.error == (
        "the start is refused: a real-money start needs a final NautilusTrader release, "
        f"and {nautilus_trader.__version__} is not a final release"
    )
    assert not (live_dir / records.SECURITY_MAP).exists()


def test_a_real_money_start_without_its_arming_file_is_refused(live_dir, run_dir):
    (live_dir / "live.toml").write_text(
        f'run = "{run_dir}"\nbroker = "ibkr_real"\naccount_id = "U1234567"\ngateway_port = 4001\nreal_money = true\n'
    )
    report = run_node(live_dir, DATE)

    assert not report.started
    assert "the arming file is missing" in report.error
    assert "initial_capital is not set" in report.error


def test_a_restart_that_lost_its_cache_claims_what_the_days_decision_trades(live_dir):
    # Held C at the last close; today's decision buys A and B, so a restart without
    # Redis must claim the positions and orders of A and B too (they are its own).
    held = pd.DataFrame({"quantity": [5.0]}, index=pd.Index([C], name="symbol"))
    records.write_snapshot(live_dir, records.Snapshot(BAR, held, 9_750.0))
    run_node(live_dir, DATE, dry_run=True, broker=fake_profile(PRICES)[0], redis=False)
    decision = records.read_decision(live_dir / DRY_RUN_DIR, BAR)

    claims = claimed_instruments(records.read_security_map(live_dir), records.latest_snapshot(live_dir, DATE), decision)

    assert {i.value for i in claims} == {"AAA=STK.SMART", "BBB=STK.SMART", "CCC=STK.SMART"}


def test_an_earlier_unverified_outcome_whose_orders_the_cache_lost_stays_unverified_and_alerts(live_dir):
    earlier = BARS[-2]
    decision = pd.DataFrame(
        {"weight": [0.5], "close": [11.0], "position_quantity": [0.0], "target_quantity": [100.0],
         "order_quantity": [100.0], "skip": [""]},
        index=pd.Index([A], name="symbol"),
    )
    records.write_decision(live_dir, earlier, decision, 10_000.0)
    unverified = pd.DataFrame(
        {"order_quantity": [100.0], "filled_quantity": [0.0], "avg_px": [float("nan")], "fees": [0.0],
         "status": ["unverified"]},
        index=pd.Index([A], name="symbol"),
    )
    records.write_outcomes(live_dir, earlier, unverified)

    report = run_node(live_dir, DATE, settle=True, broker=fake_profile(PRICES)[0], redis=False)

    assert report.started and report.error is None
    assert records.read_outcomes(live_dir, earlier)["status"].tolist() == ["unverified"]
    assert any("stay unverified: the cache holds none of its orders" in alert for alert in report.alerts)


@pytest.mark.parametrize(
    "initial_capital, equity_cash, capital",
    [
        (5_000.0, None, 5_000.0),  # the takeover: no P&L yet
        (5_000.0, 6_000.0, 6_000.0),  # 1,000 of P&L valued at the last close, nothing held
        (20_000.0, None, 10_000.0),  # capped by the account's equity
    ],
)
def test_orders_size_on_the_initial_capital_plus_the_strategys_pnl_capped_by_the_account(
    live_dir, initial_capital, equity_cash, capital
):
    (live_dir / "live.toml").write_text((live_dir / "live.toml").read_text() + f"initial_capital = {initial_capital}\n")
    if equity_cash is not None:
        records.append_equity(live_dir, BAR, cash=equity_cash, holdings=0.0)

    report = run_node(live_dir, DATE, dry_run=True, broker=fake_profile(PRICES)[0], redis=False)

    assert report.error is None
    assert records.decision_capital(live_dir / DRY_RUN_DIR, BAR) == capital
    decision = records.read_decision(live_dir / DRY_RUN_DIR, BAR)
    spent = (decision["target_quantity"] * decision["close"]).sum()
    assert capital - 50.0 < spent <= capital  # whole shares of the top two, half each
