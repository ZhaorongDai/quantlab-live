"""``live.day`` end to end, in-process, with the four fake ports.

The clock is fake (each test day runs in milliseconds), ``prepare`` and the
node launcher are scripted and the mailer records. What is checked is what
an operator sees: the mails, the day state and the node starts.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quantlab_live import records
from quantlab_live.live import Ports, SupervisorRunning, day, status
from tests.live_fakes import (
    FakeClock,
    RecordingMailer,
    ScriptedLauncher,
    ScriptedPrepare,
    SupervisorKilled,
    crashes_at,
    fails_to_start,
    finishes_at,
    ny,
)

DAY = "2024-01-03"
BAR = pd.Timestamp("2024-01-02")


@pytest.fixture
def live_dir(tmp_path):
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    (live_dir / "live.toml").write_text('run = "../run"\nat = "open_auction"\n')
    return live_dir


def ports(live_dir, *, start="06:00", ready_at="06:00", trade=None, settle=None, mailer=None, stop_at=None):
    clock = FakeClock(ny(DAY, start), stop_at=stop_at)
    return Ports(
        prepare=ScriptedPrepare(clock, BAR, None if ready_at is None else ny(DAY, ready_at)),
        launch_node=ScriptedLauncher(clock, live_dir, trade or [finishes_at(ny(DAY, "16:05"), bar=BAR)], settle),
        mailer=mailer or RecordingMailer(),
        clock=clock,
    )


def state(live_dir) -> dict:
    return json.loads((live_dir / "day" / f"{DAY}.json").read_text())


def test_a_normal_day_trades_and_mails_after_the_close(live_dir):
    p = ports(live_dir)
    assert day(live_dir, p) == "traded"
    assert p.launch_node.modes() == ["trade"]
    assert p.launch_node.started[0].at == ny(DAY, "07:15")
    assert len(p.mailer.mails) == 1 and "after the close" in p.mailer.subjects[0]
    assert "filled" in p.mailer.mails[0][1]
    assert state(live_dir)["done"] == "traded"


def test_a_start_that_fails_until_the_latest_start_is_not_traded_with_one_mail(live_dir):
    p = ports(live_dir, trade=[fails_to_start()])
    result = day(live_dir, p)
    assert result.startswith("not traded")
    starts = [s.at for s in p.launch_node.started]
    assert starts[0] == ny(DAY, "07:15") and starts[-1] == ny(DAY, "08:55")
    assert all(b - a == pd.Timedelta(minutes=5) for a, b in zip(starts, starts[1:]))
    assert len(p.mailer.mails) == 1
    subject, body = p.mailer.mails[0]
    assert "not traded" in subject and "latest start 09:00" in body and "refused the connection" in body


def test_a_crash_is_restarted_until_the_close(live_dir):
    p = ports(
        live_dir,
        trade=[crashes_at(ny(DAY, "10:00")), crashes_at(ny(DAY, "15:58")), finishes_at(ny(DAY, "16:05"), bar=BAR)],
    )
    assert day(live_dir, p) == "traded"
    assert [s.at for s in p.launch_node.started] == [ny(DAY, "07:15"), ny(DAY, "10:00"), ny(DAY, "15:58")]
    assert [s for s in p.mailer.subjects if "crashed" in s] == [
        f"quantlab-live {DAY}: the node crashed, restarting"
    ] * 2
    assert "after the close" in p.mailer.subjects[-1]


def test_a_crash_at_the_close_settles_the_outcomes(live_dir):
    p = ports(live_dir, trade=[crashes_at(ny(DAY, "16:01"))], settle=[finishes_at(None, bar=BAR)])
    assert day(live_dir, p) == "traded"
    assert p.launch_node.modes() == ["trade", "settle"]
    assert records.has_outcomes(live_dir, BAR)


def test_outcomes_missing_after_settling_need_checking(live_dir):
    p = ports(live_dir, trade=[finishes_at(ny(DAY, "16:05"), outcomes=False)], settle=[fails_to_start()])
    assert "need checking" in day(live_dir, p)
    assert p.launch_node.modes()[0] == "trade" and set(p.launch_node.modes()[1:]) == {"settle"}
    assert "NEEDS CHECKING" in p.mailer.subjects[-1]


def test_late_data_leaves_the_day_untraded(live_dir):
    p = ports(live_dir, ready_at=None)
    assert day(live_dir, p).startswith("not traded")
    assert p.launch_node.started == []
    polls = p.prepare.calls
    assert polls[0] == ny(DAY, "06:00") and polls[-1] == ny(DAY, "08:45") and len(polls) == 12
    assert len(p.mailer.mails) == 1 and "data update is running" in p.mailer.mails[0][1]


def test_data_late_but_before_the_cut_off_still_trades(live_dir):
    p = ports(live_dir, ready_at="08:40")
    assert day(live_dir, p) == "traded"
    assert p.launch_node.started[0].at == ny(DAY, "08:45")


def test_a_second_supervisor_exits_on_the_lock_and_manual_commands_are_refused(live_dir):
    p = ports(live_dir)
    seen = {}

    def meanwhile(now):
        if now == ny(DAY, "10:00"):
            second = ports(live_dir, start="10:00")
            seen["day"] = day(live_dir, second)
            seen["mails"] = second.mailer.mails
            with pytest.raises(SupervisorRunning, match="supervisor"):
                status(live_dir, DAY)

    p.clock.hooks.append(meanwhile)
    assert day(live_dir, p) == "traded"
    assert seen["day"].startswith("locked") and seen["mails"] == []


def test_a_stale_heartbeat_alerts_from_the_next_cron_call(live_dir):
    p = ports(live_dir)
    seen = {}

    def hung(now):
        if now == ny(DAY, "10:00"):
            # The next hourly call, while this supervisor has not beaten for half an hour.
            later = ports(live_dir, start="10:30")
            seen["day"] = day(live_dir, later)
            seen["subjects"] = later.mailer.subjects

    p.clock.hooks.append(hung)
    day(live_dir, p)
    assert seen["day"].startswith("locked")
    assert seen["subjects"] == [f"quantlab-live {DAY}: the supervisor's heartbeat is stale"]


def test_a_dead_supervisor_is_resumed_from_the_day_state(live_dir):
    first = ports(live_dir, stop_at=ny(DAY, "10:00"))
    with pytest.raises(SupervisorKilled):
        day(live_dir, first)
    assert first.launch_node.modes() == ["trade"]

    # The next cron call, with the node still running: it is polled, not started again.
    again = ports(live_dir, start="11:00")
    again.launch_node.started = first.launch_node.started
    assert day(live_dir, again) == "traded"
    assert again.launch_node.modes() == ["trade"]
    assert again.prepare.calls == []
    assert "was down" in again.mailer.subjects[0]


def test_a_failing_mailer_never_fails_a_step(live_dir):
    p = ports(live_dir, trade=[crashes_at(ny(DAY, "10:00")), finishes_at(ny(DAY, "16:05"), bar=BAR)],
              mailer=RecordingMailer(failing=True))
    assert day(live_dir, p) == "traded"
    mails = state(live_dir)["mails"]
    assert [m["sent"] for m in mails] == [False, False]
    assert "the mail server is down" in (live_dir / "logs" / f"{DAY}.log").read_text()


def test_unmapped_and_foreign_targets_and_node_alerts_are_mailed_once(live_dir):
    def decide(live_dir):
        decision = pd.DataFrame(
            {"weight": [0.5, 0.5, 0.0], "close": [10.0, 20.0, 30.0], "position_quantity": [0.0, 0.0, 5.0],
             "target_quantity": [50.0, 25.0, 0.0], "order_quantity": [50.0, 25.0, -5.0],
             "skip": ["", "unmapped", "foreign"]},
            index=pd.Index([10001, 10002, 10003], name="symbol"),
        )
        records.write_decision(live_dir, BAR, decision)

    def alerting(launcher, node):
        report = finishes_at(ny(DAY, "16:05"), bar=BAR, on_start=decide)(launcher, node)
        return report and records.StepReport(**{**report.__dict__, "alerts": ("held 7 shares of IBM.NYSE, unmapped",)})

    p = ports(live_dir, trade=[alerting])
    assert day(live_dir, p) == "traded"
    subjects = p.mailer.subjects
    assert subjects.count(f"quantlab-live {DAY}: unmapped or foreign securities") == 1
    body = dict(p.mailer.mails)[f"quantlab-live {DAY}: unmapped or foreign securities"]
    assert "10002" in body and "unmapped" in body and "10003" in body and "foreign" in body
    assert any("IBM.NYSE" in body for _, body in p.mailer.mails)


def test_nothing_runs_on_a_day_the_exchange_is_closed(live_dir):
    p = ports(live_dir)
    p.clock.t = ny("2024-07-04", "08:00")
    assert "not an XNYS trading day" in day(live_dir, p)
    assert p.mailer.mails == [] and not (live_dir / "day").exists()


def test_status_prints_the_state_and_the_effective_configuration_and_writes_nothing(live_dir):
    day(live_dir, ports(live_dir))
    before = (live_dir / "day" / f"{DAY}.json").read_bytes()
    text = status(live_dir, DAY)
    assert '"done": "traded"' in text
    assert "start_margin_minutes = 20" in text and "trader_id = \"QL-LIVE\"" in text
    assert (live_dir / "day" / f"{DAY}.json").read_bytes() == before
