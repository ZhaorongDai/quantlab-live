"""live.start_refusals: a node start is refused on an account or Gateway port of the other mode,
and a real-money start unless every one of its switches is set."""

from __future__ import annotations

import dataclasses

import pytest

from quantlab_live.brokers import profile
from quantlab_live.live import start_refusals
from quantlab_live.live.guards import ARMED, ARMING_FILE, Switches, arming_state


@pytest.mark.parametrize(
    ("name", "account", "port", "reported", "refused"),
    [
        ("ibkr_paper", "DU123", 4002, None, []),
        ("ibkr_paper", "DU123", 7497, ["IB-DU123"], []),
        ("ibkr_paper", "U123", 4002, None, ["account U123 is live"]),
        ("ibkr_paper", None, 4002, None, ["names no IB account"]),
        ("ibkr_paper", "DU123", 4001, None, ["port 4001 is the live Gateway port"]),
        ("ibkr_paper", "DU123", 4002, [], ["reported accounts none, not DU123"]),
        ("ibkr_real", "U123", 4001, ["IB-U123"], []),
        ("ibkr_real", "DU123", 4001, None, ["account DU123 is paper"]),
        ("ibkr_real", "U123", 7497, None, ["port 7497 is the paper Gateway port"]),
        ("ibkr_real", "U123", 4001, ["IB-U999"], ["not U123"]),
        ("ib_sandbox", None, 4002, [], []),  # its account is the sandbox's own
        ("ib_sandbox", None, 4001, None, ["port 4001 is the live Gateway port"]),
        ("ibkr_paper", "DU123", 4004, None, []),  # a port of a custom mapping: the reported account decides
    ],
)
def test_start_refusals(name, account, port, reported, refused):
    broker = profile(name, port=port, account_id=account, starting_cash=1.0)
    switches = ARMED_SWITCHES if broker.real_money else Switches()
    refusals = start_refusals(broker, account_id=account, port=port, reported=reported, switches=switches, version="2.0.0")
    assert len(refusals) == len(refused)
    for refusal, expected in zip(refusals, refused, strict=True):
        assert expected in refusal


#: A real-money start that passes every guard.
REAL = dict(account_id="U123", port=4001, reported=["IB-U123"], version="2.0.0")
ARMED_SWITCHES = Switches(
    real_money=True, arming=ARMED, initial_capital=10_000.0,
    max_notional_per_order=2_000.0, max_order_submit_rate="10/00:00:01", redis=True,
)


def test_a_fully_armed_real_money_start_passes():
    assert start_refusals(profile("ibkr_real", port=4001, account_id="U123"), switches=ARMED_SWITCHES, **REAL) == []


@pytest.mark.parametrize(
    ("change", "refused"),
    [
        (dict(version="2.0.0rc8"), "2.0.0rc8 is not a final release"),
        (dict(version="2.0.0.dev20261020"), "is not a final release"),
        (dict(version="1.230.0"), "older than 2.0.0"),
        (dict(version=None), "NautilusTrader's version is unknown"),
        (dict(real_money=False), "real_money = true is not set"),
        (dict(arming="missing"), "the arming file is missing"),
        (dict(arming="readable by others (mode 0644)"), "the arming file is readable by others"),
        (dict(initial_capital=None), "initial_capital is not set"),
        (dict(initial_capital=0.0), "initial_capital is not set"),
        (dict(max_notional_per_order=None), "max_notional_per_order is not set"),
        (dict(max_order_submit_rate=None), "max_order_submit_rate is not set"),
        (dict(redis=False), "keeps no Redis cache"),
        (dict(account_id="DU123", reported=["IB-DU123"]), "account DU123 is paper"),
        (dict(port=4002), "port 4002 is the paper Gateway port"),
        (dict(reported=["IB-U999"]), "not U123"),
    ],
)
def test_each_missing_real_money_switch_or_mismatch_refuses_alone(change, refused):
    switches = {k: v for k, v in change.items() if k in Switches.__dataclass_fields__}
    start = {**REAL, **{k: v for k, v in change.items() if k not in switches}}
    broker = profile("ibkr_real", port=start["port"], account_id=start["account_id"])
    refusals = start_refusals(broker, switches=dataclasses.replace(ARMED_SWITCHES, **switches), **start)
    assert len(refusals) == 1 and refused in refusals[0], refusals


def test_a_real_money_start_without_switches_is_refused_for_every_missing_one():
    broker = profile("ibkr_real", port=4001, account_id="U123")
    refusals = start_refusals(broker, account_id="U123", port=4001)
    assert len(refusals) == 7  # unknown version, real_money, arming, capital, both caps, Redis


@pytest.mark.parametrize("name", ["ibkr_paper", "ib_sandbox"])
def test_real_money_true_on_a_paper_profile_is_refused(name):
    broker = profile(name, port=4002, account_id="DU123", starting_cash=1.0)
    refusals = start_refusals(broker, account_id="DU123", port=4002, switches=ARMED_SWITCHES, version="2.0.0")
    assert refusals == [f"real_money = true is set, but the {name} profile trades paper"]


def test_a_paper_start_needs_no_real_money_switch():
    broker = profile("ibkr_paper", port=4002, account_id="DU123")
    assert start_refusals(broker, account_id="DU123", port=4002, switches=Switches(), version="2.0.0rc8") == []


def test_the_arming_file_must_be_an_owner_only_regular_file(tmp_path):
    path = tmp_path / ARMING_FILE
    assert arming_state(path) == "missing"
    path.write_text("")
    path.chmod(0o644)
    assert arming_state(path) == "readable by others (mode 0644)"
    path.chmod(0o600)
    assert arming_state(path) == ARMED
    path.unlink()
    path.mkdir()
    assert arming_state(path) == "not a regular file"
