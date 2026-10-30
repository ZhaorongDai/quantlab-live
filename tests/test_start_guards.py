"""live.start_refusals: a node start is refused on an account or Gateway port of the other mode."""

from __future__ import annotations

import pytest

from quantlab_live.brokers import profile
from quantlab_live.live import start_refusals


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
    refusals = start_refusals(broker, account_id=account, port=port, reported=reported)
    assert len(refusals) == len(refused)
    for refusal, expected in zip(refusals, refused, strict=True):
        assert expected in refusal
