"""order_outcomes: one outcome per ordered symbol of a decision, priced only from fills."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from quantlab_live.trading.outcomes import ORDER_COLUMNS, FILL_COLUMNS, order_outcomes

SYMBOL = 10001


def decision(order_quantity=100.0, skip="", position=0.0):
    return pd.DataFrame(
        {
            "weight": [0.5],
            "close": [10.0],
            "position_quantity": [position],
            "target_quantity": [position + order_quantity],
            "order_quantity": [order_quantity],
            "skip": [skip],
        },
        index=pd.Index([SYMBOL], name="symbol"),
    )


def orders(*rows):
    """Orders as (id, primary, child, status, leaves, cancel_requested)."""
    frame = pd.DataFrame(
        [
            {"client_order_id": i, "symbol": SYMBOL, "primary": p, "child": c, "status": s,
             "leaves": leaves, "cancel_requested": asked}
            for i, p, c, s, leaves, asked in rows
        ],
        columns=["client_order_id", *ORDER_COLUMNS],
    )
    return frame.set_index("client_order_id")


def fills(*rows):
    """Fills as (order id, signed quantity, price)."""
    return pd.DataFrame(
        [{"client_order_id": i, "symbol": SYMBOL, "quantity": q, "price": px} for i, q, px in rows],
        columns=FILL_COLUMNS,
    )


PRIMARY = ("P", "P", None, "INITIALIZED", 0.0, False)
KEPT = ("P", "P", None, "INITIALIZED", 100.0, False)


def child(kind, status, leaves=0.0, asked=False, name="C"):
    return (name, "P", kind, status, leaves, asked)


CASES = {
    "filled": (orders(PRIMARY, child("open_auction", "FILLED")), fills(("C", 100.0, 10.5))),
    "partially_filled": (
        orders(PRIMARY, child("day", "EXPIRED", 40.0)),
        fills(("C", 60.0, 10.5)),
    ),
    "cancelled": (orders(PRIMARY, child("day", "CANCELED", 100.0, asked=True)), fills()),
    "venue_cancelled": (orders(PRIMARY, child("close_auction", "EXPIRED", 100.0)), fills()),
    "venue_cancelled_fallback": (
        orders(PRIMARY, child("open_auction", "CANCELED", 100.0), child("fallback", "FILLED", name="F")),
        fills(("F", 100.0, 10.6)),
    ),
    "rejected": (orders(PRIMARY, child("day", "REJECTED", 100.0)), fills()),
    "not_sent": (orders(KEPT), fills()),
    "unverified": (orders(PRIMARY, child("day", "ACCEPTED", 100.0)), fills()),
}


@pytest.mark.parametrize("status", CASES)
def test_each_way_an_order_ends_gives_its_outcome(status):
    got = order_outcomes(decision(), *CASES[status], positions=None)
    assert got.loc[SYMBOL, "status"] == status
    assert list(got.index) == [SYMBOL]


@pytest.mark.parametrize("skip", ["unmapped", "foreign"])
def test_a_skipped_symbol_is_its_skip_reason(skip):
    got = order_outcomes(decision(skip=skip), orders(), fills(), positions=None)
    assert got.loc[SYMBOL, ["order_quantity", "filled_quantity", "status"]].tolist() == [100.0, 0.0, skip]
    assert math.isnan(got.loc[SYMBOL, "avg_px"])


def test_an_ordered_symbol_without_any_order_is_not_sent():
    got = order_outcomes(decision(), orders(), fills(), positions=None)
    assert got.loc[SYMBOL, "status"] == "not_sent"


def test_a_symbol_the_decision_does_not_order_has_no_outcome():
    got = order_outcomes(decision(order_quantity=0.0), orders(), fills(), positions=None)
    assert got.empty


def test_the_average_price_is_taken_from_the_fills_only():
    two = fills(("C", -30.0, 10.0), ("C", -70.0, 11.0))
    got = order_outcomes(decision(-100.0, position=100.0), orders(PRIMARY, child("day", "FILLED")), two, None)
    assert got.loc[SYMBOL, ["filled_quantity", "avg_px"]].tolist() == [-100.0, 10.7]


def test_an_order_without_fills_has_no_price():
    got = order_outcomes(decision(), *CASES["venue_cancelled"], positions=None)
    assert math.isnan(got.loc[SYMBOL, "avg_px"])


def test_a_restart_replacing_a_kept_primary_is_judged_on_all_the_bars_orders():
    replaced = orders(
        ("P", "P", None, "CANCELED", 100.0, True),
        ("Q", "Q", None, "INITIALIZED", 0.0, False),
        ("C", "Q", "day", "FILLED", 0.0, False),
    )
    got = order_outcomes(decision(), replaced, fills(("C", 100.0, 10.5)), positions=None)
    assert got.loc[SYMBOL, ["filled_quantity", "status"]].tolist() == [100.0, "filled"]


def test_more_filled_than_ordered_is_unverified():
    twice = fills(("C", 100.0, 10.5), ("D", 100.0, 10.5))
    two = orders(PRIMARY, child("day", "FILLED"), child("day", "FILLED", name="D"))
    assert order_outcomes(decision(), two, twice, positions=None).loc[SYMBOL, "status"] == "unverified"


def test_a_position_its_fills_do_not_explain_is_unverified():
    done = CASES["filled"]
    assert order_outcomes(decision(), *done, positions=pd.Series({SYMBOL: 100.0})).loc[SYMBOL, "status"] == "filled"
    assert order_outcomes(decision(), *done, positions=pd.Series({SYMBOL: 90.0})).loc[SYMBOL, "status"] == "unverified"
