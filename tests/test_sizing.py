"""Order sizing: target weights sized in whole shares on the strategy capital."""

from __future__ import annotations

import pandas as pd

from quantlab_live.trading.strategy import size_orders, strategy_capital

CLOSE = pd.Series({1: 10.0, 2: 20.0})
HELD = pd.DataFrame({"quantity": [5.0]}, index=[2])


def test_the_strategy_capital_is_the_initial_capital_plus_its_pnl():
    assert strategy_capital(account_equity=50_000.0, initial_capital=10_000.0, pnl=1_250.0) == 11_250.0


def test_the_strategy_capital_is_capped_by_account_equity():
    assert strategy_capital(account_equity=9_000.0, initial_capital=10_000.0, pnl=500.0) == 9_000.0


def test_without_an_initial_capital_orders_size_on_account_equity():
    assert strategy_capital(account_equity=9_000.0, initial_capital=None, pnl=500.0) == 9_000.0


def test_a_loss_lowers_the_strategy_capital():
    assert strategy_capital(account_equity=50_000.0, initial_capital=10_000.0, pnl=-2_000.0) == 8_000.0


def test_targets_are_whole_shares_of_the_capital_at_the_bars_closes():
    weights = pd.Series({1: 0.55, 2: 0.25})
    decision = size_orders(weights, CLOSE, HELD, capital=1_000.0)
    # 0.55 x 1,000 / 10 = 55 shares; 0.25 x 1,000 / 20 = 12.5, truncated to 12.
    assert decision["target_quantity"].to_dict() == {1: 55.0, 2: 12.0}
    assert decision["order_quantity"].to_dict() == {1: 55.0, 2: 7.0}
    assert decision["position_quantity"].to_dict() == {1: 0.0, 2: 5.0}


def test_a_short_target_truncates_toward_zero():
    decision = size_orders(pd.Series({1: -0.055}), CLOSE, HELD.iloc[:0], capital=1_000.0)
    assert decision.loc[1, "target_quantity"] == -5.0


def test_a_nan_weight_keeps_the_holding_and_an_unweighted_holding_is_kept():
    decision = size_orders(pd.Series({1: float("nan")}), CLOSE, HELD, capital=1_000.0)
    assert decision["order_quantity"].to_dict() == {1: 0.0, 2: 0.0}
    assert decision.loc[2, "target_quantity"] == 5.0

