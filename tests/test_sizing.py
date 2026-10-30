"""Order sizing: target weights sized in whole shares on the strategy capital, and the resume formula."""

from __future__ import annotations

import pandas as pd

from quantlab_live.trading.strategy import remaining_orders, size_orders, strategy_capital

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



def _decision(target, order, skip=None):
    frame = pd.DataFrame({"target_quantity": target, "order_quantity": order})
    frame["skip"] = skip or [""] * len(frame)
    return frame.rename_axis("symbol")


def test_a_resumed_decision_orders_its_target_less_the_position_and_working_orders():
    decision = _decision({1: 500.0, 2: 0.0, 3: 40.0}, {1: 500.0, 2: -150.0, 3: 40.0})
    position = pd.Series({1: 200.0, 2: 150.0})
    working = pd.Series({1: 100.0, 2: -150.0})
    # 1: 500 - 200 - 100; 2: sold by its working order; 3: nothing has traded.
    assert remaining_orders(decision, position, working).to_dict() == {1: 200.0, 3: 40.0}


def test_a_resumed_decision_trades_only_the_symbols_it_ordered():
    # 2 was kept (no order); a later change of its position is not traded back.
    decision = _decision({1: 500.0, 2: 30.0}, {1: 500.0, 2: 0.0})
    position = pd.Series({1: 500.0, 2: 10.0})
    assert remaining_orders(decision, position, pd.Series(dtype=float)).empty


def test_a_resumed_decision_never_trades_a_skipped_symbol():
    decision = _decision({1: 500.0, 2: 40.0}, {1: 500.0, 2: 40.0}, skip=["unmapped", "foreign"])
    assert remaining_orders(decision, pd.Series(dtype=float), pd.Series(dtype=float)).empty


def test_resuming_after_everything_filled_orders_nothing():
    decision = _decision({1: 500.0, 2: 0.0}, {1: 500.0, 2: -150.0})
    assert remaining_orders(decision, pd.Series({1: 500.0}), pd.Series(dtype=float)).empty
