"""``trading``: the strategy, the execution algorithm and the actors of every run.

``parts`` builds them for a ``LiveNode`` or a ``BacktestEngine``.
"""

from quantlab_live.trading.components import TradingParts, parts

__all__ = ["TradingParts", "parts"]
