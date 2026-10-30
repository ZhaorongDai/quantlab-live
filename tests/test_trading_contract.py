"""trading: the strategy and its execution algorithm do not know which run they are in."""

from __future__ import annotations

import ast
from pathlib import Path

from nautilus_trader.trading import Strategy

from quantlab_live import trading
from quantlab_live.trading.strategy import QuantlabStrategy

TRADING = Path(trading.__file__).parent
#: Modules a run-agnostic trading module must not import.
RUN_SPECIFIC = ("quantlab_live.replay", "quantlab_live.live", "nautilus_trader.backtest")


def test_trading_imports_nothing_run_specific():
    for path in TRADING.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                assert not module.startswith(RUN_SPECIFIC), f"{path.name} imports {module}"


def test_the_strategy_takes_decision_bars_through_on_data_only():
    handlers = {
        name
        for name in vars(QuantlabStrategy)
        if name.startswith("on_") and hasattr(Strategy, name) and name not in ("on_start", "on_stop")
    }
    assert handlers == {"on_data"}


def test_live_and_replay_build_their_trading_components_through_parts_only():
    package = TRADING.parent
    built = ("QuantlabStrategy", "ScheduledExecution", "DecisionSource")
    for path in package.rglob("*.py"):
        if path == TRADING / "components.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in built, f"{path.relative_to(package)} builds {node.func.id}"
    sources = {name: (package / name).read_text() for name in ("replay.py", "live/node.py")}
    assert all("trading.parts(" in text for text in sources.values())
