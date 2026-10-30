"""The trading components of a node or a replay, built by one function, ``parts``.

A live node and a replay run the same ``QuantlabStrategy`` and
``ScheduledExecution``; they differ only in the target source, the venue and
how the decision bars arrive. ``parts`` builds the components of both, so
what a replay runs is what trades live:

- in a replay the bars are added to the ``BacktestEngine`` as data;
- live, the ``DecisionSource`` Actor publishes the day's bars once the node
  has started (after start-up reconciliation), and only if its start check
  passes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from nautilus_trader.common import DataActor, DataActorConfig, Environment
from nautilus_trader.model import ActorId, ClientId, CustomData, DataType, ExecAlgorithmId, InstrumentId

from quantlab_live.trading.decision_bar import DecisionBar
from quantlab_live.trading.scheduled_execution import ExecutionPlan, ScheduledExecution, Session
from quantlab_live.trading.strategy import DecisionRecords, QuantlabStrategy, SecurityMap
from quantlab_live.trading.targets import ConstructorTargets, RebalanceTableTargets

#: The id every order names to reach the execution algorithm.
EXEC_ALGORITHM_ID = ExecAlgorithmId("SCHEDULED")
#: How long after its start the ``DecisionSource`` publishes, in nanoseconds.
PUBLISH_DELAY_NS = 1_000_000_000


class DecisionSource(DataActor):
    """Publish the day's ``DecisionBar`` records once the node runs (live).

    Shortly after the node starts, the actor runs its start ``check`` on the
    node's cache. When the check returns a reason, nothing is published:
    ``refused`` keeps the reason and ``on_refused(reason)`` is called (a
    node stops itself there). Otherwise every bar is published as data, so
    the strategy decides on it in ``on_data``, exactly as in a replay.

    Parameters
    ----------
    bars : sequence of DecisionBar
        The bars to publish.
    check : callable, optional
        ``check(cache) -> str | None``: why the node must not trade, or
        ``None``.
    on_refused : callable, optional
        Called with the reason when ``check`` refuses.

    Examples
    --------
    ::

        source = DecisionSource([bar], check=lambda cache: None)
    """

    def __new__(cls, *args, **kwargs):
        # The native base takes only a configuration; __init__ passes it.
        return super().__new__(cls)

    def __init__(
        self,
        bars: Sequence[DecisionBar],
        *,
        check: Callable[[Any], str | None] | None = None,
        on_refused: Callable[[str], None] | None = None,
    ):
        super().__init__(DataActorConfig(actor_id=ActorId("DECISION-SOURCE-001")))
        self._bars = list(bars)
        self._check = check
        self._on_refused = on_refused
        #: Why the start check refused the node, or ``None``.
        self.refused: str | None = None
        #: Whether the bars were published.
        self.published = False

    def on_start(self) -> None:
        # The strategy subscribes in its own on_start, after the actors start.
        self.clock.set_time_alert_ns("publish", self.clock.timestamp_ns() + PUBLISH_DELAY_NS, self._publish)

    def _publish(self, event) -> None:
        reason = None
        try:
            reason = self._check(self.cache) if self._check is not None else None
        except Exception as error:  # a check that fails refuses
            reason = f"the start check failed: {error!r}"
        if reason is not None:
            self.refused = reason
            self.log.error(f"the node is refused: {reason}")
            if self._on_refused is not None:
                self._on_refused(reason)
            return
        now = self.clock.timestamp_ns()
        for bar in self._bars:
            record = DecisionBar(bar=bar.bar, payload=bar.payload, ts_event=now, ts_init=now)
            self.publish_data(DataType(DecisionBar.__name__), CustomData(DataType(DecisionBar.__name__), record))
        self.published = True


@dataclass
class TradingParts:
    """The trading components of one node or replay.

    Attributes
    ----------
    strategy : QuantlabStrategy
    exec_algorithm : ScheduledExecution
    decision_source : DecisionSource or None
        Live, the actor that runs the start check and then publishes the
        day's bars; ``None`` in a replay.
    preloaded : callable
        ``preloaded(start, end)``: a context manager the engine runs inside
        (a closed-loop replay preloads quantlab's decision inputs over its
        window; live reads them directly).
    """

    strategy: QuantlabStrategy
    exec_algorithm: ScheduledExecution
    decision_source: DecisionSource | None = None
    preloaded: Callable[[Any, Any], contextlib.AbstractContextManager] = lambda start, end: contextlib.nullcontext()

    def add_to(self, host) -> None:
        """Add every component to ``host``, a ``LiveNode`` or a ``BacktestEngine``."""
        if self.decision_source is not None:
            host.add_actor(self.decision_source)
        host.add_exec_algorithm(self.exec_algorithm)
        host.add_strategy(self.strategy)


def parts(
    run,
    records: DecisionRecords,
    *,
    loop: str,
    plan: str | ExecutionPlan,
    security_map: SecurityMap,
    environment: Environment,
    dry_run: bool = False,
    end=None,
    bars: Sequence[DecisionBar] | None = None,
    check: Callable[[Any], str | None] | None = None,
    on_refused: Callable[[str], None] | None = None,
    external_order_claims: Sequence[InstrumentId] = (),
    instrument_client: ClientId | None = None,
    session: Session | None = None,
) -> TradingParts:
    """Return the strategy, execution algorithm and actors of a node or a replay.

    Parameters
    ----------
    run : quantlab_gate.Run
        The quantlab run traded.
    records : DecisionRecords
        Where the strategy keeps its decisions.
    loop : {"closed", "open"}
        The target source: quantlab's construction on the holdings
        (``ConstructorTargets``; replay's closed loop, and live) or the run's
        rebalance table (``RebalanceTableTargets``).
    plan : str or ExecutionPlan
        The execution plan of ``ScheduledExecution``.
    security_map : SecurityMap
        Symbols to instruments and back.
    environment : nautilus_trader.common.Environment
        ``BACKTEST`` for a replay; ``LIVE`` or ``SANDBOX`` for a node.
    dry_run : bool, default False
        Decide and write the decision, submit nothing.
    end : datetime-like, optional
        A closed-loop replay's last bar (``Run.targets``).
    bars : sequence of DecisionBar, optional
        Live: the bars the ``DecisionSource`` publishes. ``None`` adds no
        actor (a replay adds its bars as data).
    check, on_refused : callable, optional
        The ``DecisionSource``'s start check and what to do on a refusal.
    external_order_claims : sequence of InstrumentId, optional
        The instruments the strategy claims at reconciliation.
    instrument_client : ClientId, optional
        The data client that loads an instrument at runtime.
    session : Session, optional
        A node's one session (``ScheduledExecution``); a replay's orders
        trade in the session of their time.

    Raises
    ------
    ValueError
        For an unknown ``loop``.

    Examples
    --------
    ::

        trading = parts(run, decisions, loop="closed", plan="open_auction",
                        security_map=security_map, environment=Environment.LIVE, bars=[bar])
        trading.add_to(node)
    """
    if loop not in ("closed", "open"):
        raise ValueError(f"loop must be 'closed' or 'open', not {loop!r}")
    plan = plan if isinstance(plan, ExecutionPlan) else ExecutionPlan(plan)
    if loop == "closed":
        inputs = run.targets("closed", end=end)
        targets = ConstructorTargets(inputs)

        def preloaded(start, end):
            return inputs.preloaded(pd.Timestamp(start), pd.Timestamp(end))
    else:
        targets = RebalanceTableTargets(run.targets("open"))

        def preloaded(start, end):
            return contextlib.nullcontext()

    strategy = QuantlabStrategy(
        targets,
        security_map,
        EXEC_ALGORITHM_ID,
        records,
        external_order_claims=list(external_order_claims),
        instrument_client=instrument_client,
        dry_run=dry_run,
    )
    source = None if bars is None else DecisionSource(bars, check=check, on_refused=on_refused)
    return TradingParts(strategy, ScheduledExecution(EXEC_ALGORITHM_ID, plan, environment, session), source, preloaded)
