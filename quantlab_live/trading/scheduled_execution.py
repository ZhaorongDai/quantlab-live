"""``ScheduledExecution``: sends a decision's orders at the planned time.

Orders reach it through their ``exec_algorithm_id`` only; the plan lives in
the algorithm. It is stateless: an order's action follows from now, the plan
and the exchange's sessions, and the order itself is read back from the
cache when its time comes.

This first version knows one plan, the opening auction, and its backtest
form: at the next session's opening print it spawns a ``DAY`` market order
for the whole quantity, which fills at that print. A security without a
print at the open (halted, or no data) is left unsent, so no fill is ever
taken at an earlier, stale price.
"""

from __future__ import annotations

import exchange_calendars
import pandas as pd
from nautilus_trader.model import ClientOrderId, ExecAlgorithmId, TimeInForce
from nautilus_trader.trading import ExecutionAlgorithm, ExecutionAlgorithmConfig

#: The plans this version knows.
PLANS = ("open_auction",)
#: The prefix of an order's alert name; its client order id follows it.
ALERT = "open-auction:"


def next_open_ns(now_ns: int) -> int:
    """Return the UNIX nanoseconds of the first XNYS session open after ``now_ns``.

    Examples
    --------
    >>> friday_close = pd.Timestamp("2024-01-05 16:00", tz="America/New_York").value
    >>> pd.Timestamp(next_open_ns(friday_close), tz="UTC").tz_convert("America/New_York")
    Timestamp('2024-01-08 09:30:00-0500', tz='America/New_York')
    """
    calendar = exchange_calendars.get_calendar("XNYS")
    return calendar.next_open(pd.Timestamp(now_ns, tz="UTC")).value


class ScheduledExecution(ExecutionAlgorithm):
    """Send each order at its plan's time.

    Parameters
    ----------
    exec_algorithm_id : ExecAlgorithmId
        The id orders name to reach this algorithm.
    plan : {"open_auction"}
        When the orders trade.

    Raises
    ------
    ValueError
        For a plan this version does not know.

    Examples
    --------
    >>> ScheduledExecution(ExecAlgorithmId("SCHEDULED"), plan="15:30")
    Traceback (most recent call last):
    ...
    ValueError: plan must be one of ('open_auction',), not '15:30'
    """

    def __new__(cls, *args, **kwargs):
        # The native base takes only a configuration; __init__ passes it.
        return super().__new__(cls)

    def __init__(self, exec_algorithm_id: ExecAlgorithmId, plan: str = "open_auction"):
        if plan not in PLANS:
            raise ValueError(f"plan must be one of {PLANS}, not {plan!r}")
        super().__init__(ExecutionAlgorithmConfig(exec_algorithm_id=exec_algorithm_id))
        self._plan = plan

    def on_order(self, order) -> None:
        self.clock.set_time_alert_ns(
            ALERT + order.client_order_id.value,
            next_open_ns(self.clock.timestamp_ns()),
            self._at_open,
        )

    def _at_open(self, event) -> None:
        primary = self.cache.order(ClientOrderId(event.name.removeprefix(ALERT)))
        last = self.cache.trade(primary.instrument_id)
        if last is None or last.ts_event != event.ts_event:
            self.log.warning(f"{primary.instrument_id} has no opening print; {primary.client_order_id} not sent")
            return
        child = self.spawn_market(
            primary, primary.quantity, TimeInForce.DAY, tags=list(primary.tags or [])
        )
        self.submit_order(child)
