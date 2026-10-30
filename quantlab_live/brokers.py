"""``brokers``: what quantlab-live knows of each broker.

This version holds the brokers' fee models, which replay uses to charge a
broker's commissions (``replay(broker=...)``). The IBKR model charges per raw
share, so a venue that keeps its positions in split-adjusted units (replay)
gives it the raw shares per unit of each instrument; live, a unit is a share.
"""

from __future__ import annotations

from collections.abc import Callable

from nautilus_trader.execution import FeeModel
from nautilus_trader.model import InstrumentId, Money

#: IBKR's fixed pricing for US stocks, in USD: per share, the minimum per
#: order, and the maximum per order as a fraction of the trade value.
IBKR_PER_SHARE = 0.005
IBKR_MINIMUM = 1.0
IBKR_MAXIMUM_RATE = 0.01
#: The client order id prefix of the orders a venue generates to close the
#: positions in an expired instrument (a delisting); they carry no fee.
EXPIRATION = "EXPIRATION"


def ibkr_commission(shares: float, value: float) -> float:
    """Return IBKR's fixed-pricing commission on an order, in USD.

    USD 0.005 per share, at least USD 1.00 and at most 1% of the trade value;
    when 1% of the value is below the minimum, the maximum is charged.

    Parameters
    ----------
    shares : float
        The order's raw shares, unsigned.
    value : float
        The order's trade value, unsigned.

    Examples
    --------
    >>> ibkr_commission(500, 5_250.0), ibkr_commission(40, 1_920.0), ibkr_commission(1_000, 2_000.0)
    (2.5, 1.0, 5.0)
    >>> ibkr_commission(10, 50.0)  # 1% of the value is below the minimum
    0.5
    >>> ibkr_commission(0, 0.0)
    0.0
    """
    if shares == 0:
        return 0.0
    return min(max(IBKR_PER_SHARE * shares, IBKR_MINIMUM), IBKR_MAXIMUM_RATE * value)


def is_expiration(order) -> bool:
    """Whether ``order`` is a venue's closing order for an expired instrument.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> from nautilus_trader.model import ClientOrderId
    >>> is_expiration(SimpleNamespace(client_order_id=ClientOrderId("EXPIRATION-SIM-1")))
    True
    """
    return order.client_order_id.value.startswith(EXPIRATION)


class IbkrFeeModel(FeeModel):
    """IBKR's fixed per-share commission, as a NautilusTrader ``FeeModel``.

    A fill is charged what it adds to its order's commission, so the minimum
    and the maximum apply once per order however it fills. Orders a venue
    generates for an expiring instrument (``EXPIRATION-*``) are free.

    Parameters
    ----------
    shares_per_unit : callable, optional
        ``shares_per_unit(instrument_id, ts_ns)``: the raw shares in one unit
        of quantity of the instrument at ``ts_ns`` (an order is charged at its
        creation time). Defaults to one share per unit.

    Examples
    --------
    ::

        engine.add_venue(..., fee_model=IbkrFeeModel(splits.shares_per_unit))
    """

    def __new__(cls, *args, **kwargs):
        # The native base takes no arguments; __init__ keeps them.
        return super().__new__(cls)

    def __init__(self, shares_per_unit: Callable[[InstrumentId, int], float] | None = None):
        super().__init__()
        self._shares_per_unit = shares_per_unit or (lambda instrument_id, ts_ns: 1.0)

    def get_commission(self, order, fill_quantity, fill_px, instrument) -> Money:
        currency = instrument.quote_currency
        if is_expiration(order):
            return Money(0, currency)
        per_unit = self._shares_per_unit(order.instrument_id, order.ts_init)
        filled = float(order.filled_qty)
        filled_value = filled * float(order.avg_px) if filled else 0.0
        units = filled + float(fill_quantity)
        value = filled_value + float(fill_quantity) * float(fill_px)
        charged = ibkr_commission(filled * per_unit, filled_value)
        return Money(ibkr_commission(units * per_unit, value) - charged, currency)


#: The fee model of each broker a replay can charge, by name.
FEE_MODELS: dict[str, type[FeeModel]] = {"ibkr": IbkrFeeModel}


def fee_model(broker: str, shares_per_unit: Callable[[InstrumentId, int], float] | None = None) -> FeeModel:
    """Return the fee model of ``broker``.

    Parameters
    ----------
    broker : {"ibkr"}
        The broker.
    shares_per_unit : callable, optional
        See ``IbkrFeeModel``.

    Raises
    ------
    ValueError
        For a broker without a fee model.

    Examples
    --------
    >>> type(fee_model("ibkr")).__name__
    'IbkrFeeModel'
    >>> fee_model("tradestation")
    Traceback (most recent call last):
    ...
    ValueError: broker must be one of ('ibkr',), not 'tradestation'
    """
    if broker not in FEE_MODELS:
        raise ValueError(f"broker must be one of {tuple(FEE_MODELS)}, not {broker!r}")
    return FEE_MODELS[broker](shares_per_unit)
