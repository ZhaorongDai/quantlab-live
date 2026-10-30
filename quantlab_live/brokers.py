"""``brokers``: what quantlab-live knows of each broker.

A broker is a profile, ``profile(name) -> BrokerProfile``: its NautilusTrader
data and execution clients (with their instrument providers), its
security-map builder and its fee model. Nothing outside this module names a
broker's adapter, so the strategy, the day, replay and reconciliation do not
change when the broker does. Profiles exist for real brokers only:

- ``ibkr_paper`` and ``ibkr_real``: Interactive Brokers through the IB
  Gateway, with IB's simplified primary-exchange instrument ids
  (``BRK-B.NYSE``);
- ``ib_sandbox``: IB's market data with NautilusTrader's sandbox execution
  client, on IB's raw ids on one ``SMART`` venue (``BRK B=STK.SMART``), so
  one sandbox venue and one account cover every instrument.

A security map is built before the node starts from the run's listings
(``Run.listings(date)``: the vendor's ticker and exchange names), on top of
the map of the day before, so an id a renamed ticker had still maps back.

The brokers' fee models are here too, which replay also uses to charge a
broker's commissions (``replay(broker=...)``). The IBKR model charges per raw
share, so a venue that keeps its positions in split-adjusted units (replay)
gives it the raw shares per unit of each instrument; live, a unit is a share.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

import pandas as pd
from nautilus_trader.adapters.interactive_brokers import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersDataClientFactory,
    InteractiveBrokersExecutionClientConfig,
    InteractiveBrokersExecutionClientFactory,
    InteractiveBrokersInstrumentProviderConfig,
    SymbologyMethod,
)
from nautilus_trader.adapters.sandbox import SandboxExecutionClientConfig, SandboxExecutionClientFactory
from nautilus_trader.execution import FeeModel
from nautilus_trader.model import AccountType, Currency, InstrumentId, Money, OmsType, Venue

from quantlab_live.records import SecurityMap

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


#: IB's primary-exchange code of each exchange name the listings use; a
#: listing on another exchange (over the counter, say) is unmapped.
IB_EXCHANGES = {
    "NYSE": "NYSE",
    "NASDAQ": "NASDAQ",
    "NYSEMKT": "AMEX",
    "NYSEARCA": "ARCA",
    "BATS": "BATS",
}
#: The one venue of every ``ib_sandbox`` instrument.
SMART = Venue("SMART")
USD = Currency.from_str("USD")

#: ``build_map(listings, date, history=None) -> SecurityMap``.
MapBuilder = Callable[[pd.DataFrame, Any, SecurityMap | None], SecurityMap]


def ib_local_symbol(ticker: str) -> str:
    """Return IB's local symbol of a listing's ticker: a class after a space.

    Examples
    --------
    >>> ib_local_symbol("BRK.B"), ib_local_symbol("AAPL")
    ('BRK B', 'AAPL')
    """
    return ticker.replace(".", " ")


def primary_exchange_id(ticker: str | None, exchange: str | None) -> InstrumentId | None:
    """Return IB's simplified id of a listing, ``<symbol>.<primary exchange>``.

    The symbol is IB's local symbol with its space as a hyphen.

    Examples
    --------
    >>> primary_exchange_id("BRK.B", "NYSE"), primary_exchange_id("IMO", "NYSEMKT")
    (InstrumentId('BRK-B.NYSE'), InstrumentId('IMO.AMEX'))
    >>> primary_exchange_id("XYZ", "OTC") is None, primary_exchange_id("FB", None) is None
    (True, True)
    """
    if ticker is None or exchange not in IB_EXCHANGES:
        return None
    return InstrumentId.from_str(f"{ib_local_symbol(ticker).replace(' ', '-')}.{IB_EXCHANGES[exchange]}")


def smart_id(ticker: str | None, exchange: str | None) -> InstrumentId | None:
    """Return IB's raw id of a listing on the ``SMART`` venue, ``<local symbol>=STK.SMART``.

    Examples
    --------
    >>> smart_id("BRK.B", "NYSE")
    InstrumentId('BRK B=STK.SMART')
    >>> smart_id("XYZ", "OTC") is None
    True
    """
    if ticker is None or exchange not in IB_EXCHANGES:
        return None
    return InstrumentId.from_str(f"{ib_local_symbol(ticker)}=STK.{SMART}")


def map_builder(listing_id: Callable[[str | None, str | None], InstrumentId | None]) -> MapBuilder:
    """Return the security-map builder that gives each listing ``listing_id(ticker, exchange)``.

    The builder takes ``Run.listings(date)``, the date and the map of the
    previous build (``None`` for the first), and returns the map on the date
    with the previous map's history kept.

    Examples
    --------
    >>> listings = pd.DataFrame({"ticker": ["META", None], "exchange": ["NASDAQ", None]},
    ...                         index=pd.Index([1, 2], name="symbol"))
    >>> security_map = map_builder(primary_exchange_id)(listings, "2024-01-02")
    >>> security_map.to_id(1), security_map.to_id(2)
    (InstrumentId('META.NASDAQ'), None)
    """

    def build_map(listings: pd.DataFrame, date, history: SecurityMap | None = None) -> SecurityMap:
        ids = {
            int(symbol): listing_id(ticker, exchange)
            for symbol, ticker, exchange in zip(listings.index, listings["ticker"], listings["exchange"])
        }
        return (history or SecurityMap()).updated(date, ids)

    return build_map


#: The trading mode of the IB Gateway's and TWS's default API ports.
IB_PORT_MODES = {4001: "live", 7496: "live", 4002: "paper", 7497: "paper"}


def ib_account_mode(account_id: str | None) -> str | None:
    """Return the trading mode of an IB account id: ``paper`` (``DU…``), ``live`` (``U…``) or ``None``.

    NautilusTrader's account ids carry the venue first (``IB-DU123``); it is ignored.

    Examples
    --------
    >>> ib_account_mode("DU123"), ib_account_mode("IB-U456"), ib_account_mode("SANDBOX-001")
    ('paper', 'live', None)
    """
    if not account_id:
        return None
    number = account_id.rsplit("-", 1)[-1]
    if number.startswith("DU"):
        return "paper"
    if number.startswith("U"):
        return "live"
    return None


@dataclass(frozen=True)
class Client:
    """A NautilusTrader client of a node: ``LiveNodeBuilder.add_data_client`` or ``add_exec_client``.

    Attributes
    ----------
    name : str
        The client's name.
    factory : object
        Its NautilusTrader client factory.
    config : object
        Its NautilusTrader configuration, instrument provider included.
    simulated : bool
        Whether it is a simulated execution client
        (``add_simulated_exec_client``).
    account_id : str, optional
        The NautilusTrader account id an execution client trades for
        (``IB-DU123``): the account the broker must report at the start.
    """

    name: str
    factory: Any
    config: Any
    simulated: bool = False
    account_id: str | None = None


@dataclass(frozen=True)
class BrokerProfile:
    """A broker, as a node and replay use it.

    The clients are functions of the instruments to load, which are known
    only once the security map is built: the held and target instruments,
    loaded through the broker's own instrument provider.

    Attributes
    ----------
    name : str
        The profile's name.
    real_money : bool
        Whether the profile trades real money.
    data_clients, exec_clients : callable
        ``(instrument_ids) -> tuple[Client, ...]``.
    build_map : callable
        ``build_map(listings, date, history=None) -> SecurityMap``; see
        ``map_builder``.
    fee_model : callable
        ``fee_model(shares_per_unit=None) -> FeeModel``.
    simulated : bool
        Whether its orders fill in a simulated venue (a sandbox), never at
        the broker.
    """

    name: str
    real_money: bool
    data_clients: Callable[[Collection[InstrumentId]], tuple[Client, ...]]
    exec_clients: Callable[[Collection[InstrumentId]], tuple[Client, ...]]
    build_map: MapBuilder
    fee_model: Callable[..., FeeModel]
    simulated: bool = False


#: The profiles ``profile`` knows.
PROFILES = ("ibkr_paper", "ibkr_real", "ib_sandbox")


def profile(
    name: str,
    *,
    host: str | None = None,
    port: int | None = None,
    client_id: int = 1,
    account_id: str | None = None,
    starting_cash: float | None = None,
) -> BrokerProfile:
    """Return the broker profile ``name``.

    Parameters
    ----------
    name : {"ibkr_paper", "ibkr_real", "ib_sandbox"}
        The profile.
    host, port : optional
        The IB Gateway's API address. Building a client needs the port:
        NautilusTrader would otherwise fall back to its own default port,
        which may be the other trading mode's.
    client_id : int, default 1
        The IB API client id of the data client; the IB execution client
        takes the next one.
    account_id : str, optional
        The IB account (``ibkr_*``) or the sandbox account (``ib_sandbox``).
    starting_cash : float, optional
        The sandbox account's starting USD cash; ``ib_sandbox`` requires it.

    Raises
    ------
    ValueError
        For an unknown profile.

    Examples
    --------
    >>> paper = profile("ibkr_paper", port=4002, account_id="DU123")
    >>> paper.real_money, [c.name for c in paper.exec_clients({InstrumentId.from_str("BRK-B.NYSE")})]
    (False, ['IB'])
    >>> profile("ib_sandbox", starting_cash=10_000).exec_clients(set())[0].config.venue
    Venue('SMART')
    >>> profile("ibkr_real", account_id="U123").exec_clients(set())
    Traceback (most recent call last):
    ...
    ValueError: the ibkr_real profile needs the IB Gateway's port
    >>> profile("sim")
    Traceback (most recent call last):
    ...
    ValueError: profile must be one of ('ibkr_paper', 'ibkr_real', 'ib_sandbox'), not 'sim'
    """
    if name not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, not {name!r}")
    gateway = {"host": host, "port": port}
    sandbox = name == "ib_sandbox"

    def provider(instrument_ids: Collection[InstrumentId]) -> InteractiveBrokersInstrumentProviderConfig:
        if sandbox:
            return InteractiveBrokersInstrumentProviderConfig(
                symbology_method=SymbologyMethod.RAW,
                load_ids=set(instrument_ids),
                symbol_to_mic_venue={"": SMART.value},  # every symbol's prefix
            )
        return InteractiveBrokersInstrumentProviderConfig(
            symbology_method=SymbologyMethod.SIMPLIFIED, load_ids=set(instrument_ids)
        )

    def need_port() -> None:
        if port is None:
            raise ValueError(f"the {name} profile needs the IB Gateway's port")

    def data_clients(instrument_ids: Collection[InstrumentId]) -> tuple[Client, ...]:
        need_port()
        config = InteractiveBrokersDataClientConfig(
            **gateway, client_id=client_id, instrument_provider=provider(instrument_ids)
        )
        return (Client("IB", InteractiveBrokersDataClientFactory(), config),)

    def exec_clients(instrument_ids: Collection[InstrumentId]) -> tuple[Client, ...]:
        if not sandbox:
            need_port()
            config = InteractiveBrokersExecutionClientConfig(
                **gateway,
                client_id=client_id + 1,
                account_id=account_id,
                instrument_provider=provider(instrument_ids),
            )
            account = None if account_id is None else f"IB-{account_id}"
            return (Client("IB", InteractiveBrokersExecutionClientFactory(), config, account_id=account),)
        if starting_cash is None:
            raise ValueError("the ib_sandbox profile needs its starting_cash")
        config = SandboxExecutionClientConfig(
            venue=SMART,
            starting_balances=[Money(starting_cash, USD)],
            **({} if account_id is None else {"account_id": account_id}),
            base_currency=USD,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            fee_model=IbkrFeeModel(),
        )
        return (Client("SANDBOX", SandboxExecutionClientFactory(), config, simulated=True),)

    return BrokerProfile(
        name=name,
        real_money=name == "ibkr_real",
        data_clients=data_clients,
        exec_clients=exec_clients,
        build_map=map_builder(smart_id if sandbox else primary_exchange_id),
        fee_model=IbkrFeeModel,
        simulated=sandbox,
    )
