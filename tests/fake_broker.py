"""A broker profile for in-process node tests: a Python data client and NautilusTrader's sandbox.

The data client stands in for a broker's market data: it serves an equity
instrument for any id it is asked for (at connect for the ids to load, and at
runtime on a request) and, on a quote subscription, one quote around the
instrument's price. The orders fill in NautilusTrader's sandbox execution
client on the ``SMART`` venue, with the IBKR fee model, as in ``ib_sandbox``.
"""

from __future__ import annotations

from nautilus_trader.adapters.sandbox import SandboxExecutionClientConfig, SandboxExecutionClientFactory
from nautilus_trader.config import DataClientConfig
from nautilus_trader.live import InstrumentResponse
from nautilus_trader.live.clients import DataClientFactory, MarketDataClient
from nautilus_trader.model import (
    AccountType,
    Currency,
    Equity,
    Money,
    OmsType,
    Price,
    Quantity,
    QuoteTick,
    Symbol,
)

from quantlab_live.brokers import SMART, BrokerProfile, Client, IbkrFeeModel, map_builder, smart_id

USD = Currency.from_str("USD")


class FakeDataConfig(DataClientConfig):
    pass


def equity(instrument_id) -> Equity:
    return Equity(instrument_id, Symbol(instrument_id.symbol.value), USD, 2, Price(0.01, 2), 0, 0)


class FakeData(MarketDataClient):
    """Instruments for any id, and a quote at the configured price on a subscription."""

    def __init__(self, *, prices, load, requests, **kwargs):
        super().__init__(**kwargs)
        self._prices, self._load, self._requests = prices, load, requests

    async def _connect(self):
        for instrument_id in self._load:
            self._handle_instrument(equity(instrument_id))

    async def _disconnect(self):
        pass

    async def _request_instrument(self, request):
        for instrument_id in request.instrument_ids:
            self._requests.append(instrument_id)
            instrument = equity(instrument_id)
            self._handle_instrument(instrument)
            now = self.clock.timestamp_ns()
            self._handle_response(InstrumentResponse(self.client_id, instrument_id, instrument, request.request_id, now))

    async def _subscribe_quotes(self, command):
        for instrument_id in command.instrument_ids:
            price = self._prices.get(instrument_id.value, 10.0)
            now = self.clock.timestamp_ns()
            self._handle_data(
                QuoteTick(
                    instrument_id, Price(price - 0.01, 2), Price(price + 0.01, 2),
                    Quantity(10**6, 0), Quantity(10**6, 0), now, now,
                )
            )

    async def _subscribe_trades(self, command):
        pass


class FakeDataFactory(DataClientFactory):
    def __init__(self, prices, load, requests):
        self._prices, self._load, self._requests = prices, load, requests

    def create(self, *, name, config, cache, clock):
        return FakeData(
            prices=self._prices, load=self._load, requests=self._requests,
            name=name, config=config, cache=cache, clock=clock, venue=SMART,
        )


def fake_profile(prices: dict[str, float], *, starting_cash: float = 10_000.0, account_id: str | None = None):
    """Return a broker profile with the fake data client and a sandbox on ``SMART``.

    ``prices`` gives the quote price by instrument id text. ``account_id``
    names an account the broker must report: the sandbox never reports it,
    so the node's start check refuses. Returns the profile and the list of
    the instruments requested from it at runtime.
    """
    requests: list = []

    def data_clients(instrument_ids):
        return (Client("FAKE", FakeDataFactory(prices, list(instrument_ids), requests), FakeDataConfig()),)

    def exec_clients(instrument_ids):
        config = SandboxExecutionClientConfig(
            venue=SMART, starting_balances=[Money(starting_cash, USD)], base_currency=USD,
            oms_type=OmsType.NETTING, account_type=AccountType.MARGIN, fee_model=IbkrFeeModel(),
        )
        return (Client("SANDBOX", SandboxExecutionClientFactory(), config, simulated=True, account_id=account_id),)

    profile = BrokerProfile(
        name="ibkr_paper" if account_id else "fake",
        real_money=False,
        data_clients=data_clients,
        exec_clients=exec_clients,
        build_map=map_builder(smart_id),
        fee_model=IbkrFeeModel,
        simulated=account_id is None,
    )
    return profile, requests
