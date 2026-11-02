"""brokers: broker profiles and their security-map builders, on a tiny real quantlab run's listings."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.adapters.interactive_brokers import MarketDataType, SymbologyMethod
from nautilus_trader.model import InstrumentId

import quantlab_live
from quantlab_live import brokers, records
from quantlab_live.brokers import IbkrFeeModel, profile
from quantlab_live.live.config import LiveConfig
from quantlab_live.quantlab_gate import open_run
from quantlab_live.replay import IdentityMap
from tests.quantlab_run_fixture import build_quantlab_run

BARS = pd.bdate_range("2024-01-02", periods=6)
RENAMED = BARS[3]
#: Each permaticker's (start, ticker, exchange) spans, in the vendor's names.
LISTINGS = {
    10001: [(None, "FB", "NASDAQ"), (RENAMED.strftime("%Y-%m-%d"), "META", "NASDAQ")],
    10002: [(None, "BRK.B", "NYSE")],
    10003: [(None, "IMO", "NYSEMKT")],
    10004: [(None, "GLD", "NYSEARCA")],
    10005: [(None, "CBOE", "BATS")],
}
SYMBOLS = (*LISTINGS, 10006)  # quantlab cannot place 10006


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    close = {s: [20.0 + k] * len(BARS) for k, s in enumerate(SYMBOLS)}
    weights = {s: [1 / len(SYMBOLS)] + [np.nan] * (len(BARS) - 1) for s in SYMBOLS}
    return open_run(
        build_quantlab_run(tmp_path_factory.mktemp("listed"), BARS, close, close, weights, listings=LISTINGS)
    )


def _build(name, run, *days):
    """The map ``name``'s builder keeps after building on each of ``days`` in turn."""
    security_map = None
    for day in days:
        security_map = profile(name).build_map(run.listings(day), day, security_map)
    return security_map


def _ids(security_map):
    return {s: None if (i := security_map.to_id(s)) is None else i.value for s in SYMBOLS}


def test_ibkr_maps_symbols_to_simplified_primary_exchange_ids(run):
    security_map = _build("ibkr_paper", run, BARS[0])
    assert _ids(security_map) == {
        10001: "FB.NASDAQ",
        10002: "BRK-B.NYSE",
        10003: "IMO.AMEX",
        10004: "GLD.ARCA",
        10005: "CBOE.BATS",
        10006: None,
    }
    assert security_map.to_symbol(InstrumentId.from_str("BRK-B.NYSE")) == 10002
    assert security_map.to_symbol(InstrumentId.from_str("AAPL.NASDAQ")) is None
    assert _ids(_build("ibkr_real", run, BARS[0])) == _ids(security_map)


def test_ib_sandbox_maps_symbols_to_raw_ids_on_one_smart_venue(run):
    assert _ids(_build("ib_sandbox", run, BARS[0])) == {
        10001: "FB=STK.SMART",
        10002: "BRK B=STK.SMART",
        10003: "IMO=STK.SMART",
        10004: "GLD=STK.SMART",
        10005: "CBOE=STK.SMART",
        10006: None,
    }


@pytest.mark.parametrize(("name", "before", "after"), [
    ("ibkr_paper", "FB.NASDAQ", "META.NASDAQ"),
    ("ib_sandbox", "FB=STK.SMART", "META=STK.SMART"),
])
def test_a_renamed_ticker_still_maps_back_through_the_history(run, name, before, after, tmp_path):
    security_map = _build(name, run, BARS[2], RENAMED)
    assert security_map.to_id(10001) == InstrumentId.from_str(after)
    assert security_map.to_symbol(InstrumentId.from_str(before)) == 10001
    assert security_map.to_symbol(InstrumentId.from_str(after)) == 10001

    records.write_security_map(tmp_path, security_map)
    again = records.read_security_map(tmp_path)
    assert again.to_id(10001) == InstrumentId.from_str(after)
    assert again.to_symbol(InstrumentId.from_str(before)) == 10001


def test_a_symbol_unmapped_today_keeps_its_old_id_and_a_reused_ticker_maps_to_its_latest_holder():
    day1, day2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")
    def listings(rows):
        return pd.DataFrame(rows, index=pd.Index([1, 2], name="symbol"), columns=["ticker", "exchange"])

    build = profile("ibkr_paper").build_map
    security_map = build(listings([["XYZ", "NYSE"], ["ABC", "NASDAQ"]]), day1)
    # 1 moves over the counter; its old ticker goes to 2.
    security_map = build(listings([["XYZ", "OTC"], ["XYZ", "NYSE"]]), day2, security_map)
    assert security_map.to_id(1) is None
    assert security_map.to_id(2) == InstrumentId.from_str("XYZ.NYSE")
    assert security_map.to_symbol(InstrumentId.from_str("XYZ.NYSE")) == 2
    assert security_map.to_symbol(InstrumentId.from_str("ABC.NASDAQ")) == 2


def test_a_map_cannot_be_built_on_a_day_before_its_history():
    listings = pd.DataFrame({"ticker": ["XYZ"], "exchange": ["NYSE"]}, index=pd.Index([1], name="symbol"))
    build = profile("ibkr_paper").build_map
    security_map = build(listings, pd.Timestamp("2024-01-03"))
    with pytest.raises(ValueError, match="2024-01-03"):
        build(listings, pd.Timestamp("2024-01-02"), security_map)


def test_replay_maps_every_symbol_to_itself_on_the_sim_venue():
    identity = IdentityMap()
    assert identity.to_id(10002) == InstrumentId.from_str("10002.SIM")
    assert identity.to_symbol(InstrumentId.from_str("10002.SIM")) == 10002
    assert identity.to_symbol(InstrumentId.from_str("BRK-B.NYSE")) is None


LOADED = {InstrumentId.from_str("BRK-B.NYSE"), InstrumentId.from_str("META.NASDAQ")}


def test_ibkr_profiles_trade_through_ib_with_simplified_ids():
    paper, real = profile("ibkr_paper", port=4002, account_id="DU123"), profile("ibkr_real", port=4001, account_id="U123")
    assert (paper.real_money, real.real_money) == (False, True)
    for broker in (paper, real):
        (data,) = broker.data_clients(LOADED)
        (execution,) = broker.exec_clients(LOADED)
        assert (data.name, execution.name) == ("IB", "IB")
        assert not execution.simulated
        for client in (data, execution):
            provider = client.config.instrument_provider
            assert provider.symbology_method == SymbologyMethod.SIMPLIFIED
            assert provider.load_ids == LOADED
        assert data.config.client_id != execution.config.client_id
        assert isinstance(broker.fee_model(), IbkrFeeModel)
    assert paper.exec_clients(LOADED)[0].config.account_id == "DU123"
    assert real.exec_clients(LOADED)[0].config.account_id == "U123"


def test_ib_sandbox_takes_ib_data_and_fills_in_a_sandbox_on_the_smart_venue():
    loaded = {InstrumentId.from_str("BRK B=STK.SMART")}
    sandbox = profile("ib_sandbox", port=4002, starting_cash=10_000.0)
    assert not sandbox.real_money
    (data,) = sandbox.data_clients(loaded)
    (execution,) = sandbox.exec_clients(loaded)
    provider = data.config.instrument_provider
    assert provider.symbology_method == SymbologyMethod.RAW
    assert provider.symbol_to_mic_venue == {"": "SMART"}  # every stock on one venue
    assert provider.load_ids == loaded
    assert (execution.name, execution.simulated) == ("SANDBOX", True)
    assert execution.config.venue.value == "SMART"
    assert [str(m) for m in execution.config.starting_balances] == ["10000.00 USD"]
    assert isinstance(execution.config.fee_model, IbkrFeeModel)


def test_ib_sandbox_fills_on_its_own_account_whatever_ib_account_is_set():
    # TWS_ACCOUNT names the Gateway's IB account; the sandbox's simulated account is not it.
    sandbox = profile("ib_sandbox", port=4002, account_id="DU123", starting_cash=10_000.0)
    (execution,) = sandbox.exec_clients(set())
    assert "DU123" not in str(execution.config.account_id)


def test_ib_sandbox_needs_its_starting_cash():
    with pytest.raises(ValueError, match="starting_cash"):
        profile("ib_sandbox", port=4002).exec_clients(set())


@pytest.mark.parametrize("name", ["ibkr_paper", "ibkr_real", "ib_sandbox"])
def test_an_ib_client_needs_an_explicit_port(name):
    broker = profile(name, account_id="DU123", starting_cash=10_000.0)
    with pytest.raises(ValueError, match="port"):
        broker.data_clients(set())


@pytest.mark.parametrize(("name", "market_data"), [
    ("ibkr_paper", MarketDataType.DELAYED), ("ib_sandbox", MarketDataType.DELAYED),
    ("ibkr_real", MarketDataType.REALTIME),
])
def test_only_real_money_insists_on_real_time_market_data(name, market_data):
    # IB answers a delayed request with real-time data when the account subscribes to it.
    broker = profile(name, port=4002, account_id="DU123", starting_cash=10_000.0)
    (data,) = broker.data_clients(set())
    assert data.config.market_data_type == market_data


def test_a_profile_exists_for_real_brokers_only():
    with pytest.raises(ValueError, match="'sim'"):
        profile("sim")


def test_nothing_outside_brokers_names_a_broker_adapter():
    package = Path(quantlab_live.__file__).parent
    naming = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            if any(m.startswith("nautilus_trader.adapters") for m in modules):
                naming.add(path)
    assert naming == {Path(brokers.__file__)}


def test_a_second_partial_fill_is_charged_when_the_order_has_no_average_price_yet():
    # The sandbox's matching engine fills a market order level by level, and hands
    # the fee model an order already partly filled but without its average price.
    from types import SimpleNamespace

    from nautilus_trader.model import ClientOrderId, Currency, Price, Quantity

    order = SimpleNamespace(
        client_order_id=ClientOrderId("O-1"), instrument_id=InstrumentId.from_str("AAPL=STK.SMART"),
        ts_init=0, filled_qty=Quantity.from_int(80), avg_px=None,
    )
    instrument = SimpleNamespace(quote_currency=Currency.from_str("USD"))
    fee = IbkrFeeModel().get_commission(order, Quantity.from_int(45), Price.from_str("337.54"), instrument)
    # 125 shares at USD 0.005 is 0.625, below the order's USD 1.00 minimum, charged on its first fill.
    assert float(fee.as_decimal()) == 0.0


@pytest.mark.parametrize(("name", "market_data", "expected"), [
    ("ibkr_real", "delayed", MarketDataType.DELAYED), ("ibkr_paper", "realtime", MarketDataType.REALTIME),
])
def test_live_toml_chooses_real_time_or_delayed_market_data_on_any_profile(name, market_data, expected):
    config = LiveConfig(run="run", broker=name, gateway_port=4002, account_id="DU123", market_data=market_data)
    (data,) = config.profile().data_clients(set())
    assert data.config.market_data_type == expected


def test_live_toml_refuses_an_unknown_market_data_type():
    with pytest.raises(ValueError, match="market_data must be 'realtime' or 'delayed'"):
        LiveConfig(run="run", market_data="frozen")
