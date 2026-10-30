"""``records``: quantlab-live's own durable records, typed parquet in raw units.

Every record is in raw units (shares and prices as traded, not split
adjusted) and its rows are keyed by bar timestamp, so nothing assumes one
decision per day; files are per date. Layout of a live or replay directory:

- ``decisions/<date>.parquet``: the decision on each bar, by symbol;
- ``outcomes/<date>.parquet``: what each bar's orders did, by symbol;
- ``snapshots/<date>.parquet``: positions by symbol and cash, at a bar's close;
- ``equity.parquet``: the strategy's equity, one row per bar;
- ``security_map.parquet``: the security map's history (live only).

Each file is written to a temporary file, fsynced and renamed, so a crash
leaves either no file or a complete one. NautilusTrader owns orders, fills
and positions; these records hold what it does not.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nautilus_trader.model import InstrumentId

DECISIONS = "decisions"
OUTCOMES = "outcomes"
SNAPSHOTS = "snapshots"
EQUITY = "equity.parquet"
SECURITY_MAP = "security_map.parquet"

_KEY = [("bar", pa.timestamp("ns")), ("symbol", pa.int64())]
DECISION_SCHEMA = pa.schema(
    _KEY
    + [
        ("weight", pa.float64()),
        ("close", pa.float64()),
        ("position_quantity", pa.float64()),
        ("target_quantity", pa.float64()),
        ("order_quantity", pa.float64()),
    ]
)
OUTCOME_SCHEMA = pa.schema(
    _KEY
    + [
        ("order_quantity", pa.float64()),
        ("filled_quantity", pa.float64()),
        ("avg_px", pa.float64()),
        ("status", pa.string()),
    ]
)
SNAPSHOT_SCHEMA = pa.schema(_KEY + [("quantity", pa.float64())])
EQUITY_SCHEMA = pa.schema(
    [
        ("bar", pa.timestamp("ns")),
        ("cash", pa.float64()),
        ("holdings", pa.float64()),
        ("equity", pa.float64()),
    ]
)
SECURITY_MAP_SCHEMA = pa.schema(
    [
        ("symbol", pa.int64()),
        ("instrument_id", pa.string()),
        ("first", pa.date32()),
        ("last", pa.date32()),
    ]
)


@dataclass(frozen=True)
class Snapshot:
    """Positions and cash at a bar's close.

    Attributes
    ----------
    bar : pandas.Timestamp
        The bar.
    positions : pandas.DataFrame
        ``quantity`` by ``symbol``, signed raw shares; held symbols only.
    cash : float
        The account's cash.
    """

    bar: pd.Timestamp
    positions: pd.DataFrame
    cash: float


def write_decision(directory, bar: pd.Timestamp, decision: pd.DataFrame) -> Path:
    """Write the decision on ``bar``; a bar is decided at most once.

    Parameters
    ----------
    directory : path-like
        The live or replay directory.
    bar : pandas.Timestamp
        The decision bar.
    decision : pandas.DataFrame
        By ``symbol``: ``weight`` (NaN where kept), ``close``,
        ``position_quantity``, ``target_quantity`` and ``order_quantity``.

    Raises
    ------
    FileExistsError
        If the bar already has a decision.

    Examples
    --------
    >>> import tempfile
    >>> directory = tempfile.mkdtemp()
    >>> decision = pd.DataFrame(
    ...     {"weight": [0.5], "close": [10.0], "position_quantity": [0.0],
    ...      "target_quantity": [50.0], "order_quantity": [50.0]},
    ...     index=pd.Index([10001], name="symbol"),
    ... )
    >>> _ = write_decision(directory, pd.Timestamp("2024-01-02"), decision)
    >>> read_decision(directory, pd.Timestamp("2024-01-02"))["order_quantity"].to_dict()
    {10001: 50.0}
    >>> write_decision(directory, pd.Timestamp("2024-01-02"), decision)
    Traceback (most recent call last):
    ...
    FileExistsError: bar 2024-01-02 already has a decision at ...
    """
    path = _path(directory, DECISIONS, bar)
    if path.exists():
        raise FileExistsError(f"bar {pd.Timestamp(bar).date()} already has a decision at {path}")
    return _write(path, _keyed(bar, decision), DECISION_SCHEMA)


def read_decision(directory, bar: pd.Timestamp) -> pd.DataFrame:
    """Return the decision on ``bar``, by ``symbol``.

    Raises
    ------
    FileNotFoundError
        If the bar has no decision.

    Examples
    --------
    See ``write_decision``.
    """
    return _read(_path(directory, DECISIONS, bar))


def write_outcomes(directory, bar: pd.Timestamp, outcomes: pd.DataFrame) -> Path:
    """Write what ``bar``'s orders did.

    Parameters
    ----------
    outcomes : pandas.DataFrame
        By ``symbol``: the signed ``order_quantity`` and ``filled_quantity``,
        the fills' ``avg_px`` (NaN without a fill) and the ``status``.

    Examples
    --------
    >>> import tempfile
    >>> directory = tempfile.mkdtemp()
    >>> outcomes = pd.DataFrame(
    ...     {"order_quantity": [50.0], "filled_quantity": [50.0], "avg_px": [10.5], "status": ["filled"]},
    ...     index=pd.Index([10001], name="symbol"),
    ... )
    >>> _ = write_outcomes(directory, pd.Timestamp("2024-01-02"), outcomes)
    >>> read_outcomes(directory, pd.Timestamp("2024-01-02")).loc[10001, "status"]
    'filled'
    """
    return _write(_path(directory, OUTCOMES, bar), _keyed(bar, outcomes), OUTCOME_SCHEMA)


def read_outcomes(directory, bar: pd.Timestamp) -> pd.DataFrame:
    """Return the outcomes of ``bar``'s orders, by ``symbol``.

    Examples
    --------
    See ``write_outcomes``.
    """
    return _read(_path(directory, OUTCOMES, bar))


def write_snapshot(directory, snapshot: Snapshot) -> Path:
    """Write the positions and cash at ``snapshot.bar``'s close.

    Examples
    --------
    >>> import tempfile
    >>> directory = tempfile.mkdtemp()
    >>> positions = pd.DataFrame({"quantity": [50.0]}, index=pd.Index([10001], name="symbol"))
    >>> _ = write_snapshot(directory, Snapshot(pd.Timestamp("2024-01-02"), positions, cash=475.0))
    >>> again = read_snapshot(directory, pd.Timestamp("2024-01-02"))
    >>> again.cash, again.positions["quantity"].to_dict()
    (475.0, {10001: 50.0})
    """
    table = _keyed(snapshot.bar, snapshot.positions)
    metadata = {b"cash": repr(float(snapshot.cash)).encode()}
    return _write(_path(directory, SNAPSHOTS, snapshot.bar), table, SNAPSHOT_SCHEMA, metadata)


def read_snapshot(directory, bar: pd.Timestamp) -> Snapshot:
    """Return the snapshot at ``bar``'s close.

    Examples
    --------
    See ``write_snapshot``.
    """
    path = _path(directory, SNAPSHOTS, bar)
    cash = float(pq.read_schema(path).metadata[b"cash"])
    return Snapshot(pd.Timestamp(bar).normalize(), _read(path), cash)


def append_equity(directory, bar: pd.Timestamp, cash: float, holdings: float) -> Path:
    """Append the strategy's equity at ``bar``: ``cash`` plus ``holdings``.

    Raises
    ------
    ValueError
        If ``bar`` is not after the last bar recorded.

    Examples
    --------
    >>> import tempfile
    >>> directory = tempfile.mkdtemp()
    >>> _ = append_equity(directory, pd.Timestamp("2024-01-02"), cash=500.0, holdings=9_600.0)
    >>> read_equity(directory)["equity"].tolist()
    [10100.0]
    """
    path = Path(directory) / EQUITY
    rows = read_equity(directory) if path.exists() else None
    if rows is not None and len(rows) and rows.index[-1] >= pd.Timestamp(bar):
        raise ValueError(f"equity is recorded up to {rows.index[-1].date()}; cannot append {bar}")
    row = pd.DataFrame(
        {"cash": [float(cash)], "holdings": [float(holdings)], "equity": [float(cash) + float(holdings)]},
        index=pd.DatetimeIndex([pd.Timestamp(bar)], name="bar"),
    )
    frame = row if rows is None else pd.concat([rows, row])
    return _write(path, frame.reset_index(), EQUITY_SCHEMA)


def read_equity(directory) -> pd.DataFrame:
    """Return the strategy's equity: ``cash``, ``holdings`` and ``equity`` by ``bar``.

    Examples
    --------
    See ``append_equity``.
    """
    return pq.read_table(Path(directory) / EQUITY).to_pandas().set_index("bar")


class SecurityMap:
    """Symbols to instrument ids and back, with every id a symbol has had.

    A map is built on a day from that day's ids (``updated``) and keeps its
    history: each ``(symbol, instrument_id)`` pair with the first and last
    day it held. ``to_id`` gives a symbol's id on the map's day, and
    ``to_symbol`` maps any id the history holds back to its symbol, so the
    id of a renamed ticker still maps back. An id two symbols have held
    (a reused ticker) maps back to the one that held it last.

    Parameters
    ----------
    history : pandas.DataFrame, optional
        Columns ``symbol``, ``instrument_id`` (text), ``first`` and ``last``
        (days); empty by default.
    date : datetime-like, optional
        The map's day, the last day its history holds; ``None`` when empty.

    Examples
    --------
    >>> monday, tuesday = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    >>> fb, meta = InstrumentId.from_str("FB.NASDAQ"), InstrumentId.from_str("META.NASDAQ")
    >>> security_map = SecurityMap().updated(monday, {1: fb}).updated(tuesday, {1: meta, 2: None})
    >>> security_map.to_id(1), security_map.to_id(2), security_map.to_symbol(fb)
    (InstrumentId('META.NASDAQ'), None, 1)
    >>> security_map.to_symbol(InstrumentId.from_str("AAPL.NASDAQ")) is None
    True
    """

    def __init__(self, history: pd.DataFrame | None = None, date=None):
        if history is None:
            history = pd.DataFrame({"symbol": [], "instrument_id": [], "first": [], "last": []})
        self.history = history.astype(
            {"symbol": "int64", "instrument_id": object, "first": "datetime64[ns]", "last": "datetime64[ns]"}
        ).reset_index(drop=True)
        self.date = None if date is None else pd.Timestamp(date).normalize()
        ordered = self.history.sort_values(["last", "first"], kind="stable")
        self._back = dict(zip(ordered["instrument_id"], ordered["symbol"]))
        today = ordered[ordered["last"] == self.date]
        self._today = {int(s): InstrumentId.from_str(i) for s, i in zip(today["symbol"], today["instrument_id"])}

    def to_id(self, symbol: int) -> InstrumentId | None:
        """Return ``symbol``'s instrument id on the map's day; ``None`` when unmapped."""
        return self._today.get(int(symbol))

    def to_symbol(self, instrument_id: InstrumentId) -> int | None:
        """Return the symbol that last held ``instrument_id``; ``None`` for an id never mapped."""
        symbol = self._back.get(instrument_id.value)
        return None if symbol is None else int(symbol)

    def updated(self, date, ids: Mapping[int, InstrumentId | None]) -> SecurityMap:
        """Return this map built on ``date`` from that day's ``ids`` (``None`` where unmapped).

        Building again on the map's own day adds to that day's ids.

        Raises
        ------
        ValueError
            If ``date`` is before the map's day.

        Examples
        --------
        See ``SecurityMap``.
        """
        date = pd.Timestamp(date).normalize()
        if self.date is not None and date < self.date:
            raise ValueError(f"the security map is built up to {self.date.date()}; cannot build it on {date.date()}")
        history = self.history.copy()
        rows = {(s, i): k for k, (s, i) in enumerate(zip(history["symbol"], history["instrument_id"]))}
        new = []
        for symbol, instrument_id in ids.items():
            if instrument_id is None:
                continue
            key = (int(symbol), instrument_id.value)
            if key in rows:
                history.loc[rows[key], "last"] = date
            else:
                new.append({"symbol": key[0], "instrument_id": key[1], "first": date, "last": date})
        if new:
            history = pd.concat([history, pd.DataFrame(new)], ignore_index=True)
        return SecurityMap(history, date)


def write_security_map(directory, security_map: SecurityMap) -> Path:
    """Write ``security_map`` with its history, so the next day's map keeps it.

    Examples
    --------
    >>> import tempfile
    >>> directory = tempfile.mkdtemp()
    >>> fb = InstrumentId.from_str("FB.NASDAQ")
    >>> _ = write_security_map(directory, SecurityMap().updated(pd.Timestamp("2024-01-02"), {1: fb}))
    >>> again = read_security_map(directory)
    >>> again.to_id(1), again.date.date().isoformat()
    (InstrumentId('FB.NASDAQ'), '2024-01-02')
    """
    if security_map.date is None:
        raise ValueError("an empty security map has no day to write")
    metadata = {b"date": security_map.date.date().isoformat().encode()}
    return _write(Path(directory) / SECURITY_MAP, security_map.history, SECURITY_MAP_SCHEMA, metadata)


def read_security_map(directory) -> SecurityMap | None:
    """Return the security map written in ``directory``; ``None`` before the first.

    Examples
    --------
    See ``write_security_map``.
    """
    path = Path(directory) / SECURITY_MAP
    if not path.exists():
        return None
    table = pq.read_table(path)
    history = table.to_pandas()
    for column in ("first", "last"):
        history[column] = pd.to_datetime(history[column])
    return SecurityMap(history, table.schema.metadata[b"date"].decode())


def _path(directory, kind: str, bar: pd.Timestamp) -> Path:
    return Path(directory) / kind / f"{pd.Timestamp(bar).date().isoformat()}.parquet"


def _keyed(bar: pd.Timestamp, frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` by symbol with the bar as its first column."""
    keyed = frame.rename_axis("symbol").reset_index()
    keyed.insert(0, "bar", pd.Timestamp(bar))
    return keyed


def _write(path: Path, frame: pd.DataFrame, schema: pa.Schema, metadata=None) -> Path:
    """Write ``frame`` with ``schema`` by tmp, fsync and rename."""
    if metadata:
        schema = schema.with_metadata(metadata)
    table = pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as file:
        pq.write_table(table, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)
    return path


def _read(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas().drop(columns="bar").set_index("symbol")
