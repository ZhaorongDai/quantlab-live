"""``records``: quantlab-live's own durable records, typed parquet in raw units.

Every record is in raw units (shares and prices as traded, not split
adjusted) and its rows are keyed by bar timestamp, so nothing assumes one
decision per day; files are per date. Layout of a live or replay directory:

- ``decisions/<date>.parquet``: the decision on each bar, by symbol;
- ``outcomes/<date>.parquet``: what each bar's orders did, by symbol;
- ``snapshots/<date>.parquet``: positions by symbol and cash, at a bar's close;
- ``equity.parquet``: the strategy's equity, one row per bar;
- ``security_map.parquet``: the security map's history (live only).
- ``archive/``: NautilusTrader's streaming archive of every event (live
  only, ``live.node``).

The account history (``account.toml``, one per account) lists the account's
strategy segments in order; ``record_segment`` starts one.

Each file is written to a temporary file, fsynced and renamed, so a crash
leaves either no file or a complete one. NautilusTrader owns orders, fills
and positions; these records hold what it does not.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
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
ACCOUNT_HISTORY = "account.toml"

_KEY = [("bar", pa.timestamp("ns")), ("symbol", pa.int64())]
DECISION_SCHEMA = pa.schema(
    _KEY
    + [
        ("weight", pa.float64()),
        ("close", pa.float64()),
        ("position_quantity", pa.float64()),
        ("target_quantity", pa.float64()),
        ("order_quantity", pa.float64()),
        ("skip", pa.string()),
    ]
)
OUTCOME_SCHEMA = pa.schema(
    _KEY
    + [
        ("order_quantity", pa.float64()),
        ("filled_quantity", pa.float64()),
        ("avg_px", pa.float64()),
        ("fees", pa.float64()),
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
    foreign : dict of str to float
        The account's foreign positions: signed raw shares by instrument id,
        positions the strategy does not own; never traded, listed in the
        close briefing.
    """

    bar: pd.Timestamp
    positions: pd.DataFrame
    cash: float
    foreign: dict[str, float] = field(default_factory=dict)


def write_decision(directory, bar: pd.Timestamp, decision: pd.DataFrame, capital: float | None = None) -> Path:
    """Write the decision on ``bar``; a bar is decided at most once.

    Parameters
    ----------
    directory : path-like
        The live or replay directory.
    bar : pandas.Timestamp
        The decision bar.
    decision : pandas.DataFrame
        By ``symbol``: ``weight`` (NaN where kept), ``close``,
        ``position_quantity``, ``target_quantity``, ``order_quantity`` and
        ``skip``: why an ordered symbol is not traded (``"unmapped"``,
        ``"foreign"``), empty when it is.
    capital : float, optional
        The strategy capital the decision sized on (``decision_capital``),
        which the decision recheck divides the holdings by.

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
    ...      "target_quantity": [50.0], "order_quantity": [50.0], "skip": [""]},
    ...     index=pd.Index([10001], name="symbol"),
    ... )
    >>> _ = write_decision(directory, pd.Timestamp("2024-01-02"), decision)
    >>> read_decision(directory, pd.Timestamp("2024-01-02"))["order_quantity"].to_dict()
    {10001: 50.0}
    >>> decision_capital(directory, pd.Timestamp("2024-01-02")) is None
    True
    >>> write_decision(directory, pd.Timestamp("2024-01-02"), decision)
    Traceback (most recent call last):
    ...
    FileExistsError: bar 2024-01-02 already has a decision at ...
    """
    path = _path(directory, DECISIONS, bar)
    if path.exists():
        raise FileExistsError(f"bar {pd.Timestamp(bar).date()} already has a decision at {path}")
    metadata = None if capital is None else {b"capital": repr(float(capital)).encode()}
    return _write(path, _keyed(bar, decision), DECISION_SCHEMA, metadata)


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


def decision_capital(directory, bar: pd.Timestamp) -> float | None:
    """Return the strategy capital the decision on ``bar`` sized on; ``None`` if not recorded.

    Raises
    ------
    FileNotFoundError
        If the bar has no decision.

    Examples
    --------
    >>> import tempfile
    >>> directory = tempfile.mkdtemp()
    >>> decision = pd.DataFrame(
    ...     {"weight": [1.0], "close": [10.0], "position_quantity": [0.0],
    ...      "target_quantity": [100.0], "order_quantity": [100.0], "skip": [""]},
    ...     index=pd.Index([10001], name="symbol"),
    ... )
    >>> _ = write_decision(directory, pd.Timestamp("2024-01-02"), decision, capital=1_000.0)
    >>> decision_capital(directory, pd.Timestamp("2024-01-02"))
    1000.0
    """
    metadata = pq.read_schema(_path(directory, DECISIONS, bar)).metadata or {}
    return float(metadata[b"capital"]) if b"capital" in metadata else None


def write_outcomes(directory, bar: pd.Timestamp, outcomes: pd.DataFrame) -> Path:
    """Write what ``bar``'s orders did.

    Parameters
    ----------
    outcomes : pandas.DataFrame
        By ``symbol``: the signed ``order_quantity`` and ``filled_quantity``,
        the fills' ``avg_px`` (NaN without a fill), their ``fees`` in cash
        and the ``status`` (``trading.outcomes``).

    Examples
    --------
    >>> import tempfile
    >>> directory = tempfile.mkdtemp()
    >>> outcomes = pd.DataFrame(
    ...     {"order_quantity": [50.0], "filled_quantity": [50.0], "avg_px": [10.5], "fees": [1.0],
    ...      "status": ["filled"]},
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
    >>> again.cash, again.positions["quantity"].to_dict(), again.foreign
    (475.0, {10001: 50.0}, {})
    """
    table = _keyed(snapshot.bar, snapshot.positions)
    foreign = {str(k): float(v) for k, v in snapshot.foreign.items()}
    metadata = {b"cash": repr(float(snapshot.cash)).encode(), b"foreign": json.dumps(foreign).encode()}
    return _write(_path(directory, SNAPSHOTS, snapshot.bar), table, SNAPSHOT_SCHEMA, metadata)


def read_snapshot(directory, bar: pd.Timestamp) -> Snapshot:
    """Return the snapshot at ``bar``'s close.

    Examples
    --------
    See ``write_snapshot``.
    """
    path = _path(directory, SNAPSHOTS, bar)
    metadata = pq.read_schema(path).metadata
    foreign = json.loads(metadata.get(b"foreign", b"{}"))
    return Snapshot(pd.Timestamp(bar).normalize(), _read(path), float(metadata[b"cash"]), foreign)


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


@dataclass(frozen=True)
class Book:
    """Every bar a directory records, in raw units: what a report is computed on.

    Attributes
    ----------
    equity : pandas.DataFrame
        ``read_equity``: ``cash``, ``holdings`` and ``equity`` by ``bar``; the
        book's bars.
    positions : pandas.DataFrame
        The signed raw shares held at each bar's close (its snapshot), by
        ``bar`` and one column per symbol; 0 where not held, NaN on a bar
        without a snapshot.
    closes : pandas.DataFrame
        The raw close each bar's decision marks every symbol at (the price
        equity is valued at), by ``bar`` and symbol; NaN where a decision
        has none.
    weights : pandas.DataFrame
        The target weight of each bar's decision, by ``bar`` and symbol; NaN
        where the decision keeps a holding.
    outcomes : pandas.DataFrame
        Every bar's outcomes (``write_outcomes``), with their ``bar`` and
        ``symbol`` as columns; ``fees`` is 0 in a file written without it.
    """

    equity: pd.DataFrame
    positions: pd.DataFrame
    closes: pd.DataFrame
    weights: pd.DataFrame
    outcomes: pd.DataFrame


def read_book(directory) -> Book:
    """Return every bar ``directory`` records: its equity, snapshots, decisions and outcomes.

    Raises
    ------
    FileNotFoundError
        If the directory records no equity.

    Examples
    --------
    >>> import tempfile
    >>> directory = tempfile.mkdtemp()
    >>> bar = pd.Timestamp("2024-01-02")
    >>> positions = pd.DataFrame({"quantity": [50.0]}, index=pd.Index([10001], name="symbol"))
    >>> _ = write_snapshot(directory, Snapshot(bar, positions, cash=500.0))
    >>> _ = append_equity(directory, bar, cash=500.0, holdings=500.0)
    >>> book = read_book(directory)
    >>> book.equity["equity"].tolist(), book.positions.loc[bar].to_dict()
    ([1000.0], {10001: 50.0})
    """
    equity = read_equity(directory)
    snapshots = {bar: read_snapshot(directory, bar).positions["quantity"] for bar in _bars(directory, SNAPSHOTS)}
    decisions = {bar: read_decision(directory, bar) for bar in _bars(directory, DECISIONS)}
    by_bar = {
        name: pd.DataFrame(columns).T.reindex(equity.index)
        for name, columns in (
            ("positions", snapshots),
            ("closes", {bar: d["close"] for bar, d in decisions.items()}),
            ("weights", {bar: d["weight"] for bar, d in decisions.items()}),
        )
    }
    symbols = by_bar["positions"].columns.union(by_bar["closes"].columns).astype("int64")
    positions, closes, weights = (
        by_bar[name].reindex(columns=symbols).astype(float).rename_axis(index="bar", columns="symbol")
        for name in ("positions", "closes", "weights")
    )
    snapshot_bars = pd.Series(positions.index.isin(list(snapshots)), index=positions.index)
    positions = positions.mask(positions.isna() & snapshot_bars.to_numpy()[:, None], 0.0)
    frames = [_keyed(bar, read_outcomes(directory, bar)) for bar in _bars(directory, OUTCOMES)]
    outcomes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTCOME_SCHEMA.names)
    if "fees" not in outcomes:
        outcomes["fees"] = 0.0
    outcomes["fees"] = outcomes["fees"].fillna(0.0)
    return Book(equity, positions, closes, weights, outcomes)


def _bars(directory, kind: str) -> list[pd.Timestamp]:
    """The bars ``kind`` records in ``directory``, in order."""
    return sorted(pd.Timestamp(path.stem) for path in (Path(directory) / kind).glob("*.parquet"))


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

    def ids(self) -> list[InstrumentId]:
        """Return every instrument id mapped on the map's day.

        Examples
        --------
        >>> SecurityMap().updated("2024-01-02", {1: InstrumentId.from_str("FB.NASDAQ"), 2: None}).ids()
        [InstrumentId('FB.NASDAQ')]
        """
        return list(self._today.values())

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


@dataclass(frozen=True)
class StepReport:
    """What one node process reports to the supervisor, its only channel to it.

    The node writes it to the path it is given, at least once when its
    start-up has passed (``started``) and again when it ends; a process that
    ends with ``started`` set and ``finished`` unset crashed.

    Attributes
    ----------
    mode : str
        ``"trade"`` or ``"settle"`` (settle-only).
    started : bool
        The start guards passed and the node runs.
    finished : bool
        The node ran to its end.
    outcomes : bool
        The day's outcomes are written.
    error : str, optional
        Why the node failed.
    alerts : tuple of str
        Things to mail (a held position no symbol maps to, a foreign
        position on a target, an outcome that needs checking); each goes out
        once.

    Examples
    --------
    >>> import tempfile
    >>> path = Path(tempfile.mkdtemp()) / "1-trade.json"
    >>> _ = write_step_report(path, StepReport("trade", started=True, alerts=("5 shares of X unmapped",)))
    >>> read_step_report(path)
    StepReport(mode='trade', started=True, finished=False, outcomes=False, error=None, alerts=('5 shares of X unmapped',))
    """

    mode: str
    started: bool = False
    finished: bool = False
    outcomes: bool = False
    error: str | None = None
    alerts: tuple[str, ...] = ()


def write_step_report(path, report: StepReport) -> Path:
    """Write a node's step report by tmp, fsync and rename.

    Examples
    --------
    See ``StepReport``.
    """
    text = json.dumps({**report.__dict__, "alerts": list(report.alerts)}, indent=1)
    return _write_text(Path(path), text)


def read_step_report(path) -> StepReport:
    """Read a node's step report.

    Examples
    --------
    See ``StepReport``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return StepReport(**{**raw, "alerts": tuple(raw.get("alerts", ()))})


def has_outcomes(directory, bar: pd.Timestamp) -> bool:
    """Whether ``bar``'s outcomes are written.

    Examples
    --------
    >>> import tempfile
    >>> has_outcomes(tempfile.mkdtemp(), pd.Timestamp("2024-01-02"))
    False
    """
    return _path(directory, OUTCOMES, bar).exists()


def has_decision(directory, bar: pd.Timestamp) -> bool:
    """Whether ``bar`` is decided.

    Examples
    --------
    >>> import tempfile
    >>> has_decision(tempfile.mkdtemp(), pd.Timestamp("2024-01-02"))
    False
    """
    return _path(directory, DECISIONS, bar).exists()


def decision_bars(directory) -> list[pd.Timestamp]:
    """Return every decided bar of ``directory``, in order.

    Examples
    --------
    >>> import tempfile
    >>> decision_bars(tempfile.mkdtemp())
    []
    """
    return _bars(directory, DECISIONS)


def outcome_bars(directory) -> list[pd.Timestamp]:
    """Return every bar of ``directory`` with outcomes, in order.

    Examples
    --------
    >>> import tempfile
    >>> outcome_bars(tempfile.mkdtemp())
    []
    """
    return _bars(directory, OUTCOMES)


def latest_snapshot(directory, before) -> Snapshot | None:
    """Return the latest snapshot at a bar before the day ``before``; ``None`` without one.

    Examples
    --------
    >>> import tempfile
    >>> directory = tempfile.mkdtemp()
    >>> positions = pd.DataFrame({"quantity": [50.0]}, index=pd.Index([10001], name="symbol"))
    >>> _ = write_snapshot(directory, Snapshot(pd.Timestamp("2024-01-02"), positions, cash=475.0))
    >>> latest_snapshot(directory, pd.Timestamp("2024-01-03")).bar.date().isoformat()
    '2024-01-02'
    >>> latest_snapshot(directory, pd.Timestamp("2024-01-02")) is None
    True
    """
    folder = Path(directory) / SNAPSHOTS
    limit = pd.Timestamp(before).date().isoformat()
    days = sorted(p.stem for p in folder.glob("*.parquet")) if folder.exists() else []
    days = [d for d in days if d < limit]
    return read_snapshot(directory, pd.Timestamp(days[-1])) if days else None


#: The outcomes of an order that never reached the broker.
_NEVER_SENT = ("not_sent", "unmapped", "foreign")


@dataclass(frozen=True)
class Segment:
    """One strategy segment of an account: a strategy trading it from a decision bar on.

    Attributes
    ----------
    start : pandas.Timestamp
        The first decision bar whose orders reached the broker.
    live_dir : str
        The live directory that traded it, absolute.
    run : str
        The quantlab run traded, absolute.
    broker : str
        The broker profile.
    initial_capital : float, optional
        The strategy capital's start; unset when it sized on the account's equity.
    """

    start: pd.Timestamp
    live_dir: str
    run: str
    broker: str
    initial_capital: float | None = None

    def same_strategy(self, other: Segment) -> bool:
        """Whether ``other`` is this segment's strategy: the same live directory, run and broker."""
        return (self.live_dir, self.run, self.broker) == (other.live_dir, other.run, other.broker)


def read_account_history(path) -> list[Segment]:
    """Return the account's strategy segments in order; empty before the first.

    Examples
    --------
    See ``record_segment``.
    """
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "rb") as file:
        raw = tomllib.load(file)
    return [
        Segment(pd.Timestamp(item["start"]), item["live_dir"], item["run"], item["broker"], item.get("initial_capital"))
        for item in raw.get("segment", [])
    ]


def record_segment(path, segment: Segment, outcomes: pd.DataFrame) -> bool:
    """Start ``segment`` in the account history at ``path``, if its orders start one.

    A segment starts only with real orders: ``outcomes`` (of the decision on
    ``segment.start``) must hold an order that reached the broker, so a
    dry run or a day whose orders were all skipped starts none. The same
    strategy trading on continues its segment; another strategy (another
    live directory, run or broker) starts the next one.

    Returns
    -------
    bool
        Whether a segment was started.

    Examples
    --------
    >>> import tempfile
    >>> path = Path(tempfile.mkdtemp()) / "account.toml"
    >>> segment = Segment(pd.Timestamp("2024-01-02"), "/live/a", "/runs/a", "ibkr_paper", 10_000.0)
    >>> outcomes = pd.DataFrame({"status": ["filled"]}, index=pd.Index([10001], name="symbol"))
    >>> record_segment(path, segment, outcomes), record_segment(path, segment, outcomes)
    (True, False)
    >>> [s.start.date().isoformat() for s in read_account_history(path)]
    ['2024-01-02']
    """
    if not (~outcomes["status"].isin(_NEVER_SENT)).any():
        return False
    history = read_account_history(path)
    if history and history[-1].same_strategy(segment):
        return False
    history.append(segment)
    lines = [
        "# The account's strategy segments, in order. quantlab-live starts one when a",
        "# strategy's orders first reach the broker.",
    ]
    for item in history:
        lines += ["", "[[segment]]", f"start = {item.start.date().isoformat()}"]
        lines += [f"{name} = {json.dumps(getattr(item, name))}" for name in ("live_dir", "run", "broker")]
        if item.initial_capital is not None:
            lines.append(f"initial_capital = {float(item.initial_capital)!r}")
    _write_text(Path(path), "\n".join(lines) + "\n")
    return True


def current_segment(path, live_dir, run) -> Segment | None:
    """Return the account's current segment when it is the strategy of ``live_dir`` and ``run``.

    Examples
    --------
    >>> import tempfile
    >>> current_segment(Path(tempfile.mkdtemp()) / "account.toml", "/live/a", "/runs/a") is None
    True
    """
    history = read_account_history(path)
    if history and (history[-1].live_dir, history[-1].run) == (str(live_dir), str(run)):
        return history[-1]
    return None


def _write_text(path: Path, text: str) -> Path:
    """Write ``text`` by tmp, fsync and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)
    return path


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
