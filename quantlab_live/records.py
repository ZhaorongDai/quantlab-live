"""``records``: quantlab-live's own durable records, typed parquet in raw units.

Every record is in raw units (shares and prices as traded, not split
adjusted) and its rows are keyed by bar timestamp, so nothing assumes one
decision per day; files are per date. Layout of a live or replay directory:

- ``decisions/<date>.parquet``: the decision on each bar, by symbol;
- ``outcomes/<date>.parquet``: what each bar's orders did, by symbol;
- ``snapshots/<date>.parquet``: positions by symbol and cash, at a bar's close;
- ``equity.parquet``: the strategy's equity, one row per bar.

Each file is written to a temporary file, fsynced and renamed, so a crash
leaves either no file or a complete one. NautilusTrader owns orders, fills
and positions; these records hold what it does not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DECISIONS = "decisions"
OUTCOMES = "outcomes"
SNAPSHOTS = "snapshots"
EQUITY = "equity.parquet"

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
        ``position_quantity``, ``target_quantity``, ``order_quantity`` and
        ``skip``: why an ordered symbol is not traded (``"unmapped"``,
        ``"foreign"``), empty when it is.

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
        the fills' ``avg_px`` (NaN without a fill) and the ``status``
        (``trading.outcomes``).

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
