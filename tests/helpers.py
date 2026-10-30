"""Small helpers the tests share: New York times and a replay's fills."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NEW_YORK = "America/New_York"


def ny(day: str, hh_mm: str) -> pd.Timestamp:
    """The New York time ``hh_mm`` on ``day``, in UTC."""
    return pd.Timestamp(f"{day} {hh_mm}", tz=NEW_YORK).tz_convert("UTC")


#: The fields ``replay_fills`` can give, read from NautilusTrader's fills report.
FILL_FIELDS = {
    "ts": lambda row: pd.to_datetime(row.ts_event, utc=True),
    "instrument_id": lambda row: row.instrument_id,
    "side": lambda row: row.order_side,
    "qty": lambda row: float(row.last_qty),
    "px": lambda row: float(row.last_px),
    "fee": lambda row: float(str(row.commission).split()[0]),
}


def replay_fills(out_dir, *fields: str) -> list[tuple]:
    """A replay's fills, sorted, each the tuple of ``fields`` (``FILL_FIELDS``)."""
    frame = pd.read_parquet(Path(out_dir) / "reports" / "fills.parquet")
    return sorted(tuple(FILL_FIELDS[name](row) for name in fields) for row in frame.itertuples())
