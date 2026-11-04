"""``DecisionBar``: one decision bar's quantlab inputs, as NautilusTrader custom data.

The same record reaches the strategy live (published by an actor) and in a
replay (added to the engine as data), so the strategy never knows which run
it is in. It is small: the bar's predictions and the marks orders are sized
and valued at. The rest of a closed-loop decision's inputs (the price window,
tradability, factors, risk) are built by quantlab when the strategy decides.
NautilusTrader v2 custom data has no ``ndarray`` fields and refuses to
register an opaque ``bytes`` field, so the bar's rows travel as an Arrow IPC
stream in base64 text.
"""

from __future__ import annotations

import base64
import io

import pandas as pd
import pyarrow as pa
import xarray as xr
from nautilus_trader.model import register_custom_data_class
from nautilus_trader.model.custom import customdataclass

#: Where the decision on a bar is taken: the bar's close, New York time.
DECISION_TIME = pd.Timedelta(hours=16)
NEW_YORK = "America/New_York"

#: The prefix of a prediction column; a label's name follows it.
PREDICTION = "prediction:"
#: The marks every bar carries, by symbol.
MARKS = ("close", "delisted")


def decision_ns(bar: pd.Timestamp) -> int:
    """Return the UNIX nanoseconds of the decision on ``bar``: its close in New York.

    Examples
    --------
    >>> decision_ns(pd.Timestamp("2024-01-02"))
    1704229200000000000
    """
    return (pd.Timestamp(bar).normalize().tz_localize(NEW_YORK) + DECISION_TIME).value


@customdataclass
class DecisionBar:
    """One decision bar's predictions and marks, by symbol.

    Attributes
    ----------
    bar : int
        The quantlab bar's timestamp, in UNIX nanoseconds (a date at midnight).
    payload : str
        The bar's rows as an Arrow IPC stream, base64-encoded: ``symbol``, the marks
        (``close``, the raw close at which orders are sized and holdings
        valued, a delisted security marked at its last valuation; ``delisted``)
        and one ``prediction:<label>`` column per label (none in an open-loop
        run).

    Examples
    --------
    >>> frame = pd.DataFrame(
    ...     {"close": [10.0, 20.0], "delisted": [False, False], "prediction:ret_5": [0.1, -0.2]},
    ...     index=pd.Index([10001, 10002], name="symbol"),
    ... )
    >>> bar = DecisionBar.from_frame(pd.Timestamp("2024-01-02"), frame)
    >>> bar.timestamp, bar.ts_event == decision_ns(bar.timestamp)
    (Timestamp('2024-01-02 00:00:00'), True)
    >>> bar.marks()["close"].tolist()
    [10.0, 20.0]
    >>> bar.predictions()["ret_5"].values.tolist()
    [0.1, -0.2]
    """

    bar: int = 0
    payload: str = ""

    @classmethod
    def from_frame(
        cls,
        bar: pd.Timestamp,
        frame: pd.DataFrame,
        *,
        ts_event: int | None = None,
        ts_init: int | None = None,
    ) -> DecisionBar:
        """Build the record of ``bar`` from its rows.

        Parameters
        ----------
        bar : pandas.Timestamp
            The quantlab bar.
        frame : pandas.DataFrame
            Indexed by ``symbol``, with the ``MARKS`` columns and any
            ``prediction:<label>`` columns.
        ts_event, ts_init : int, optional
            UNIX nanoseconds; both default to ``decision_ns(bar)``.

        Raises
        ------
        ValueError
            If a mark is missing or a column is neither a mark nor a prediction.

        Examples
        --------
        >>> DecisionBar.from_frame(pd.Timestamp("2024-01-02"), pd.DataFrame({"close": [1.0]}))
        Traceback (most recent call last):
        ...
        ValueError: a decision bar needs the marks ('close', 'delisted'); missing ['delisted']
        """
        missing = [name for name in MARKS if name not in frame.columns]
        if missing:
            raise ValueError(f"a decision bar needs the marks {MARKS}; missing {missing}")
        unknown = [c for c in frame.columns if c not in MARKS and not c.startswith(PREDICTION)]
        if unknown:
            raise ValueError(f"columns {unknown} are neither marks {MARKS} nor {PREDICTION}<label>")
        table = pa.Table.from_pandas(frame.rename_axis("symbol").reset_index(), preserve_index=False)
        sink = io.BytesIO()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        stamp = decision_ns(bar)
        return cls(
            ts_event=stamp if ts_event is None else ts_event,
            ts_init=stamp if ts_init is None else ts_init,
            bar=pd.Timestamp(bar).normalize().value,
            payload=base64.b64encode(sink.getvalue()).decode("ascii"),
        )

    @property
    def timestamp(self) -> pd.Timestamp:
        """The quantlab bar.

        Examples
        --------
        >>> DecisionBar(bar=pd.Timestamp("2024-01-02").value).timestamp
        Timestamp('2024-01-02 00:00:00')
        """
        return pd.Timestamp(self.bar)

    def frame(self) -> pd.DataFrame:
        """Return the bar's rows, indexed by ``symbol``.

        Examples
        --------
        >>> frame = pd.DataFrame({"close": [1.5], "delisted": [True]}, index=[7])
        >>> DecisionBar.from_frame(pd.Timestamp("2024-01-02"), frame).frame().loc[7, "delisted"]
        np.True_
        """
        with pa.ipc.open_stream(base64.b64decode(self.payload)) as reader:
            return reader.read_all().to_pandas().set_index("symbol")

    def marks(self) -> pd.DataFrame:
        """Return the marks: ``close`` and ``delisted`` by ``symbol``.

        Examples
        --------
        >>> frame = pd.DataFrame({"close": [1.5], "delisted": [False]}, index=[7])
        >>> list(DecisionBar.from_frame(pd.Timestamp("2024-01-02"), frame).marks().columns)
        ['close', 'delisted']
        """
        return self.frame()[list(MARKS)]

    def predictions(self) -> xr.Dataset:
        """Return the predictions as quantlab takes them: one variable per label on ``symbol``.

        Symbols without a finite prediction are left out; a bar without
        predictions (open loop) gives an empty dataset.

        Examples
        --------
        >>> frame = pd.DataFrame(
        ...     {"close": [1.0, 2.0], "delisted": [False, False], "prediction:ret_5": [0.3, float("nan")]},
        ...     index=pd.Index([1, 2], name="symbol"),
        ... )
        >>> DecisionBar.from_frame(pd.Timestamp("2024-01-02"), frame).predictions()["ret_5"].symbol.values.tolist()
        [1]
        """
        frame = self.frame()
        columns = [c for c in frame.columns if c.startswith(PREDICTION)]
        predicted = frame[columns].dropna(how="all")
        predicted.columns = [c.removeprefix(PREDICTION) for c in columns]
        return xr.Dataset.from_dataframe(predicted.rename_axis("symbol"))


register_custom_data_class(DecisionBar)
