"""``brief``: the after-open mail, with the previous close's briefing.

The after-open mail of a New York day carries:

- the day's decision and its orders;
- the close briefing of the bar before the day: the strategy's equity at
  quantlab's close beside a closed-loop replay of the current strategy
  segment (``records.current_segment``) at the prints, and the gap between
  them split by ``execution attribution``;
- the decision recheck: quantlab's construction re-run on the day's
  decision bar and the holdings actually held must give the decision's
  weights bit for bit;
- the run's benchmark, by its ticker (``Run.benchmark``);
- the account's foreign positions.

**Execution attribution.** Every order of a decision on a segment bar before
the briefed bar fills on the next bar and is worth its shares at the briefed
bar's close mark (split adjusted) less what it cost. Per ordered symbol, with
the live (L) and replay (R) ordered quantity ``o``, filled quantity ``q``,
average price ``p`` and fees ``f``, and a reference price ``p*`` (the
replay's fill price, else the live one, else the fill bar's close mark):

- decisions: ``(oL - oR) * (mark - p*)``, what live ordered that the replay
  did not; the positions live held at the segment's first bar count as
  ordered there at its close mark, since the replay starts in cash;
- unfilled orders: ``((qL - oL) - (qR - oR)) * (mark - p*)``, skipped orders
  included;
- fill prices: ``qL * (p* - pL)``;
- fees: ``fR - fL``;
- residual: the gap less the four (dividends, delistings, cash flows,
  rounding), flagged when it exceeds ``LARGE_RESIDUAL`` of the equity.

Each amount is in cents, and the residual is computed in cents, so the five
sum to the gap exactly.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab_live import records
from quantlab_live.live.config import LiveConfig, load_config
from quantlab_live.quantlab_gate import open_run
from quantlab_live.trading.targets import ConstructorTargets

#: A residual larger than this fraction of the strategy's equity needs checking.
LARGE_RESIDUAL = 0.001
#: Where the briefings' replays are written, per briefed bar, in the live directory.
BRIEFINGS_DIR = "briefings"
#: The fee model the replay charges: every broker profile is IBKR's.
REPLAY_BROKER = "ibkr"
COMPONENTS = ("decisions", "unfilled orders", "fill prices", "fees", "residual")
CENT = Decimal("0.01")


@dataclass(frozen=True)
class Briefing:
    """The after-open mail of a day: the decision, the close briefing and the checks.

    Attributes
    ----------
    date : pandas.Timestamp
        The New York day of the mail.
    bar : pandas.Timestamp or None
        The close briefed: the last bar before ``date`` with equity.
    decision_bar : pandas.Timestamp or None
        The day's decision bar: the last decided bar before ``date``.
    decision : pandas.DataFrame or None
        The day's decision.
    segment : records.Segment or None
        The current strategy segment.
    live_equity, replay_equity : float or None
        The strategy's equity at ``bar`` and the segment replay's.
    gap : Decimal or None
        ``live_equity - replay_equity``, in cents.
    attribution : dict of str to Decimal
        The gap's ``COMPONENTS``, which sum to it exactly; empty without a
        replay.
    returns : dict of str to float
        The strategy's ``day`` and ``segment`` returns and the replay's
        ``replay segment`` return, where known.
    benchmark : tuple of (str, float, float) or None
        The benchmark's ticker, day return and segment return (NaN without
        a segment).
    recheck : str
        The decision recheck's verdict.
    foreign : dict of str to float
        The foreign positions at ``bar``: signed shares by instrument id.
    checks : tuple of str
        What needs checking; empty when nothing does.
    """

    date: pd.Timestamp
    bar: pd.Timestamp | None = None
    decision_bar: pd.Timestamp | None = None
    decision: pd.DataFrame | None = None
    segment: records.Segment | None = None
    live_equity: float | None = None
    replay_equity: float | None = None
    gap: Decimal | None = None
    attribution: dict[str, Decimal] = field(default_factory=dict)
    returns: dict[str, float] = field(default_factory=dict)
    benchmark: tuple[str, float, float] | None = None
    recheck: str = "not run"
    foreign: dict[str, float] = field(default_factory=dict)
    checks: tuple[str, ...] = ()

    def subject(self) -> str:
        """The mail's subject: ``after the open``, with ``NEEDS CHECKING`` when something does."""
        return "after the open" + (": NEEDS CHECKING" if self.checks else "")

    def text(self) -> str:
        """The mail's body."""
        lines = [f"After the open, {self.date.date()}"]
        if self.checks:
            lines += ["", "Needs checking:"] + [f"- {check}" for check in self.checks]
        lines += ["", f"The decision on bar {_day(self.decision_bar)}:"]
        if self.decision is None:
            lines.append("  no decision yet")
        else:
            ordered = self.decision[self.decision["order_quantity"] != 0]
            lines.append(f"  {len(ordered)} orders")
            if len(ordered):
                lines.append(_indent(ordered.to_string()))
        lines.append(f"  decision recheck: {self.recheck}")
        lines += ["", f"Close briefing, {_day(self.bar)}:"]
        if self.live_equity is not None:
            lines.append(f"  strategy equity at quantlab's close: {self.live_equity:,.2f}")
        if self.segment is None:
            lines.append("  no strategy segment yet: no order of this strategy has reached the broker")
        else:
            lines.append(
                f"  segment since {self.segment.start.date()}: {Path(self.segment.run).name} "
                f"({self.segment.broker}, initial capital {_money(self.segment.initial_capital)})"
            )
        if self.replay_equity is not None:
            lines.append(f"  closed-loop replay of the segment: {self.replay_equity:,.2f}")
            lines.append(f"  gap: {self.gap:+,.2f}")
            lines += [f"    {name:<16}{amount:+,.2f}" for name, amount in self.attribution.items()]
        for name, value in self.returns.items():
            lines.append(f"  {name} return: {value:+.2%}")
        if self.benchmark is not None:
            ticker, day, segment = self.benchmark
            lines.append(f"  benchmark {ticker}: day {day:+.2%}" + ("" if np.isnan(segment) else f", segment {segment:+.2%}"))
        lines.append("  foreign positions: " + (", ".join(f"{k} {v:+g}" for k, v in self.foreign.items()) or "none"))
        return "\n".join(lines) + "\n"


def brief(live_dir, date, *, mailer=None) -> Briefing:
    """Compose and send the after-open mail of the New York day ``date``.

    Parameters
    ----------
    live_dir : path-like
        The live directory, with its ``live.toml``.
    date : datetime-like
        The day of the mail; its close briefing is of the last bar before it.
    mailer : optional
        Sends the mail (``send(subject, body)``); the SMTP mailer of the
        owner's mail settings by default.

    Returns
    -------
    Briefing
        What the mail said.

    Examples
    --------
    ::

        brief(live_dir, "2026-10-30")  # briefs the close of 2026-10-29
    """
    composed = briefing(live_dir, date)
    if mailer is None:
        from quantlab_live.live.ports import SmtpMailer

        mailer = SmtpMailer.from_env_file()
    mailer.send(f"quantlab-live {composed.date.date()}: {composed.subject()}", composed.text())
    return composed


def briefing(live_dir, date) -> Briefing:
    """Return the after-open mail of the New York day ``date``, without sending it.

    A part that fails (the replay, the recheck, the benchmark) is reported
    in ``checks``; composing never raises once the configuration reads.

    Examples
    --------
    ::

        print(briefing(live_dir, "2026-10-30").text())
    """
    live_dir = Path(live_dir)
    config = load_config(live_dir)
    date = pd.Timestamp(date).normalize()
    checks: list[str] = []
    values: dict = {"date": date}

    decided = [bar for bar in records.decision_bars(live_dir) if bar < date]
    if decided:
        values["decision_bar"] = decided[-1]
        values["decision"] = records.read_decision(live_dir, decided[-1])
        try:
            values["recheck"] = _recheck(live_dir, config, decided[-1])
        except Exception as error:  # noqa: BLE001 - a failing part is reported, not raised
            values["recheck"] = f"failed: {error!r}"
        if values["recheck"] != SAME:
            checks.append(f"the decision recheck on {decided[-1].date()}: {values['recheck']}")
    else:
        checks.append(f"no decision before {date.date()}")

    equity = records.read_equity(live_dir) if (live_dir / records.EQUITY).exists() else None
    closed = equity.index[equity.index < date] if equity is not None else pd.DatetimeIndex([])
    segment = records.current_segment(config.account_path, live_dir.absolute(), config.run_dir)
    values["segment"] = segment
    if len(closed):
        bar = closed[-1]
        values.update(bar=bar, live_equity=float(equity.at[bar, "equity"]))
        if len(closed) > 1:
            values["returns"] = {"day": float(equity.at[bar, "equity"] / equity.at[closed[-2], "equity"] - 1)}
        try:
            snapshot = records.read_snapshot(live_dir, bar)
            values["foreign"] = dict(snapshot.foreign)
        except FileNotFoundError:
            pass
        if segment is not None and segment.start <= bar:
            try:
                values.update(_against_replay(live_dir, config, segment, bar, equity, checks))
            except Exception as error:  # noqa: BLE001
                checks.append(f"the segment replay failed: {error!r}")
        try:
            values["benchmark"] = _benchmark(config, bar, closed, segment)
        except Exception as error:  # noqa: BLE001
            checks.append(f"the benchmark failed: {error!r}")
    values["checks"] = tuple(checks)
    return Briefing(**values)


SAME = "same weights bit for bit"


def _recheck(live_dir: Path, config: LiveConfig, bar: pd.Timestamp) -> str:
    """Re-run quantlab's construction on ``bar`` and the holdings held; ``SAME`` when it agrees."""
    decision = records.read_decision(live_dir, bar)
    capital = records.decision_capital(live_dir, bar)
    if capital is None:
        return "the decision records no capital to recheck it on"
    run = open_run(config.run_dir)
    (decision_bar,) = run.bars(bar, bar, **_store(run, config, bar))
    try:
        held = records.read_snapshot(live_dir, bar).positions["quantity"]
    except FileNotFoundError:  # the first day: the holdings the decision found
        held = decision.loc[decision["position_quantity"] != 0, "position_quantity"]
    close = decision_bar.marks()["close"]
    current = held * close.reindex(held.index) / capital
    weights = ConstructorTargets(run.targets("closed")).weights(decision_bar, current)
    extra = weights.index.difference(decision.index)
    extra = extra[weights.reindex(extra).notna()]
    expected = weights.reindex(decision.index).to_numpy(dtype=float)
    recorded = decision["weight"].to_numpy(dtype=float)
    differ = decision.index[~((expected == recorded) | (np.isnan(expected) & np.isnan(recorded)))]
    if len(differ) or len(extra):
        symbols = [str(s) for s in [*differ, *extra]]
        return f"quantlab's construction gives other weights for symbols {', '.join(symbols)}"
    return SAME


def _store(run, config: LiveConfig, bar: pd.Timestamp) -> dict:
    """The live prediction store, for a bar past the run's end."""
    return {"store": config.store_dir} if bar > run.window[1] else {}


def _against_replay(
    live_dir: Path, config: LiveConfig, segment: records.Segment, bar: pd.Timestamp, equity: pd.DataFrame,
    checks: list[str],
) -> dict:
    """Replay the segment up to ``bar`` and attribute the gap between it and the live equity."""
    start = segment.start
    if start in equity.index:
        initial = float(equity.at[start, "equity"])
    elif segment.initial_capital is not None:  # the takeover day: the strategy's equity is its initial capital
        initial = float(segment.initial_capital)
    else:  # the takeover day: no close valued before it, so the capital it sized on
        initial = records.decision_capital(live_dir, start)
    if initial is None:
        raise ValueError(f"the segment's first bar {start.date()} has neither equity nor a recorded capital")
    run = open_run(config.run_dir)
    out_dir = live_dir / BRIEFINGS_DIR / bar.date().isoformat() / "replay"
    shutil.rmtree(out_dir, ignore_errors=True)
    replay_dir = _replay(config, run, out_dir, start, bar, initial)
    live, replayed = records.read_book(live_dir), records.read_book(replay_dir)
    live_equity = float(equity.at[bar, "equity"])
    replay_equity = float(replayed.equity.at[bar, "equity"])
    split_factor, _ = run.corporate_actions(start, bar)
    attribution = _attribution(live, replayed, live_dir, start, bar, run.marks(bar), split_factor)
    gap = _cents(live_equity - replay_equity)
    attribution["residual"] = gap - sum(attribution.values())
    if abs(attribution["residual"]) > _cents(LARGE_RESIDUAL * abs(live_equity)):
        checks.append(f"the attribution's residual {attribution['residual']:+,.2f} is large")
    returns = {
        "segment": live_equity / initial - 1,
        "replay segment": replay_equity / initial - 1,
    }
    return {"replay_equity": replay_equity, "gap": gap, "attribution": attribution, "returns": returns}


def _replay(config: LiveConfig, run, out_dir: Path, start, end, initial_cash: float) -> Path:
    from quantlab_live.replay import replay

    return replay(
        config.run_dir, out_dir, loop="closed", plan=config.plan, start=start, end=end, broker=REPLAY_BROKER,
        initial_cash=initial_cash, **_store(run, config, end),
    )


def _attribution(
    live: records.Book, replayed: records.Book, live_dir: Path, start: pd.Timestamp, bar: pd.Timestamp,
    marks: pd.Series, split_factor: pd.DataFrame,
) -> dict[str, Decimal]:
    """The decisions, unfilled orders, fill prices and fees parts of the gap at ``bar``, in cents."""
    bars = replayed.equity.index
    factor = split_factor.reindex(index=bars).fillna(1.0)
    factor.iloc[0] = 1.0
    ratio = factor.cumprod()  # raw shares per share of the first bar

    def growth(at, symbol) -> float:
        """The raw shares at ``bar`` of one raw share at ``at``."""
        if symbol not in ratio.columns:
            return 1.0
        return float(ratio.at[bar, symbol] / ratio.at[at, symbol])

    columns = ["order_quantity", "filled_quantity", "avg_px", "fees"]

    def orders(book: records.Book) -> pd.DataFrame:
        frame = book.outcomes[(book.outcomes["bar"] >= start) & (book.outcomes["bar"] < bar)]
        return frame.set_index(["bar", "symbol"])[columns].astype(float)

    both = orders(live).join(orders(replayed), how="outer", suffixes=("_l", "_r"))
    for side in ("_l", "_r"):
        for name in ("order_quantity", "filled_quantity", "fees"):
            both[name + side] = both[name + side].fillna(0.0)
    parts = dict.fromkeys(COMPONENTS[:-1], 0.0)
    for (decided, symbol), row in both.iterrows():
        filled_at = bars[bars.searchsorted(decided, "right")]
        mark = float(marks.get(symbol, np.nan)) * growth(filled_at, symbol)
        if row["filled_quantity_r"]:
            reference = row["avg_px_r"]
        elif row["filled_quantity_l"]:
            reference = row["avg_px_l"]
        else:
            reference = float(replayed.closes.at[filled_at, symbol])
        move = mark - reference
        parts["decisions"] += (row["order_quantity_l"] - row["order_quantity_r"]) * move
        unfilled = (row["filled_quantity_l"] - row["order_quantity_l"]) - (row["filled_quantity_r"] - row["order_quantity_r"])
        parts["unfilled orders"] += unfilled * move if unfilled else 0.0
        if row["filled_quantity_l"]:
            parts["fill prices"] += row["filled_quantity_l"] * (reference - row["avg_px_l"])
        parts["fees"] += row["fees_r"] - row["fees_l"]
    # The positions live held at the segment's start, which the replay starts without.
    held = _held_at(live_dir, start)
    for symbol, quantity in held.items():
        mark = float(marks.get(symbol, np.nan)) * growth(start, symbol)
        parts["decisions"] += quantity * (mark - float(replayed.closes.at[start, symbol]))
    return {name: _cents(amount) for name, amount in parts.items()}


def _held_at(live_dir: Path, bar: pd.Timestamp) -> pd.Series:
    """The strategy's positions at ``bar``'s close: its snapshot, else its decision's."""
    try:
        held = records.read_snapshot(live_dir, bar).positions["quantity"]
    except FileNotFoundError:
        decision = records.read_decision(live_dir, bar)
        held = decision["position_quantity"]
    return held[held != 0]


def _benchmark(config: LiveConfig, bar: pd.Timestamp, closed: pd.DatetimeIndex, segment) -> tuple | None:
    """The benchmark's ticker, day return and segment return at ``bar``."""
    run = open_run(config.run_dir)
    first = closed[-2] if len(closed) > 1 else bar
    if segment is not None and segment.start < first:
        first = segment.start
    found = run.benchmark(first, bar)
    if found is None:
        return None
    ticker, prices = found
    prices = prices.dropna()
    day = float(prices.iloc[-1] / prices.iloc[-2] - 1) if len(prices) > 1 else float("nan")
    since = float("nan")
    if segment is not None:
        before = prices[prices.index <= max(segment.start, prices.index[0])]
        since = float(prices.iloc[-1] / before.iloc[-1] - 1)
    return ticker, day, since


def _cents(amount: float) -> Decimal:
    return Decimal(float(amount)).quantize(CENT, rounding=ROUND_HALF_EVEN)


def _day(bar) -> str:
    return "none" if bar is None else str(pd.Timestamp(bar).date())


def _money(amount) -> str:
    return "unset" if amount is None else f"{amount:,.2f}"


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())
