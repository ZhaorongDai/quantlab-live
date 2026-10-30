"""``live.brief``: the after-open mail with the previous close's briefing.

The fixture live directory is a closed-loop replay that filled 1% worse than
the prints (``ProportionalSlippage``), standing in for a live account whose
fills slipped: a replay of its segment at the prints then differs from it by
the fill prices alone.
"""

from __future__ import annotations

import shutil
from decimal import Decimal

import pandas as pd
import pytest
from quantlab.portfolio.config import TopkDropoutConfig
from quantlab.portfolio.predefined.topk_dropout import TopkDropoutConstructor
from quantlab.runs.prediction_panel import LabelSpec

from quantlab_live import records
from quantlab_live.live import brief
from quantlab_live.replay import ProportionalSlippage, replay
from tests.live_fakes import RecordingMailer
from tests.quantlab_run_fixture import build_constructor_run

A, B, C, BENCHMARK = 10001, 10002, 10003, 90001
BARS = pd.bdate_range("2024-01-02", periods=4)
OPEN = {A: [10.0, 10.5, 11.5, 12.5], B: [20.0, 20.5, 21.0, 24.0], C: [50.0, 48.0, 41.0, 45.0]}
CLOSE = {A: [10.0, 11.0, 12.0, 13.0], B: [20.0, 20.0, 25.0, 25.0], C: [50.0, 40.0, 40.0, 50.0]}
PREDICTIONS = {"ret_5": {A: [0.03] * 4, B: [0.02] * 4, C: [0.01] * 4}}
LABELS = [LabelSpec("ret_5", "raw", 1, 5)]
BENCHMARK_CLOSE = [100.0, 101.0, 102.0, 103.0]
#: The day of the after-open mail; it briefs the close of the bar before it.
TODAY, CLOSED = BARS[3], BARS[2]
CENT = Decimal("0.01")


def _rule():
    return TopkDropoutConstructor(TopkDropoutConfig(direction="long_only", topk=2, n_drop=1))


@pytest.fixture(scope="module")
def traded(tmp_path_factory):
    """A live directory that traded the run from its first bar, with its account history."""
    root = tmp_path_factory.mktemp("brief")
    run_dir, _ = build_constructor_run(
        root, BARS, OPEN, CLOSE, PREDICTIONS, _rule(), LABELS,
        benchmark=({BENCHMARK: BENCHMARK_CLOSE}, {BENCHMARK: BENCHMARK_CLOSE}), benchmark_ticker="SPY",
    )
    live_dir = replay(run_dir, root / "live", fill_model=ProportionalSlippage(0.01))
    (live_dir / "live.toml").write_text(f'run = "{run_dir}"\n')
    segment = records.Segment(BARS[0], str(live_dir.absolute()), str(run_dir.absolute()), "ibkr_paper")
    assert records.record_segment(live_dir / "account.toml", segment, records.read_outcomes(live_dir, BARS[0]))
    return live_dir


@pytest.fixture
def live_dir(traded, tmp_path):
    copy = tmp_path / "live"
    shutil.copytree(traded, copy)
    segment = records.read_account_history(traded / "account.toml")[0]
    (copy / "account.toml").unlink()
    moved = records.Segment(segment.start, str(copy.absolute()), segment.run, segment.broker)
    records.record_segment(copy / "account.toml", moved, records.read_outcomes(copy, BARS[0]))
    return copy


def _rewrite_decision(live_dir, bar, change):
    decision = records.read_decision(live_dir, bar)
    capital = records.decision_capital(live_dir, bar)
    (live_dir / records.DECISIONS / f"{bar.date()}.parquet").unlink()
    change(decision)
    records.write_decision(live_dir, bar, decision, capital)


def test_the_attribution_splits_the_gap_to_the_replay_exactly(live_dir):
    mailer = RecordingMailer()
    briefing = brief(live_dir, TODAY, mailer=mailer)
    assert briefing.bar == CLOSED
    # The slipped fills left cash at -484.125 against the replay's -378.75.
    assert briefing.live_equity == pytest.approx(11_765.875)
    assert briefing.replay_equity == pytest.approx(11_871.25)
    assert briefing.gap == Decimal("-105.38")
    assert sum(briefing.attribution.values()) == briefing.gap
    parts = briefing.attribution
    assert list(parts) == ["decisions", "unfilled orders", "fill prices", "fees", "residual"]
    # 1% on 500 A at 10.5 and 250 B at 20.5, to the cent; the same orders and fees as the replay.
    assert abs(parts["fill prices"] - Decimal("-105.375")) <= CENT
    assert parts["decisions"] == parts["unfilled orders"] == parts["fees"] == 0
    assert abs(parts["residual"]) <= CENT
    assert briefing.checks == ()
    ((subject, body),) = mailer.mails
    assert subject == f"quantlab-live {TODAY.date()}: after the open"
    assert "fill prices" in body and "-105.38" in body and "same weights bit for bit" in body


def test_a_large_residual_is_flagged(live_dir):
    equity = records.read_equity(live_dir)
    equity.loc[CLOSED, ["cash", "equity"]] += 50.0  # money no trade explains
    equity.reset_index().to_parquet(live_dir / records.EQUITY)
    briefing = brief(live_dir, TODAY, mailer=RecordingMailer())
    assert abs(briefing.attribution["residual"] - Decimal("50.00")) <= CENT
    assert sum(briefing.attribution.values()) == briefing.gap == Decimal("-55.38")
    assert any("residual" in check for check in briefing.checks)
    assert briefing.subject().endswith("NEEDS CHECKING")


def test_a_recheck_mismatch_needs_checking(live_dir):
    def nudge(decision):
        decision.loc[A, "weight"] += 1e-12

    _rewrite_decision(live_dir, CLOSED, nudge)
    mailer = RecordingMailer()
    briefing = brief(live_dir, TODAY, mailer=mailer)
    assert any("recheck" in check and str(A) in check for check in briefing.checks)
    assert mailer.subjects == [f"quantlab-live {TODAY.date()}: after the open: NEEDS CHECKING"]


def test_the_benchmark_appears_by_its_ticker(live_dir):
    briefing = brief(live_dir, TODAY, mailer=RecordingMailer())
    ticker, day, segment = briefing.benchmark
    assert ticker == "SPY"
    assert day == pytest.approx(102.0 / 101.0 - 1)
    assert segment == pytest.approx(102.0 / 100.0 - 1)
    assert "SPY" in briefing.text()


def test_foreign_positions_are_listed(live_dir):
    snapshot = records.read_snapshot(live_dir, CLOSED)
    records.write_snapshot(live_dir, records.Snapshot(CLOSED, snapshot.positions, snapshot.cash, {"IBM.NYSE": 7.0}))
    text = brief(live_dir, TODAY, mailer=RecordingMailer()).text()
    assert "IBM.NYSE" in text and "7" in text


def test_without_a_segment_there_is_nothing_to_replay(live_dir):
    (live_dir / "account.toml").unlink()
    briefing = brief(live_dir, TODAY, mailer=RecordingMailer())
    assert briefing.replay_equity is None and briefing.attribution == {}
    assert "no strategy segment" in briefing.text()
    assert briefing.checks == ()


def test_an_order_live_made_beyond_the_replay_and_left_unfilled_nets_out(live_dir):
    outcomes = records.read_outcomes(live_dir, BARS[0])
    outcomes.loc[A, ["order_quantity", "status"]] = [600.0, "partially_filled"]
    records.write_outcomes(live_dir, BARS[0], outcomes)
    parts = brief(live_dir, TODAY, mailer=RecordingMailer()).attribution
    # 100 more A ordered than the replay did, worth 12.00 at the close against the replay's 10.50 fill.
    assert parts["decisions"] == Decimal("150.00") and parts["unfilled orders"] == Decimal("-150.00")
    assert abs(parts["residual"]) <= CENT
