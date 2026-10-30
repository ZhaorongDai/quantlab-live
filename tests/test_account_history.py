"""The account history: one ``account.toml`` per account, its strategy segments in order."""

from __future__ import annotations

import pandas as pd

from quantlab_live import records

BAR = pd.Timestamp("2024-01-02")


def _outcomes(*statuses: str) -> pd.DataFrame:
    n = len(statuses)
    return pd.DataFrame(
        {"order_quantity": [10.0] * n, "filled_quantity": [10.0 if s == "filled" else 0.0 for s in statuses],
         "avg_px": [20.0] * n, "fees": [1.0] * n, "status": list(statuses)},
        index=pd.Index(range(10001, 10001 + n), name="symbol"),
    )


def _segment(start, run="/runs/a", live_dir="/live/a") -> records.Segment:
    return records.Segment(pd.Timestamp(start), live_dir=live_dir, run=run, broker="ibkr_paper", initial_capital=10_000.0)


def test_an_account_without_history_has_no_segments(tmp_path):
    assert records.read_account_history(tmp_path / "account.toml") == []


def test_a_segment_starts_only_with_real_orders(tmp_path):
    path = tmp_path / "account.toml"
    assert not records.record_segment(path, _segment(BAR), _outcomes("not_sent", "unmapped", "foreign"))
    assert not records.record_segment(path, _segment(BAR), _outcomes())
    assert records.read_account_history(path) == []
    assert records.record_segment(path, _segment("2024-01-03"), _outcomes("not_sent", "venue_cancelled"))
    assert records.read_account_history(path) == [_segment("2024-01-03")]


def test_the_same_strategy_trading_on_continues_its_segment(tmp_path):
    path = tmp_path / "account.toml"
    assert records.record_segment(path, _segment(BAR), _outcomes("filled"))
    assert not records.record_segment(path, _segment("2024-01-03"), _outcomes("filled"))
    assert records.read_account_history(path) == [_segment(BAR)]


def test_a_strategy_switch_starts_the_next_segment_in_order(tmp_path):
    path = tmp_path / "account.toml"
    records.record_segment(path, _segment(BAR), _outcomes("filled"))
    records.record_segment(path, _segment("2024-02-01", run="/runs/b"), _outcomes("filled"))
    records.record_segment(path, _segment("2024-03-01", run="/runs/b", live_dir="/live/b"), _outcomes("rejected"))
    history = records.read_account_history(path)
    assert [(s.start.date().isoformat(), s.run, s.live_dir) for s in history] == [
        ("2024-01-02", "/runs/a", "/live/a"), ("2024-02-01", "/runs/b", "/live/a"), ("2024-03-01", "/runs/b", "/live/b"),
    ]
    assert "[[segment]]" in path.read_text() and "start = 2024-02-01" in path.read_text()


def test_a_segment_without_initial_capital_round_trips(tmp_path):
    path = tmp_path / "account.toml"
    segment = records.Segment(BAR, live_dir="/live/a", run="/runs/a", broker="ibkr_paper")
    records.record_segment(path, segment, _outcomes("filled"))
    assert records.read_account_history(path) == [segment]
    assert records.current_segment(path, "/live/a", "/runs/a") == segment
    assert records.current_segment(path, "/live/a", "/runs/b") is None
