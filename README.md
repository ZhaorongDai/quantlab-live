# quantlab-live

Live trading and replay of [quantlab](https://github.com/ZhaorongDai/quantlab) strategies, built
NautilusTrader first.

One strategy runs in three ways: live on a broker account, in a closed-loop replay (quantlab's portfolio
construction on every bar, against the holdings the replay actually has) and in an open-loop replay
(a finished rebalance table). They differ only in where targets come from and in the venue
configuration. Decisions follow quantlab's daily bars, and execution can be set to any time of the
trading day or to the opening or closing auction.

Status: under construction. NautilusTrader is pinned to an exact v2 release candidate.

## Development

quantlab is a path dependency on a sibling checkout:

```
projects/
  quantlab/        https://github.com/ZhaorongDai/quantlab
  quantlab-live/   this repository
```

Until quantlab's changes for this repository reach its `main`, the quantlab checkout is on its
branch `quantlab-live-support` (CI checks out the same branch).

```
uv sync
uv run pytest -n auto
```

Public functions carry NumPy docstrings with an Examples section; the examples run as doctests.

## Layout

- `quantlab_live/quantlab_gate.py`: the only module that imports quantlab (an exact allowlist, enforced by a test). `open_run(run_dir)` reads a run in-process (decision bars, target inputs, listings); `prepare_day(run_dir, date, store=...)` waits for quantlab's daily data update and runs quantlab's live prediction job, bound to NUMA node 1 with `taskset`.
- `quantlab_live/trading/`: NautilusTrader extension classes, the same live and in replay: `DecisionBar` (one decision bar's quantlab inputs as custom data), `QuantlabStrategy` (which writes each decision before its first order and resumes a decided bar after a restart), its target sources, the `ScheduledExecution` execution algorithm and `order_outcomes` (one outcome per ordered symbol, priced only from fills).
- `quantlab_live/records.py`: the repository's own records (decisions, outcomes, snapshots, equity), typed parquet in raw units.
- `quantlab_live/replay.py`: `replay(run_dir, loop="closed" | "open", start=None, end=None)` trades a quantlab run, or a window of it, on NautilusTrader's `BacktestEngine` and writes the records and NautilusTrader's reports. `restarts=[...]` restarts the node mid-replay as after a crash, and a replay into a decided directory resumes its decisions.

- `quantlab_live/live/`: the unattended live day. `day(live_dir)` is the hourly cron entry: under a per-day lock it prepares quantlab's data and prediction, values the latest snapshot, starts the node in a fresh process (retried until the latest start, restarted after a crash until the close), settles missing outcomes and mails every failure. `status`, `run-node` and `brief` take the same lock. The configuration is one `live.toml` (`run`, `initial_capital`, `at`, `order`, `broker`, `real_money`; everything else defaults, and `status` prints it all).
- `python -m quantlab_live day | status | run-node | brief | replay`: the command line.

The domain vocabulary is in [GLOSSARY.md](GLOSSARY.md).
