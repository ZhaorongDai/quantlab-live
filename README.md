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

```
uv sync
uv run pytest -n auto
```

Public functions carry NumPy docstrings with an Examples section; the examples run as doctests.

## Layout

- `quantlab_live/quantlab_gate.py`: the only module that imports quantlab (an exact allowlist, enforced by a test). `open_run(run_dir)` reads a run in-process (decision bars, target inputs, listings), and `run.report(directory)` writes a live or replay directory's `metrics.json` and `report.html` with quantlab's metric and report functions, from the records alone; `prepare_day(run_dir, date, store=...)` waits for quantlab's daily data update and runs quantlab's live prediction job, bound to NUMA node 1 with `taskset`.
- `quantlab_live/trading/`: NautilusTrader extension classes, the same live and in replay: `DecisionBar` (one decision bar's quantlab inputs as custom data), `QuantlabStrategy` (which writes each decision before its first order and resumes a decided bar after a restart), its target sources, the `ScheduledExecution` execution algorithm and `order_outcomes` (one outcome per ordered symbol, priced only from fills).
- `quantlab_live/records.py`: the repository's own records (decisions, outcomes, snapshots, equity, the account history `account.toml`), typed parquet in raw units; `read_book` reads them all back for a report.
- `quantlab_live/replay.py`: `replay(run_dir, loop="closed" | "open", start=None, end=None)` trades a quantlab run, or a window of it, on NautilusTrader's `BacktestEngine` and writes the records, `metrics.json` and `report.html` from them, and NautilusTrader's reports for audit. `restarts=[...]` restarts the node mid-replay as after a crash, and a replay into a decided directory resumes its decisions.

- `quantlab_live/live/`: the unattended live day. `day(live_dir)` is the hourly cron entry: under a per-day lock it prepares quantlab's data and prediction, values the latest snapshot, starts the node in a fresh process (retried until the latest start, restarted after a crash until the close), settles missing outcomes and mails every failure. After the open it mails the day's decision with the previous close's briefing (`brief`): the strategy's equity beside a closed-loop replay of its current segment, the gap split by execution attribution, the decision recheck, the benchmark by ticker and any foreign positions. `status`, `run-node` and `brief` take the same lock. The configuration is one `live.toml` (`run`, `initial_capital`, `at`, `order`, `broker`, `real_money`; everything else defaults, and `status` prints it all). Every start is guarded (`start_refusals`): the account and the Gateway port must match the profile's mode, and an `ibkr_real` start also needs `real_money = true`, an owner-only `real_money.armed` file in the live directory, `initial_capital`, the risk engine's `max_notional_per_order` and `max_order_submit_rate`, a Redis cache and a final NautilusTrader release.
- `python -m quantlab_live day | status | run-node | brief | replay`: the command line.

The domain vocabulary is in [GLOSSARY.md](GLOSSARY.md).
