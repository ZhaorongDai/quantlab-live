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

## Deployment

The live server runs three things: a checkout of this repository, one Docker Compose project with the
IB Gateway and Redis, and cron.

**Checkouts.** Clone quantlab (`main`) and this repository side by side in their own directory, apart
from any other quantlab checkout on the server, and run `uv sync` in this one. Add `--extra mosek` when
the run's portfolio construction uses the MOSEK solver, which needs a MOSEK licence. A deploy is
`git pull` in both, then the same `uv sync`, outside the live window (US weekdays, roughly 06:00 to
16:30 New York time).

**Compose project.** It holds two services:

- the IB Gateway, with its image pinned by digest, its API and VNC ports bound to 127.0.0.1 and its
  credentials in an owner-only `.env` beside the compose file; and
- Redis 7, with its image pinned by digest, bound to 127.0.0.1 on its default port, with AOF on and
  `appendfsync everysec`, its data directory under the data root, `cpuset` on the cores the jobs use, and
  the deploying user as its user, so the backup can read its files.

**Settings.** They live in owner-only files in `~/.config/quantlab/`, never in the repository:

- `ibkr.env` sets `TWS_ACCOUNT`.
- `mail.env` sets `QUANTLAB_SMTP_HOST`, `QUANTLAB_SMTP_PORT` (465 by default; SMTP over SSL),
  `QUANTLAB_SMTP_USER`, `QUANTLAB_SMTP_PASSWORD`, `QUANTLAB_MAIL_TO` and optionally `QUANTLAB_MAIL_FROM`.
- `healthchecks.url` holds the ping URL of a healthchecks.io check with a one-hour period. The check
  mails when the pings stop.

**cron.** Every job runs hourly and checks the New York hour itself, so daylight-saving changes need no
edit:

| Job | When | What it does |
| --- | --- | --- |
| Gateway restart | 07:00 | `docker compose up -d --force-recreate ib-gateway`: a fresh login before the live day. Redis keeps running. |
| Upkeep | 18:00 | Takes a Redis `BGSAVE` and archives the whole data directory (RDB and AOF) under the data root, keeping 14 days. Gzips each live directory's `logs/<day>.log` after 7 days and deletes it after 365. |
| Ping | every hour | `curl` of the healthchecks.io URL: the dead-man signal. |
| Live day | every hour | `taskset -c <cores> uv run python -m quantlab_live day <live_dir>`. |

The code binds no process to cores. To keep a day on chosen cores, the live day's cron line starts the
supervisor under `taskset`; the node and quantlab's prediction job are its child processes and inherit the
binding, and Redis gets the same cores through its `cpuset`. `<cores>` is a CPU list such as `0-7`; drop
`taskset -c <cores>` to use every core.

**Restore.** Stop Redis (`docker compose stop redis`), move its data directory aside, unpack an archive
into a fresh one, and start it again.

## Layout

- `quantlab_live/quantlab_gate.py`: the only module that imports quantlab (an exact allowlist, enforced by a test). `open_run(run_dir)` reads a run in-process (decision bars, target inputs, listings), and `run.report(directory)` writes a live or replay directory's `metrics.json` and `report.html` with quantlab's metric and report functions, from the records alone; `prepare_day(run_dir, date, store=...)` waits for quantlab's daily data update and runs quantlab's live prediction job.
- `quantlab_live/trading/`: NautilusTrader extension classes, the same live and in replay: `DecisionBar` (one decision bar's quantlab inputs as custom data), `QuantlabStrategy` (which writes each decision before its first order and resumes a decided bar after a restart), its target sources, the `ScheduledExecution` execution algorithm and `order_outcomes` (one outcome per ordered symbol, priced only from fills).
- `quantlab_live/records.py`: the repository's own records (decisions, outcomes, snapshots, equity, the account history `account.toml`), typed parquet in raw units; `read_book` reads them all back for a report.
- `quantlab_live/replay.py`: `replay(run_dir, loop="closed" | "open", start=None, end=None)` trades a quantlab run, or a window of it, on NautilusTrader's `BacktestEngine` and writes the records, `metrics.json` and `report.html` from them, and NautilusTrader's reports for audit. `restarts=[...]` restarts the node mid-replay as after a crash, and a replay into a decided directory resumes its decisions.

- `quantlab_live/live/`: the unattended live day. `day(live_dir)` is the hourly cron entry: under a per-day lock it prepares quantlab's data and prediction, values the latest snapshot, starts the node in a fresh process (retried until the latest start, restarted after a crash until the close), settles missing outcomes and mails every failure. After the open it mails the day's decision with the previous close's briefing (`brief`): the strategy's equity beside a closed-loop replay of its current segment, the gap split by execution attribution, the decision recheck, the benchmark by ticker and any foreign positions. `status`, `run-node` and `brief` take the same lock. The configuration is one `live.toml` (`run`, `initial_capital`, `at`, `order`, `broker`, `real_money`; everything else defaults, and `status` prints it all). Every start is guarded (`start_refusals`): the account and the Gateway port must match the profile's mode, and an `ibkr_real` start also needs `real_money = true`, an owner-only `real_money.armed` file in the live directory, `initial_capital`, the risk engine's `max_notional_per_order` and `max_order_submit_rate`, a Redis cache and a final NautilusTrader release.
- `python -m quantlab_live day | status | run-node | brief | replay`: the command line.

The domain vocabulary is in [GLOSSARY.md](GLOSSARY.md).
