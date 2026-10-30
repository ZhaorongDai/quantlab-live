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

- `quantlab_live/quantlab_gate.py`: the only module that imports quantlab (an exact allowlist, enforced by a test).
- `quantlab_live/trading/`: NautilusTrader extension classes, the same live and in replay: `DecisionBar` (one decision bar's quantlab inputs as custom data), `QuantlabStrategy`, its target sources and the `ScheduledExecution` execution algorithm.
- `quantlab_live/records.py`: the repository's own records (decisions, outcomes, snapshots, equity), typed parquet in raw units.
- `quantlab_live/replay.py`: `replay(run_dir, loop="open")` trades a quantlab run on NautilusTrader's `BacktestEngine` and writes the records and NautilusTrader's reports.

The domain vocabulary is in [GLOSSARY.md](GLOSSARY.md).
