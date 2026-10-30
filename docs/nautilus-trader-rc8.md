# NautilusTrader 2.0.0rc8: limitations and workarounds

quantlab-live is pinned to NautilusTrader `2.0.0rc8`. Each entry below is a behaviour of that release that the code works around. When the pin moves (to a later release candidate or to `2.0.0` final), check every entry: read the release notes, run the probe, and remove the workaround once the limitation is gone. An entry that still holds stays, with the version it was checked on.

Each entry gives the limitation, the workaround and where it lives, and how to check it.

## Strategy and execution algorithm

1. **Python handler exceptions are swallowed.** An exception in a Python `on_data` handler is logged, and `BacktestEngine.run()` returns normally, with or without `shutdown_on_error`.
   - Workaround: `QuantlabStrategy` keeps its first failure in `failure`, and `replay` raises it (`trading/strategy.py`, `replay.py`).
   - Check: raise in `on_data` and see whether `run()` raises or stops.

2. **Configuration is read in `__new__`.** A Python `Strategy` subclass reads its `StrategyConfig` in `__new__`. An `ExecutionAlgorithm` takes its config in `__init__`. Python `FeeModel`, `FillModel` and `SimulationModule` subclasses must override `__new__` to take no arguments.
   - Workaround: the subclasses override `__new__` (`trading/strategy.py`, `brokers.py`, `replay.py`).
   - Check: drop an override and construct the class with arguments.

3. **An execution algorithm cannot see the kernel environment.** A Python `ExecutionAlgorithm` cannot read whether it runs live or in a backtest.
   - Workaround: `ScheduledExecution` takes `environment` as an argument (`trading/scheduled_execution.py`). The spec originally had it read the kernel environment.
   - Check: look for the environment on the algorithm's clock, cache or kernel.

4. **An execution algorithm has no data subscriptions and no state hooks.** A Python `ExecutionAlgorithm` has no data subscriptions and no `on_save`/`on_load`.
   - Workaround: the algorithm reads the cache's last trade and keeps no state. Its action is a pure function of now, the plan and the session (`trading/scheduled_execution.py`).
   - Check: subscribe from an algorithm, and implement `on_save`.

5. **Opaque `bytes` fields are refused.** `register_custom_data_class` refuses a `bytes` field.
   - Workaround: `DecisionBar` carries base64 Arrow IPC text (`trading/decision_bar.py`).
   - Check: register a custom data class with a `bytes` field.

6. **Mixed time types.** `spawn_limit(expire_time=)` takes a datetime, while `order_factory` takes nanoseconds.
   - Workaround: converted at the call site (`trading/scheduled_execution.py`).

7. **Order-event accessors are irregular.** `order.events()` is a method, and a spawned child's `init_event` can be None.
   - Workaround: the outcome code does not rely on `init_event` (`trading/outcomes.py`).

8. **Orders are cancelled through the order object.** `Strategy.cancel_order` takes the order. It cancels a kept primary that was never sent locally.
   - Workaround: used as is on resume (`trading/strategy.py`).

9. **Two strategies need distinct order id tags.** Two strategies on one engine collide without distinct `order_id_tag` values.
   - Workaround: the tests set distinct tags.

## Backtest engine and simulated venue

10. **`BacktestNode` takes no custom data.**
    - Workaround: replay uses `BacktestEngine` (`replay.py`).
    - Check: add `CustomData` through a `BacktestDataConfig`.

11. **The simulator never expires `DAY` orders.**
    - Workaround: backtest auction children are `GTD` limit orders that expire 1 second after the print. Backtest `DAY` limit children are given an expiry at the close (`trading/scheduled_execution.py`). The spec originally asked for `DAY` market orders, or marketable `DAY` limit orders for LOO/LOC.
    - Check: leave a `DAY` limit order working past the session close in a backtest.

12. **GTD expiry is checked late.** GTD expiry is checked only on the instrument's next trade, after matching.
    - Workaround: replay repeats each auction print 1 second later at the same price, so an unfilled child expires before anything else trades (`replay.py`).
    - Check: a GTD order with no later trade on its instrument.

13. **Auction time-in-force is rejected.** The simulator rejects `AT_THE_OPEN` and `AT_THE_CLOSE` orders.
    - Workaround: as in entry 11.
    - Check: submit a market-on-open order in a backtest.

14. **An empty fill-model book rejects market orders.** A market order given an empty fill-model book is rejected ("No market"), not expired.
    - Workaround: the fallback tests block auction children, which are limit orders (`tests/test_replay_plans.py`).

15. **An L1 book walks past a print's size.** A 100-share order against a 1-share print fills 99 shares one tick worse.
    - Workaround: replay prints carry an unlimited size, `PRINT_SIZE` (`replay.py`).
    - Check: a print of size 1 against a larger market order.

16. **A missing print fills at a stale price.** A market child for a security with no opening print fills against its last, stale print.
    - Workaround: `ScheduledExecution` leaves the order unsent when the cache's last trade is not at the print time (`trading/scheduled_execution.py`).

17. **Timers fire after the print.** A timer at the exact timestamp of a print fires after the print is processed, so a child spawned then fills at that print.
    - Workaround: relied on by the plans.
    - Check: whether the order of timers and data at one timestamp changed.

18. **Simulation modules run after the handlers.** Simulation modules run once per data timestamp, after that timestamp's handlers, custom data included.
    - Workaround: dividends are paid after the decision on the bar before the ex-date (`replay.py`, `DividendModule`).

19. **The fee model has no fill timestamp.**
    - Workaround: the split ratio is looked up at the order's `ts_init` (`brokers.py`).

20. **Expiry orders carry no tags.** Expiry fills come from venue-generated `EXPIRATION-*` market orders with no tags.
    - Workaround: the fee model recognises them by their client order id and charges nothing (`brokers.py`).

21. **Empty data is refused.** `add_data([])` raises "data was empty".
    - Workaround: replay adds data only when there is some (`replay.py`).

22. **Strategies cannot be swapped mid-run.** A `BacktestEngine` can run in pieces (`run(end=, streaming=True)`, then `run(start=)`) and keeps its state. But a component added between pieces is never started, and starting it by hand fails with "Already borrowed".
    - Workaround: `replay(restarts=...)` clears the algorithm's timers rather than replacing the strategy (`replay.py`).
    - Check: add a strategy between two `run` calls.

23. **A venue requires an explicit fee model.**
    - Workaround: replay always passes one (`replay.py`).

24. **Reports need the venue.** `generate_account_report()` needs the venue. Nested report values are written as JSON text.
    - Workaround: in `replay.py`.

25. **Log level parameter.** `LoggerConfig` takes `stdout_level=LogLevel.X`.

## Live node and Interactive Brokers adapter

26. **A cache database needs the Rust live node.** A cache database needs `LiveNode.run()` with native adapters.

27. **Upstream NautilusTrader issue 5428.** An IB start with positions held overnight aborts on an average-cost mismatch. This blocks the paper acceptance; it is tracked in #3.
    - Check: the release notes for a fix, then the `ib_sandbox` smoke test with held positions.

28. **SMART ids carry a SMART primary exchange.** Loading a raw `...=STK.SMART` id makes the IB adapter build a contract with primary exchange `SMART`, which IB may reject.
    - Check: the `ib_sandbox` smoke test (#11).

29. **The sandbox calls Python fee models.** Checked on rc8 (#11): a `LiveNode` with the sandbox execution client charged `IbkrFeeModel`'s USD 1.00 minimum on a 5-share fill. No workaround is needed.
    - Check: the `ib_sandbox` smoke test asserts every outcome's `fees` is at least the minimum (`tests/test_ib_sandbox.py`).

30. **`SymbologyMethod` has no name.** A Python `SymbologyMethod` has no `.name`.

31. **IB configs default the port.** With no port given, the IB configs fall back to NautilusTrader's default port.
    - Workaround: a profile builds no IB client without an explicit port, and `live.toml` names it (`gateway_port`) (`brokers.py`, `live/config.py`).
    - Check: build an IB client config without a port and read its `port`.

32. **A market order needs a cached price.** The risk engine denies a `MARKET` order (`MARKET_PRICE_UNAVAILABLE`) on an instrument with no quote or trade in the cache, on every account, with or without a notional cap. A node just started, or an instrument just loaded at runtime, has none.
    - Workaround: the node's market-data actor subscribes to the quotes and trades of every instrument loaded, and a live `ScheduledExecution` child waits up to `PRICE_WAIT_SECONDS` for a first price before it is sent (`live/node.py`, `trading/scheduled_execution.py`).
    - Check: submit a market order on an instrument without a quote in a `LiveNode`.

33. **The sandbox rejects auction time in force.** The sandbox execution client rejects `AT_THE_OPEN` and `AT_THE_CLOSE` orders ("time in force AT_THE_OPEN is not currently supported"), as the simulator does (entry 13).
    - Workaround: in `Environment.SANDBOX` an auction child is a `DAY` order sent at the auction's time (`trading/scheduled_execution.py`).
    - Check: submit a market-on-open order through the sandbox execution client.

34. **A live node hides Python failures.** An exception in a Python actor's `on_start` fails the hosted run ("Hosted run failed"), yet `LiveNode.run()` returns normally; an exception in a Python timer callback is logged and swallowed.
    - Workaround: `run_node` reports a node whose start check never passed as failed, and the strategy keeps its first failure (entry 1) (`live/node.py`).
    - Check: raise in an actor's `on_start` and see whether `run()` raises.

35. **The Redis cache cannot select a database.** `RedisCacheConfig` takes no database index.
    - Workaround: `live.toml` refuses a `redis_db` other than 0, so paper and real money use separate Redis instances; the fixed `trader_id` per live directory separates the keys of directories sharing one (`live/config.py`).
    - Check: the parameters of `RedisCacheConfig`.

36. **The Python cache cannot be seeded.** A node's Python `Cache` has no `add_instrument` and no account listing.
    - Workaround: the in-process node tests serve instruments from a Python data client (`tests/fake_broker.py`), and the start check looks up the configured account by its id (`live/node.py`).
    - Check: the methods of `nautilus_trader.common.Cache`.

37. **The notional cap is per instrument.** `LiveRiskEngineConfig.max_notional_per_order` maps an instrument id to a cap (in its quote currency); there is no cap for every instrument, and an instrument it does not name is not capped. Its order submit rate defaults to 100 a second.
    - Workaround: `live.toml`'s `max_notional_per_order` is set on every instrument of the day's security map, the only instruments the strategy trades, and its `max_order_submit_rate` is passed only when set; a real-money start needs both (`live/config.py`, `live/node.py`, `live/guards.py`).
    - Check: the parameters of `LiveRiskEngineConfig`; the `ib_sandbox` smoke test's capped node (`tests/test_ib_sandbox.py`) sends nothing above the cap.

38. **A request's callback runs once.** `request_instrument` with several instrument ids calls its callback on the first response only, so instruments that arrive in later responses go unnoticed by it.
    - Workaround: while orders wait for their instruments, `QuantlabStrategy` also looks at the cache every second, and leaves unsent only the orders whose instrument has not arrived `INSTRUMENT_WAIT_SECONDS` after the request (`trading/strategy.py`).
    - Check: request two instruments from a data client that answers each in its own response (`tests/fake_broker.py`), and count the callbacks.

## Corporate actions in replay

39. **A position's quantity cannot change from Python.** NautilusTrader has no Python call that changes an open position's quantity, so a split cannot be applied to a position held through it.
    - Workaround: replay trades in split-adjusted engine units, one unit being the raw shares of one share at the replay's first bar. The prints and marks it is fed are split adjusted (`SplitTable`), the fee model converts units back to raw shares at the order's `ts_init` (entry 19), the `DividendModule` pays dividends per unit, and the records are converted to raw units when written (`replay.py`, `brokers.py`). An upstream patch for corporate actions is optional later work.
    - Check: look for a split or position-adjustment event on positions, or a cache method that sets a position's quantity, and replay a split without the conversion.
