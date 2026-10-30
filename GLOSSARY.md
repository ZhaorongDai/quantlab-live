# Glossary

**Symbol**:
The security key of a quantlab run's dataset (a Sharadar permaticker, or a CRSP PERMNO); the strategy's only identity for a security. An instrument id exists only where an order is submitted or a position is read back.
_Avoid_: ticker, PERMNO (as the general term)

**Decision bar**:
One bar of quantlab's daily calendar on which the strategy decides, at the bar's close; also the custom data record carrying that bar's predictions and marks.

**Marks**:
A decision bar's raw close per symbol (a delisting security marked at its last valuation) and its delisting flags; what orders are sized and holdings valued at.

**Closed-loop replay**:
A replay in which quantlab's portfolio construction decides on every rebalance bar against the holdings the replay actually has; the same loop as live trading.

**Open-loop replay**:
A replay that executes a finished rebalance table; fills never feed back into decisions.

**Strategy capital**:
What a decision sizes its orders on: `initial_capital` plus the strategy's cumulative profit and loss, valued at quantlab's close and capped by the account's equity; the account's equity when `initial_capital` is unset.

**Rebalance table**:
Target portfolio weights per decision bar and symbol, finished by quantlab.
_Avoid_: signal file, order list

**Listing**:
A symbol's ticker and primary exchange as of a date, as quantlab records them; what a broker's instrument id is built from.
_Avoid_: contract (the broker's), name (quantlab's display name)

**Security map**:
Symbols to a broker's instrument ids and back, built each day from the listings by the broker profile, keeping every id a symbol has had so a renamed ticker's id still maps back.

**Broker profile**:
A broker as configuration: its NautilusTrader data and execution clients, its security-map builder and its fee model; paper and real money are separate profiles.

**Live prediction store**:
quantlab's append-only store of a run's predictions past its end, one bar per live day, written by quantlab's prediction job.

**Raw units**:
Shares and prices as traded on each date, not split adjusted; every record is in raw units.

**Engine unit**:
A replay engine's unit of quantity: the raw shares of one share at the replay's first bar, so a split never changes a position's quantity. Replay converts engine units to raw units at the edges (the fee model, dividends, the records).
_Avoid_: adjusted share

**Foreign position**:
A position in the account that the strategy does not own (a manual trade, another holding); never traded, outside the strategy capital, and a target on its instrument is skipped.

**Unmapped**:
A symbol with no instrument on the day; a target on it is skipped and stays in cash.

**Order outcome**:
What an ordered symbol of a decision came to, exactly one per symbol: filled, partially filled, cancelled (by us), venue cancelled (with or without a fallback), rejected, unmapped, foreign, not sent or unverified.

**Live day**:
One New York trading day of a live directory, run by its supervisor: prepare, value, start the node, settle, the post-close mail.

**Latest start**:
The plan's execution time (the opening or closing auction's cut-off, or the `HH:MM` time) minus the start margin; a node not started by then leaves the day not traded. It is also the cut-off of quantlab's data and prediction.

**Day state**:
The supervisor's record of a live day (prepare polls, node attempts, mails, heartbeat); only the supervisor writes it.

**Step report**:
What one node process tells the supervisor (started, finished, outcomes written, error, alerts); the node's only channel to it.

**Strategy segment**:
One strategy (a live directory, its run and broker) trading an account, from the first decision bar whose orders reached the broker until another strategy starts the next segment.
_Avoid_: era, epoch

**Account history**:
An account's strategy segments in order, one `account.toml` per account; a strategy switch is written down there and nowhere else.

**Close briefing**:
The after-open mail's account of the previous bar's close: the strategy's equity at quantlab's close beside a closed-loop replay of the current strategy segment, the execution attribution of the gap, the decision recheck, the benchmark by ticker and the foreign positions.

**Execution attribution**:
The live-versus-replay gap at a close split into decisions, unfilled orders, fill prices, fees and a residual, in cents, summing to the gap exactly; a large residual needs checking.

**Decision recheck**:
quantlab's construction re-run on a decision's bar and the holdings actually held; it must give the decision's weights bit for bit, and a mismatch needs checking.
