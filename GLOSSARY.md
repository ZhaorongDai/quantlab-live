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
