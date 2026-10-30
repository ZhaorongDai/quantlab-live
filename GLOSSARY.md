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
