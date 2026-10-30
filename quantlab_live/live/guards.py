"""The start guards: every node start is refused, before any order, on a mismatch.

``start_refusals`` is a pure function of the broker profile, the configured
account and Gateway port, and the account the broker reports once the node
is connected. It runs twice: before the node is built (``reported=None``)
and inside the node, after start-up reconciliation and before the decision
bar is published, with the accounts the execution clients reported.
"""

from __future__ import annotations

from collections.abc import Collection

from quantlab_live.brokers import IB_PORT_MODES, BrokerProfile, ib_account_mode


def start_refusals(
    profile: BrokerProfile,
    *,
    account_id: str | None,
    port: int | None,
    reported: Collection[str] | None = None,
) -> list[str]:
    """Return why a node start with ``profile`` must be refused; empty when it may start.

    The trading mode is ``live`` for a real-money profile and ``paper``
    otherwise (``ib_sandbox`` reads the paper Gateway's data). A start is
    refused when:

    - the configured IB account is of the other mode (``DU…`` is paper,
      ``U…`` real money), or a profile that trades at IB names none;
    - the Gateway port is one of IB's default ports of the other mode;
    - once connected (``reported`` given), the broker did not report the
      configured account.

    Parameters
    ----------
    profile : BrokerProfile
        The broker profile.
    account_id : str, optional
        The configured IB account (``ibkr_*``); unused by ``ib_sandbox``,
        whose account is its own.
    port : int, optional
        The configured IB Gateway port.
    reported : collection of str, optional
        The account ids the node's execution clients reported
        (NautilusTrader's, such as ``IB-DU123``); ``None`` before the node
        is built.

    Examples
    --------
    >>> from quantlab_live.brokers import profile
    >>> paper = profile("ibkr_paper", port=4002, account_id="DU123")
    >>> start_refusals(paper, account_id="DU123", port=4002)
    []
    >>> start_refusals(paper, account_id="U123", port=4001)
    ['the ibkr_paper profile trades paper, but account U123 is live', 'the ibkr_paper profile trades paper, but port 4001 is the live Gateway port']
    >>> start_refusals(paper, account_id="DU123", port=4002, reported=["IB-DU999"])
    ['the broker reported accounts IB-DU999, not DU123']
    """
    mode = "live" if profile.real_money else "paper"
    simulated = profile.simulated
    refusals = []
    if not simulated:
        account_mode = ib_account_mode(account_id)
        if account_id is None:
            refusals.append(f"the {profile.name} profile names no IB account")
        elif account_mode != mode:
            refusals.append(f"the {profile.name} profile trades {mode}, but account {account_id} is {account_mode or 'unknown'}")
    port_mode = IB_PORT_MODES.get(port)
    if port_mode is not None and port_mode != mode:
        refusals.append(f"the {profile.name} profile trades {mode}, but port {port} is the {port_mode} Gateway port")
    if reported is not None and not simulated and account_id is not None:
        numbers = {r.rsplit("-", 1)[-1] for r in reported}
        if account_id not in numbers:
            refusals.append(f"the broker reported accounts {', '.join(sorted(reported)) or 'none'}, not {account_id}")
    return refusals

