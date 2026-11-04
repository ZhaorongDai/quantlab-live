"""The start guards: every node start is refused, before any order, on a mismatch.

``start_refusals`` is a pure function of the broker profile, the configured
account and Gateway port, the account the broker reports once the node is
connected, NautilusTrader's version and the real-money switches. It runs
twice: before the node is built (``reported=None``) and inside the node,
after start-up reconciliation and before the decision bar is published, with
the accounts the execution clients reported.

A real-money start needs every switch at once, so no single configuration
change starts one: the ``ibkr_real`` profile, ``real_money = true``, an
owner-only arming file in the live directory, an ``initial_capital``, the
risk engine's per-order notional cap and order submit rate, a Redis cache,
a live account and Gateway, and a final NautilusTrader release.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from quantlab_live.brokers import IB_PORT_MODES, BrokerProfile, ib_account_mode

#: The arming file, in the live directory: a real-money start needs it owner-only.
ARMING_FILE = "real_money.armed"
#: ``arming_state`` of an arming file that arms.
ARMED = "owner-only"
#: The first NautilusTrader release a real-money start may run on.
FIRST_REAL_MONEY_RELEASE = (2, 0, 0)


@dataclass(frozen=True)
class Switches:
    """The real-money switches of a start, read from ``live.toml`` and the live directory.

    Parameters
    ----------
    real_money : bool, default False
        ``real_money`` of ``live.toml``.
    arming : str, default ``"missing"``
        ``arming_state`` of the live directory's arming file.
    initial_capital : float, optional
        ``initial_capital`` of ``live.toml``.
    max_notional_per_order : float, optional
        The risk engine's per-order notional cap, in the instrument's quote
        currency.
    max_order_submit_rate : str, optional
        The risk engine's order submit rate (``"10/00:00:01"``).
    redis : bool, default False
        Whether the node keeps its cache in Redis.

    Examples
    --------
    >>> Switches().real_money, Switches().arming
    (False, 'missing')
    """

    real_money: bool = False
    arming: str = "missing"
    initial_capital: float | None = None
    max_notional_per_order: float | None = None
    max_order_submit_rate: str | None = None
    redis: bool = False


def arming_state(path) -> str:
    """Return ``ARMED`` when ``path`` is a regular file of the running user that no one else may access.

    Otherwise say what it is: ``"missing"``, ``"not a regular file"``,
    ``"owned by another user"`` or ``"readable by others (mode 0644)"``.

    Examples
    --------
    >>> import tempfile
    >>> path = Path(tempfile.mkdtemp()) / ARMING_FILE
    >>> arming_state(path)
    'missing'
    >>> _ = path.write_text("")
    >>> path.chmod(0o600)
    >>> arming_state(path)
    'owner-only'
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return "missing"
    if not stat.S_ISREG(info.st_mode):
        return "not a regular file"
    if info.st_uid != os.getuid():
        return "owned by another user"
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        return f"readable by others (mode {mode:04o})"
    return ARMED


def start_refusals(
    profile: BrokerProfile,
    *,
    account_id: str | None,
    port: int | None,
    reported: Collection[str] | None = None,
    version: str | None = None,
    switches: Switches = Switches(),
) -> list[str]:
    """Return why a node start with ``profile`` must be refused; empty when it may start.

    The trading mode is ``live`` for a real-money profile (``ibkr_real``)
    and ``paper`` otherwise (``ib_sandbox`` reads the paper Gateway's
    data). A start is refused when:

    - the configured IB account is of the other mode (``DU…`` is paper,
      ``U…`` real money), or a profile that trades at IB names none;
    - the Gateway port is one of IB's default ports of the other mode;
    - once connected (``reported`` given), the broker did not report the
      configured account;
    - ``real_money = true`` is set on a paper profile.

    A real-money start is refused, besides, unless NautilusTrader is a final
    release of 2.0.0 or later, ``real_money = true`` is set, the arming file
    is owner-only, ``initial_capital`` is set, the risk engine's per-order
    notional cap and order submit rate are configured, and the node keeps its
    cache in Redis. Each missing switch is its own refusal.

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
    version : str, optional
        NautilusTrader's version (``nautilus_trader.__version__``); unknown
        when ``None``, which refuses real money.
    switches : Switches, optional
        The real-money switches; none set by default.

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
    >>> real = profile("ibkr_real", port=4001, account_id="U123")
    >>> armed = Switches(True, ARMED, 10_000.0, 2_000.0, "10/00:00:01", redis=True)
    >>> start_refusals(real, account_id="U123", port=4001, version="2.0.0", switches=armed)
    []
    >>> start_refusals(real, account_id="U123", port=4001, version="2.0.0rc8", switches=armed)
    ['a real-money start needs a final NautilusTrader release, and 2.0.0rc8 is not a final release']
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
    if profile.real_money:
        refusals.extend(_real_money_refusals(version, switches))
    elif switches.real_money:
        refusals.append(f"real_money = true is set, but the {profile.name} profile trades paper")
    return refusals


def _real_money_refusals(version: str | None, switches: Switches) -> list[str]:
    """Return a refusal for each real-money switch missing."""
    needs = "a real-money start needs"
    refusals = []
    if version is None:
        refusals.append(f"{needs} a final NautilusTrader release, and NautilusTrader's version is unknown")
    elif not re.fullmatch(r"\d+(\.\d+)*", version):
        refusals.append(f"{needs} a final NautilusTrader release, and {version} is not a final release")
    elif tuple(int(p) for p in version.split(".")) < FIRST_REAL_MONEY_RELEASE:
        first = ".".join(map(str, FIRST_REAL_MONEY_RELEASE))
        refusals.append(f"{needs} NautilusTrader {first} or later, and {version} is older than {first}")
    if not switches.real_money:
        refusals.append(f"{needs} real_money = true in live.toml, and real_money = true is not set")
    if switches.arming != ARMED:
        refusals.append(f"{needs} an owner-only {ARMING_FILE} in the live directory, and the arming file is {switches.arming}")
    if not switches.initial_capital or switches.initial_capital <= 0:
        refusals.append(f"{needs} a positive initial_capital, and initial_capital is not set")
    if not switches.max_notional_per_order or switches.max_notional_per_order <= 0:
        refusals.append(f"{needs} the risk engine's notional cap, and max_notional_per_order is not set")
    if not switches.max_order_submit_rate:
        refusals.append(f"{needs} the risk engine's submit rate, and max_order_submit_rate is not set")
    if not switches.redis:
        refusals.append(f"{needs} its cache in Redis, and the node keeps no Redis cache")
    return refusals
