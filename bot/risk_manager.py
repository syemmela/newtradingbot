"""ATR-based position sizing, dual stop-loss (trailing + hard), and the
correlation filter.

Spec tension, resolved explicitly: sizing is defined so 1 ATR of adverse
move = 1% of equity, but strategies also specify trailing stops at 2x/3x
ATR. Those two rules conflict (a 2x/3x ATR trailing stop would be a 2%/3%
equity loss, not the "1% hard stop, no exceptions" the spec also demands).
Resolution used here: the hard 1%-equity stop is checked every tick
alongside the wider ATR trailing stop, and whichever is tighter fires
first. In practice the 1% hard stop will usually be the one that triggers
on adverse moves; the ATR trailing stop mostly matters for locking in
profit as a position runs (it only ratchets favorably, never loosens).
"""

from __future__ import annotations

import math
from typing import Callable

import config
from bot.portfolio import Portfolio
from bot.types import Position, Signal


def position_qty(equity: float, atr: float, kind: str) -> float:
    if atr is None or atr <= 0 or math.isnan(atr):
        return 0.0
    raw = (config.RISK_PER_TRADE_PCT * equity) / atr
    if kind == "equity":
        return float(math.floor(raw))
    return raw  # crypto: fractional qty allowed


def hard_stop_price(entry_price: float, side: str, equity: float, qty: float) -> float:
    if qty <= 0:
        return entry_price
    distance = (config.RISK_PER_TRADE_PCT * equity) / qty
    return entry_price - distance if side == "long" else entry_price + distance


def initial_trailing_stop(entry_price: float, atr: float, side: str, atr_mult: float) -> float:
    distance = atr * atr_mult
    return entry_price - distance if side == "long" else entry_price + distance


def update_trailing_stop(position: Position, current_price: float, atr: float, atr_mult: float) -> None:
    """Ratchet the trailing stop in the favorable direction only."""
    distance = atr * atr_mult
    if position.side == "long":
        position.trailing_stop_price = max(position.trailing_stop_price, current_price - distance)
    else:
        position.trailing_stop_price = min(position.trailing_stop_price, current_price + distance)


def stop_triggered(position: Position, current_price: float) -> bool:
    if position.side == "long":
        effective_stop = max(position.trailing_stop_price, position.hard_stop_price)
        return current_price <= effective_stop
    effective_stop = min(position.trailing_stop_price, position.hard_stop_price)
    return current_price >= effective_stop


# Correlation filter: a flat list of predicates, not a rule-engine class.
# Extending it later means appending one lambda.
CORRELATION_RULES: list[Callable[[Portfolio, Signal], bool]] = [
    lambda pf, sig: (
        sig.symbol == "BTC/USD"
        and sig.action == "long"
        and pf.is_long("SPY")
        and pf.is_long("QQQ")
    ),
]


def is_blocked(portfolio: Portfolio, signal: Signal) -> bool:
    return any(rule(portfolio, signal) for rule in CORRELATION_RULES)


def circuit_breaker_tripped(portfolio: Portfolio) -> bool:
    return portfolio.drawdown() >= config.MAX_DRAWDOWN_PCT


def open_risk_dollars(portfolio: Portfolio) -> float:
    """Sum of (entry-to-hard-stop distance * qty) across all open
    positions -- the total dollar amount at risk if every stop were hit
    simultaneously. By construction each position's own risk is exactly
    RISK_PER_TRADE_PCT of the equity at entry (that's what hard_stop_price
    is built to guarantee), but this sums the actual stored stop distances
    rather than assuming that invariant, so it stays correct even if
    sizing logic changes later."""
    return sum(abs(pos.entry_price - pos.hard_stop_price) * pos.qty for pos in portfolio.positions.values())


def would_exceed_portfolio_risk_cap(portfolio: Portfolio, new_position_risk_dollars: float) -> bool:
    """Would adding a new position with this much dollar risk push total
    open risk across the whole portfolio past config.MAX_TOTAL_OPEN_RISK_PCT
    of equity? Distinct from the circuit breaker, which reacts to REALIZED
    drawdown after the fact -- this caps prospective risk before it's ever
    realized, so several strategies opening positions at once can't all
    stack their 1%-per-trade risk into a much larger simultaneous exposure
    than intended (e.g. 5 positions at 1% each = 5% open risk with no cap)."""
    if portfolio.equity <= 0:
        return True
    projected = open_risk_dollars(portfolio) + new_position_risk_dollars
    return (projected / portfolio.equity) > config.MAX_TOTAL_OPEN_RISK_PCT
