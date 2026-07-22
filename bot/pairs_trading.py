"""Two-symbol joint strategies: pairs trading and relative-strength
rotation between SPY and QQQ. Deliberately separate from bot/backtest.py
and bot/strategies/*.py -- every strategy elsewhere in this bot decides
based on ONE symbol's own bars (evaluate(df, symbol, position)); these
two decide based on the RELATIONSHIP between two symbols' bars jointly,
which doesn't fit that interface or the shared _simulate() event loop.

Pairs trading: model the spread between SPY and QQQ via a rolling hedge
ratio (OLS beta of one on the other), trade convergence when the spread
deviates from its own recent norm -- long the laggard, short the
leader, exit when it reverts. This is a genuinely different bet than
either symbol's own direction: it can profit even if both fall or both
rise, as long as their RELATIONSHIP reverts.

Relative-strength rotation: simpler -- hold whichever of SPY/QQQ has the
stronger recent momentum (single long position, one symbol at a time),
move to cash if both are weak. No spread math, no short leg.

Both use their own lightweight simulator below rather than
bot.backtest._simulate, since neither maps onto per-symbol single-signal
events. Fill costs (slippage+spread) and next-bar-open execution match
the conventions in bot/backtest.py for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

import config
from bot import indicators

SLIPPAGE_SPREAD_PCT = config.BACKTEST_SLIPPAGE_PCT + config.BACKTEST_SPREAD_PCT / 2


@dataclass
class PairTradeRecord:
    timestamp: datetime
    kind: str  # "long_spread" | "short_spread" | "hold_a" | "hold_b"
    pnl: float


def prepare_pair_bars(bars_a: pd.DataFrame, bars_b: pd.DataFrame, hedge_lookback: int = 60) -> pd.DataFrame:
    """Align both symbols on common timestamps and compute the rolling
    hedge ratio, spread, and spread z-score (for pairs trading), plus
    each symbol's own momentum (for rotation)."""
    merged = pd.DataFrame({"close_a": bars_a["close"], "close_b": bars_b["close"]}).dropna()
    roll_cov = merged["close_a"].rolling(hedge_lookback).cov(merged["close_b"])
    roll_var = merged["close_b"].rolling(hedge_lookback).var()
    merged["hedge_ratio"] = roll_cov / roll_var
    merged["spread"] = merged["close_a"] - merged["hedge_ratio"] * merged["close_b"]
    spread_mean = merged["spread"].rolling(hedge_lookback).mean()
    spread_std = merged["spread"].rolling(hedge_lookback).std()
    merged["spread_zscore"] = (merged["spread"] - spread_mean) / spread_std
    merged["spread_change_std"] = merged["spread"].diff().rolling(hedge_lookback).std()
    merged["momentum_a"] = indicators.momentum(merged["close_a"], hedge_lookback)
    merged["momentum_b"] = indicators.momentum(merged["close_b"], hedge_lookback)
    # Precomputed here (not inline in simulate_rotation's loop) so sizing
    # never silently degrades to qty=0: recomputing .rolling(20).std() on
    # an already-dropna'd subset inside the loop is NaN right at the start
    # of that subset (not enough warmup within the filtered rows), which
    # looked like "a position was taken" but was actually sized at zero.
    merged["vol_a"] = merged["close_a"].diff().rolling(hedge_lookback).std()
    merged["vol_b"] = merged["close_b"].diff().rolling(hedge_lookback).std()
    return merged


def _fill_mult(action: str) -> float:
    return (1 + SLIPPAGE_SPREAD_PCT) if action == "buy" else (1 - SLIPPAGE_SPREAD_PCT)


def simulate_pairs_trade(
    joint: pd.DataFrame, entry_z: float = 2.0, exit_z: float = 0.5, initial_equity: float = 100_000.0
) -> tuple[list[PairTradeRecord], list[float]]:
    """Long the spread (long A, short hedge_ratio*B) when the spread is
    unusually low (z <= -entry_z), short the spread when unusually high
    (z >= entry_z), exit when it reverts to within exit_z of its mean.
    Position sized so a spread_change_std move against the trade equals
    ~1% of equity, matching the rest of the bot's risk convention."""
    trades: list[PairTradeRecord] = []
    equity = initial_equity
    equity_curve = [equity]
    position = None  # "long_spread" | "short_spread"
    entry_spread = 0.0
    qty_a = 0.0
    hedge_at_entry = 0.0

    # execute at the NEXT row's close (approximating the next-bar-open
    # convention used elsewhere in this bot, since this simulator works
    # off aligned closes rather than full OHLC bars)
    rows = joint.dropna(subset=["spread_zscore", "spread_change_std"])
    for i in range(1, len(rows) - 1):
        z = rows["spread_zscore"].iloc[i]
        vol = rows["spread_change_std"].iloc[i]
        next_price_a = rows["close_a"].iloc[i + 1]
        next_price_b = rows["close_b"].iloc[i + 1]
        next_hedge = rows["hedge_ratio"].iloc[i + 1]

        if position is not None:
            if abs(z) <= exit_z:
                # unwind: opposite-direction fill from whichever side opened each leg
                a_action = "sell" if position == "long_spread" else "buy"
                b_action = "buy" if position == "long_spread" else "sell"
                exit_price_a = next_price_a * _fill_mult(a_action)
                exit_price_b = next_price_b * _fill_mult(b_action)
                exit_spread = exit_price_a - hedge_at_entry * exit_price_b
                sign = 1 if position == "long_spread" else -1
                pnl = sign * (exit_spread - entry_spread) * qty_a
                equity += pnl
                trades.append(PairTradeRecord(rows.index[i + 1], position, pnl))
                position = None
            equity_curve.append(equity)
            continue

        if pd.isna(vol) or vol <= 0:
            equity_curve.append(equity)
            continue
        qty_a_candidate = (config.RISK_PER_TRADE_PCT * equity) / vol
        if z <= -entry_z or z >= entry_z:
            position = "long_spread" if z <= -entry_z else "short_spread"
            qty_a = qty_a_candidate
            hedge_at_entry = next_hedge
            a_action = "buy" if position == "long_spread" else "sell"
            b_action = "sell" if position == "long_spread" else "buy"
            entry_price_a = next_price_a * _fill_mult(a_action)
            entry_price_b = next_price_b * _fill_mult(b_action)
            entry_spread = entry_price_a - hedge_at_entry * entry_price_b
        equity_curve.append(equity)

    if position is not None:
        # force-close whatever's still open at the end of the data (same
        # convention as bot/backtest.py's _simulate) -- otherwise a spread
        # position still open when the window ends never gets its P&L
        # realized, silently understating the result.
        a_action = "sell" if position == "long_spread" else "buy"
        b_action = "buy" if position == "long_spread" else "sell"
        exit_price_a = rows["close_a"].iloc[-1] * _fill_mult(a_action)
        exit_price_b = rows["close_b"].iloc[-1] * _fill_mult(b_action)
        exit_spread = exit_price_a - hedge_at_entry * exit_price_b
        sign = 1 if position == "long_spread" else -1
        pnl = sign * (exit_spread - entry_spread) * qty_a
        equity += pnl
        trades.append(PairTradeRecord(rows.index[-1], position, pnl))
        equity_curve.append(equity)

    return trades, equity_curve


def simulate_rotation(joint: pd.DataFrame, min_momentum: float = 0.0, initial_equity: float = 100_000.0) -> tuple[list[PairTradeRecord], list[float]]:
    """Hold whichever of A/B has stronger momentum (long only, one at a
    time), move to cash if both are below min_momentum. Re-evaluates
    every bar; switches or exits whenever the ranking/threshold changes."""
    trades: list[PairTradeRecord] = []
    equity = initial_equity
    equity_curve = [equity]
    held = None  # "a" | "b" | None
    entry_price = 0.0
    qty = 0.0

    rows = joint.dropna(subset=["momentum_a", "momentum_b", "vol_a", "vol_b"])
    for i in range(len(rows) - 1):
        mom_a = rows["momentum_a"].iloc[i]
        mom_b = rows["momentum_b"].iloc[i]
        next_price_a = rows["close_a"].iloc[i + 1]
        next_price_b = rows["close_b"].iloc[i + 1]

        target = None
        if mom_a >= min_momentum or mom_b >= min_momentum:
            target = "a" if mom_a >= mom_b else "b"

        if held != target:
            if held is not None:
                exit_price = next_price_a if held == "a" else next_price_b
                pnl = (exit_price * _fill_mult("sell") - entry_price) * qty
                equity += pnl
                trades.append(PairTradeRecord(rows.index[i + 1], f"hold_{held}", pnl))
            if target is not None:
                entry_price = (next_price_a if target == "a" else next_price_b) * _fill_mult("buy")
                vol_proxy = rows["vol_a"].iloc[i] if target == "a" else rows["vol_b"].iloc[i]
                qty = (config.RISK_PER_TRADE_PCT * equity) / vol_proxy if vol_proxy and vol_proxy > 0 else 0.0
            held = target
        equity_curve.append(equity)

    if held is not None:
        # force-close whatever's still open at the end of the data, same
        # convention as bot/backtest.py's _simulate -- otherwise a position
        # that's still open when the window ends never gets its P&L
        # realized into equity/trades, silently understating the result.
        last_price = rows["close_a"].iloc[-1] if held == "a" else rows["close_b"].iloc[-1]
        pnl = (last_price * _fill_mult("sell") - entry_price) * qty
        equity += pnl
        trades.append(PairTradeRecord(rows.index[-1], f"hold_{held}", pnl))
        equity_curve.append(equity)

    return trades, equity_curve


def _stats(trades: list[PairTradeRecord], equity_curve: list[float], initial_equity: float) -> dict:
    n = len(trades)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    win_rate = (len(wins) / n * 100) if n else 0.0
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    curve = np.array(equity_curve, dtype=float)
    returns = np.diff(curve) / curve[:-1] if len(curve) > 1 else np.array([])
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0.0
    peak = np.maximum.accumulate(curve) if len(curve) else curve
    dd = (curve - peak) / peak if len(curve) else curve
    max_dd = float(dd.min() * 100) if len(dd) else 0.0
    total_return_pct = (curve[-1] - initial_equity) / initial_equity * 100 if len(curve) else 0.0
    return {"trades": n, "win_rate": win_rate, "profit_factor": pf, "sharpe": sharpe, "max_dd": max_dd, "total_return_pct": total_return_pct}


def print_pair_report(name: str, trades: list[PairTradeRecord], equity_curve: list[float], initial_equity: float = 100_000.0) -> dict:
    s = _stats(trades, equity_curve, initial_equity)
    pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    print(f"{name}: trades={s['trades']} win%={s['win_rate']:.1f} PF={pf} sharpe={s['sharpe']:.2f} max_dd={s['max_dd']:.1f}% return={s['total_return_pct']:+.1f}%")
    return s
