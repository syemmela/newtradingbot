"""Trend pullback: 1-hour bars. Trade in the direction of a fast/slow EMA
trend, but time entries off a pullback to the fast EMA rather than the
crossover moment itself -- this is a state-based, continuous-entry
strategy (many entries per trend), unlike trend_following.py's
crossover-only entries (one entry per direction change).

Serves two of the user's specified strategies with one parameterized
module, since they share identical mechanics and only differ in period
choice and whether an ADX gate is used:
- SPY ("Strategy 1"): EMA 50/200, no ADX gate (per spec).
- QQQ ("Strategy 2"): EMA 20/50, gated by ADX >= adx_min (per spec, exact
  threshold unspecified -- swept via the validation framework rather
  than guessed).
Per-symbol config: config.INSTRUMENTS[symbol]["params"] needs
fast_ema_period, slow_ema_period, pullback_lookback, and adx_min (use 0
to disable the gate, since ADX is always computed regardless).

Entry definition (resolving the spec's qualitative "wait for pullback,
enter when trend resumes" into something concrete and testable):
- Trend: fast EMA above/below slow EMA.
- Pullback: within the last `pullback_lookback` bars (not including the
  current one), price actually reached the fast EMA (low <= ema_fast for
  an uptrend, high >= ema_fast for a downtrend) -- a genuine touch, not
  just "got closer."
- Resumption: the CURRENT bar closes back on the trend side of the fast
  EMA (close > ema_fast for long, < for short) with a bar in the trend's
  direction (close > open for long, < for open for short).
Exit: trend reversal (fast EMA crosses the slow EMA the other way) --
matches trend_following.py's exit convention. The ATR trailing/hard stop
(shared risk_manager machinery, not strategy-specific code) handles
day-to-day profit protection and loss cutting.
"""

from __future__ import annotations

import pandas as pd

import config
from bot import indicators
from bot.types import Position, Signal


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    return df  # already at the strategy's native 1-hour granularity


def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    params = config.INSTRUMENTS[symbol]["params"]
    fast_period = params["fast_ema_period"]
    slow_period = params["slow_ema_period"]

    df = df.copy()
    df["ema_fast"] = indicators.ema(df["close"], fast_period)
    df["ema_slow"] = indicators.ema(df["close"], slow_period)
    df["atr"] = indicators.atr(df, config.ATR_PERIOD)
    df["adx"] = indicators.adx(df, config.ADX_PERIOD)
    df["touched_up"] = df["low"] <= df["ema_fast"]
    df["touched_down"] = df["high"] >= df["ema_fast"]
    return df


def evaluate(df: pd.DataFrame, symbol: str, position: Position | None) -> Signal | None:
    params = config.INSTRUMENTS[symbol]["params"]
    slow_period = params["slow_ema_period"]
    pullback_lookback = params["pullback_lookback"]
    adx_min = params.get("adx_min", 0)

    if len(df) < slow_period + pullback_lookback + 1:
        return None
    latest = df.iloc[-1]
    if pd.isna(latest["ema_slow"]) or pd.isna(latest["ema_fast"]):
        return None

    price = latest["close"]
    ts = df.index[-1]
    trend_up = latest["ema_fast"] > latest["ema_slow"]
    trend_down = latest["ema_fast"] < latest["ema_slow"]

    if position is not None:
        if position.side == "long" and trend_down:
            return Signal(symbol, "trend_pullback", "exit", price, ts, "trend reversed down")
        if position.side == "short" and trend_up:
            return Signal(symbol, "trend_pullback", "exit", price, ts, "trend reversed up")
        return None

    adx = latest.get("adx")
    if adx_min and (pd.isna(adx) or adx < adx_min):
        return None  # trend not strong enough to trust per this symbol's ADX gate

    recent = df.iloc[-(pullback_lookback + 1) : -1]  # excludes the current (resumption) bar
    if trend_up:
        pulled_back = bool(recent["touched_up"].any())
        resumed = price > latest["ema_fast"] and price > latest["open"]
        if pulled_back and resumed:
            adx_note = f", adx={adx:.1f}" if adx_min else ""
            return Signal(symbol, "trend_pullback", "long", price, ts, f"pullback resumed up{adx_note}")
    elif trend_down:
        pulled_back = bool(recent["touched_down"].any())
        resumed = price < latest["ema_fast"] and price < latest["open"]
        if pulled_back and resumed:
            adx_note = f", adx={adx:.1f}" if adx_min else ""
            return Signal(symbol, "trend_pullback", "short", price, ts, f"pullback resumed down{adx_note}")
    return None
