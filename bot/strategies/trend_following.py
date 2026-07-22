"""Trend following: 4-hour bars, EMA10/EMA30 crossover. Alpaca has no
native 4-hour bar timeframe, so this strategy fetches 1-hour bars and
resamples them locally in prepare_bars().

Period history, each step driven by backtesting a longer window than
the last and finding the previous choice didn't hold up:
- Original spec (EMA50/200): barely ever crosses on 4H bars (200 bars
  alone is ~33 days of warmup) — 0-1 trades over 6 months.
- Shortened to 20/50: looked good at 6-12mo (positive Sharpe both
  symbols), but extending to 12mo also exposed USO whipsawing through 8
  straight losing crossovers in a choppy stretch (ADX ~12-21) — added a
  per-symbol ADX floor (config.INSTRUMENTS[symbol]["params"]["trend_adx_min"])
  to gate entries on real trend strength.
- Testing 20/50 further out (24/60/66mo) showed it degrading badly for
  GLD (Sharpe -0.48 to -0.62) regardless of ADX floor value — the 6-12mo
  result was itself an overfit to a short window, not a real edge.
  Swept both ADX floor and EMA period against all three longer windows
  simultaneously (the bar for "robust": wins across all three, not just
  one) and found EMA10/30 consistently best or least-bad for GLD, and
  at least as good as 20/50 for USO too — so it's now the shared
  default rather than a per-symbol split.

Exits are never gated by ADX; getting out of a position doesn't depend
on regime, only entries do.
"""

from __future__ import annotations

import pandas as pd

import config
from bot import indicators
from bot.types import Position, Signal

FAST_PERIOD = 10
SLOW_PERIOD = 30
RESAMPLE_RULE = "4h"


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    return indicators.resample_ohlcv(df, RESAMPLE_RULE)


def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = indicators.ema(df["close"], FAST_PERIOD)
    df["ema_slow"] = indicators.ema(df["close"], SLOW_PERIOD)
    df["atr"] = indicators.atr(df, config.ATR_PERIOD)
    df["adx"] = indicators.adx(df, config.ADX_PERIOD)
    return df


def evaluate(df: pd.DataFrame, symbol: str, position: Position | None) -> Signal | None:
    if len(df) < SLOW_PERIOD + 1:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if pd.isna(latest["ema_slow"]) or pd.isna(prev["ema_slow"]):
        return None

    price = latest["close"]
    ts = df.index[-1]
    cross_up = prev["ema_fast"] <= prev["ema_slow"] and latest["ema_fast"] > latest["ema_slow"]
    cross_down = prev["ema_fast"] >= prev["ema_slow"] and latest["ema_fast"] < latest["ema_slow"]

    if cross_down:
        if position is not None and position.side == "long":
            return Signal(symbol, "trend_following", "exit", price, ts, "death cross")
    if cross_up:
        if position is not None and position.side == "short":
            return Signal(symbol, "trend_following", "exit", price, ts, "golden cross")
    if position is not None:
        return None

    adx = latest.get("adx")
    adx_min = config.INSTRUMENTS[symbol]["params"]["trend_adx_min"]
    if pd.isna(adx) or adx < adx_min:
        return None  # not enough trend strength to trust this crossover
    if cross_down:
        return Signal(symbol, "trend_following", "short", price, ts, f"death cross adx={adx:.1f}")
    if cross_up:
        return Signal(symbol, "trend_following", "long", price, ts, f"golden cross adx={adx:.1f}")
    return None
