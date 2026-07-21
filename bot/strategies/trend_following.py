"""Trend following: 4-hour bars, EMA20/EMA50 crossover. Alpaca has no
native 4-hour bar timeframe, so this strategy fetches 1-hour bars and
resamples them locally in prepare_bars().

Originally EMA50/EMA200 per the initial spec, but a 6-month backtest
showed that pairing barely ever crosses on 4H bars (200 bars alone is
~33 days of warmup) — GLD got 0 trades and USO got 1 over the whole
window. Shortened to 20/50, which produced 5-6 trades with a positive
Sharpe on both symbols in that backtest.

Extending to 12 months exposed a problem 20/50 alone didn't catch: USO
whipsawed through 8 straight losing crossovers (Sept-Nov 2025) at ADX as
low as ~12-21 — a choppy stretch with no real trend, which a fast
crossover pair trades right through. GLD didn't have this problem over
the same window. So entries are now gated per-symbol by ADX
(config.INSTRUMENTS[symbol]["params"]["trend_adx_min"]) rather than a
shared threshold — GLD and USO need very different floors. Exits are
never gated; getting out of a position doesn't depend on regime.
"""

from __future__ import annotations

import pandas as pd

import config
from bot import indicators
from bot.types import Position, Signal

FAST_PERIOD = 20
SLOW_PERIOD = 50
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
