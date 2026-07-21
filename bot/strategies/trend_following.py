"""Trend following: 4-hour bars, EMA50/EMA200 crossover. Alpaca has no
native 4-hour bar timeframe, so this strategy fetches 1-hour bars and
resamples them locally in prepare_bars().
"""

from __future__ import annotations

import pandas as pd

import config
from bot import indicators
from bot.types import Position, Signal

FAST_PERIOD = 50
SLOW_PERIOD = 200
RESAMPLE_RULE = "4h"


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    return indicators.resample_ohlcv(df, RESAMPLE_RULE)


def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = indicators.ema(df["close"], FAST_PERIOD)
    df["ema_slow"] = indicators.ema(df["close"], SLOW_PERIOD)
    df["atr"] = indicators.atr(df, config.ATR_PERIOD)
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
        if position is None:
            return Signal(symbol, "trend_following", "short", price, ts, "death cross")
    if cross_up:
        if position is not None and position.side == "short":
            return Signal(symbol, "trend_following", "exit", price, ts, "golden cross")
        if position is None:
            return Signal(symbol, "trend_following", "long", price, ts, "golden cross")
    return None
