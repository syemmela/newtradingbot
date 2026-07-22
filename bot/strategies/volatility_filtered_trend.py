"""Volatility-filtered trend following: 4-hour bars (resampled from 1-hour,
same mechanism as trend_following.py -- Alpaca has no native 4H bars).
EMA50/EMA200 for the long-term trend, ADX to confirm real trend strength.

Unlike trend_following.py (which only enters on the crossover EVENT),
this is state-based per the spec's literal wording ("go long when the 50
EMA is above the 200 EMA and ADX is above the minimum threshold"): it
enters whenever flat and the trend+strength condition holds, not only at
the moment of crossing. That means it can re-enter mid-trend after a
stop-out, not just once per crossover.

Exit: trend reversal (fast EMA crosses to the other side) -- getting out
doesn't depend on ADX, only getting in does ("avoid opening new
positions when ADX indicates a weak or sideways market" is explicitly
entry-only per the spec). The ATR trailing/hard stop (shared
risk_manager machinery) handles day-to-day stop management; per spec
this symbol uses a 2x ATR trailing stop
(config.INSTRUMENTS["BTC/USD"]["params"]["trailing_atr_mult"]).

BTC/USD can't be shorted on Alpaca -- short signals are only ever
emitted if config.INSTRUMENTS[symbol]["params"]["shortable"] is True.
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
    df["adx"] = indicators.adx(df, config.ADX_PERIOD)
    return df


def evaluate(df: pd.DataFrame, symbol: str, position: Position | None) -> Signal | None:
    if len(df) < SLOW_PERIOD + 1:
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
            return Signal(symbol, "volatility_filtered_trend", "exit", price, ts, "trend reversed down")
        if position.side == "short" and trend_up:
            return Signal(symbol, "volatility_filtered_trend", "exit", price, ts, "trend reversed up")
        return None

    adx = latest.get("adx")
    adx_min = config.INSTRUMENTS[symbol]["params"].get("adx_min", 0)
    if adx_min and (pd.isna(adx) or adx < adx_min):
        return None  # weak/sideways market -- avoid opening a new position

    shortable = config.INSTRUMENTS[symbol]["params"].get("shortable", True)
    if trend_up:
        return Signal(symbol, "volatility_filtered_trend", "long", price, ts, f"uptrend adx={adx:.1f}")
    if trend_down and shortable:
        return Signal(symbol, "volatility_filtered_trend", "short", price, ts, f"downtrend adx={adx:.1f}")
    return None
