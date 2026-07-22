"""Bollinger squeeze + volatility breakout, 1-hour bars: wait for Bollinger
bandwidth to compress relative to its own recent norm (a "squeeze" --
unusually quiet), then trade a breakout of the bands in whichever
direction it happens, on the theory that a squeeze resolves into an
expansion move. Completely different trigger mechanism from
momentum_breakout.py's fixed 20-period Donchian channel.

BTC/USD can't be shorted on Alpaca -- short signals only ever emitted if
config.INSTRUMENTS[symbol]["params"]["shortable"] is True (matches the
existing convention in momentum_breakout.py/volatility_filtered_trend.py).
"""

from __future__ import annotations

import pandas as pd

import config
from bot import indicators
from bot.types import Position, Signal


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    return df  # native 1-hour bars, no resampling needed


def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    params = config.INSTRUMENTS[symbol]["params"]
    bb_period = params.get("bb_period", 20)
    bb_std = params.get("bb_std", 2.0)

    df = df.copy()
    bb = indicators.bollinger_bands(df["close"], period=bb_period, num_std=bb_std)
    df["bb_upper"] = bb["upper"]
    df["bb_lower"] = bb["lower"]
    df["bb_bandwidth"] = bb["bandwidth"]
    df["bandwidth_ratio"] = indicators.bandwidth_ratio(df["bb_bandwidth"], config.VOLATILITY_LOOKBACK)
    df["atr"] = indicators.atr(df, config.ATR_PERIOD)
    return df


def evaluate(df: pd.DataFrame, symbol: str, position: Position | None) -> Signal | None:
    params = config.INSTRUMENTS[symbol]["params"]
    squeeze_max_ratio = params.get("squeeze_max_ratio", 0.7)

    if len(df) < config.VOLATILITY_LOOKBACK + params.get("bb_period", 20) + 1:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if pd.isna(latest["bandwidth_ratio"]) or pd.isna(prev["bandwidth_ratio"]):
        return None

    price = latest["close"]
    ts = df.index[-1]

    if position is not None:
        return None  # exits handled by the shared ATR trailing/hard stop only

    was_squeezed = prev["bandwidth_ratio"] <= squeeze_max_ratio
    if not was_squeezed:
        return None  # no recent compression -- nothing to break out of

    shortable = config.INSTRUMENTS[symbol]["params"].get("shortable", True)
    if price > latest["bb_upper"]:
        return Signal(symbol, "bollinger_squeeze_breakout", "long", price, ts, f"squeeze breakout up, bw_ratio={prev['bandwidth_ratio']:.2f}")
    if shortable and price < latest["bb_lower"]:
        return Signal(symbol, "bollinger_squeeze_breakout", "short", price, ts, f"squeeze breakout down, bw_ratio={prev['bandwidth_ratio']:.2f}")
    return None
