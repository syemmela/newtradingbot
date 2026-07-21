"""Mean reversion: 15-min bars, 20-period z-score. Entry beyond +/-z_entry
standard deviations, exit when price reverts back through the mean.
"""

from __future__ import annotations

import pandas as pd

import config
from bot import indicators
from bot.types import Position, Signal

PERIOD = 20


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    return df  # already at the strategy's native 15-min granularity


def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["zscore"] = indicators.zscore(df["close"], PERIOD)
    df["atr"] = indicators.atr(df, config.ATR_PERIOD)
    return df


def evaluate(df: pd.DataFrame, symbol: str, position: Position | None) -> Signal | None:
    if len(df) < PERIOD + 1:
        return None
    latest = df.iloc[-1]
    if pd.isna(latest["zscore"]):
        return None
    z = latest["zscore"]
    price = latest["close"]
    ts = df.index[-1]
    z_entry = config.INSTRUMENTS[symbol]["params"]["z_entry"]

    if position is None:
        if z <= -z_entry:
            return Signal(symbol, "mean_reversion", "long", price, ts, f"z={z:.2f} <= -{z_entry}")
        if z >= z_entry:
            return Signal(symbol, "mean_reversion", "short", price, ts, f"z={z:.2f} >= {z_entry}")
        return None

    if position.side == "long" and z >= 0:
        return Signal(symbol, "mean_reversion", "exit", price, ts, "reverted to mean")
    if position.side == "short" and z <= 0:
        return Signal(symbol, "mean_reversion", "exit", price, ts, "reverted to mean")
    return None
