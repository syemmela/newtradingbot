"""VWAP reversion, 15-min bars: fade deviations from the session VWAP,
expecting a return toward it -- structurally the same shape as
mean_reversion.py (deviation threshold entry, revert-to-target exit,
ADX regime gate on entries only), but referenced to VWAP instead of a
rolling SMA/std z-score, and reset each session rather than a continuous
rolling window.

Deviation is ATR-normalized ((close - vwap) / atr) rather than a
percentage, so the entry threshold means the same thing regardless of
the instrument's price level or typical volatility.

Gated by an ADX regime filter, same rationale as mean_reversion.py: only
trade reversion when the market ISN'T trending (a real trend will keep
pulling price away from VWAP, not back to it). Exits are never gated.
"""

from __future__ import annotations

import pandas as pd

import config
from bot import indicators
from bot.types import Position, Signal


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    return df  # already at the strategy's native 15-min granularity


def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["vwap"] = indicators.vwap(df)
    df["atr"] = indicators.atr(df, config.ATR_PERIOD)
    df["adx"] = indicators.adx(df, config.ADX_PERIOD)
    df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["atr"]
    return df


def evaluate(df: pd.DataFrame, symbol: str, position: Position | None) -> Signal | None:
    params = config.INSTRUMENTS[symbol]["params"]
    entry_threshold = params["vwap_deviation_entry"]
    adx_max = params.get("vwap_adx_max", 20)

    if len(df) < config.ATR_PERIOD + 1:
        return None
    latest = df.iloc[-1]
    deviation = latest.get("vwap_deviation")
    if pd.isna(deviation):
        return None
    price = latest["close"]
    ts = df.index[-1]

    if position is None:
        adx = latest.get("adx")
        if pd.isna(adx) or adx >= adx_max:
            return None  # trending too hard to safely fade
        if deviation <= -entry_threshold:
            return Signal(symbol, "vwap_reversion", "long", price, ts, f"deviation={deviation:.2f} <= -{entry_threshold}, adx={adx:.1f}")
        if deviation >= entry_threshold:
            return Signal(symbol, "vwap_reversion", "short", price, ts, f"deviation={deviation:.2f} >= {entry_threshold}, adx={adx:.1f}")
        return None

    if position.side == "long" and deviation >= 0:
        return Signal(symbol, "vwap_reversion", "exit", price, ts, "reverted to VWAP")
    if position.side == "short" and deviation <= 0:
        return Signal(symbol, "vwap_reversion", "exit", price, ts, "reverted to VWAP")
    return None
