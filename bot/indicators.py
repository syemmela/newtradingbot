"""Pure pandas/numpy indicator functions. No TA-Lib — every indicator here
is a one-line rolling/ewm op, not worth a compiled dependency.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def zscore(close: pd.Series, period: int) -> pd.Series:
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - mean) / std


def donchian(df: pd.DataFrame, period: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hi": df["high"].rolling(period).max(),
            "lo": df["low"].rolling(period).min(),
            "vol_avg": df["volume"].rolling(period).mean(),
        },
        index=df.index,
    )


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend-strength regime filter.

    Conventionally: ADX < 20 means the market is ranging/choppy (favorable
    for mean reversion), ADX > 25 means a real trend is underway
    (favorable for momentum breakout continuation). Wilder's smoothing is
    approximated here with ewm(alpha=1/period), the common practical
    substitute.
    """
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift()
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    smoothed_tr = true_range.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / smoothed_tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / smoothed_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def volatility_ratio(atr_series: pd.Series, lookback: int = 50) -> pd.Series:
    """Current ATR relative to its own trailing rolling average. >1 means
    volatility is expanding relative to this instrument's recent norm, <1
    means it's contracting. Distinct from ADX, which measures directional
    trend STRENGTH, not the magnitude of price movement -- a market can be
    ranging (low ADX) yet unusually choppy (high vol_ratio), which is a
    different kind of danger for mean reversion than a real trend is."""
    return atr_series / atr_series.rolling(lookback).mean()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-bar-granularity OHLCV into a coarser timeframe (e.g. '4h').

    Used for trend_following: Alpaca has no native 4-hour bar timeframe, so
    we fetch 1-hour bars and build 4-hour bars locally.
    """
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
