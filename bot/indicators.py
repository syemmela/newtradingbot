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
