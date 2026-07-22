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


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, reset at the start of each calendar
    day (a session boundary) -- not a plain rolling average. Uses typical
    price (H+L+C)/3, the standard VWAP convention."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    day = df.index.normalize()
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()
    return cum_pv / cum_vol


def opening_range(df: pd.DataFrame, bars_per_session: int) -> pd.DataFrame:
    """High/low of the first `bars_per_session` bars of each calendar day,
    broadcast across every bar in that same day -- e.g. bars_per_session=2
    on 15-min bars gives the first 30 minutes of each session. Bars after
    the opening window still see that day's range; the opening window's
    own bars see a range still being formed (their own high/low so far)."""
    day = df.index.normalize()
    grouped_high = df.groupby(day)["high"]
    grouped_low = df.groupby(day)["low"]
    or_high = grouped_high.transform(lambda s: s.iloc[:bars_per_session].max())
    or_low = grouped_low.transform(lambda s: s.iloc[:bars_per_session].min())
    return pd.DataFrame({"or_high": or_high, "or_low": or_low}, index=df.index)


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "bandwidth": bandwidth}, index=close.index)


def bandwidth_ratio(bandwidth: pd.Series, lookback: int = 50) -> pd.Series:
    """Current Bollinger bandwidth relative to its own trailing rolling
    average -- <1 means a squeeze (compressed relative to recent norm),
    the setup a Bollinger-squeeze breakout strategy waits for."""
    return bandwidth / bandwidth.rolling(lookback).mean()


def momentum(close: pd.Series, period: int) -> pd.Series:
    """Simple N-period rate of change: how much price has moved over the
    last `period` bars, as a fraction of the price `period` bars ago."""
    return close / close.shift(period) - 1


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
