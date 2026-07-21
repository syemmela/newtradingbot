"""Momentum breakout: 1-hour bars, 20-period Donchian channel + volume
confirmation. BTC/USD only, and Alpaca does not allow shorting crypto —
a breakdown below the 20-low only ever exits an existing long, never
opens a short (see config.INSTRUMENTS["BTC/USD"]["params"]["shortable"]).

Gated by an ADX regime filter: new entries only fire when ADX is above
config.MOMENTUM_BREAKOUT_ADX_MIN (a real trend is underway) — a 6-month
backtest showed this strategy losing consistently across every
period/volume/ATR combination tried, with a low win rate suggesting
most "breakouts" were false ones in a choppy, non-trending market. Exits
are never gated; getting out of a position doesn't depend on regime.
"""

from __future__ import annotations

import pandas as pd

import config
from bot import indicators
from bot.types import Position, Signal

PERIOD = 20
VOLUME_CONFIRM_MULT = 1.5


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    return df  # native 1-hour bars, no resampling needed


def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    channel = indicators.donchian(df, PERIOD)
    # shift(1): compare current bar's close against the channel formed by
    # the PRIOR 20 bars, not including the current bar itself.
    df["donchian_hi"] = channel["hi"].shift(1)
    df["donchian_lo"] = channel["lo"].shift(1)
    df["vol_avg"] = channel["vol_avg"].shift(1)
    df["atr"] = indicators.atr(df, config.ATR_PERIOD)
    df["adx"] = indicators.adx(df, config.ADX_PERIOD)
    return df


def evaluate(df: pd.DataFrame, symbol: str, position: Position | None) -> Signal | None:
    if len(df) < PERIOD + 2:
        return None
    latest = df.iloc[-1]
    if pd.isna(latest["donchian_hi"]) or pd.isna(latest["vol_avg"]) or latest["vol_avg"] <= 0:
        return None

    price = latest["close"]
    ts = df.index[-1]
    vol_ratio = latest["volume"] / latest["vol_avg"]
    shortable = config.INSTRUMENTS[symbol]["params"].get("shortable", True)
    volume_confirmed = vol_ratio >= VOLUME_CONFIRM_MULT

    if position is not None and position.side == "long":
        if price < latest["donchian_lo"] and volume_confirmed:
            return Signal(symbol, "momentum_breakout", "exit", price, ts, f"breakdown vol={vol_ratio:.2f}x")
        return None

    if position is not None and position.side == "short":
        if price > latest["donchian_hi"] and volume_confirmed:
            return Signal(symbol, "momentum_breakout", "exit", price, ts, f"breakout vol={vol_ratio:.2f}x")
        return None

    # flat: look for a fresh breakout, but only when a real trend is underway
    adx = latest.get("adx")
    if pd.isna(adx) or adx < config.MOMENTUM_BREAKOUT_ADX_MIN:
        return None
    if price > latest["donchian_hi"] and volume_confirmed:
        return Signal(symbol, "momentum_breakout", "long", price, ts, f"breakout vol={vol_ratio:.2f}x, adx={adx:.1f}")
    if shortable and price < latest["donchian_lo"] and volume_confirmed:
        return Signal(symbol, "momentum_breakout", "short", price, ts, f"breakdown vol={vol_ratio:.2f}x, adx={adx:.1f}")
    return None
