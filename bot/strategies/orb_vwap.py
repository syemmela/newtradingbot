"""Opening range breakout + VWAP direction filter, 15-min bars. Serves two
of the user's specified combinations with one parameterized module, since
they share identical mechanics and only differ in the confirmation signal:
- SPY ("Combination 1"): confirmed by volatility expansion (ATR ratio).
- QQQ ("Combination 4"): confirmed by volume expansion.
Per-symbol config: config.INSTRUMENTS[symbol]["params"] needs
or_bars (bars in the opening range), confirmation_type ("volatility" or
"volume"), and the matching threshold (vol_ratio_min or volume_mult).

Entry: once the day's opening range has formed (skip bars still inside
it), price breaks above the range high with price also above VWAP (long)
or below the range low with price below VWAP (short), confirmed by
whichever signal this symbol uses.

Exit: ATR trailing/hard stop (shared risk_manager machinery), OR forced
flat at the first bar of a new session if still open overnight -- this is
an intraday strategy by construction (the opening range itself only
means something within a single session), so it shouldn't carry
positions across a session boundary the way a swing strategy would.
"""

from __future__ import annotations

import pandas as pd

import config
from bot import indicators
from bot.types import Position, Signal


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    return df  # already at the strategy's native 15-min granularity


def compute_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    params = config.INSTRUMENTS[symbol]["params"]
    or_bars = params["or_bars"]

    df = df.copy()
    df["vwap"] = indicators.vwap(df)
    ranges = indicators.opening_range(df, or_bars)
    df["or_high"] = ranges["or_high"]
    df["or_low"] = ranges["or_low"]
    df["atr"] = indicators.atr(df, config.ATR_PERIOD)
    df["atr_ratio"] = indicators.volatility_ratio(df["atr"], config.VOLATILITY_LOOKBACK)
    df["vol_avg"] = df["volume"].rolling(config.VOLATILITY_LOOKBACK).mean()
    df["_day"] = df.index.normalize()
    df["_bar_of_day"] = df.groupby("_day").cumcount()
    return df


def evaluate(df: pd.DataFrame, symbol: str, position: Position | None) -> Signal | None:
    params = config.INSTRUMENTS[symbol]["params"]
    or_bars = params["or_bars"]

    if len(df) < config.VOLATILITY_LOOKBACK + config.ATR_PERIOD + 1:
        return None
    latest = df.iloc[-1]
    price = latest["close"]
    ts = df.index[-1]

    if position is not None:
        entry_day = pd.Timestamp(position.entry_time).normalize()
        if latest["_day"] != entry_day:
            return Signal(symbol, "orb_vwap", "exit", price, ts, "new session, flatten overnight position")
        return None

    if latest["_bar_of_day"] < or_bars:
        return None  # still inside the opening range itself -- nothing to break out of yet
    if pd.isna(latest["or_high"]) or pd.isna(latest["vwap"]):
        return None

    confirmation_type = params["confirmation_type"]
    if confirmation_type == "volatility":
        atr_ratio = latest.get("atr_ratio")
        confirmed = pd.notna(atr_ratio) and atr_ratio >= params["vol_ratio_min"]
        note = f"atr_ratio={atr_ratio:.2f}" if pd.notna(atr_ratio) else "atr_ratio=nan"
    else:
        vol_avg = latest.get("vol_avg")
        vol_ratio = latest["volume"] / vol_avg if vol_avg else 0
        confirmed = vol_avg and vol_ratio >= params["volume_mult"]
        note = f"vol={vol_ratio:.2f}x"

    if not confirmed:
        return None

    if price > latest["or_high"] and price > latest["vwap"]:
        return Signal(symbol, "orb_vwap", "long", price, ts, f"ORB breakout above {latest['or_high']:.2f}, {note}")
    if price < latest["or_low"] and price < latest["vwap"]:
        return Signal(symbol, "orb_vwap", "short", price, ts, f"ORB breakdown below {latest['or_low']:.2f}, {note}")
    return None
