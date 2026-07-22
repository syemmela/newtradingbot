"""Time-series momentum, 1-hour bars: trade an asset's OWN historical
directional momentum (not a breakout level, not a moving-average
crossover) -- go long when its trailing N-bar return is strongly
positive, short when strongly negative, exit when momentum fades back
toward zero. Deliberately simple and doesn't depend on any specific
price level, unlike the Donchian-channel-based momentum_breakout.py.

"Volatility scaling" from the spec is already handled -- every strategy
in this bot sizes positions via risk_manager.position_qty(), which is
ATR-based (qty = 1% equity / ATR): a quiet period gets a bigger position,
a volatile one gets a smaller one, automatically. No extra code needed
here for that part of the design.

BTC/USD can't be shorted on Alpaca -- short signals only ever emitted if
config.INSTRUMENTS[symbol]["params"]["shortable"] is True.
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
    momentum_period = params.get("momentum_period", 100)

    df = df.copy()
    df["momentum"] = indicators.momentum(df["close"], momentum_period)
    df["atr"] = indicators.atr(df, config.ATR_PERIOD)
    return df


def evaluate(df: pd.DataFrame, symbol: str, position: Position | None) -> Signal | None:
    params = config.INSTRUMENTS[symbol]["params"]
    momentum_period = params.get("momentum_period", 100)
    entry_threshold = params.get("momentum_entry_threshold", 0.05)
    exit_threshold = params.get("momentum_exit_threshold", 0.0)

    if len(df) < momentum_period + 1:
        return None
    latest = df.iloc[-1]
    mom = latest.get("momentum")
    if pd.isna(mom):
        return None
    price = latest["close"]
    ts = df.index[-1]

    if position is not None:
        if position.side == "long" and mom <= exit_threshold:
            return Signal(symbol, "timeseries_momentum", "exit", price, ts, f"momentum faded, mom={mom:.3f}")
        if position.side == "short" and mom >= -exit_threshold:
            return Signal(symbol, "timeseries_momentum", "exit", price, ts, f"momentum faded, mom={mom:.3f}")
        return None

    shortable = config.INSTRUMENTS[symbol]["params"].get("shortable", True)
    if mom >= entry_threshold:
        return Signal(symbol, "timeseries_momentum", "long", price, ts, f"strong positive momentum, mom={mom:.3f}")
    if shortable and mom <= -entry_threshold:
        return Signal(symbol, "timeseries_momentum", "short", price, ts, f"strong negative momentum, mom={mom:.3f}")
    return None
