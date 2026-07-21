"""Thin synchronous wrapper around alpaca-trade-api's REST client.

All methods are blocking (the SDK is requests-based, not asyncio-native) —
callers in the engine invoke these via asyncio.to_thread(), not directly.
"""

from __future__ import annotations

import pandas as pd
from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

import config

_TIMEFRAME_MAP = {
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
}


def is_crypto(symbol: str) -> bool:
    return "/" in symbol


class Broker:
    def __init__(self) -> None:
        self._api = REST(
            key_id=config.API_KEY_ID,
            secret_key=config.API_SECRET_KEY,
            base_url=config.API_BASE_URL,
        )

    def get_account(self):
        return self._api.get_account()

    def get_clock(self):
        return self._api.get_clock()

    def get_bars(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV bars as a DataFrame indexed by timestamp.

        Equities and crypto use different endpoints on this SDK version
        (get_bars vs get_crypto_bars) — branch on symbol shape once here so
        every other module can call broker.get_bars() uniformly.
        """
        tf = _TIMEFRAME_MAP[timeframe]
        if is_crypto(symbol):
            bars = self._api.get_crypto_bars(symbol, tf, limit=limit)
        else:
            bars = self._api.get_bars(symbol, tf, limit=limit, feed=config.DATA_FEED)
        df = bars.df
        if df.empty:
            return df
        return df[["open", "high", "low", "close", "volume"]]

    def get_historical_bars(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """Fetch a date-ranged run of bars for backtesting (as opposed to
        get_bars()'s live 'last N bars' usage)."""
        tf = _TIMEFRAME_MAP[timeframe]
        if is_crypto(symbol):
            bars = self._api.get_crypto_bars(symbol, tf, start=start, end=end)
        else:
            bars = self._api.get_bars(symbol, tf, start=start, end=end, feed=config.DATA_FEED)
        df = bars.df
        if df.empty:
            return df
        return df[["open", "high", "low", "close", "volume"]]

    def submit_order(self, symbol: str, side: str, qty: float) -> None:
        self._api.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="day" if not is_crypto(symbol) else "gtc")

    def get_position_qty(self, symbol: str) -> float:
        try:
            pos = self._api.get_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0
