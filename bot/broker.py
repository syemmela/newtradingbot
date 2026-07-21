"""Thin synchronous wrapper around alpaca-py.

All methods are blocking (alpaca-py's REST clients are requests-based, not
asyncio-native) — callers in the engine invoke these via asyncio.to_thread(),
not directly.
"""

from __future__ import annotations

import pandas as pd
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

import config

_TIMEFRAME_MAP = {
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
}


def is_crypto(symbol: str) -> bool:
    return "/" in symbol


def _bars_df(bar_set, symbol: str) -> pd.DataFrame:
    """alpaca-py's BarSet.df is always MultiIndex(symbol, timestamp), even
    for a single-symbol request — slice down to a plain timestamp-indexed
    frame so every other module can treat get_bars() uniformly."""
    df = bar_set.df
    if df.empty or symbol not in df.index.get_level_values("symbol"):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df.xs(symbol, level="symbol")
    return df[["open", "high", "low", "close", "volume"]]


class Broker:
    def __init__(self) -> None:
        self._trading = TradingClient(config.API_KEY_ID, config.API_SECRET_KEY, paper=config.PAPER_TRADING)
        self._stock_data = StockHistoricalDataClient(config.API_KEY_ID, config.API_SECRET_KEY)
        self._crypto_data = CryptoHistoricalDataClient(config.API_KEY_ID, config.API_SECRET_KEY)

    def get_account(self):
        return self._trading.get_account()

    def get_clock(self):
        return self._trading.get_clock()

    def get_bars(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        tf = _TIMEFRAME_MAP[timeframe]
        if is_crypto(symbol):
            request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
            bar_set = self._crypto_data.get_crypto_bars(request)
        else:
            request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit, feed=config.DATA_FEED)
            bar_set = self._stock_data.get_stock_bars(request)
        return _bars_df(bar_set, symbol)

    def get_historical_bars(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """Fetch a date-ranged run of bars for backtesting (as opposed to
        get_bars()'s live 'last N bars' usage)."""
        tf = _TIMEFRAME_MAP[timeframe]
        if is_crypto(symbol):
            request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end)
            bar_set = self._crypto_data.get_crypto_bars(request)
        else:
            request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end, feed=config.DATA_FEED)
            bar_set = self._stock_data.get_stock_bars(request)
        return _bars_df(bar_set, symbol)

    def submit_order(self, symbol: str, side: str, qty: float) -> None:
        time_in_force = TimeInForce.GTC if is_crypto(symbol) else TimeInForce.DAY
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=time_in_force,
        )
        self._trading.submit_order(request)

    def get_position_qty(self, symbol: str) -> float:
        try:
            position = self._trading.get_open_position(symbol)
            return float(position.qty)
        except Exception:
            return 0.0
