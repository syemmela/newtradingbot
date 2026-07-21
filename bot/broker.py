"""Thin synchronous wrapper around alpaca-py.

All methods are blocking (alpaca-py's REST clients are requests-based, not
asyncio-native) — callers in the engine invoke these via asyncio.to_thread(),
not directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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

_MINUTES_PER_BAR = {"15Min": 15, "1Hour": 60, "1Day": 1440}

# Regular equity session is ~390 minutes/day, 5 of 7 calendar days. Padded
# generously (1.4x) so holiday clusters don't force a retry in steady state.
_EQUITY_SESSION_MINUTES_PER_DAY = 390
_EQUITY_CALENDAR_PADDING = (7 / 5) * 1.4


def is_crypto(symbol: str) -> bool:
    return "/" in symbol


def _initial_lookback_minutes(timeframe: str, limit: int, crypto: bool) -> float:
    total_bar_minutes = _MINUTES_PER_BAR[timeframe] * limit
    if crypto:
        return total_bar_minutes * 1.5  # trades 24/7, just pad for gaps
    trading_days_needed = total_bar_minutes / _EQUITY_SESSION_MINUTES_PER_DAY
    calendar_days_needed = trading_days_needed * _EQUITY_CALENDAR_PADDING
    return calendar_days_needed * 1440


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
        """Fetch the most recent `limit` bars.

        alpaca-py's `limit` alone (no `start`) does NOT mean "last N bars
        ending now" — without a `start` it silently caps to a narrow
        default window, and passing `start` + `limit` together returns the
        EARLIEST `limit` bars from `start` forward, not the most recent
        ones. So instead: fetch everything from a generously-sized `start`
        to now with no `limit`, then take `.tail(limit)` ourselves. One
        retry with a 3x wider window covers holiday clusters etc.
        """
        tf = _TIMEFRAME_MAP[timeframe]
        crypto = is_crypto(symbol)
        lookback_minutes = _initial_lookback_minutes(timeframe, limit, crypto)
        df = pd.DataFrame()
        for attempt in range(2):
            start = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
            if crypto:
                request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start.isoformat())
                bar_set = self._crypto_data.get_crypto_bars(request)
            else:
                request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start.isoformat(), feed=config.DATA_FEED)
                bar_set = self._stock_data.get_stock_bars(request)
            df = _bars_df(bar_set, symbol)
            if len(df) >= limit:
                break
            lookback_minutes *= 3
        return df.tail(limit)

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
