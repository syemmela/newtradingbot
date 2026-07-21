"""Central configuration: env vars, instrument list, strategy parameters."""

import os

from dotenv import load_dotenv

load_dotenv()

API_KEY_ID = os.environ.get("APCA_API_KEY_ID", "")
API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY", "")
API_BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
DATA_FEED = os.environ.get("APCA_DATA_FEED", "iex")
PAPER_TRADING = os.environ.get("PAPER_TRADING", "true").lower() == "true"

TRADES_CSV = "trades.csv"
DAILY_PNL_CSV = "daily_pnl.csv"
LOG_DIR = "logs"

# Per-instrument metadata: which strategy runs it, its asset kind, and the
# strategy-specific parameters it needs. Instrument kind controls order
# rounding (equities round to whole shares; crypto stays fractional and can
# never be shorted on Alpaca).
INSTRUMENTS = {
    "SPY": {
        "kind": "equity",
        "strategy": "mean_reversion",
        "params": {"z_entry": 1.5},
    },
    "QQQ": {
        "kind": "equity",
        "strategy": "mean_reversion",
        "params": {"z_entry": 1.8},
    },
    "BTC/USD": {
        "kind": "crypto",
        "strategy": "momentum_breakout",
        "params": {"trailing_atr_mult": 2.0, "shortable": False},
    },
    "GLD": {
        "kind": "equity",
        "strategy": "trend_following",
        "params": {"trailing_atr_mult": 3.0},
    },
    "USO": {
        "kind": "equity",
        "strategy": "trend_following",
        "params": {"trailing_atr_mult": 3.0},
    },
}

STRATEGY_SYMBOLS = {
    "mean_reversion": [s for s, m in INSTRUMENTS.items() if m["strategy"] == "mean_reversion"],
    "momentum_breakout": [s for s, m in INSTRUMENTS.items() if m["strategy"] == "momentum_breakout"],
    "trend_following": [s for s, m in INSTRUMENTS.items() if m["strategy"] == "trend_following"],
}

# Source bar timeframe fetched from Alpaca per strategy, and the polling
# cadence each StrategyRunner sleeps between ticks. trend_following fetches
# 1-hour bars and resamples to 4H locally (Alpaca has no native 4H bars).
STRATEGY_TIMEFRAMES = {
    "mean_reversion": {"source": "15Min", "poll_seconds": 15 * 60, "needs_market_hours": True},
    "momentum_breakout": {"source": "1Hour", "poll_seconds": 60 * 60, "needs_market_hours": False},
    "trend_following": {"source": "1Hour", "poll_seconds": 60 * 60, "needs_market_hours": True},
}

# Bars fetched per tick. 250 gives EMA200 enough warm-up headroom after
# resampling 1H bars 4:1 for trend_following (~1000 1H bars needed).
LOOKBACK_BARS = {
    "mean_reversion": 60,
    "momentum_breakout": 60,
    "trend_following": 1000,
}

RISK_PER_TRADE_PCT = 0.01  # 1% of equity per ATR of adverse move, and the hard stop distance
ATR_PERIOD = 14

MAX_DRAWDOWN_PCT = 0.10  # circuit breaker: halt new entries past this drawdown from peak equity
