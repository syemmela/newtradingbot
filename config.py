"""Central configuration: env vars, instrument list, strategy parameters."""

import os

from dotenv import load_dotenv

load_dotenv()

API_KEY_ID = os.environ.get("APCA_API_KEY_ID", "")
API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY", "")
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
        # Per-symbol ADX floor for entries (see MEAN_REVERSION_ADX_MAX below
        # for why this is per-symbol, not shared): a 12-month backtest sweep
        # showed GLD's crossovers stay good down to a low bar (15), while a
        # higher one starts cutting its winners.
        "params": {"trailing_atr_mult": 3.0, "trend_adx_min": 15},
    },
    "USO": {
        "kind": "equity",
        "strategy": "trend_following",
        # USO whipsawed badly (8 straight losing trades, Sept-Nov 2025) at
        # ADX as high as ~21 — needs a much higher floor than GLD to avoid
        # trading its crossovers during chop. Note: at 25, the 12-month
        # backtest only left 1 trade — directionally right, not yet proven.
        "params": {"trailing_atr_mult": 3.0, "trend_adx_min": 25},
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

# Bars fetched per tick. trend_following's 400 1H bars resample down to
# ~100 4H bars — comfortable headroom over EMA30's warmup.
LOOKBACK_BARS = {
    "mean_reversion": 60,
    "momentum_breakout": 60,
    "trend_following": 400,
}

RISK_PER_TRADE_PCT = 0.01  # 1% of equity per ATR of adverse move, and the hard stop distance
MAX_TOTAL_OPEN_RISK_PCT = 0.03  # cap on SUM of all open positions' risk-to-stop, across every strategy at once
ATR_PERIOD = 14

MAX_DRAWDOWN_PCT = 0.10  # circuit breaker: halt new entries past this drawdown from peak equity

# Regime filter (ADX): mean reversion only enters when the market ISN'T
# trending (ranging favors reversion), momentum breakout only enters when
# it IS trending (a real trend favors breakout continuation over chop).
# Exits are never gated by this — only new entries.
ADX_PERIOD = 14
MEAN_REVERSION_ADX_MAX = 20
MOMENTUM_BREAKOUT_ADX_MIN = 25

# Backtest fill assumptions: Alpaca is commission-free, but every fill still
# has real costs beyond the broker's own fee.
# - Slippage: price impact/movement between the decision and the fill.
# - Spread: the bid-ask spread crossed on every market order (half paid on
#   entry, half on exit) -- a typical rough figure for liquid ETFs/large
#   caps, not instrument-specific.
# - Fills execute against the NEXT bar's open, not the decision bar's own
#   close (see _simulate in bot/backtest.py) -- the standard fix for
#   lookahead bias in an event-driven backtest.
BACKTEST_SLIPPAGE_PCT = 0.0005
BACKTEST_SPREAD_PCT = 0.0002
BACKTEST_COMMISSION_PER_TRADE = 0.0
BACKTEST_MONTHS = 6
