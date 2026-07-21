"""Shared dataclasses used across strategies, engine, risk, portfolio, and TUI.

Kept separate from portfolio.py to avoid a circular import: strategies need
Signal, portfolio needs Position/TradeRecord, and risk_manager needs both.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Signal:
    symbol: str
    strategy: str
    action: str  # "long" | "short" | "exit"
    price: float
    timestamp: datetime
    reason: str


@dataclass
class Position:
    symbol: str
    strategy: str
    side: str  # "long" | "short"
    qty: float
    entry_price: float
    entry_time: datetime
    atr_at_entry: float
    trailing_stop_price: float
    hard_stop_price: float


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    qty: float
    strategy: str
