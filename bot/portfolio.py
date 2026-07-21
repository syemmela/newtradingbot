"""Single source of truth for account state. Read by strategies (via engine),
risk_manager, and every TUI screen; mutated only through the async methods
here, each guarded by one lock (cheap insurance against interleaved writes,
not a distributed-systems problem for a single-process bot).

broker=None puts this in backtest-sim mode: no CSV writes, no live account
sync, equity is simulated from cash + unrealized P&L instead.
"""

from __future__ import annotations

import asyncio
import csv
import os
from collections import deque
from datetime import date, datetime, timezone

import config
from bot.types import Position, TradeRecord


class Portfolio:
    def __init__(self, broker=None, starting_equity: float = 100_000.0):
        self.broker = broker
        self.equity = starting_equity
        self.cash = starting_equity
        self.peak_equity = starting_equity
        self.positions: dict[str, Position] = {}
        self.trade_log: deque[TradeRecord] = deque(maxlen=50)
        self.equity_curve: list[float] = [starting_equity]
        self._lock = asyncio.Lock()
        self._current_day = date.today()
        self._day_start_equity = starting_equity
        self._ensure_csv_headers()

    def _ensure_csv_headers(self) -> None:
        if self.broker is None:
            return
        if not os.path.exists(config.TRADES_CSV):
            with open(config.TRADES_CSV, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "symbol", "direction", "entry_price", "exit_price", "pnl", "qty", "strategy"]
                )
        if not os.path.exists(config.DAILY_PNL_CSV):
            with open(config.DAILY_PNL_CSV, "w", newline="") as f:
                csv.writer(f).writerow(["date", "pnl", "equity"])

    def sync_from_broker(self) -> None:
        if self.broker is None:
            return
        acct = self.broker.get_account()
        self.equity = float(acct.equity)
        self.cash = float(acct.cash)
        self.peak_equity = max(self.peak_equity, self.equity)

    @property
    def day_start_equity(self) -> float:
        return self._day_start_equity

    def is_long(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and pos.side == "long"

    def is_short(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and pos.side == "short"

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    async def open_position(self, position: Position) -> None:
        async with self._lock:
            self.positions[position.symbol] = position

    async def close_position(self, symbol: str, exit_price: float, timestamp: datetime | None = None) -> TradeRecord | None:
        async with self._lock:
            pos = self.positions.pop(symbol, None)
            if pos is None:
                return None
            sign = 1 if pos.side == "long" else -1
            pnl = (exit_price - pos.entry_price) * pos.qty * sign
            record = TradeRecord(
                timestamp=timestamp or datetime.now(timezone.utc),
                symbol=symbol,
                direction=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                qty=pos.qty,
                strategy=pos.strategy,
            )
            self.trade_log.append(record)
            self.cash += pnl
            self._append_trade_csv(record)
            return record

    def _append_trade_csv(self, record: TradeRecord) -> None:
        if self.broker is None:
            return
        with open(config.TRADES_CSV, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    record.timestamp.isoformat(),
                    record.symbol,
                    record.direction,
                    record.entry_price,
                    record.exit_price,
                    record.pnl,
                    record.qty,
                    record.strategy,
                ]
            )

    def unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            price = current_prices.get(sym, pos.entry_price)
            sign = 1 if pos.side == "long" else -1
            total += (price - pos.entry_price) * pos.qty * sign
        return total

    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def mark_to_market(self, current_prices: dict[str, float]) -> None:
        if self.broker is not None:
            self.sync_from_broker()
        else:
            self.equity = self.cash + self.unrealized_pnl(current_prices)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)
        self.roll_daily_pnl_if_new_day()

    def roll_daily_pnl_if_new_day(self) -> None:
        today = date.today()
        if today != self._current_day:
            pnl = self.equity - self._day_start_equity
            if self.broker is not None:
                with open(config.DAILY_PNL_CSV, "a", newline="") as f:
                    csv.writer(f).writerow([self._current_day.isoformat(), pnl, self.equity])
            self._current_day = today
            self._day_start_equity = self.equity
