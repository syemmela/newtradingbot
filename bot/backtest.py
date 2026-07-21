"""Backtest engine. Replays the same strategy.evaluate()/compute_indicators()
functions used live against historical bars — no duplicated strategy logic.
Fills are simulated at the signal bar's close price (a modeling choice, not
an attempt to model slippage/partial fills).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from bot import risk_manager
from bot.engine import STRATEGY_MODULES
from bot.portfolio import Portfolio
from bot.types import Position


@dataclass
class BacktestResult:
    symbol: str
    trades: int
    win_rate: float
    profit_factor: float
    sharpe: float
    max_dd: float
    total_return_pct: float
    equity_curve: list[float]


def _empty_result(symbol: str, initial_equity: float) -> BacktestResult:
    return BacktestResult(symbol, 0, 0.0, 0.0, 0.0, 0.0, 0.0, [initial_equity])


async def run_backtest(
    broker, strategy_name: str, symbol: str, months: int = 6, initial_equity: float = 100_000.0
) -> BacktestResult:
    module = STRATEGY_MODULES[strategy_name]
    cfg = config.STRATEGY_TIMEFRAMES[strategy_name]

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * months)
    raw = await asyncio.to_thread(
        broker.get_historical_bars, symbol, cfg["source"], start.isoformat(), end.isoformat()
    )
    if raw is None or raw.empty:
        return _empty_result(symbol, initial_equity)

    bars = module.prepare_bars(raw)
    bars = module.compute_indicators(bars, symbol)
    if bars.empty:
        return _empty_result(symbol, initial_equity)

    sim = Portfolio(broker=None, starting_equity=initial_equity)
    kind = config.INSTRUMENTS[symbol]["kind"]
    atr_mult = config.INSTRUMENTS[symbol]["params"].get("trailing_atr_mult", 2.0)
    completed_trades = []

    for i in range(1, len(bars)):
        window = bars.iloc[: i + 1]
        latest = window.iloc[-1]
        price = float(latest["close"])
        atr_val = float(latest["atr"]) if not pd.isna(latest["atr"]) else None

        position = sim.get_position(symbol)
        if position is not None and atr_val is not None:
            risk_manager.update_trailing_stop(position, price, atr_val, atr_mult)
            if risk_manager.stop_triggered(position, price):
                record = await sim.close_position(symbol, price)
                if record:
                    completed_trades.append(record)
                position = None

        signal = module.evaluate(window, symbol, position)
        if signal is not None:
            if signal.action == "exit":
                record = await sim.close_position(symbol, price)
                if record:
                    completed_trades.append(record)
            else:
                if position is not None and position.side != signal.action:
                    record = await sim.close_position(symbol, price)
                    if record:
                        completed_trades.append(record)
                if atr_val:
                    qty = risk_manager.position_qty(sim.equity, atr_val, kind)
                    if qty > 0:
                        hard_stop = risk_manager.hard_stop_price(price, signal.action, sim.equity, qty)
                        trailing_stop = risk_manager.initial_trailing_stop(price, atr_val, signal.action, atr_mult)
                        new_position = Position(
                            symbol=symbol,
                            strategy=strategy_name,
                            side=signal.action,
                            qty=qty,
                            entry_price=price,
                            entry_time=signal.timestamp,
                            atr_at_entry=atr_val,
                            trailing_stop_price=trailing_stop,
                            hard_stop_price=hard_stop,
                        )
                        await sim.open_position(new_position)

        sim.mark_to_market({symbol: price})

    if sim.get_position(symbol) is not None:
        record = await sim.close_position(symbol, float(bars.iloc[-1]["close"]))
        if record:
            completed_trades.append(record)

    trades_n = len(completed_trades)
    wins = [t for t in completed_trades if t.pnl > 0]
    win_rate = (len(wins) / trades_n * 100) if trades_n else 0.0
    gross_profit = sum(t.pnl for t in completed_trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in completed_trades if t.pnl < 0)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    equity_curve = sim.equity_curve
    curve = np.array(equity_curve, dtype=float)
    returns = np.diff(curve) / curve[:-1] if len(curve) > 1 else np.array([])
    sharpe = float(returns.mean() / returns.std() * math.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0.0
    running_peak = np.maximum.accumulate(curve) if len(curve) else curve
    drawdowns = (curve - running_peak) / running_peak if len(curve) else curve
    max_dd = float(drawdowns.min() * 100) if len(drawdowns) else 0.0
    total_return_pct = (sim.equity - initial_equity) / initial_equity * 100

    return BacktestResult(
        symbol=symbol,
        trades=trades_n,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe=sharpe,
        max_dd=max_dd,
        total_return_pct=total_return_pct,
        equity_curve=equity_curve,
    )


async def run_all_backtests(broker, months: int = 6) -> dict[str, BacktestResult]:
    results: dict[str, BacktestResult] = {}
    for strategy_name, symbols in config.STRATEGY_SYMBOLS.items():
        for symbol in symbols:
            results[symbol] = await run_backtest(broker, strategy_name, symbol, months=months)
    return results


def combined_summary(results: dict[str, BacktestResult]) -> dict:
    """Average per-symbol stats into a single 'combined portfolio' row.

    This is an approximation (mean return/Sharpe, worst-case drawdown), not
    a true correlated portfolio-level backtest — a real combined equity
    curve would need position sizing to run across all 5 symbols
    simultaneously against one shared account, which the per-symbol
    backtest above deliberately doesn't attempt.
    """
    if not results:
        return {"total_return_pct": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    return {
        "total_return_pct": sum(r.total_return_pct for r in results.values()) / len(results),
        "sharpe": sum(r.sharpe for r in results.values()) / len(results),
        "max_dd": min(r.max_dd for r in results.values()),
    }
