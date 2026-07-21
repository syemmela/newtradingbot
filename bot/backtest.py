"""Backtest engine. Replays the same strategy.evaluate()/compute_indicators()
functions used live against historical bars — no duplicated strategy logic.

Two modes share one event-driven simulator (_simulate):
- run_backtest(): one symbol in isolation. The correlation filter is a
  structural no-op here since it needs OTHER symbols to be long too.
- run_combined_backtest(): all 5 symbols in one shared Portfolio, events
  merged chronologically across their different timeframes (15Min for
  SPY/QQQ, 1Hour for BTC/USD, 4H-resampled for GLD/USO). This is what
  makes the correlation filter (blocking new BTC longs when SPY and QQQ
  are both already long) actually mean something — it's checked against
  real cross-symbol state at the moment each signal fires, not simulated
  independently and averaged after the fact.

Fills include 0.05% slippage (config.BACKTEST_SLIPPAGE_PCT) against the
signal's bar-close price, and $0 commission (Alpaca is commission-free,
but the hook exists via config.BACKTEST_COMMISSION_PER_TRADE).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from bot import risk_manager
from bot.portfolio import Portfolio
from bot.strategies import mean_reversion, momentum_breakout, trend_following
from bot.types import Position, TradeRecord

STRATEGY_MODULES = {
    "mean_reversion": mean_reversion,
    "momentum_breakout": momentum_breakout,
    "trend_following": trend_following,
}

STRATEGY_LABELS = {
    "mean_reversion": "Mean Reversion",
    "momentum_breakout": "Momentum Breakout",
    "trend_following": "Trend Following",
}


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe: float
    max_dd: float
    total_return_pct: float
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list = field(default_factory=list)


@dataclass
class CombinedBacktestResult:
    trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe: float
    max_dd: float
    total_return_pct: float
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    per_symbol_trade_counts: dict = field(default_factory=dict)


def _empty_result(symbol: str, strategy_name: str, initial_equity: float) -> BacktestResult:
    return BacktestResult(symbol, strategy_name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [initial_equity], [])


def _fill_price(price: float, action: str) -> float:
    """action is the side actually executed for this fill: 'buy' or 'sell'."""
    if action == "buy":
        return price * (1 + config.BACKTEST_SLIPPAGE_PCT)
    return price * (1 - config.BACKTEST_SLIPPAGE_PCT)


async def _fetch_prepared_bars(broker, strategy_name: str, symbol: str, months: int) -> pd.DataFrame:
    module = STRATEGY_MODULES[strategy_name]
    cfg = config.STRATEGY_TIMEFRAMES[strategy_name]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * months)
    raw = await asyncio.to_thread(
        broker.get_historical_bars, symbol, cfg["source"], start.isoformat(), end.isoformat()
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    bars = module.prepare_bars(raw)
    bars = module.compute_indicators(bars, symbol)
    return bars


async def _open_sim_position(sim: Portfolio, signal, symbol: str, strategy_name: str, atr_val: float, qty: float) -> None:
    action = "buy" if signal.action == "long" else "sell"
    fill_price = _fill_price(signal.price, action)
    atr_mult = config.INSTRUMENTS[symbol]["params"].get("trailing_atr_mult", 2.0)
    position = Position(
        symbol=symbol,
        strategy=strategy_name,
        side=signal.action,
        qty=qty,
        entry_price=fill_price,
        entry_time=signal.timestamp,
        atr_at_entry=atr_val,
        trailing_stop_price=risk_manager.initial_trailing_stop(fill_price, atr_val, signal.action, atr_mult),
        hard_stop_price=risk_manager.hard_stop_price(fill_price, signal.action, sim.equity, qty),
    )
    await sim.open_position(position)


async def _close_sim_position(sim: Portfolio, symbol: str, price: float, timestamp) -> TradeRecord | None:
    position = sim.get_position(symbol)
    if position is None:
        return None
    action = "sell" if position.side == "long" else "buy"
    fill_price = _fill_price(price, action)
    record = await sim.close_position(symbol, fill_price, timestamp=timestamp)
    if record is not None:
        record.pnl -= config.BACKTEST_COMMISSION_PER_TRADE * 2  # entry + exit; $0 for Alpaca, kept for realism
    return record


async def _simulate(symbol_bars: dict[str, pd.DataFrame], initial_equity: float) -> tuple[Portfolio, list[TradeRecord], list]:
    """Event-driven simulator shared by single-symbol and combined backtests.

    symbol_bars maps symbol -> prepared+indicator bars. Pass one entry for
    an isolated backtest (correlation filter is a no-op with only one
    symbol in play) or all 5 for the combined portfolio backtest (the
    filter becomes real, since it reads cross-symbol state off the same
    shared `sim` Portfolio every other symbol's events also mutate).
    """
    events = []
    for symbol, bars in symbol_bars.items():
        if bars.empty:
            continue
        strategy_name = config.INSTRUMENTS[symbol]["strategy"]
        for i in range(1, len(bars)):
            events.append((bars.index[i], symbol, strategy_name, i))
    events.sort(key=lambda e: e[0])

    sim = Portfolio(broker=None, starting_equity=initial_equity)
    last_price: dict[str, float] = {}
    trades: list[TradeRecord] = []
    timestamps: list = []

    for timestamp, symbol, strategy_name, idx in events:
        module = STRATEGY_MODULES[strategy_name]
        bars = symbol_bars[symbol]
        window = bars.iloc[: idx + 1]
        row = window.iloc[-1]
        price = float(row["close"])
        atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else None
        last_price[symbol] = price

        position = sim.get_position(symbol)
        if position is not None and atr_val is not None:
            atr_mult = config.INSTRUMENTS[symbol]["params"].get("trailing_atr_mult", 2.0)
            risk_manager.update_trailing_stop(position, price, atr_val, atr_mult)
            if risk_manager.stop_triggered(position, price):
                record = await _close_sim_position(sim, symbol, price, timestamp)
                if record:
                    trades.append(record)
                position = None

        signal = module.evaluate(window, symbol, position)
        if signal is not None:
            if signal.action == "exit":
                record = await _close_sim_position(sim, symbol, signal.price, timestamp)
                if record:
                    trades.append(record)
            else:
                if position is not None and position.side != signal.action:
                    record = await _close_sim_position(sim, symbol, signal.price, timestamp)
                    if record:
                        trades.append(record)
                if atr_val and not risk_manager.circuit_breaker_tripped(sim) and not risk_manager.is_blocked(sim, signal):
                    kind = config.INSTRUMENTS[symbol]["kind"]
                    qty = risk_manager.position_qty(sim.equity, atr_val, kind)
                    if qty > 0:
                        await _open_sim_position(sim, signal, symbol, strategy_name, atr_val, qty)

        sim.mark_to_market(last_price)
        timestamps.append(timestamp)

    final_timestamp = timestamps[-1] if timestamps else None
    for symbol in list(sim.positions.keys()):
        record = await _close_sim_position(sim, symbol, last_price.get(symbol, sim.positions[symbol].entry_price), final_timestamp)
        if record:
            trades.append(record)
    if timestamps:
        sim.mark_to_market(last_price)
        timestamps.append(timestamps[-1])

    return sim, trades, timestamps


def _compute_stats(trades: list[TradeRecord], equity_curve: list[float], initial_equity: float) -> dict:
    trades_n = len(trades)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    win_rate = (len(wins) / trades_n * 100) if trades_n else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    curve = np.array(equity_curve, dtype=float)
    returns = np.diff(curve) / curve[:-1] if len(curve) > 1 else np.array([])
    # Annualized with sqrt(252) regardless of each strategy's actual bar
    # frequency — a standard simplification for a quick retail backtest,
    # not a duration-weighted Sharpe.
    sharpe = float(returns.mean() / returns.std() * math.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0.0
    running_peak = np.maximum.accumulate(curve) if len(curve) else curve
    drawdowns = (curve - running_peak) / running_peak if len(curve) else curve
    max_dd = float(drawdowns.min() * 100) if len(drawdowns) else 0.0
    total_return_pct = (curve[-1] - initial_equity) / initial_equity * 100 if len(curve) else 0.0

    return {
        "trades": trades_n,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return_pct": total_return_pct,
    }


async def run_backtest(
    broker, strategy_name: str, symbol: str, months: int = config.BACKTEST_MONTHS, initial_equity: float = 100_000.0
) -> BacktestResult:
    bars = await _fetch_prepared_bars(broker, strategy_name, symbol, months)
    if bars.empty:
        return _empty_result(symbol, strategy_name, initial_equity)

    sim, trades, timestamps = await _simulate({symbol: bars}, initial_equity)
    stats = _compute_stats(trades, sim.equity_curve, initial_equity)
    return BacktestResult(symbol=symbol, strategy=strategy_name, equity_curve=sim.equity_curve, timestamps=timestamps, **stats)


async def run_all_backtests(broker, months: int = config.BACKTEST_MONTHS) -> dict[str, BacktestResult]:
    results: dict[str, BacktestResult] = {}
    for strategy_name, symbols in config.STRATEGY_SYMBOLS.items():
        for symbol in symbols:
            results[symbol] = await run_backtest(broker, strategy_name, symbol, months=months)
    return results


async def run_combined_backtest(
    broker, months: int = config.BACKTEST_MONTHS, initial_equity: float = 100_000.0
) -> CombinedBacktestResult:
    symbol_bars: dict[str, pd.DataFrame] = {}
    for strategy_name, symbols in config.STRATEGY_SYMBOLS.items():
        for symbol in symbols:
            symbol_bars[symbol] = await _fetch_prepared_bars(broker, strategy_name, symbol, months)

    sim, trades, timestamps = await _simulate(symbol_bars, initial_equity)
    stats = _compute_stats(trades, sim.equity_curve, initial_equity)

    per_symbol_trade_counts: dict[str, int] = {}
    for t in trades:
        per_symbol_trade_counts[t.symbol] = per_symbol_trade_counts.get(t.symbol, 0) + 1

    return CombinedBacktestResult(
        equity_curve=sim.equity_curve,
        timestamps=timestamps,
        per_symbol_trade_counts=per_symbol_trade_counts,
        **stats,
    )


def render_equity_curve_chart(
    combined: CombinedBacktestResult,
    per_symbol: dict[str, BacktestResult],
    path: str = "backtest_results.png",
    months: int = config.BACKTEST_MONTHS,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    if combined.timestamps:
        combined_curve = combined.equity_curve[-len(combined.timestamps):]
        ax.plot(combined.timestamps, combined_curve, label="Combined Portfolio", linewidth=2.5, color="black")
    for symbol, result in per_symbol.items():
        if result.timestamps:
            curve = result.equity_curve[-len(result.timestamps):]
            ax.plot(result.timestamps, curve, label=symbol, linewidth=1, alpha=0.6)

    ax.set_title(f"Backtest Equity Curves ({months} Months)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary_table(
    per_symbol: dict[str, BacktestResult], combined: CombinedBacktestResult, months: int = config.BACKTEST_MONTHS
) -> None:
    columns = (
        f"{'Symbol':10s} {'Strategy':18s} {'Trades':>7s} {'Win %':>7s} "
        f"{'Avg Win':>10s} {'Avg Loss':>10s} {'PF':>6s} {'Sharpe':>7s} {'Max DD':>8s} {'Return':>8s}"
    )
    print(columns)
    print("-" * len(columns))
    for symbol, r in per_symbol.items():
        pf_display = "inf" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
        print(
            f"{symbol:10s} {STRATEGY_LABELS[r.strategy]:18s} {r.trades:7d} {r.win_rate:6.1f}% "
            f"{r.avg_win:10.2f} {r.avg_loss:10.2f} {pf_display:>6s} {r.sharpe:7.2f} {r.max_dd:7.1f}% {r.total_return_pct:+7.1f}%"
        )
    print("-" * len(columns))
    pf_display = "inf" if combined.profit_factor == float("inf") else f"{combined.profit_factor:.2f}"
    print(
        f"{'COMBINED':10s} {'(correlation ON)':18s} {combined.trades:7d} {combined.win_rate:6.1f}% "
        f"{combined.avg_win:10.2f} {combined.avg_loss:10.2f} {pf_display:>6s} "
        f"{combined.sharpe:7.2f} {combined.max_dd:7.1f}% {combined.total_return_pct:+7.1f}%"
    )
    print()

    flagged = [(symbol, r.strategy, r.sharpe) for symbol, r in per_symbol.items() if r.sharpe < 0]
    if combined.sharpe < 0:
        flagged.append(("COMBINED", "portfolio", combined.sharpe))
    if flagged:
        print(f"FLAGGED (negative Sharpe over {months} months — consider adjusting parameters):")
        for symbol, strategy_name, sharpe in flagged:
            label = STRATEGY_LABELS.get(strategy_name, strategy_name)
            print(f"  - {symbol} ({label}): Sharpe {sharpe:.2f}")
    else:
        print(f"No strategies flagged — all Sharpe ratios positive over the {months}-month window.")


async def run_full_report(
    broker, months: int = config.BACKTEST_MONTHS, output_path: str = "backtest_results.png"
) -> tuple[dict[str, BacktestResult], CombinedBacktestResult]:
    per_symbol = await run_all_backtests(broker, months=months)
    combined = await run_combined_backtest(broker, months=months)
    render_equity_curve_chart(combined, per_symbol, path=output_path, months=months)
    print_summary_table(per_symbol, combined, months=months)
    return per_symbol, combined


if __name__ == "__main__":
    from bot.broker import Broker
    from bot.logs import setup_logging

    setup_logging()
    asyncio.run(run_full_report(Broker()))
