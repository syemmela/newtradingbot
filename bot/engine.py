"""Trading engine: one StrategyRunner per strategy, each an independent
asyncio coroutine polling at its own cadence (15min / 1hour / 1hour-then-
resampled-to-4h). Crypto never waits on market hours; equities do — this
is what lets BTC/USD keep ticking through an equity market close without
stalling the other runners, all on a single event loop.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time

import config
from bot import risk_manager
from bot.strategies import mean_reversion, momentum_breakout, trend_following
from bot.types import Position

logger = logging.getLogger(__name__)

STRATEGY_MODULES = {
    "mean_reversion": mean_reversion,
    "momentum_breakout": momentum_breakout,
    "trend_following": trend_following,
}


class StrategyRunner:
    def __init__(self, broker, portfolio, strategy_name: str, symbols: list[str]):
        self.broker = broker
        self.portfolio = portfolio
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.module = STRATEGY_MODULES[strategy_name]

        cfg = config.STRATEGY_TIMEFRAMES[strategy_name]
        self.source_timeframe = cfg["source"]
        self.poll_seconds = cfg["poll_seconds"]
        self.needs_market_hours = cfg["needs_market_hours"]
        self.lookback = config.LOOKBACK_BARS[strategy_name]

        self.paused = False
        self.stopped = False
        self.latest_signals: dict[str, object] = {s: None for s in symbols}
        self.latest_bars: dict[str, object] = {}
        self._market_open_cache: tuple[float, bool] | None = None

    async def market_open(self) -> bool:
        now = time.monotonic()
        if self._market_open_cache is not None and now - self._market_open_cache[0] < 30:
            return self._market_open_cache[1]
        clock = await asyncio.to_thread(self.broker.get_clock)
        is_open = bool(clock.is_open)
        self._market_open_cache = (now, is_open)
        return is_open

    async def run(self) -> None:
        while not self.stopped:
            if self.paused:
                await asyncio.sleep(5)
                continue
            if self.needs_market_hours:
                try:
                    open_now = await self.market_open()
                except Exception:
                    logger.exception("%s: clock check failed, retrying", self.strategy_name)
                    await asyncio.sleep(30)
                    continue
                if not open_now:
                    await asyncio.sleep(60)
                    continue
            for symbol in self.symbols:
                try:
                    await self.tick_symbol(symbol)
                except Exception:
                    logger.exception("%s/%s: tick failed", self.strategy_name, symbol)
            await asyncio.sleep(self.poll_seconds)

    async def tick_symbol(self, symbol: str) -> None:
        raw = await asyncio.to_thread(self.broker.get_bars, symbol, self.source_timeframe, self.lookback)
        if raw is None or raw.empty:
            return
        bars = self.module.prepare_bars(raw)
        bars = self.module.compute_indicators(bars, symbol)
        if bars.empty:
            return
        self.latest_bars[symbol] = bars

        position = self.portfolio.get_position(symbol)
        latest_price = float(bars.iloc[-1]["close"])
        latest_atr = float(bars.iloc[-1]["atr"])

        if position is not None:
            atr_mult = config.INSTRUMENTS[symbol]["params"].get("trailing_atr_mult")
            if atr_mult is not None and not math.isnan(latest_atr):
                risk_manager.update_trailing_stop(position, latest_price, latest_atr, atr_mult)
            if risk_manager.stop_triggered(position, latest_price):
                logger.info("%s: stop triggered @ %.2f", symbol, latest_price)
                await self.close(symbol, latest_price)
                self.latest_signals[symbol] = None
                return

        signal = self.module.evaluate(bars, symbol, position)
        self.latest_signals[symbol] = signal
        if signal is not None:
            await self.handle_signal(signal, latest_atr)

    async def handle_signal(self, signal, latest_atr: float) -> None:
        if signal.action == "exit":
            logger.info("%s: exit signal (%s)", signal.symbol, signal.reason)
            await self.close(signal.symbol, signal.price)
            return

        if risk_manager.circuit_breaker_tripped(self.portfolio):
            logger.warning("%s: circuit breaker tripped, blocking new entry", signal.symbol)
            return

        if risk_manager.is_blocked(self.portfolio, signal):
            logger.info("%s: blocked by correlation filter", signal.symbol)
            return

        existing = self.portfolio.get_position(signal.symbol)
        if existing is not None and existing.side != signal.action:
            await self.close(signal.symbol, signal.price)

        kind = config.INSTRUMENTS[signal.symbol]["kind"]
        qty = risk_manager.position_qty(self.portfolio.equity, latest_atr, kind)
        if qty <= 0:
            return

        atr_mult = config.INSTRUMENTS[signal.symbol]["params"].get("trailing_atr_mult", 2.0)
        hard_stop = risk_manager.hard_stop_price(signal.price, signal.action, self.portfolio.equity, qty)
        trailing_stop = risk_manager.initial_trailing_stop(signal.price, latest_atr, signal.action, atr_mult)

        position = Position(
            symbol=signal.symbol,
            strategy=self.strategy_name,
            side=signal.action,
            qty=qty,
            entry_price=signal.price,
            entry_time=signal.timestamp,
            atr_at_entry=latest_atr,
            trailing_stop_price=trailing_stop,
            hard_stop_price=hard_stop,
        )
        order_side = "buy" if signal.action == "long" else "sell"
        logger.info("%s: %s signal (%s) qty=%s", signal.symbol, signal.action, signal.reason, qty)
        await asyncio.to_thread(self.broker.submit_order, signal.symbol, order_side, qty)
        await self.portfolio.open_position(position)

    async def close(self, symbol: str, price: float) -> None:
        position = self.portfolio.get_position(symbol)
        if position is None:
            return
        order_side = "sell" if position.side == "long" else "buy"
        await asyncio.to_thread(self.broker.submit_order, symbol, order_side, position.qty)
        record = await self.portfolio.close_position(symbol, price)
        if record is not None:
            logger.info("%s: closed %s qty=%s pnl=%.2f", symbol, position.side, position.qty, record.pnl)


class Engine:
    def __init__(self, broker, portfolio):
        self.broker = broker
        self.portfolio = portfolio
        self.runners = [
            StrategyRunner(broker, portfolio, name, symbols)
            for name, symbols in config.STRATEGY_SYMBOLS.items()
        ]

    async def run(self) -> None:
        await asyncio.gather(*(runner.run() for runner in self.runners))

    def pause(self) -> None:
        for runner in self.runners:
            runner.paused = True

    def resume(self) -> None:
        for runner in self.runners:
            runner.paused = False

    async def kill_switch(self) -> None:
        self.pause()
        logger.warning("KILL SWITCH: flattening all positions")
        for runner in self.runners:
            for symbol in list(runner.symbols):
                position = self.portfolio.get_position(symbol)
                if position is None:
                    continue
                bars = runner.latest_bars.get(symbol)
                price = float(bars.iloc[-1]["close"]) if bars is not None else position.entry_price
                await runner.close(symbol, price)

    def all_symbols(self) -> list[str]:
        return list(config.INSTRUMENTS.keys())
