"""Textual TUI. Runs on the same asyncio event loop as the trading engine —
Textual drives the loop, the engine runs as a background worker task. There
is no thread boundary between the UI and the engine; both are coroutines,
and the shared Portfolio object is the only thing passed between them.
"""

from __future__ import annotations

from textual.app import App

from bot.engine import Engine, reconcile_positions
from bot.portfolio import Portfolio
from bot.tui.screens import (
    BacktestScreen,
    DashboardScreen,
    LogsScreen,
    PortfolioScreen,
    ReportsScreen,
    RiskScreen,
    StrategiesScreen,
)


class TradingApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    .panel {
        border: round $primary;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    .title {
        text-style: bold;
        color: $accent;
    }
    """

    BINDINGS = [
        ("1", "switch_dashboard", "Dashboard"),
        ("2", "switch_portfolio", "Portfolio"),
        ("3", "switch_strategies", "Strategies"),
        ("4", "switch_risk", "Risk"),
        ("5", "switch_backtest", "Backtest"),
        ("6", "switch_reports", "Reports"),
        ("7", "switch_logs", "Logs"),
        ("p", "toggle_pause", "Pause/Resume"),
        ("k", "kill_switch", "Kill Switch"),
        ("r", "manual_refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
        "portfolio": PortfolioScreen,
        "strategies": StrategiesScreen,
        "risk": RiskScreen,
        "backtest": BacktestScreen,
        "reports": ReportsScreen,
        "logs": LogsScreen,
    }

    TITLE = "ALPACA QUANT BOT"

    def __init__(self, broker, portfolio: Portfolio, engine: Engine):
        super().__init__()
        self.broker = broker
        self.portfolio = portfolio
        self.engine = engine
        self.paused = False

    def on_mount(self) -> None:
        self.push_screen("dashboard")
        self.run_worker(self._start_engine(), name="engine", exclusive=True)
        self.set_interval(5.0, self._mark_to_market_tick)

    async def _start_engine(self) -> None:
        await reconcile_positions(self.broker, self.portfolio)
        await self.engine.run()

    def _mark_to_market_tick(self) -> None:
        prices: dict[str, float] = {}
        for runner in self.engine.runners:
            for symbol, bars in runner.latest_bars.items():
                if bars is not None and not bars.empty:
                    prices[symbol] = float(bars.iloc[-1]["close"])
        self.portfolio.mark_to_market(prices)

    def action_switch_dashboard(self) -> None:
        self.switch_screen("dashboard")

    def action_switch_portfolio(self) -> None:
        self.switch_screen("portfolio")

    def action_switch_strategies(self) -> None:
        self.switch_screen("strategies")

    def action_switch_risk(self) -> None:
        self.switch_screen("risk")

    def action_switch_backtest(self) -> None:
        self.switch_screen("backtest")

    def action_switch_reports(self) -> None:
        self.switch_screen("reports")

    def action_switch_logs(self) -> None:
        self.switch_screen("logs")

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self.engine.pause()
        else:
            self.engine.resume()

    async def action_kill_switch(self) -> None:
        self.paused = True
        await self.engine.kill_switch()

    def action_manual_refresh(self) -> None:
        if hasattr(self.screen, "refresh_content"):
            self.screen.refresh_content()
