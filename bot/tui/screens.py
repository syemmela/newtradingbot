"""The 7 TUI screens. Each reads shared state off self.app (broker,
portfolio, engine) on a periodic timer — none of them mutate state directly
except via the App-level pause/kill actions (bot/tui/app.py).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Sparkline, Static

import config
from bot import backtest, logs, risk_manager
from bot.strategies import trend_following

STRATEGY_LABELS = {
    "mean_reversion": "Mean Reversion",
    "momentum_breakout": "Momentum Breakout",
    "trend_following": "Trend Following",
}

EASTERN = ZoneInfo("America/New_York")


def _latest_price(app, symbol: str) -> float | None:
    for runner in app.engine.runners:
        bars = runner.latest_bars.get(symbol)
        if bars is not None and not bars.empty:
            return float(bars.iloc[-1]["close"])
    return None


def _bot_status_line(app) -> str:
    state = "PAUSED" if app.paused else "RUNNING"
    now_et = datetime.now(EASTERN).strftime("%A, %b %d %Y %I:%M:%S %p ET")
    mode = "Paper Trading" if config.PAPER_TRADING else "LIVE TRADING"
    if app.market_open is None:
        market_line = "MARKET STATUS UNKNOWN"
    elif app.market_open:
        market_line = "MARKET OPENED 🟢"
    else:
        market_line = "MARKET CLOSED 🔴"
    return f"[b]ALPACA QUANT BOT[/b]   ● {state}   {mode}   {market_line}\n{now_et}"


class DashboardScreen(Screen):
    def compose(self):
        yield Header()
        yield Static(id="account_panel", classes="panel")
        yield DataTable(id="positions_table", classes="panel")
        yield Static(id="strategy_status", classes="panel")
        yield DataTable(id="recent_trades", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        positions = self.query_one("#positions_table", DataTable)
        positions.add_columns("Symbol", "Strategy", "Side", "Qty", "Entry", "Current", "Unrealized")
        trades = self.query_one("#recent_trades", DataTable)
        trades.add_columns("Time", "Symbol", "Direction", "Qty", "Strategy", "P&L")
        self.refresh_content()
        self.set_interval(2.0, self.refresh_content)

    def refresh_content(self) -> None:
        app = self.app
        pf = app.portfolio

        daily_pnl = pf.equity - pf.day_start_equity
        daily_pnl_pct = (daily_pnl / pf.day_start_equity * 100) if pf.day_start_equity else 0.0
        total_return_pct = (pf.equity - 100_000.0) / 100_000.0 * 100
        account = self.query_one("#account_panel", Static)
        account.update(
            _bot_status_line(app)
            + f"\n\nEquity ${pf.equity:,.2f}   Cash ${pf.cash:,.2f}   "
            f"Today's P&L {daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)   "
            f"Total Return {total_return_pct:+.2f}%   Drawdown {pf.drawdown()*100:-.2f}%"
        )

        positions = self.query_one("#positions_table", DataTable)
        positions.clear()
        for symbol in config.INSTRUMENTS:
            pos = pf.get_position(symbol)
            price = _latest_price(app, symbol)
            if pos is None:
                positions.add_row(symbol, STRATEGY_LABELS[config.INSTRUMENTS[symbol]["strategy"]], "-", "-", "-", "-", "-")
                continue
            current = price if price is not None else pos.entry_price
            sign = 1 if pos.side == "long" else -1
            unrealized = (current - pos.entry_price) * pos.qty * sign
            positions.add_row(
                symbol,
                STRATEGY_LABELS[pos.strategy],
                pos.side.upper(),
                f"{pos.qty:g}",
                f"{pos.entry_price:.2f}",
                f"{current:.2f}",
                f"{unrealized:+.2f}",
            )

        status_lines = []
        for strategy_name, symbols in config.STRATEGY_SYMBOLS.items():
            runner = next(r for r in app.engine.runners if r.strategy_name == strategy_name)
            parts = []
            for symbol in symbols:
                signal = runner.latest_signals.get(symbol)
                state = "ACTIVE" if pf.get_position(symbol) else "WAITING"
                sig_text = signal.action.upper() if signal else "HOLD"
                parts.append(f"{symbol} ● {state} ({sig_text})")
            status_lines.append(f"{STRATEGY_LABELS[strategy_name].upper()}: " + "  ".join(parts))
        self.query_one("#strategy_status", Static).update("\n".join(status_lines))

        trades_table = self.query_one("#recent_trades", DataTable)
        trades_table.clear()
        for record in list(pf.trade_log)[-10:][::-1]:
            trades_table.add_row(
                record.timestamp.strftime("%H:%M:%S"),
                record.symbol,
                record.direction.upper(),
                f"{record.qty:g}",
                STRATEGY_LABELS[record.strategy],
                f"{record.pnl:+.2f}",
            )


class PortfolioScreen(Screen):
    def compose(self):
        yield Header()
        yield DataTable(id="portfolio_table", classes="panel")
        yield Static(id="portfolio_summary", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#portfolio_table", DataTable)
        table.add_columns("Symbol", "Strategy", "Side", "Qty", "Entry", "Price", "P&L")
        self.refresh_content()
        self.set_interval(2.0, self.refresh_content)

    def refresh_content(self) -> None:
        app = self.app
        pf = app.portfolio
        table = self.query_one("#portfolio_table", DataTable)
        table.clear()
        unrealized_total = 0.0
        for symbol, pos in pf.positions.items():
            price = _latest_price(app, symbol) or pos.entry_price
            sign = 1 if pos.side == "long" else -1
            pnl = (price - pos.entry_price) * pos.qty * sign
            unrealized_total += pnl
            table.add_row(symbol, STRATEGY_LABELS[pos.strategy], pos.side.upper(), f"{pos.qty:g}", f"{pos.entry_price:.2f}", f"{price:.2f}", f"{pnl:+.2f}")

        realized_total = sum(t.pnl for t in pf.trade_log)
        self.query_one("#portfolio_summary", Static).update(
            f"Total Unrealized P&L: {unrealized_total:+,.2f}\n"
            f"Total Realized P&L (recent): {realized_total:+,.2f}\n"
            f"Net P&L: {unrealized_total + realized_total:+,.2f}"
        )


class StrategiesScreen(Screen):
    def compose(self):
        yield Header()
        yield VerticalScroll(Static(id="strategy_detail", classes="panel"))
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()
        self.set_interval(2.0, self.refresh_content)

    def refresh_content(self) -> None:
        app = self.app
        pf = app.portfolio
        sections = []
        for runner in app.engine.runners:
            lines = [f"[b]{STRATEGY_LABELS[runner.strategy_name].upper()}[/b]"]
            for symbol in runner.symbols:
                bars = runner.latest_bars.get(symbol)
                signal = runner.latest_signals.get(symbol)
                sig_text = signal.action.upper() if signal else "HOLD"
                if bars is None or bars.empty:
                    lines.append(f"  {symbol}: no data yet")
                    continue
                latest = bars.iloc[-1]
                if runner.strategy_name == "mean_reversion":
                    z = latest.get("zscore")
                    adx = latest.get("adx")
                    atr_ratio = latest.get("atr_ratio")
                    threshold = config.INSTRUMENTS[symbol]["params"]["z_entry"]
                    trending = pd.notna(adx) and adx >= config.MEAN_REVERSION_ADX_MAX
                    choppy = pd.notna(atr_ratio) and atr_ratio > config.MEAN_REVERSION_MAX_VOL_RATIO
                    regime = "TRENDING (blocked)" if trending else ("CHOPPY (blocked)" if choppy else "RANGING")
                    lines.append(
                        f"  {symbol}  Z-Score: {z:+.2f}   Threshold: +/-{threshold}   "
                        f"ADX: {adx:.1f}  ATR Ratio: {atr_ratio:.2f} ({regime})   Signal: {sig_text}"
                    )
                elif runner.strategy_name == "momentum_breakout":
                    hi = latest.get("donchian_hi")
                    vol_avg = latest.get("vol_avg")
                    vol_ratio = latest["volume"] / vol_avg if vol_avg else 0
                    adx = latest.get("adx")
                    atr_ratio = latest.get("atr_ratio")
                    not_trending = pd.isna(adx) or adx < config.MOMENTUM_BREAKOUT_ADX_MIN
                    not_expanding = pd.isna(atr_ratio) or atr_ratio < config.MOMENTUM_BREAKOUT_MIN_VOL_RATIO
                    regime = "TRENDING+EXPANDING" if not (not_trending or not_expanding) else "BLOCKED"
                    lines.append(
                        f"  {symbol}  20H High: {hi:.2f}  Price: {latest['close']:.2f}  "
                        f"Volume: {vol_ratio:.2f}x  ATR: {latest['atr']:.2f}  ADX: {adx:.1f}  "
                        f"ATR Ratio: {atr_ratio:.2f} ({regime})  Signal: {sig_text}"
                    )
                elif runner.strategy_name == "trend_following":
                    lines.append(
                        f"  {symbol}  EMA{trend_following.FAST_PERIOD}: {latest['ema_fast']:.2f}  "
                        f"EMA{trend_following.SLOW_PERIOD}: {latest['ema_slow']:.2f}  Signal: {sig_text}"
                    )
                pos = pf.get_position(symbol)
                if pos is not None:
                    lines.append(f"        Position: {pos.side.upper()} {pos.qty:g} @ {pos.entry_price:.2f}   Trailing Stop: {pos.trailing_stop_price:.2f}   Hard Stop: {pos.hard_stop_price:.2f}")
            sections.append("\n".join(lines))
        self.query_one("#strategy_detail", Static).update("\n\n".join(sections))


class RiskScreen(Screen):
    def compose(self):
        yield Header()
        yield Static(id="risk_panel", classes="panel")
        yield Static(id="position_risk_panel", classes="panel")
        yield Static(id="correlation_panel", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()
        self.set_interval(2.0, self.refresh_content)

    def refresh_content(self) -> None:
        app = self.app
        pf = app.portfolio
        tripped = risk_manager.circuit_breaker_tripped(pf)
        self.query_one("#risk_panel", Static).update(
            f"Account Equity      ${pf.equity:,.2f}\n"
            f"Peak Equity         ${pf.peak_equity:,.2f}\n"
            f"Current Drawdown    {pf.drawdown()*100:.2f}%\n"
            f"Max Drawdown Limit  {config.MAX_DRAWDOWN_PCT*100:.2f}%\n"
            f"Circuit Breaker     {'● TRIPPED' if tripped else '● ARMED'}"
        )

        lines = []
        for symbol, pos in pf.positions.items():
            price = _latest_price(app, symbol) or pos.entry_price
            risk_dollars = abs(price - pos.hard_stop_price) * pos.qty
            risk_pct = (risk_dollars / pf.equity * 100) if pf.equity else 0.0
            lines.append(f"{symbol:10s} {risk_pct:.2f}%")
        self.query_one("#position_risk_panel", Static).update(
            "POSITION RISK\n" + ("\n".join(lines) if lines else "no open positions")
        )

        blocked = pf.is_long("SPY") and pf.is_long("QQQ")
        self.query_one("#correlation_panel", Static).update(
            "CORRELATION FILTER\n"
            f"SPY LONG            {'YES' if pf.is_long('SPY') else 'NO'}\n"
            f"QQQ LONG            {'YES' if pf.is_long('QQQ') else 'NO'}\n"
            f"BTC/USD NEW LONGS   {'BLOCKED' if blocked else 'ALLOWED'}\n"
            f"STATUS              {'● BLOCKING' if blocked else '● CLEAR'}"
        )


class BacktestScreen(Screen):
    BINDINGS = [("b", "run_backtest", "Run Backtest")]

    def compose(self):
        yield Header()
        yield Static("Press [b]B[/b] to run a 6-month backtest across all 5 instruments.", id="backtest_hint", classes="panel")
        yield DataTable(id="backtest_table", classes="panel")
        yield Static(id="backtest_summary", classes="panel")
        yield Sparkline([], id="backtest_curve", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#backtest_table", DataTable)
        table.add_columns("Symbol", "Trades", "Win %", "PF", "Sharpe", "Max DD", "Return")

    def action_run_backtest(self) -> None:
        self.query_one("#backtest_hint", Static).update("Running backtest against Alpaca historical data...")
        self.run_worker(self._do_backtest(), exclusive=True)

    async def _do_backtest(self) -> None:
        app = self.app
        results = await backtest.run_all_backtests(app.broker, months=config.BACKTEST_MONTHS)
        table = self.query_one("#backtest_table", DataTable)
        table.clear()
        for symbol, result in results.items():
            pf_display = "inf" if result.profit_factor == float("inf") else f"{result.profit_factor:.2f}"
            table.add_row(
                symbol, str(result.trades), f"{result.win_rate:.1f}%", pf_display,
                f"{result.sharpe:.2f}", f"{result.max_dd:.1f}%", f"{result.total_return_pct:+.1f}%",
            )
        combined = await backtest.run_combined_backtest(app.broker, months=config.BACKTEST_MONTHS)
        self.query_one("#backtest_summary", Static).update(
            f"COMBINED PORTFOLIO (correlation filter active)   Return {combined.total_return_pct:+.1f}%   "
            f"Sharpe {combined.sharpe:.2f}   Max DD {combined.max_dd:.1f}%   Trades {combined.trades}"
        )
        self.query_one("#backtest_hint", Static).update("Backtest complete. Press B to re-run.")
        if combined.equity_curve:
            self.query_one("#backtest_curve", Sparkline).data = combined.equity_curve


class ReportsScreen(Screen):
    BINDINGS = [("e", "toggle_evening", "Toggle Morning/Evening")]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evening = False

    def compose(self):
        yield Header()
        yield Static(id="report_panel", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()
        self.set_interval(5.0, self.refresh_content)

    def action_toggle_evening(self) -> None:
        self.evening = not self.evening
        self.refresh_content()

    def refresh_content(self) -> None:
        app = self.app
        pf = app.portfolio
        label = "EVENING WRAP-UP" if self.evening else "MORNING BRIEFING"
        now = datetime.now(EASTERN).strftime("%I:%M %p ET")
        lines = [f"{label}                                  {now}", "", "OPEN POSITIONS"]
        for symbol, pos in pf.positions.items():
            price = _latest_price(app, symbol) or pos.entry_price
            sign = 1 if pos.side == "long" else -1
            pnl = (price - pos.entry_price) * pos.qty * sign
            lines.append(f"{symbol} {pos.side.upper():5s} {pnl:+.2f} unrealized")
        if not pf.positions:
            lines.append("(none)")

        daily_pnl = pf.equity - pf.day_start_equity
        lines += ["", "TODAY", f"Total P&L: {daily_pnl:+,.2f}"]
        per_symbol = []
        for symbol in config.INSTRUMENTS:
            symbol_trades = [t for t in pf.trade_log if t.symbol == symbol]
            per_symbol.append(f"{symbol}: {sum(t.pnl for t in symbol_trades):+,.2f}")
        lines.append("   ".join(per_symbol))

        tripped = risk_manager.circuit_breaker_tripped(pf)
        lines += [
            "",
            "RISK",
            f"Portfolio Drawdown: {pf.drawdown()*100:.2f}%",
            f"Circuit Breaker: {'TRIPPED' if tripped else 'ARMED'}",
            "",
            "● No critical risk flags" if not tripped else "● CIRCUIT BREAKER TRIPPED",
        ]
        self.query_one("#report_panel", Static).update("\n".join(lines))


class LogsScreen(Screen):
    def compose(self):
        yield Header()
        yield Static(id="logs_panel", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()
        self.set_interval(1.0, self.refresh_content)

    def refresh_content(self) -> None:
        lines = list(logs.BUFFER)[-40:]
        self.query_one("#logs_panel", Static).update("\n".join(lines) if lines else "(no log entries yet)")
