"""Entry point: builds the broker, portfolio, and engine, then hands off to
the Textual TUI, which drives the asyncio event loop from here on.
"""

from __future__ import annotations

from bot.broker import Broker
from bot.engine import Engine
from bot.logs import setup_logging
from bot.portfolio import Portfolio
from bot.tui.app import TradingApp


def main() -> None:
    setup_logging()
    broker = Broker()
    portfolio = Portfolio(broker=broker)
    engine = Engine(broker, portfolio)
    app = TradingApp(broker, portfolio, engine)
    app.run()


if __name__ == "__main__":
    main()
