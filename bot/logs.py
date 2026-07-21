"""Logging setup: writes to logs/bot.log and keeps an in-memory ring buffer
the TUI's Logs screen reads directly (no need to tail the file).
"""

from __future__ import annotations

import logging
import os
from collections import deque

import config

BUFFER: deque[str] = deque(maxlen=200)


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        BUFFER.append(self.format(record))


def setup_logging() -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(config.LOG_DIR, "bot.log"))
    file_handler.setFormatter(fmt)

    buffer_handler = _BufferHandler()
    buffer_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(buffer_handler)
