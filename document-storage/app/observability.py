from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


def setup_json_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)

    # reset default handlers
    root.handlers = [handler]


class TraceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Simple filter for now (can add OpenTelemetry later if needed)
        return True










