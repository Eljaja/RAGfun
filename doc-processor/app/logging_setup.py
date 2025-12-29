from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger


def setup_json_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    fmt = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s %(extra)s")
    handler.setFormatter(fmt)

    root.handlers = [handler]
















