from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml
from app.config import settings

_logging_configured: bool = False


class JSONFormatter(logging.Formatter):
    """Custom formatter to output log messages in JSON format."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: N802
        log_record: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record, ensure_ascii=False)


def setup_logging() -> None:
    """Initialise logging once from ``logger.yaml``.

    Call this at application startup (e.g. inside ``main.py``).
    Falls back to :func:`logging.basicConfig` when the YAML file is missing.
    """
    global _logging_configured  # noqa: PLW0603

    if _logging_configured:
        return

    # Resolve relative to src/app/ (parent of utils/)
    config_path = (Path(__file__).parent.parent / "logger.yaml").resolve()

    if not config_path.exists():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d in %(funcName)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.warning(
            "Logging config '%s' not found. Using basicConfig.", config_path
        )
        _logging_configured = True
        return

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    log_dir = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve file-handler paths to the absolute log directory
    handlers = config.get("handlers", {})
    for handler_cfg in handlers.values():
        if "filename" in handler_cfg:
            filename = Path(handler_cfg["filename"]).name
            handler_cfg["filename"] = str(log_dir / filename)

    logging.config.dictConfig(config)
    _logging_configured = True
