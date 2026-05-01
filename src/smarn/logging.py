from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

from smarn.config import Settings, get_settings

_TELEGRAM_BOT_TOKEN_PATTERN = re.compile(r"/bot([^/\s\"]+)")
_OPENAI_API_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")

_STANDARD_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": _redact_sensitive_values(record.getMessage()),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_KEYS and not key.startswith("_"):
                payload[key] = _redact_sensitive_values(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Settings | None = None) -> None:
    resolved_settings = settings or get_settings()
    root_logger = logging.getLogger()

    if getattr(root_logger, "_smarn_logging_configured", False):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(resolved_settings.log_level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    setattr(root_logger, "_smarn_logging_configured", True)


def _redact_sensitive_values(value: Any) -> Any:
    if isinstance(value, str):
        redacted = _TELEGRAM_BOT_TOKEN_PATTERN.sub("/bot[REDACTED]", value)
        return _OPENAI_API_KEY_PATTERN.sub("sk-[REDACTED]", redacted)
    if isinstance(value, dict):
        return {key: _redact_sensitive_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_values(item) for item in value)
    return value
