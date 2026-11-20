"""Reliable CLI logging even when imported libraries configured root logging."""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from queue import SimpleQueue

import numpy as np

_action_logger = logging.getLogger("lerobot.inference_time_rtc.actions")


def _numpy_to_json(value):
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    raise TypeError(f"Unsupported RTC action log value: {type(value).__name__}")


class _ActionJSON:
    def __init__(self, values):
        self.values = values

    def __str__(self):
        return json.dumps(self.values, separators=(",", ":"), default=_numpy_to_json)


class _ActionQueueHandler(QueueHandler):
    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        # The listener is in this process. Defer JSON formatting as well as file writes.
        return record


def log_action_event(event: str, **values):
    _action_logger.info("RTC %s %s", event, _ActionJSON(values))


@contextmanager
def log_actions_to_file(path: Path):
    """Write action records only to the configured RTC log and flush on exit."""
    file_handler = next(
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(path)
    )
    queue = SimpleQueue()
    queue_handler = _ActionQueueHandler(queue)
    listener = QueueListener(queue, file_handler)
    propagate = _action_logger.propagate
    _action_logger.propagate = False
    _action_logger.addHandler(queue_handler)
    listener.start()
    try:
        yield
    finally:
        _action_logger.removeHandler(queue_handler)
        listener.stop()
        _action_logger.propagate = propagate


def configure_logging(role: str, log_file: str | None = None) -> Path:
    path = Path(log_file) if log_file else Path("logs") / f"rtc_{role}_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )
    logging.getLogger(__name__).info("RTC log file: %s", path)
    return path
