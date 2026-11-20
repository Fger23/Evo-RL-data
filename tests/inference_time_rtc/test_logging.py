"""Imported logging configuration must not hide RTC diagnostics."""

import logging

from lerobot.inference_time_rtc.logging_utils import configure_logging


def test_client_log_is_written_when_root_was_preconfigured_at_warning(tmp_path):
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    installed = []
    try:
        root.handlers = [logging.NullHandler()]
        root.setLevel(logging.WARNING)
        path = configure_logging("client", str(tmp_path / "rtc_client.log"))
        installed = root.handlers[:]
        logging.getLogger("lerobot.inference_time_rtc.robot_client").info(
            "RTC request=1 latency=0.5s estimated_delay=15 actual_delay=15 queue=35"
        )
        for handler in installed:
            handler.flush()
        text = path.read_text(encoding="utf-8")
        assert "RTC log file:" in text and "RTC request=1" in text
        assert "actual_delay=15" in text
    finally:
        for handler in installed:
            handler.close()
        root.handlers, root.level = previous_handlers, previous_level
