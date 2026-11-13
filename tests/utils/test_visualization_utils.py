"""Preview lifecycle tests; no robot, camera, network, or GUI is started."""
import inspect
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.utils import visualization_utils as vu


@pytest.fixture(autouse=True)
def clean_preview():
    vu.close_rerun()
    yield
    vu.close_rerun()


def test_no_init_never_buffers_into_an_unconnected_sdk():
    assert vu.log_rerun_data(observation={"camera": np.zeros((8, 8, 3), dtype=np.uint8)}) is False


def test_command_cleanup_preserves_getfullargspec_for_cli_parser():
    class Config:
        pass

    @vu.with_rerun_cleanup
    def command(cfg: Config):
        return cfg

    arguments = inspect.getfullargspec(command)
    assert arguments.args[0] == "cfg"
    assert arguments.annotations["cfg"] is Config
    value = Config()
    assert command(value) is value


def test_preview_lifecycle_forwards_snapshot_without_mutation(monkeypatch):
    session = Mock()
    session.submit.return_value = True
    constructor = Mock(return_value=session)
    monkeypatch.setattr(vu, "LivePreviewSession", constructor)
    assert vu.init_rerun("recording", ip="127.0.0.1", port=9877)
    image = np.full((8, 8, 3), 37, dtype=np.uint8)
    observation, action = {"camera": image}, {"joint": 0.2}
    assert vu.log_rerun_data(observation, action)
    constructor.assert_called_once_with("recording", ip="127.0.0.1", port=9877)
    session.submit.assert_called_once_with(observation, action, compress_images=True)
    np.testing.assert_array_equal(image, np.full_like(image, 37))
    vu.close_rerun()
    vu.close_rerun()
    session.close.assert_called_once()


def test_reinitialize_closes_its_previous_preview(monkeypatch):
    first, second = Mock(), Mock()
    monkeypatch.setattr(vu, "LivePreviewSession", Mock(side_effect=[first, second]))
    vu.init_rerun("first")
    vu.init_rerun("second")
    first.close.assert_called_once()
    second.close.assert_not_called()
    vu.close_rerun()
    second.close.assert_called_once()


def test_command_exception_still_closes_its_preview(monkeypatch):
    session = Mock()
    monkeypatch.setattr(vu, "LivePreviewSession", Mock(return_value=session))

    @vu.with_rerun_cleanup
    def command():
        vu.init_rerun("recording")
        raise RuntimeError("synthetic hardware initialization failure")

    with pytest.raises(RuntimeError, match="synthetic hardware"):
        command()
    session.close.assert_called_once()


def test_unrelated_existing_preview_is_not_closed_by_a_command(monkeypatch):
    session = Mock()
    monkeypatch.setattr(vu, "LivePreviewSession", Mock(return_value=session))
    vu.init_rerun("existing")

    @vu.with_rerun_cleanup
    def command_without_preview():
        return 3

    assert command_without_preview() == 3
    session.close.assert_not_called()


def test_startup_failure_keeps_control_available(monkeypatch):
    monkeypatch.setattr(vu, "LivePreviewSession", Mock(side_effect=OSError("viewer unavailable")))
    assert vu.init_rerun() is False
    assert vu.log_rerun_data(action={"joint": 0.2}) is False


def test_preview_failure_does_not_interrupt_robot_control(monkeypatch):
    session = Mock()
    session.submit.side_effect = MemoryError("preview allocator failed")
    monkeypatch.setattr(vu, "LivePreviewSession", Mock(return_value=session))
    vu.init_rerun()
    assert vu.log_rerun_data(action={"joint": 0.2}) is False
