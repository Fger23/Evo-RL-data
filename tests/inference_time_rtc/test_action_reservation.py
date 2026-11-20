"""Check action provenance without connecting a robot or a server."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from lerobot.inference_time_rtc.action_buffer import RTCActionBuffer
from lerobot.inference_time_rtc.action_filter import filter_joint_targets
from lerobot.inference_time_rtc.protocol import RTCActionChunk


def action_chunk(request_id, start_step):
    actions = np.arange(20, dtype=np.float32).reshape(10, 2)
    return RTCActionChunk(request_id, start_step, actions, actions + 100 * request_id)


def test_reservation_tracks_absolute_slot_and_untrimmed_chunk_row():
    buffer = RTCActionBuffer()
    assert buffer.reserve(skipped_steps=2) is None
    bootstrap = action_chunk(4, 0)
    assert buffer.merge(bootstrap) == 0
    first = buffer.reserve()
    assert (first.request_id, first.step, first.chunk_index) == (4, 0, 0)
    np.testing.assert_array_equal(first.action, bootstrap.processed_actions[0])

    request_step, _, _ = buffer.snapshot()
    buffer.reserve(skipped_steps=1)
    continuation = action_chunk(5, request_step)
    assert buffer.merge(continuation) == 2

    reserved = buffer.reserve(skipped_steps=1)
    assert (reserved.request_id, reserved.step, reserved.chunk_index) == (5, 4, 3)
    np.testing.assert_array_equal(reserved.action, continuation.processed_actions[3])
    with pytest.raises(FrozenInstanceError):
        reserved.request_id = 123
    with pytest.raises(ValueError, match="read-only"):
        reserved.action[0] = -1


def test_rejected_and_stale_chunks_keep_reserved_actions_on_accepted_chunk():
    buffer = RTCActionBuffer()
    buffer.merge(action_chunk(0, 0))
    buffer.reserve()
    accepted = action_chunk(1, 0)
    assert buffer.merge(accepted) == 1
    buffer.reserve()

    with pytest.raises(RuntimeError, match="RTC prefix continuity lost"):
        buffer.merge(action_chunk(2, 0), prefix_horizon=2)
    assert buffer.last_splice.request_id == 2
    assert buffer.merge(action_chunk(0, 0)) is None

    reserved = buffer.reserve()
    assert (reserved.request_id, reserved.step, reserved.chunk_index) == (1, 2, 2)
    np.testing.assert_array_equal(reserved.action, accepted.processed_actions[2])


def test_advance_still_returns_a_writable_action_copy():
    buffer = RTCActionBuffer()
    chunk = action_chunk(0, 0)
    buffer.merge(chunk)

    action = buffer.advance()
    assert isinstance(action, np.ndarray)
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action, chunk.processed_actions[0])
    action[:] = -1
    np.testing.assert_array_equal(buffer.processed[0], chunk.processed_actions[0])


@pytest.mark.parametrize(
    ("rows", "requested", "used"),
    [(1, 7, 0), (4, 7, 0), (5, 7, 5), (6, 7, 5), (7, 7, 7), (8, 7, 7), (7, 5, 5), (7, 0, 0)],
)
def test_savgol_uses_complete_windows_and_preserves_unfiltered_targets(rows, requested, used):
    actions = np.zeros((rows, 3), dtype=np.float32)
    actions[:, 1] = np.arange(rows) * 2
    actions[:, 2] = np.arange(rows) % 2
    center = used // 2
    actions[center, 0] = 35 if used == 5 else 21
    original = actions.copy()

    filtered, applied = filter_joint_targets(actions, (0, 1), requested)

    assert applied == used
    assert filtered.dtype == actions.dtype
    np.testing.assert_array_equal(actions, original)
    np.testing.assert_array_equal(filtered[:, 1:], original[:, 1:])
    if used:
        np.testing.assert_array_equal(filtered[:center], original[:center])
        np.testing.assert_array_equal(filtered[-center:], original[-center:])
        assert filtered[center, 0] == (17 if used == 5 else 7)
    else:
        np.testing.assert_array_equal(filtered, original)


def test_savgol_bounds_the_center_to_the_local_input_range():
    actions = np.array([[0], [0], [0], [0], [0], [0], [21]], dtype=np.float32)
    filtered, _ = filter_joint_targets(actions, (0,), 7)
    # The negative last coefficient would produce a center target of -2.
    np.testing.assert_array_equal(filtered, actions)


def test_buffer_filters_after_alignment_and_keeps_the_model_prefix():
    buffer = RTCActionBuffer(sg_window=7, joint_columns=(0,))
    buffer.merge(action_chunk(0, 0))
    buffer.reserve(skipped_steps=1)
    model_actions = np.arange(18, dtype=np.float32).reshape(9, 2)
    joint_targets = [-1000, 1000, 0, 0, 0, 21, 0, 0, 0]
    targets = np.column_stack((joint_targets, model_actions[:, 1])).astype(np.float32)
    continuation = RTCActionChunk(1, 0, model_actions, targets)

    assert buffer.merge(continuation, prefix_horizon=7) == 2
    assert buffer.applied_sg_window == 7
    np.testing.assert_array_equal(buffer.snapshot()[1], model_actions[2:])
    reserved = [buffer.reserve() for _ in range(7)]

    assert [row.step for row in reserved] == list(range(2, 9))
    assert [row.chunk_index for row in reserved] == list(range(2, 9))
    np.testing.assert_array_equal([row.action[0] for row in reserved], [0, 0, 0, 7, 0, 0, 0])
    np.testing.assert_array_equal([row.action[1] for row in reserved], targets[2:, 1])
    assert continuation.processed_actions[5, 0] == 21
