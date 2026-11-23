"""Episode step intervals and per-row policy provenance, without robot hardware."""

import numpy as np
import pytest

from lerobot.inference_time_rtc.action_buffer import RTCActionBuffer
from lerobot.inference_time_rtc.configs import RTCRLTConfig
from lerobot.inference_time_rtc.protocol import (
    RTCActionChunk,
    RTCObservation,
    rlt_row_mask,
    validate_rlt_stage,
)


def config(**kwargs):
    return RTCRLTConfig(enabled=True, checkpoint="/models/actor.pt", start_step=10, end_step=20, **kwargs)


@pytest.mark.parametrize("bounds", [{}, {"start_step": 0}, {"end_step": 20}])
def test_enabled_rlt_requires_explicit_start_and_end(bounds):
    with pytest.raises(ValueError, match="start_step and rlt.end_step"):
        RTCRLTConfig(enabled=True, checkpoint="/models/actor.pt", **bounds)


@pytest.mark.parametrize("bounds", [(-1, 20), (10, 10), (20, 10), (False, 20), (0, 1.5)])
def test_invalid_rlt_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        RTCRLTConfig(enabled=True, checkpoint="/models/actor.pt", start_step=bounds[0], end_step=bounds[1])


def test_rlt_interval_is_off_on_off_and_disabled_needs_no_bounds():
    scheduled = config()
    assert [scheduled.active_at(step) for step in (0, 9, 10, 19, 20, 25)] == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    disabled = RTCRLTConfig()
    assert not any(disabled.active_at(step) for step in (0, 10, 20))


@pytest.mark.parametrize(
    "step,prefix_length,recovery,expected",
    [
        (0, 0, 0, []),
        (5, 0, 0, []),
        (12, 3, 0, list(range(3, 8))),
        (12, 8, 2, list(range(2, 8))),
        (20, 0, 0, []),
    ],
)
def test_row_mask_intersects_interval_with_unprotected_rows(step, prefix_length, recovery, expected):
    prefix = np.zeros((prefix_length, 2)) if prefix_length else None
    request = RTCObservation(0, 0, {}, prefix, 1, recovery_prefix_steps=recovery, episode_step=step)
    mask = rlt_row_mask(config(), request, count=12, horizon=4)
    assert mask.dtype == np.bool_ and np.flatnonzero(mask).tolist() == expected


@pytest.mark.parametrize("step", [None, -1, False])
def test_row_mask_requires_episode_clock(step):
    with pytest.raises(ValueError, match="episode_step"):
        rlt_row_mask(config(), RTCObservation(0, 0, {}, None, 1, episode_step=step), 10, 4)


def test_client_validation_requires_exact_mask_not_only_rlt_flag():
    request = RTCObservation(0, 0, {}, None, 1, episode_step=12)
    raw = np.zeros((12, 2), np.float32)
    expected = rlt_row_mask(config(), request, 12, 4)
    result = RTCActionChunk(0, 0, raw, raw.copy(), rlt_applied=True, rlt_action_mask=expected.copy())
    result.validate(2)
    validate_rlt_stage(result, request, config(), 4)
    result.rlt_action_mask[-1] = True  # Still rlt_applied=True, but outside the requested interval.
    with pytest.raises(ValueError, match="step interval"):
        validate_rlt_stage(result, request, config(), 4)
    result.rlt_action_mask = None
    with pytest.raises(ValueError, match="step interval"):
        validate_rlt_stage(result, request, config(), 4)


def test_buffer_reservations_report_actor_source_for_each_row():
    actions = np.arange(20, dtype=np.float32).reshape(10, 2)
    mask = np.array([False, False, True, True, True, False, False, False, False, False])
    chunk = RTCActionChunk(
        0,
        0,
        actions,
        actions.copy(),
        rlt_applied=True,
        rlt_reference_actions=actions.copy(),
        rlt_action_mask=mask,
    )
    buffer = RTCActionBuffer()
    assert buffer.merge(chunk) == 0
    reservations = [buffer.reserve() for _ in range(10)]
    assert [row.rlt_applied for row in reservations] == mask.tolist()
    assert [row.chunk_index for row in reservations] == list(range(10))


def test_protocol_rejects_row_mask_that_disagrees_with_chunk_marker():
    actions = np.zeros((5, 2), np.float32)
    chunk = RTCActionChunk(0, 0, actions, actions, rlt_applied=False, rlt_action_mask=np.ones(5, dtype=bool))
    with pytest.raises(ValueError, match="marker"):
        chunk.validate(2)
