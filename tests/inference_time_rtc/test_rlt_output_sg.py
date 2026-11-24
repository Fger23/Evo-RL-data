"""Verify final S-G for actor output, recovery, and differentiable Q actions."""

import numpy as np
import pytest
import torch

from lerobot.inference_time_rtc.action_buffer import RTCActionBuffer
from lerobot.inference_time_rtc.action_filter import filter_joint_targets, filter_joint_targets_tensor
from lerobot.inference_time_rtc.protocol import RTCActionChunk


@pytest.mark.parametrize("window", [0, 5, 7])
def test_tensor_filter_matches_physical_numpy_filter_and_preserves_gradients(window):
    torch.manual_seed(41)
    lengths = torch.tensor([0, 1, 4, 5, 6, 7, 12])
    valid = torch.arange(12)[None, :] < lengths[:, None]
    actions = torch.randn(7, 12, 3, requires_grad=True)
    result = filter_joint_targets_tensor(actions, (0, 1), window, valid)
    # Native PI05 normalization is affine per channel. Compare directly with
    # the client-side physical filter under different scales and offsets.
    scale = np.array([2.0, 7.0, 3.0])
    offset = np.array([10.0, -4.0, 20.0])
    for index, length in enumerate(lengths.tolist()):
        physical = actions[index, :length].detach().numpy() * scale + offset
        expected, _ = filter_joint_targets(physical, (0, 1), window)
        actual = result[index, :length].detach().numpy() * scale + offset
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result[:, :, 2], actions[:, :, 2])
    torch.testing.assert_close(result[~valid], actions[~valid])
    result.square().sum().backward()
    assert actions.grad.isfinite().all() and actions.grad.abs().sum() > 0


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("actor_active", [False, True])
def test_output_filter_applies_to_actor_and_bypass_after_cropping_and_on_resume(recovery, actor_active):
    actions = np.stack((np.sin(np.arange(14) * 2), np.arange(14)), axis=1).astype(np.float32)
    mask = np.full(14, actor_active, dtype=bool)
    chunk = RTCActionChunk(1, 0, actions.copy(), actions.copy(), rlt_applied=actor_active, rlt_action_mask=mask)
    buffer = RTCActionBuffer(sg_window=7, joint_columns=(0,))
    if recovery:
        buffer.started = True
        buffer._motion_paused = True
        assert buffer.resume(chunk)
        offset = 0
    else:
        buffer.started = True
        buffer.step = 3
        assert buffer.merge(chunk) == 3
        offset = 3
    expected, window = filter_joint_targets(actions[offset:], (0,), 7)
    np.testing.assert_allclose(buffer.processed, expected)
    assert not np.allclose(expected[:, 0], actions[offset:, 0])
    np.testing.assert_array_equal(buffer.processed[:, 1], actions[offset:, 1])
    np.testing.assert_array_equal(buffer.snapshot()[1], actions[offset:])
    assert buffer.applied_sg_window == window == 7
    assert buffer.reserve().rlt_applied == actor_active
