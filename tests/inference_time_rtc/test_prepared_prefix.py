"""Check the fixed-shape RTC continuation against the existing guidance path."""

import pytest
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.inference_time_rtc.modeling_rtc import InputGradientRTCProcessor
from lerobot.policies.rtc.configuration_rtc import RTCConfig


@pytest.mark.parametrize("schedule", list(RTCAttentionSchedule))
@pytest.mark.parametrize("prefix_steps", [21, 28, 35])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
        ),
    ],
)
def test_prepared_prefix_matches_original_guidance(schedule, prefix_steps, device):
    horizon = 28
    delay = 22
    chunk_size = 50
    action_dim = 3
    processor = InputGradientRTCProcessor(
        RTCConfig(enabled=True, execution_horizon=horizon, prefix_attention_schedule=schedule),
        prefix_weights_device=device,
    )
    prefix = torch.linspace(-0.5, 0.5, prefix_steps * 2, device=device).reshape(prefix_steps, 2)
    latent = torch.linspace(-1.0, 1.0, chunk_size * action_dim, device=device).reshape(
        1, chunk_size, action_dim
    )
    prepared = processor.prepare_prefix(prefix, delay, horizon, chunk_size=chunk_size, action_dim=action_dim)

    assert prepared.actions.shape == (1, chunk_size, action_dim)
    assert prepared.weights.shape == (1, chunk_size, 1)
    assert prepared.actions.device == prepared.weights.device == latent.device
    torch.testing.assert_close(prepared.actions[0, :prefix_steps, :2], prefix)
    torch.testing.assert_close(
        prepared.actions[0, prefix_steps:, :], torch.zeros_like(prepared.actions[0, prefix_steps:, :])
    )
    torch.testing.assert_close(prepared.actions[0, :, 2:], torch.zeros_like(prepared.actions[0, :, 2:]))

    effective_horizon = min(horizon, prefix_steps)
    expected_weights = processor.get_prefix_weights(delay, effective_horizon, chunk_size)
    torch.testing.assert_close(prepared.weights[0, :, 0], expected_weights)

    def denoiser(value):
        return 0.5 * value + 0.25 * value.sin()

    # Endpoints exercise the guidance clamp and the NaN/inf handling. The
    # shorter prefix exercises capping the horizon before padding.
    with torch.no_grad():
        for time in (0.0, 1e-6, 0.5, 1.0 - 1e-6, 1.0):
            expected = processor.denoise_step(latent, prefix, delay, time, denoiser, horizon)
            actual = processor.denoise_prepared_step(
                latent, prepared, torch.tensor(time, device=device), denoiser
            )
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
            assert not actual.requires_grad
