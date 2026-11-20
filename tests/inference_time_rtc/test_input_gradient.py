"""Analytical checks of the inference-time guidance, independent of robot/model."""

import pytest
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.inference_time_rtc.modeling_rtc import InputGradientRTCProcessor
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


@pytest.mark.parametrize("schedule", list(RTCAttentionSchedule))
@pytest.mark.parametrize("start,end", [(22, 28), (22, 22)])
def test_prefix_weights_stay_on_configured_device(schedule, start, end):
    processor = InputGradientRTCProcessor(
        RTCConfig(prefix_attention_schedule=schedule), prefix_weights_device="meta"
    )

    weights = processor.get_prefix_weights(start, end, 50)

    assert weights.device.type == "meta"
    assert weights.shape == (50,)


def test_guidance_retains_denoiser_input_jacobian_under_no_grad():
    processor = InputGradientRTCProcessor(
        RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=10.0)
    )
    x = torch.ones(1, 20, 1)
    previous = torch.full_like(x, 0.1)
    with torch.no_grad():
        result = processor.denoise_step(x, previous, 5, torch.tensor(0.5), lambda value: value * 0.5)
    # x_final=x-0.5*(0.5*x)=0.75*x, so its derivative must be 0.75,
    # error=-0.65, clipped multiplier=2, and corrected velocity=1.475.
    torch.testing.assert_close(result[0, :5, 0], torch.full((5,), 1.475))
    assert not result.requires_grad


def test_zero_final_estimate_jacobian_means_zero_guidance():
    processor = InputGradientRTCProcessor(RTCConfig(enabled=True))
    x = torch.ones(1, 20, 1)
    with torch.no_grad():
        result = processor.denoise_step(x, torch.ones_like(x), 5, 0.5, lambda value: 2 * value)
    # x_final=x-0.5*(2*x)=0 has zero derivative; correction must be zero.
    torch.testing.assert_close(result, torch.full_like(x, 2))


def test_first_chunk_without_prefix_preserves_original_denoiser():
    processor = InputGradientRTCProcessor(RTCConfig(enabled=True))
    x = torch.randn(1, 20, 3)
    with torch.no_grad():
        result = processor.denoise_step(x, None, 5, 0.5, lambda value: value.sin())
    torch.testing.assert_close(result, x.sin())


def test_shared_rtc_processor_is_unchanged():
    # Existing entry points retain their previous behavior.
    processor = RTCProcessor(RTCConfig(enabled=True, max_guidance_weight=10.0))
    x = torch.ones(1, 20, 1)
    with torch.no_grad():
        result = processor.denoise_step(x, torch.full_like(x, 0.1), 5, 0.5, lambda value: value * 0.5)
    torch.testing.assert_close(result[0, :5, 0], torch.full((5,), 1.8))
