"""Exercise PI05 feature export with CPU tensors and no checkpoint or hardware."""

import pytest
import torch
from torch import nn

from lerobot.inference_time_rtc.modeling_rtc import InputGradientRTCProcessor
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class TinyConfig:
    """Plain attributes support Dynamo mutation tracing on PyTorch 2.7 too."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TinyPrefixModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = TinyConfig(language_model=TinyConfig(config=TinyConfig()))
        self.calls = 0

    def forward(self, *, inputs_embeds, **kwargs):
        self.calls += 1
        # Distinct from input embeddings: consumers need the final hidden states.
        features = inputs_embeds[0] * 2 + 3
        return [features, None], features.mean(dim=(1, 2))


class TinyPI05(PI05Pytorch):
    """Use the production sampler with a tiny deterministic denoiser/prefix."""

    def __init__(self, rtc_processor=None):
        nn.Module.__init__(self)
        self.config = TinyConfig(
            num_inference_steps=3,
            chunk_size=5,
            max_action_dim=4,
            rtc_config=None if rtc_processor is None else rtc_processor.rtc_config,
        )
        self.rtc_processor = rtc_processor
        self.paligemma_with_expert = TinyPrefixModel()

    def embed_prefix(self, images, img_masks, tokens, masks):
        prefix = images[0] + tokens.unsqueeze(-1).to(torch.float32)
        return prefix, masks, torch.zeros_like(masks)

    def _prepare_attention_masks_4d(self, masks):
        return masks.unsqueeze(1)

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        return x_t * 0.25 + past_key_values[:, None, None] * 0.01 + timestep[:, None, None] * 0.02


class TinyPolicy(PI05Policy):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model
        self.config = TinyConfig(output_features={ACTION: TinyConfig(shape=(2,))})

    def _preprocess_images(self, batch):
        return [batch["image"]], [torch.ones(batch["image"].shape[0], dtype=torch.bool)]


@pytest.fixture
def batch():
    return {
        "image": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 24,
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 0], [3, 0, 0]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True, False], [True, False, False]]),
    }


def sampler_inputs(batch):
    return (
        [batch["image"]],
        [torch.ones(batch["image"].shape[0], dtype=torch.bool)],
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
    )


def test_features_are_from_same_prefix_forward_and_preserve_default_actions(batch):
    model = TinyPI05()
    noise = torch.linspace(-1, 1, 40).reshape(2, 5, 4)
    expected = model.sample_actions(*sampler_inputs(batch), noise=noise)
    assert isinstance(expected, torch.Tensor)
    assert model.paligemma_with_expert.calls == 1

    actions, features, mask = model.sample_actions(
        *sampler_inputs(batch), noise=noise, return_prefix_features=True
    )
    assert model.paligemma_with_expert.calls == 2
    torch.testing.assert_close(actions, expected)
    prefix = batch["image"] + batch[OBS_LANGUAGE_TOKENS].unsqueeze(-1)
    torch.testing.assert_close(features, prefix * 2 + 3)
    torch.testing.assert_close(mask, batch[OBS_LANGUAGE_ATTENTION_MASK])
    assert not actions.requires_grad and not features.requires_grad


@pytest.mark.parametrize("compiled", [False, True])
def test_prepared_rtc_features_keep_input_gradient_guidance(batch, compiled):
    rtc = InputGradientRTCProcessor(RTCConfig(enabled=True, execution_horizon=3))
    model = TinyPI05(rtc)
    # Prepared RTC currently represents a single remote observation.
    batch = {name: value[:1] for name, value in batch.items()}
    inputs = sampler_inputs(batch)
    noise = torch.linspace(-1, 1, 20).reshape(1, 5, 4)
    prepared = rtc.prepare_prefix(torch.full((3, 2), 0.25), 1, 3, chunk_size=5, action_dim=4)
    expected = model.sample_actions(*inputs, noise=noise, prepared_rtc_prefix=prepared)
    if compiled:
        # Keep the existing RTC autograd graph-break behavior. This test checks
        # opt-in feature export also works through the compiled entry point.
        model.sample_actions = torch.compile(model.sample_actions, backend="eager")
    actions, features, mask = model.sample_actions(
        *inputs, noise=noise, prepared_rtc_prefix=prepared, return_prefix_features=True
    )
    torch.testing.assert_close(actions, expected)
    assert features.shape == (1, 3, 4)
    assert mask.shape == (1, 3)
    assert mask.dtype == torch.bool
    assert not actions.requires_grad and not features.requires_grad
    assert torch.isfinite(actions).all()


def test_feature_export_can_compile_as_one_graph_for_first_chunk(batch):
    model = TinyPI05()
    compiled = torch.compile(model.sample_actions, backend="eager", fullgraph=True)
    noise = torch.zeros(2, 5, 4)
    expected = model.sample_actions(*sampler_inputs(batch), noise=noise, return_prefix_features=True)
    actual = compiled(*sampler_inputs(batch), noise=noise, return_prefix_features=True)
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted)


def test_policy_unpads_actions_and_owns_outputs_outside_compiled_sampler(batch):
    policy = TinyPolicy(TinyPI05())
    # Simulate reusable CUDA Graph output storage without requiring a GPU.
    source_actions = torch.randn(2, 5, 4)
    source_features = torch.randn(2, 3, 4)
    source_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].to(torch.int32)
    prepared = object()
    calls = []

    def sampler(images, img_masks, tokens, masks, **kwargs):
        calls.append(kwargs)
        return source_actions, source_features, source_mask

    policy.model.sample_actions = sampler
    actions, features, mask = policy.predict_action_chunk_with_features(batch, prepared_rtc_prefix=prepared)
    assert calls == [{"return_prefix_features": True, "prepared_rtc_prefix": prepared}]
    assert not policy.training
    assert actions.shape == (2, 5, 2)
    assert mask.dtype == torch.bool
    torch.testing.assert_close(actions, source_actions[:, :, :2])
    torch.testing.assert_close(features, source_features)
    torch.testing.assert_close(mask, source_mask.bool())
    expected = (actions.clone(), features.clone(), mask.clone())
    source_actions.zero_()
    source_features.zero_()
    source_mask.zero_()
    for actual, saved in zip((actions, features, mask), expected, strict=True):
        torch.testing.assert_close(actual, saved)
