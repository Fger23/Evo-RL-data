"""Input gradients and fixed-shape guidance for remote RTC."""

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

from lerobot.policies.rtc.modeling_rtc import RTCProcessor


class PreparedRTCPrefix(NamedTuple):
    actions: Tensor
    weights: Tensor


class InputGradientRTCProcessor(RTCProcessor):
    """Retain the denoiser's input derivative in the RTC vector-Jacobian product.

    The shared processor clones/detaches x_t, then enables its gradient after
    evaluating the denoiser. Enabling it inside this callback, before forward,
    preserves d(x - t*v(x))/dx instead of reducing that derivative to identity.
    Shared policy/RTC implementations remain unchanged for other entry points.
    """

    def prepare_prefix(
        self,
        prefix: Tensor,
        inference_delay: int,
        execution_horizon: int,
        *,
        chunk_size: int,
        action_dim: int,
    ) -> PreparedRTCPrefix:
        """Pad a single request and compute weights using the actual prefix length."""
        length, width = prefix.shape
        actions = prefix.new_zeros((1, chunk_size, action_dim))
        actions[0, :length, :width] = prefix
        horizon = min(execution_horizon, length)
        weights = self.get_prefix_weights(inference_delay, horizon, chunk_size).to(prefix.device)
        return PreparedRTCPrefix(actions=actions, weights=weights.view(1, chunk_size, 1))

    def denoise_prepared_step(
        self,
        x_t: Tensor,
        prepared_prefix: PreparedRTCPrefix,
        time: Tensor,
        denoiser: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Guide a batched latent with fixed-shape inputs and a scalar device timestep."""
        with torch.enable_grad():
            x_t = x_t.clone().detach().requires_grad_(True)
            velocity = denoiser(x_t)
            estimate = x_t - time * velocity
            error = ((prepared_prefix.actions - estimate) * prepared_prefix.weights).detach()
            correction = torch.autograd.grad(estimate, x_t, error)[0]

        tau = 1 - time
        squared_time = (1 - tau).square()
        inv_r2 = (squared_time + tau.square()) / squared_time
        limit = self.rtc_config.max_guidance_weight
        ratio = torch.nan_to_num((1 - tau) / tau, posinf=limit)
        guidance = torch.nan_to_num(ratio * inv_r2, posinf=limit).clamp(max=limit)
        return velocity - guidance * correction

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ):
        def denoise_with_input_gradient(value):
            if prev_chunk_left_over is not None and torch.is_grad_enabled():
                value.requires_grad_(True)
            return original_denoise_step_partial(value)

        return super().denoise_step(
            x_t,
            prev_chunk_left_over,
            inference_delay,
            time,
            denoise_with_input_gradient,
            execution_horizon,
        )
