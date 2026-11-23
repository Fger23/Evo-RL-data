"""Versioned payloads carried by the existing gRPC byte messages.

As with async_inference, pickle transport is only suitable for trusted networks.
No protobuf definitions or existing inference payloads are changed.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .configs import InferenceTimeRTCConfig, RTCRLTConfig

PROTOCOL_VERSION = "inference-time-rtc-v3"
SESSION_HEADER = "rtc-session-id"
CHECKPOINT_SELECTION_HEADER = "rtc-client-checkpoint"
CHECKPOINT_PATH_HEADER = "rtc-checkpoint-path-bin"
POLICY_TYPE_HEADER = "rtc-policy-type"
OBSERVATION_IMAGE_CODEC_HEADER = "rtc-observation-image-codec"
OBSERVATION_PNG_HEADER = "rtc-observation-lossless-png"
CURRENT_POSE_RECOVERY_HEADER = "rtc-current-pose-recovery"
RLT_HEADER = "rtc-rlt-direct-vla"
RLT_PHASE_HEADER = "rtc-rlt-step-interval"


@dataclass
class SessionInstructions:
    lerobot_features: dict[str, dict]
    action_names: tuple[str, ...]
    fps: float
    rtc: InferenceTimeRTCConfig
    actions_per_chunk: int | None = None
    policy_device: str | None = None
    protocol_version: str = PROTOCOL_VERSION
    pretrained_name_or_path: str | None = None
    policy_type: str | None = None
    rlt: RTCRLTConfig | None = None
    action_sg_window: int = 0


@dataclass
class RTCObservation:
    request_id: int
    start_step: int
    observation: dict[str, Any]
    prev_chunk_left_over: np.ndarray | None
    inference_delay: int
    recovery_prefix_steps: int = 0
    # Recording frame index (not the buffer clock that resets on takeover).
    episode_step: int | None = None


@dataclass
class RTCActionChunk:
    request_id: int
    start_step: int
    original_actions: np.ndarray
    processed_actions: np.ndarray
    # S-G and RLT are mutually exclusive. Never filter the actor's output.
    rlt_applied: bool = False
    applied_sg_window: int = 0
    sg_reference_actions: np.ndarray | None = None
    rlt_reference_actions: np.ndarray | None = None
    rlt_action_mask: np.ndarray | None = None

    def validate(self, action_dim: int | None = None) -> None:
        if self.request_id < 0 or self.start_step < 0:
            raise ValueError("Action request_id and start_step must be non-negative")
        raw, processed = self.original_actions, self.processed_actions
        if not isinstance(raw, np.ndarray) or not isinstance(processed, np.ndarray):
            raise ValueError("Actions must be numpy arrays")
        if raw.ndim != 2 or raw.shape != processed.shape or min(raw.shape) <= 0:
            raise ValueError("Original and processed actions must have matching nonempty (T, A) shapes")
        if action_dim is not None and raw.shape[1] != action_dim:
            raise ValueError(f"Expected {action_dim} action dimensions, got {raw.shape[1]}")
        if not np.isfinite(raw).all() or not np.isfinite(processed).all():
            raise ValueError("Policy returned non-finite actions")
        mask = self.rlt_action_mask
        if mask is not None:
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (len(raw),):
                raise ValueError("RLT action mask must be a boolean row mask matching the chunk")
            if bool(mask.any()) != self.rlt_applied:
                raise ValueError("RLT marker must match the action row mask")
        if self.rlt_applied:
            if self.applied_sg_window != 0 or self.sg_reference_actions is not None:
                raise ValueError("RLT and action S-G are mutually exclusive; update both RTC endpoints")
            reference = self.rlt_reference_actions
            if reference is not None and (
                not isinstance(reference, np.ndarray)
                or reference.shape != processed.shape
                or not np.isfinite(reference).all()
            ):
                raise ValueError("RLT reference must be a finite physical action chunk matching its output")


def rlt_row_mask(config: RTCRLTConfig, request: RTCObservation, count: int, horizon: int) -> np.ndarray:
    """Enter by observation step, then restrict rows to the active interval.

    Match the training cache: an observation outside the marked phase has no
    actor rows, even if its predicted chunk extends into a future phase.
    """
    step = request.episode_step
    if type(step) is not int or step < 0:
        raise ValueError("Scheduled RLT requires a non-negative episode_step")
    if not config.active_at(step):
        return np.zeros(count, dtype=np.bool_)
    rows = np.arange(count)
    protected = request.recovery_prefix_steps
    if not protected and request.prev_chunk_left_over is not None:
        protected = min(len(request.prev_chunk_left_over), horizon)
    return (
        (step + rows >= config.start_step)
        & (step + rows < config.end_step)
        & (rows >= protected)
    )


def validate_rlt_stage(result: RTCActionChunk, request: RTCObservation, config: RTCRLTConfig, horizon: int):
    if not config.enabled:
        if result.rlt_applied:
            raise ValueError("Server applied RLT to a session without RLT")
        return
    expected = rlt_row_mask(config, request, len(result.original_actions), horizon)
    actual = result.rlt_action_mask
    if actual is None or not np.array_equal(actual, expected) or result.rlt_applied != bool(expected.any()):
        raise ValueError("Server did not honor the requested RLT step interval or protected prefix")
