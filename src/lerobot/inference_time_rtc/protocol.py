"""Versioned payloads carried by the existing gRPC byte messages.

As with async_inference, pickle transport is only suitable for trusted networks.
No protobuf definitions or existing inference payloads are changed.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .configs import InferenceTimeRTCConfig

PROTOCOL_VERSION = "inference-time-rtc-v3"
SESSION_HEADER = "rtc-session-id"
CHECKPOINT_SELECTION_HEADER = "rtc-client-checkpoint"
CHECKPOINT_PATH_HEADER = "rtc-checkpoint-path-bin"
POLICY_TYPE_HEADER = "rtc-policy-type"
OBSERVATION_IMAGE_CODEC_HEADER = "rtc-observation-image-codec"
OBSERVATION_PNG_HEADER = "rtc-observation-lossless-png"
CURRENT_POSE_RECOVERY_HEADER = "rtc-current-pose-recovery"


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


@dataclass
class RTCObservation:
    request_id: int
    start_step: int
    observation: dict[str, Any]
    prev_chunk_left_over: np.ndarray | None
    inference_delay: int
    recovery_prefix_steps: int = 0


@dataclass
class RTCActionChunk:
    request_id: int
    start_step: int
    original_actions: np.ndarray
    processed_actions: np.ndarray

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
