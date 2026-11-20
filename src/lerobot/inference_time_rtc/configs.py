"""Configuration for the independent inference-time RTC entry points."""

import math
import re
from dataclasses import dataclass, field

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.robots.config import RobotConfig

from .action_filter import ACTION_SG_WINDOWS

SUPPORTED_POLICIES = ("pi05", "pi0", "smolvla")


def normalize_policy_device(device: str | None) -> str | None:
    if device is None:
        return None
    if device == "cuda":
        return "cuda:0"
    if device == "cpu" or (isinstance(device, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device)):
        return device
    raise ValueError("policy_device must be cpu, cuda or cuda:N (a non-negative server GPU index)")


def positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


@dataclass
class InferenceTimeRTCConfig(RTCConfig):
    """All RTC algorithm and scheduling options are supplied by the client."""

    enabled: bool = True
    execution_horizon: int = 20
    max_guidance_weight: float = 1.0
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    queue_threshold: int = 30
    inference_delay: int = 10
    auto_delay: bool = True
    latency_window: int = 20
    observation_image_codec: str = "raw"

    def __post_init__(self):
        super().__post_init__()
        if not self.enabled:
            raise ValueError(
                "This entry point requires rtc.enabled=true; use async_inference for other modes"
            )
        positive_finite("rtc.max_guidance_weight", self.max_guidance_weight)
        if not isinstance(self.prefix_attention_schedule, RTCAttentionSchedule):
            raise ValueError("rtc.prefix_attention_schedule must be ZEROS, ONES, LINEAR or EXP")
        if self.execution_horizon <= 0:
            raise ValueError("rtc.execution_horizon must be positive")
        if self.queue_threshold < self.execution_horizon:
            raise ValueError("rtc.queue_threshold must be >= rtc.execution_horizon")
        if self.inference_delay < 0 or self.inference_delay >= self.execution_horizon:
            raise ValueError("rtc.inference_delay must be >= 0 and < rtc.execution_horizon")
        if self.latency_window <= 0:
            raise ValueError("rtc.latency_window must be positive")
        if self.observation_image_codec not in ("raw", "lossless_jpeg", "lossless_png"):
            raise ValueError("rtc.observation_image_codec must be raw, lossless_jpeg or lossless_png")

    def validate_chunk_size(self, chunk_size: int) -> None:
        self.__post_init__()
        if chunk_size <= self.queue_threshold:
            raise ValueError(
                f"rtc.queue_threshold ({self.queue_threshold}) must be < actions_per_chunk ({chunk_size})"
            )


@dataclass
class RTCServerConfig:
    pretrained_name_or_path: str | None = None
    policy_type: str = "pi05"
    # Optional compatibility option: preload on this device. By default weights
    # are staged on CPU and the client chooses the server inference device.
    policy_device: str | None = None
    host: str = "0.0.0.0"
    port: int = 8100
    tokenizer_name_or_path: str | None = None
    session_timeout_s: float = 120.0
    log_file: str | None = None
    policy_dtype: str | None = None
    compile_model: bool = False
    compile_mode: str = "default"
    warmup_on_connect: bool = False
    warmup_task: str = ""
    checkpoint_root: str | None = None

    def __post_init__(self):
        if not self.host:
            raise ValueError("host must not be empty")
        if not self.pretrained_name_or_path and not self.checkpoint_root:
            raise ValueError("Set pretrained_name_or_path or checkpoint_root on the RTC server")
        for name in ("pretrained_name_or_path", "checkpoint_root"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must not be empty")
        self.policy_device = normalize_policy_device(self.policy_device)
        if self.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(f"RTC supports {SUPPORTED_POLICIES}, got {self.policy_type}")
        if self.policy_dtype not in (None, "float32", "bfloat16"):
            raise ValueError("policy_dtype must be float32 or bfloat16")
        if self.policy_type != "pi05" and (self.policy_dtype is not None or self.compile_model):
            raise ValueError("policy_dtype and compile_model overrides are supported for RTC pi05 only")
        if self.warmup_on_connect and (self.policy_type != "pi05" or not self.compile_model):
            raise ValueError("warmup_on_connect requires compiled RTC pi05")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        positive_finite("session_timeout_s", self.session_timeout_s)


@dataclass
class RTCRecoveryConfig:
    enabled: bool = False
    prefix_steps: int = 6
    max_observation_age_s: float = 0.8
    # Accepted for existing client commands.
    max_joint_delta: float | None = None
    max_gripper_delta: float | None = None

    def __post_init__(self):
        if self.prefix_steps <= 0:
            raise ValueError("recovery.prefix_steps must be positive")
        positive_finite("recovery.max_observation_age_s", self.max_observation_age_s)
        for name in ("max_joint_delta", "max_gripper_delta"):
            value = getattr(self, name)
            if value is not None:
                positive_finite(f"recovery.{name}", value)


@dataclass
class RTCClientConfig:
    robot: RobotConfig
    server_address: str = "localhost:8100"
    # None preserves an explicitly preloaded server device; otherwise the server
    # defaults to cuda:0. This always names a device on the server, not the robot.
    policy_device: str | None = None
    task: str = ""
    fps: float = 30.0
    actions_per_chunk: int | None = None
    rtc: InferenceTimeRTCConfig = field(default_factory=InferenceTimeRTCConfig)
    connect_timeout_s: float = 10.0
    rpc_timeout_s: float = 120.0
    allow_late_splice: bool = False
    duration_s: float | None = None
    log_file: str | None = None
    action_sg_window: int = 0
    pretrained_name_or_path: str | None = None
    policy_type: str | None = None
    recovery: RTCRecoveryConfig = field(default_factory=RTCRecoveryConfig)

    def __post_init__(self):
        if not self.server_address:
            raise ValueError("server_address must not be empty")
        if self.pretrained_name_or_path is not None and not self.pretrained_name_or_path.strip():
            raise ValueError("pretrained_name_or_path must not be empty")
        if self.policy_type is not None and self.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(f"RTC supports {SUPPORTED_POLICIES}, got {self.policy_type}")
        self.policy_device = normalize_policy_device(self.policy_device)
        for name in ("fps", "connect_timeout_s", "rpc_timeout_s"):
            positive_finite(name, getattr(self, name))
        if self.duration_s is not None:
            positive_finite("duration_s", self.duration_s)
        if self.action_sg_window not in ACTION_SG_WINDOWS:
            raise ValueError("action_sg_window must be 0, 5, or 7")
        self.rtc.__post_init__()
        if self.recovery.enabled and self.recovery.prefix_steps > self.rtc.execution_horizon:
            raise ValueError("recovery.prefix_steps must be <= rtc.execution_horizon")
        if self.actions_per_chunk is not None:
            self.rtc.validate_chunk_size(self.actions_per_chunk)
