"""Record RTC policy rollouts with human takeover into a local PKL capture."""

from dataclasses import dataclass, field, fields

from lerobot.configs import parser
from lerobot.scripts.lerobot_human_inloop_record import human_inloop_record
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.scripts.recording_remote_policy import RemotePolicyRecordConfig
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import InferenceTimeRTCConfig, RTCRecoveryConfig


@dataclass
class RTCRolloutConfig(RecordConfig):
    """Keep model/GPU/RTC selection on the recording client.

    dataset.fps is both the control rate and the stored sampling rate. The
    regular human-in-loop state machine handles leader mirroring and outcomes.
    """

    server_address: str = "10.61.5.1:8110"
    pretrained_name_or_path: str | None = None
    policy_type: str = "pi05"
    policy_device: str = "cuda:0"
    actions_per_chunk: int = 50
    connect_timeout_s: float = 10.0
    rpc_timeout_s: float = 360.0
    rtc: InferenceTimeRTCConfig = field(default_factory=InferenceTimeRTCConfig)
    allow_late_splice: bool = False
    action_sg_window: int = 0
    recovery: RTCRecoveryConfig = field(default_factory=lambda: RTCRecoveryConfig(enabled=True))

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError("RTC rollout requires teleop for human takeover")
        if self.dataset.storage != "pkl" or self.dataset.push_to_hub:
            raise ValueError("RTC rollout requires dataset.storage=pkl and dataset.push_to_hub=false")
        self.remote_policy = RemotePolicyRecordConfig(
            enable=True,
            backend="rtc",
            server_address=self.server_address,
            pretrained_name_or_path=self.pretrained_name_or_path or "",
            policy_type=self.policy_type,
            policy_device=self.policy_device,
            actions_per_chunk=self.actions_per_chunk,
            connect_timeout_s=self.connect_timeout_s,
            rpc_timeout_s=self.rpc_timeout_s,
            rtc=self.rtc,
            allow_late_splice=self.allow_late_splice,
            action_sg_window=self.action_sg_window,
            recovery=self.recovery,
        )
        super().__post_init__()


@parser.wrap()
def rollout(cfg: RTCRolloutConfig):
    # The record parser accepts its exact config type when called in Python.
    record_cfg = RecordConfig(**{item.name: getattr(cfg, item.name) for item in fields(RecordConfig)})
    return human_inloop_record(record_cfg)


def main():
    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
