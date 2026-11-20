"""Populate the RTC pi05 compiler cache in a separate process without a robot."""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import draccus
import numpy as np

from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR

from .configs import InferenceTimeRTCConfig, RTCServerConfig
from .protocol import RTCObservation, SessionInstructions

if TYPE_CHECKING:
    from .policy_server import RTCPolicyServer


@dataclass
class CompileWarmupConfig:
    pretrained_name_or_path: str
    policy_device: str = "cuda:0"
    policy_dtype: str = "bfloat16"
    compile_model: bool = True
    compile_mode: str = "default"
    tokenizer_name_or_path: str | None = None
    task: str = ""
    actions_per_chunk: int | None = None
    rtc: InferenceTimeRTCConfig = field(default_factory=InferenceTimeRTCConfig)
    prefix_steps: list[int] = field(default_factory=list)
    repeats: int = 3

    def __post_init__(self):
        if self.repeats < 1:
            raise ValueError("repeats must be at least one")
        if any(length < 1 for length in self.prefix_steps):
            raise ValueError("prefix_steps must contain only positive lengths; the first chunk is automatic")


logger = logging.getLogger(__name__)


def warmup(
    service: "RTCPolicyServer",
    config: CompileWarmupConfig,
    *,
    is_active: Callable[[], bool] | None = None,
):
    count = service.chunk_size if config.actions_per_chunk is None else config.actions_per_chunk
    if count > service.chunk_size:
        raise ValueError("actions_per_chunk exceeds the model chunk size")
    rtc = config.rtc
    rtc.validate_chunk_size(count)
    if rtc.auto_delay:
        logger.info(
            "Warmup uses inference_delay=%d; automatic delay may select other values", rtc.inference_delay
        )
    prefixes = config.prefix_steps or [rtc.queue_threshold, max(1, rtc.execution_horizon - 1)]
    if any(length > count for length in prefixes):
        raise ValueError("prefix_steps exceed the requested action chunk length")

    policy_config = service.policy.config
    state_names = [f"joint{i}.pos" for i in range(policy_config.input_features[OBS_STATE].shape[0])]
    hardware_features = dict.fromkeys(state_names, float)
    observation = dict.fromkeys(state_names, 0.0)
    for key, feature in policy_config.image_features.items():
        channels, height, width = feature.shape
        camera = key.removeprefix(f"{OBS_IMAGES}.")
        shape = (height, width, channels)
        hardware_features[camera] = shape
        observation[camera] = np.zeros(shape, dtype=np.uint8)
    observation["task"] = config.task
    features = hw_to_dataset_features(hardware_features, OBS_STR, use_video=False)
    actions = tuple(f"joint{i}.pos" for i in range(service.action_dim))
    instructions = SessionInstructions(features, actions, 30.0, rtc, count, policy_device=service.device)
    service._configure_model(rtc)

    first = None
    for length in dict.fromkeys([0, *prefixes]):
        prefix = None if length == 0 else first.original_actions[-length:]
        logger.info("RTC offline warmup starting: prefix_steps=%d repeats=%d", length, config.repeats)
        samples = []
        for index in range(config.repeats):
            if is_active is not None and not is_active():
                raise TimeoutError("RTC warmup cancelled before the next inference")
            request = RTCObservation(index, 0, observation, prefix, rtc.inference_delay)
            started = time.perf_counter()
            result = service._predict(request, instructions)
            samples.append((time.perf_counter() - started) * 1000)
            if first is None:
                first = result
        logger.info(
            "RTC offline warmup finished: prefix_steps=%d pipeline_ms=%s",
            length,
            [round(x, 1) for x in samples],
        )


@draccus.wrap()
def main(config: CompileWarmupConfig):
    from .policy_server import RTCPolicyServer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    server_config = RTCServerConfig(
        pretrained_name_or_path=config.pretrained_name_or_path,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        policy_dtype=config.policy_dtype,
        compile_model=config.compile_model,
        compile_mode=config.compile_mode,
    )
    device = RTCPolicyServer._resolve_device(config.policy_device)
    logger.info(
        "RTC offline warmup: device=%s dtype=%s compile=%s mode=%s cache=%s",
        device,
        config.policy_dtype,
        config.compile_model,
        config.compile_mode,
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", "PyTorch default"),
    )
    service = RTCPolicyServer(server_config)
    service._set_policy_device(device)
    warmup(service, config)


if __name__ == "__main__":
    main()
