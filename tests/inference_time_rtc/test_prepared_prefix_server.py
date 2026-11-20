"""Exercise the PI05 RTC server dispatch with synthetic observations."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.inference_time_rtc.configs import InferenceTimeRTCConfig, RTCServerConfig
from lerobot.inference_time_rtc.policy_server import RTCPolicyServer
from lerobot.inference_time_rtc.protocol import RTCObservation, SessionInstructions
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import ACTION, OBS_STR


@pytest.mark.parametrize("debug", [False, True])
def test_pi05_server_prepares_only_guided_requests_with_debug_disabled(debug):
    rtc = InferenceTimeRTCConfig(execution_horizon=28, inference_delay=22, debug=debug)
    policy = Mock(
        config=PI05Config(
            device="cpu",
            max_action_dim=3,
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        ),
        rtc_processor=RTCProcessor(rtc),
    )
    policy.predict_action_chunk.return_value = torch.zeros(1, 50, 2)
    identity = Mock(side_effect=lambda value: value)
    service = RTCPolicyServer(
        RTCServerConfig("unused", policy_device="cpu"), components=(policy, identity, identity)
    )
    service._configure_model(rtc)
    features = hw_to_dataset_features({"j1": float, "j2": float}, OBS_STR, use_video=False)
    instructions = SessionInstructions(features, ("j1", "j2"), 30, rtc)

    for request_id, prefix_steps in enumerate((0, 21, 28, 35)):
        prefix = np.full((prefix_steps, 2), 0.25, dtype=np.float32) if prefix_steps else None
        request = RTCObservation(request_id, 0, {"j1": 0.0, "j2": 0.0}, prefix, rtc.inference_delay)
        actions = service._predict(request, instructions)
        assert actions.original_actions.shape == (50, 2)
        kwargs = policy.predict_action_chunk.call_args.kwargs

        if prefix_steps and not debug:
            assert set(kwargs) == {"prepared_rtc_prefix"}
            prepared = kwargs["prepared_rtc_prefix"]
            assert prepared.actions.shape == (1, 50, 3)
            assert prepared.weights.shape == (1, 50, 1)
            assert prepared.actions.device.type == "cpu"
            torch.testing.assert_close(prepared.actions[0, :prefix_steps, :2], torch.from_numpy(prefix))
        else:
            assert "prepared_rtc_prefix" not in kwargs
            assert kwargs["execution_horizon"] == rtc.execution_horizon
            assert kwargs["inference_delay"] == rtc.inference_delay
            if prefix_steps:
                torch.testing.assert_close(kwargs["prev_chunk_left_over"], torch.from_numpy(prefix))
            else:
                assert kwargs["prev_chunk_left_over"] is None
