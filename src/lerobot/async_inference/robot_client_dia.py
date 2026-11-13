# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.processor import RobotAction
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> RobotAction:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        # Monotonic phase timings only; never record images or change control flow.
        trace = {"started_ns": time.perf_counter_ns(), "thread_id": threading.get_ident()}
        stage = "capture"
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            trace["capture_end_ns"] = time.perf_counter_ns()
            trace["capture_ms"] = (trace["capture_end_ns"] - trace["started_ns"]) / 1e6
            raw_observation["task"] = task

            stage = "latest_action_lock"
            lock_start_ns = time.perf_counter_ns()
            with self.latest_action_lock:
                trace["latest_action_lock_wait_ms"] = (time.perf_counter_ns() - lock_start_ns) / 1e6
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            stage = "action_queue_lock"
            lock_start_ns = time.perf_counter_ns()
            with self.action_queue_lock:
                trace["action_queue_lock_wait_ms"] = (time.perf_counter_ns() - lock_start_ns) / 1e6
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            trace.update(timestep=observation.get_timestep(), queue_size=current_queue_size,
                         must_go=observation.must_go)
            stage = "send_observation"
            trace["send_start_ns"] = time.perf_counter_ns()
            _ = self.send_observation(observation)
            trace["send_end_ns"] = time.perf_counter_ns()
            trace["send_observation_ms"] = (trace["send_end_ns"] - trace["send_start_ns"]) / 1e6
            trace["send_ok"] = bool(_)
            # This includes pickle serialization and synchronous RPC/server handling.
            # It is not a measurement of pure network latency.
            stage = "post_send"

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            stage = "complete"
            return raw_observation

        except Exception as e:
            trace["error_type"] = type(e).__name__
            self.logger.error(f"Error in observation sender: {e}")

        finally:
            trace["finished_ns"] = time.perf_counter_ns()
            trace["total_ms"] = (trace["finished_ns"] - trace["started_ns"]) / 1e6
            trace["stage"] = stage
def async_client(cfg: RobotClientConfig):
                try:
                    self.diag.event("observation_timing", **trace)
                except Exception:
                    # Diagnostics must not mask the existing result/error handling.
                    self.logger.debug("Could not enqueue observation timing")
    client = RobotClient(cfg)

    if client.start():

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig, diag_output: Optional[str] = None):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")
        ),
    )
    client = RobotClient(cfg, diag=diag)
    async_client()  # run the client
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")
    finally:
        session_dir = diag.close()
        client.logger.info(f"Diagnostic logs saved to {session_dir}")
        try:
            analyze_session(str(session_dir))
        except Exception as e:
            client.logger.error(f"Diagnostics analysis failed: {e}")



# Collection-only copies. Original classes/functions above remain unchanged.
# The _dia subclasses add timings and preserve synchronous calls and queue policy.
class PiperVLADiagnosticsDia(PiperVLADiagnostics):
    def send_joint_dia(self, arm_label: str, piper: Any, q_sdk: Any, joint_ctrl_fn: Callable[..., Any]) -> Any:
        """Replace one `piper.JointCtrl(*q_sdk)` call, forwarding the command unchanged.

            `q_sdk` must be the six 0.001-degree integer values already converted and validated by
            the original program. chunk_id/step_id/queue_size come from `current_context`, set by
            the control loop right before robot.send_action().
            """
        q_values = tuple(q_sdk)
        if len(q_values) != 6:
            raise ValueError('JointCtrl expects exactly six joint positions')
        t_send_ns = time.perf_counter_ns()
        with self._send_lock:
            previous_by_arm = getattr(self, '_last_send_ns_by_arm_dia', None)
            if previous_by_arm is None:
                previous_by_arm = self._last_send_ns_by_arm_dia = {}
            previous_ns = previous_by_arm.get(arm_label)
            send_dt_ms = (t_send_ns - previous_ns) / 1e6 if previous_ns is not None else None
            previous_by_arm[arm_label] = t_send_ns
        chunk_id, step_id, queue_size = self.current_context
        error = None
        try:
            return joint_ctrl_fn(*q_values)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            t_return_ns = time.perf_counter_ns()
            self._push('commands', {'arm': str(arm_label), 't_send_ns': t_send_ns, 't_return_ns': t_return_ns, 'sdk_call_ms': (t_return_ns - t_send_ns) / 1000000.0, 'send_dt_ms': send_dt_ms, 'chunk_id': int(chunk_id), 'step_id': int(step_id), 'queue_size': int(queue_size), 'q_sent_sdk': [int(v) for v in q_values], 'q_sent_rad': [float(v) * SDK_TO_RAD for v in q_values], 'send_error': error})

    send_joint = send_joint_dia


class RobotClientDia(RobotClient):
    def _emit_timing_dia(self, event_name, **metadata):
        # Non-blocking diagnostic queue; never write files in the control thread.
        if self.diag is None:
            return
        try:
            local = getattr(self, "_dia_local", None)
            self.diag.event(event_name, thread_id=threading.get_ident(),
                            thread_name=threading.current_thread().name,
                            cycle_id=getattr(local, "cycle_id", None), **metadata)
        except Exception:
            pass  # Collection failure must not change control results/exceptions.

    def _call_timed_dia(self, phase, fn, *args, _dia_metadata=None, **kwargs):
        started_ns = time.perf_counter_ns()
        result = None
        success = False
        error_type = None
        try:
            result = fn(*args, **kwargs)
            success = True
            return result
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            ended_ns = time.perf_counter_ns()
            try:
                metadata = dict(_dia_metadata or {})
                if phase == "serialize_observation" and isinstance(result, bytes):
                    metadata["payload_bytes"] = len(result)
                if phase == "get_actions_rpc" and result is not None:
                    metadata["payload_bytes"] = len(result.data)
                if phase == "deserialize_actions" and success:
                    metadata["action_count"] = len(result)
                    if result:
                        metadata["first_timestep"] = result[0].get_timestep()
                        metadata["last_timestep"] = result[-1].get_timestep()
                        metadata["source_obs_wall_ns"] = int(round(result[0].get_timestamp() * 1e9))
                if phase in ("serialize_observation", "send_observation_total") and args:
                    metadata["timestep"] = args[0].get_timestep()
                    metadata["source_obs_wall_ns"] = int(round(args[0].get_timestamp() * 1e9))
                if phase == "queue_update" and args:
                    metadata["incoming_count"] = len(args[0])
                    if args[0]:
                        metadata["first_timestep"] = args[0][0].get_timestep()
                        metadata["last_timestep"] = args[0][-1].get_timestep()
                if phase == "camera_read" and success:
                    metadata["image_bytes"] = int(getattr(result, "nbytes", 0))
                self._emit_timing_dia("phase_timing_dia", phase=phase,
                                      started_ns=started_ns, ended_ns=ended_ns,
                                      duration_ms=(ended_ns-started_ns)/1e6,
                                      success=success, error_type=error_type, **metadata)
            except Exception:
                pass

    def _lock_timed_dia(self, lock, lock_name):
        from contextlib import contextmanager

        @contextmanager
        def _lock_span_dia():
            started_ns = time.perf_counter_ns()
            acquired_ns = None
            releasing_ns = None
            state = {}
            try:
                with lock:
                    acquired_ns = time.perf_counter_ns()
                    if lock_name == "action_queue_lock":
                        state["queue_size_before"] = self.action_queue.qsize()
                        state["queue_id_before"] = id(self.action_queue)
                    try:
                        yield
                    finally:
                        releasing_ns = time.perf_counter_ns()
                        if lock_name == "action_queue_lock":
                            state["queue_size_after"] = self.action_queue.qsize()
                            state["queue_id_after"] = id(self.action_queue)
            finally:
                ended_ns = time.perf_counter_ns()
                self._emit_timing_dia("lock_timing_dia", lock=lock_name,
                                      started_ns=started_ns, acquired_ns=acquired_ns,
                                      ended_ns=ended_ns,
                                      wait_ms=(acquired_ns-started_ns)/1e6 if acquired_ns else None,
                                      held_ms=(releasing_ns-acquired_ns)/1e6 if releasing_ns else None,
                                      **state)
        return _lock_span_dia()

    def _setup_diagnostics_dia(self):
        """Start diagnostics, wrap each piper arm's JointCtrl for command logging."""
        self._dia_local = threading.local()
        if self.diag is None:
            return
        self._wall_minus_mono_ns = time.time_ns() - time.perf_counter_ns()
        self._diag_chunk_id = -1
        self._diag_chunk_starts: list[tuple[int, int]] = []
        self._diag_chunk_lock = threading.Lock()
        arms = self._find_piper_arms(self.robot)
        diag = self.diag
        for arm_label, sdk in arms.items():
            original_joint_ctrl = sdk.JointCtrl

            def make_wrapper(label, sdk_obj, joint_ctrl_fn):

                def wrapped_joint_ctrl(*q_values):
                    return diag.send_joint(label, sdk_obj, q_values, joint_ctrl_fn)
                return wrapped_joint_ctrl
            try:
                sdk.JointCtrl = make_wrapper(arm_label, sdk, original_joint_ctrl)
            except (AttributeError, TypeError) as e:
                self.logger.warning(f"Could not wrap JointCtrl of '{arm_label}' arm: {e}")
        self.diag.start(arms)
        self.diag.event('client_start', server_address=self.config.server_address, policy_type=self.config.policy_type, pretrained_name_or_path=self.config.pretrained_name_or_path, actions_per_chunk=self.config.actions_per_chunk, task=self.config.task, piper_arms=sorted(arms.keys()))
        self.logger.info(f'Diagnostics started: {self.diag.session_dir} (arms: {sorted(arms.keys())})')
        if self.diag is not None:
            arms = [(side, getattr(self.robot, side + '_arm', None)) for side in ('left', 'right')]
            if not any((arm is not None for _, arm in arms)):
                arms = [('arm', self.robot)]

            def _make_camera_reader_dia(label, reader):

                def _camera_read_dia(*args, **kwargs):
                    return self._call_timed_dia('camera_read', reader, *args, _dia_metadata={'camera': label}, **kwargs)
                return _camera_read_dia
            for side, arm in arms:
                for name, camera in getattr(arm, 'cameras', {}).items():
                    try:
                        camera.async_read = _make_camera_reader_dia(side + '/' + name, camera.async_read)
                    except (AttributeError, TypeError):
                        self._emit_timing_dia('camera_timing_unavailable_dia', camera=side + '/' + name)
            self._emit_timing_dia('collection_enabled_dia', collection_only=True, queue_strategy_changed=False, thread_model_changed=False, rpc_timing_note='includes transfer and server handling; not pure network latency')

    def send_observation_dia(self, obs: TimedObservation) -> bool:
        """Send observation to the policy server.
            Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError('Client not running. Run RobotClient.start() before sending observations.')
        if not isinstance(obs, TimedObservation):
            raise ValueError('Input observation needs to be a TimedObservation!')
        start_time = time.perf_counter()
        observation_bytes = self._call_timed_dia('serialize_observation', pickle.dumps, obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f'Observation serialization time: {serialize_time:.6f}s')
        try:
            observation_iterator = send_bytes_in_chunks(observation_bytes, services_pb2.Observation, log_prefix='[CLIENT] Observation', silent=True)
            _ = self._call_timed_dia('send_observations_rpc', self.stub.SendObservations, observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f'Sent observation #{obs_timestep} | ')
            return True
        except grpc.RpcError as e:
            self.logger.error(f'Error sending observation #{obs.get_timestep()}: {e}')
            return False

    def _aggregate_action_queues_dia(self, incoming_actions: list[TimedAction], aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None=None):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:

            def aggregate_fn(x1, x2):
                return x2
        future_action_queue = Queue()
        with self._lock_timed_dia(self.action_queue_lock, 'action_queue_lock'):
            internal_queue = self.action_queue.queue
        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}
        for new_action in incoming_actions:
            with self._lock_timed_dia(self.latest_action_lock, 'latest_action_lock'):
                latest_action = self.latest_action
            if new_action.get_timestep() <= latest_action:
                continue
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue
            future_action_queue.put(TimedAction(timestamp=new_action.get_timestamp(), timestep=new_action.get_timestep(), action=aggregate_fn(current_action_queue[new_action.get_timestep()], new_action.get_action())))
        with self._lock_timed_dia(self.action_queue_lock, 'action_queue_lock'):
            self.action_queue = future_action_queue

    def receive_actions_dia(self, verbose: bool=False):
        """Receive actions from the policy server"""
        self.start_barrier.wait()
        self.logger.info('Action receiving thread starting')
        while self.running:
            self._dia_local.cycle_id = time.perf_counter_ns()
            try:
                actions_chunk = self._call_timed_dia('get_actions_rpc', self.stub.GetActions, services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue
                receive_time = time.time()
                deserialize_start = time.perf_counter()
                timed_actions = self._call_timed_dia('deserialize_actions', pickle.loads, actions_chunk.data)
                deserialize_time = time.perf_counter() - deserialize_start
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                if self.diag is not None and len(timed_actions) > 0:
                    first_action_ts_ns = int(round(timed_actions[0].get_timestamp() * 1000000000.0))
                    obs_mono_ns = first_action_ts_ns - self._wall_minus_mono_ns
                    with self._lock_timed_dia(self._diag_chunk_lock, '_diag_chunk_lock'):
                        self._diag_chunk_id += 1
                        chunk_id = self._diag_chunk_id
                        self._diag_chunk_starts.append((chunk_id, timed_actions[0].get_timestep()))
                    self._call_timed_dia('raw_chunk_logging', self.diag.log_vla, chunk_id, obs_mono_ns, time.perf_counter_ns(), action_raw=[a.get_action() for a in timed_actions], obs_time_ns=obs_mono_ns, first_timestep=timed_actions[0].get_timestep(), last_timestep=timed_actions[-1].get_timestep(), note='client-side end-to-end (obs capture -> chunk received); policy inference runs on the server')
                if len(timed_actions) > 0 and verbose:
                    with self._lock_timed_dia(self.latest_action_lock, 'latest_action_lock'):
                        latest_action = self.latest_action
                    self.logger.debug(f'Current latest action: {latest_action}')
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]
                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000
                    self.logger.info(f'Received action chunk for step #{first_action_timestep} | Latest action: #{latest_action} | Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | Network latency (server->client): {server_to_client_latency:.2f}ms | Deserialization time: {deserialize_time * 1000:.2f}ms')
                start_time = time.perf_counter()
                self._call_timed_dia('queue_update', self._aggregate_action_queues, timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time
                self.must_go.set()
                if verbose:
                    new_size, new_timesteps = self._inspect_action_queue()
                    with self._lock_timed_dia(self.latest_action_lock, 'latest_action_lock'):
                        latest_action = self.latest_action
                    self.logger.info(f'Latest action: {latest_action} | Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}')
                    self.logger.debug(f'Queue update complete ({queue_update_time:.6f}s) | Before: {old_size} items | After: {new_size} items | ')
            except grpc.RpcError as e:
                self.logger.error(f'Error receiving actions: {e}')

    def actions_available_dia(self):
        """Check if there are actions available in the queue"""
        with self._lock_timed_dia(self.action_queue_lock, 'action_queue_lock'):
            return not self.action_queue.empty()

    def control_loop_action_dia(self, verbose: bool=False) -> RobotAction:
        """Reading and performing actions in local queue"""
        get_start = time.perf_counter()
        with self._lock_timed_dia(self.action_queue_lock, 'action_queue_lock'):
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start
        if self.diag is not None:
            with self._lock_timed_dia(self.action_queue_lock, 'action_queue_lock'):
                remaining_actions = self.action_queue.qsize()
            timestep = timed_action.get_timestep()
            with self._lock_timed_dia(self._diag_chunk_lock, '_diag_chunk_lock'):
                owning_chunk = -1
                for cid, first_timestep in self._diag_chunk_starts:
                    if first_timestep <= timestep:
                        owning_chunk = cid
                    else:
                        break
            self.diag.current_context = (owning_chunk, timestep, remaining_actions)
        _performed_action = self._call_timed_dia('robot_send_action', self.robot.send_action, self._call_timed_dia('action_tensor_conversion', self._action_tensor_to_action_dict, timed_action.get_action()))
        with self._lock_timed_dia(self.latest_action_lock, 'latest_action_lock'):
            self.latest_action = timed_action.get_timestep()
        if verbose:
            with self._lock_timed_dia(self.action_queue_lock, 'action_queue_lock'):
                current_queue_size = self.action_queue.qsize()
            self.logger.debug(f'Ts={timed_action.get_timestamp()} | Action #{timed_action.get_timestep()} performed | Queue size: {current_queue_size}')
            self.logger.debug(f'Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}')
        return _performed_action

    def _ready_to_send_observation_dia(self):
        """Flags when the client is ready to send an observation"""
        with self._lock_timed_dia(self.action_queue_lock, 'action_queue_lock'):
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation_dia(self, task: str, verbose: bool=False) -> RawObservation:
        trace = {'started_ns': time.perf_counter_ns(), 'thread_id': threading.get_ident()}
        stage = 'capture'
        try:
            start_time = time.perf_counter()
            raw_observation: RawObservation = self._call_timed_dia('capture_observation', self.robot.get_observation)
            trace['capture_end_ns'] = time.perf_counter_ns()
            trace['capture_ms'] = (trace['capture_end_ns'] - trace['started_ns']) / 1000000.0
            raw_observation['task'] = task
            stage = 'latest_action_lock'
            lock_start_ns = time.perf_counter_ns()
            with self._lock_timed_dia(self.latest_action_lock, 'latest_action_lock'):
                trace['latest_action_lock_wait_ms'] = (time.perf_counter_ns() - lock_start_ns) / 1000000.0
                latest_action = self.latest_action
            observation = TimedObservation(timestamp=time.time(), observation=raw_observation, timestep=max(latest_action, 0))
            obs_capture_time = time.perf_counter() - start_time
            stage = 'action_queue_lock'
            lock_start_ns = time.perf_counter_ns()
            with self._lock_timed_dia(self.action_queue_lock, 'action_queue_lock'):
                trace['action_queue_lock_wait_ms'] = (time.perf_counter_ns() - lock_start_ns) / 1000000.0
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()
            trace.update(timestep=observation.get_timestep(), queue_size=current_queue_size, must_go=observation.must_go)
            stage = 'send_observation'
            trace['send_start_ns'] = time.perf_counter_ns()
            _ = self._call_timed_dia('send_observation_total', self.send_observation, observation)
            trace['send_end_ns'] = time.perf_counter_ns()
            trace['send_observation_ms'] = (trace['send_end_ns'] - trace['send_start_ns']) / 1000000.0
            trace['send_ok'] = bool(_)
            stage = 'post_send'
            self.logger.debug(f'QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})')
            if observation.must_go:
                self.must_go.clear()
            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
                self.logger.info(f"Obs #{observation.get_timestep()} | Avg FPS: {fps_metrics['avg_fps']:.2f} | Target: {fps_metrics['target_fps']:.2f}")
                self.logger.debug(f'Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s')
            stage = 'complete'
            return raw_observation
        except Exception as e:
            trace['error_type'] = type(e).__name__
            self.logger.error(f'Error in observation sender: {e}')
        finally:
            trace['finished_ns'] = time.perf_counter_ns()
            trace['total_ms'] = (trace['finished_ns'] - trace['started_ns']) / 1000000.0
            trace['stage'] = stage
            if self.diag is not None:
                try:
                    self.diag.event('observation_timing', **trace)
                except Exception:
                    self.logger.debug('Could not enqueue observation timing')

    def control_loop_dia(self, task: str, verbose: bool=False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        self.start_barrier.wait()
        self.logger.info('Control loop thread starting')
        _performed_action = None
        _captured_observation = None
        while self.running:
            self._dia_local.cycle_id = time.perf_counter_ns()
            control_loop_start = time.perf_counter()
            'Control loop: (1) Performing actions, when available'
            if self._call_timed_dia('actions_available_check', self.actions_available):
                _performed_action = self._call_timed_dia('control_action_total', self.control_loop_action, verbose)
            'Control loop: (2) Streaming observations to the remote policy server'
            if self._call_timed_dia('observation_ready_check', self._ready_to_send_observation):
                _captured_observation = self._call_timed_dia('control_observation_total', self.control_loop_observation, task, verbose)
            self.logger.debug(f'Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}')
            self._call_timed_dia('control_sleep', time.sleep, max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))
        return (_captured_observation, _performed_action)

    _setup_diagnostics = _setup_diagnostics_dia
    send_observation = send_observation_dia
    _aggregate_action_queues = _aggregate_action_queues_dia
    receive_actions = receive_actions_dia
    actions_available = actions_available_dia
    control_loop_action = control_loop_action_dia
    _ready_to_send_observation = _ready_to_send_observation_dia
    control_loop_observation = control_loop_observation_dia
    control_loop = control_loop_dia


@draccus.wrap()
def async_client_dia(cfg: RobotClientConfig, diag_output: Optional[str]=None):
    logging.info(pformat(asdict(cfg)))
    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f'Robot {cfg.robot.type} not yet supported!')
    diag = PiperVLADiagnosticsDia(output_root=diag_output or './piper_vla_logs', feedback_hz=100.0, mode_note=f'async_inference client | policy={cfg.policy_type} | model={cfg.pretrained_name_or_path} | actions_per_chunk={cfg.actions_per_chunk} | server={cfg.server_address}')
    client = RobotClientDia(cfg, diag=diag)
    try:
        if not client.start():
            client.logger.error('Could not connect to policy server; exiting')
            return
        client.logger.info('Starting action receiver thread...')
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info('Client stopped')
    finally:
        session_dir = diag.close()
        client.logger.info(f'Diagnostic logs saved to {session_dir}')
        try:
            analyze_session(str(session_dir))
        except Exception as e:
            client.logger.error(f'Diagnostics analysis failed: {e}')


if __name__ == "__main__":
    diag_output, diag_analyze_only = _extract_diag_args()
    if diag_analyze_only is not None:
        # Offline analysis of an existing session; no draccus config needed.
        analyze_session(diag_analyze_only)
    else:
        async_client_dia(diag_output=diag_output)  # collection-only _dia copies
