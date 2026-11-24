"""Run robot commands independently from RTC observation capture and RPCs."""

import logging
import math
import pickle  # nosec: trusted-network transport, same as async_inference
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

import draccus
import grpc
import numpy as np

from lerobot.async_inference.helpers import map_robot_keys_to_lerobot_features
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    bi_piper_follower,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    piper_follower,
    so_follower,
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .action_buffer import RTCActionBuffer, RTCActionReservation, RTCContinuityError
from .configs import RTCClientConfig
from .logging_utils import configure_logging, log_action_event, log_actions_to_file
from .observation_codec import check_observation_image_codec, encode_observation_images
from .protocol import (
    CHECKPOINT_PATH_HEADER,
    CHECKPOINT_SELECTION_HEADER,
    CURRENT_POSE_RECOVERY_HEADER,
    POLICY_TYPE_HEADER,
    PROTOCOL_VERSION,
    RLT_HEADER,
    RLT_OUTPUT_SG_HEADER,
    RLT_PHASE_HEADER,
    SESSION_HEADER,
    RTCActionChunk,
    RTCObservation,
    SessionInstructions,
    validate_rlt_stage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RecoveryCandidate:
    chunk: RTCActionChunk
    source_request_id: int
    prefix_steps: int
    captured_ns: int


class RTCRobotClient:
    def __init__(self, config: RTCClientConfig, *, robot=None):
        config.__post_init__()
        self.config = config
        self.robot = robot if robot is not None else make_robot_from_config(config.robot)
        self._action_names = tuple(self.robot.action_features)
        self._dispatch_lock = threading.Lock()
        self._last_verified_request_id: int | None = None
        self._pending_recovery: _RecoveryCandidate | None = None
        self._joint_columns = [
            index
            for index, name in enumerate(self._action_names)
            if name.endswith(".pos") and not name.endswith("gripper.pos")
        ]
        self._joint_names = tuple(self._action_names[index] for index in self._joint_columns)
        if config.recovery.enabled:
            if not self._action_names or any(
                not name.endswith(".pos") or name not in self.robot.observation_features
                for name in self._action_names
            ):
                raise ValueError("RTC recovery requires a position observation for each commanded position")
            if not self._joint_columns:
                raise ValueError("RTC recovery requires joint position actions")
        if config.action_sg_window and not self._joint_columns:
            raise ValueError("action_sg_window requires joint position actions ending in .pos")
        self.buffer = RTCActionBuffer(sg_window=config.action_sg_window, joint_columns=tuple(self._joint_columns))
        self.stop_event = threading.Event()
        self.refill_event = threading.Event()
        self.refill_event.set()
        self.worker = None
        self.error = None
        self.connected = False
        self.connection_attempted = False
        self.session_started = False
        self.server_checkpoint = None
        self.request_id = 0
        self._rlt_phase: bool | None = None
        self._rlt_epoch = 0
        self._phase_refresh = False
        self.latencies = deque(maxlen=config.rtc.latency_window)
        self.metadata = ((SESSION_HEADER, uuid.uuid4().hex),)
        self.channel = grpc.insecure_channel(
            config.server_address, options=grpc_channel_options(enable_retries=False)
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

    def start(self):
        grpc.channel_ready_future(self.channel).result(timeout=self.config.connect_timeout_s)
        _, call = self.stub.Ready.with_call(services_pb2.Empty(), timeout=self.config.connect_timeout_s)
        ready = dict(call.initial_metadata())
        if ready.get("rtc-protocol") != PROTOCOL_VERSION:
            raise RuntimeError(
                "This client requires lerobot.inference_time_rtc.policy_server on a separate port"
            )
        check_observation_image_codec(self.config.rtc.observation_image_codec, ready)
        if self.config.rlt.enabled and ready.get(RLT_HEADER) != "1":
            raise RuntimeError("Restart the updated RTC server to use direct VLA RLT")
        if self.config.rlt.enabled and ready.get(RLT_PHASE_HEADER) != "1":
            raise RuntimeError("Restart the updated RTC server to use RLT step intervals")
        if self.config.rlt.enabled and self.config.action_sg_window and ready.get(RLT_OUTPUT_SG_HEADER) != "1":
            raise RuntimeError("Restart the updated RTC server to use RLT with client output S-G")
        if self.config.recovery.enabled and ready.get(CURRENT_POSE_RECOVERY_HEADER) != "1":
            raise RuntimeError("Restart the updated RTC server to use current-pose recovery")
        checkpoint = self.config.pretrained_name_or_path
        if checkpoint is not None:
            if ready.get(CHECKPOINT_SELECTION_HEADER) != "1":
                raise RuntimeError(
                    "This RTC server needs --checkpoint_root to accept a client-selected model"
                )
        else:
            available = int(ready["rtc-chunk-size"])
            if not available:
                raise ValueError("This RTC server needs the client flag --pretrained_name_or_path")
            requested = available if self.config.actions_per_chunk is None else self.config.actions_per_chunk
            if requested > available:
                raise ValueError(f"actions_per_chunk ({requested}) exceeds model chunk_size ({available})")
            self.config.rtc.validate_chunk_size(requested)
        if self.config.policy_type is not None and ready.get(POLICY_TYPE_HEADER) != self.config.policy_type:
            raise ValueError(
                f"Expected server policy_type={self.config.policy_type}; "
                f"server advertised {ready.get(POLICY_TYPE_HEADER, 'no policy type')}"
            )
        # Robot feature metadata is available from its configuration before
        # connecting. Prepare the server GPU before activating hardware.
        instructions = SessionInstructions(
            lerobot_features=map_robot_keys_to_lerobot_features(self.robot),
            action_names=tuple(self.robot.action_features),
            fps=self.config.fps,
            rtc=self.config.rtc,
            actions_per_chunk=self.config.actions_per_chunk,
            policy_device=self.config.policy_device,
            pretrained_name_or_path=checkpoint,
            policy_type=self.config.policy_type,
            rlt=self.config.rlt if self.config.rlt.enabled else None,
            action_sg_window=self.config.action_sg_window,
        )
        _, call = self.stub.SendPolicyInstructions.with_call(
            services_pb2.PolicySetup(data=pickle.dumps(instructions, protocol=pickle.HIGHEST_PROTOCOL)),
            metadata=self.metadata,
            timeout=self.config.rpc_timeout_s,
        )
        self.session_started = True
        server_metadata = dict(call.initial_metadata())
        self.chunk_size = int(server_metadata["rtc-chunk-size"])
        self.server_policy_device = server_metadata["rtc-policy-device"]
        if self.config.policy_device and self.server_policy_device != self.config.policy_device:
            raise RuntimeError("Server did not honor the client-selected policy_device")
        if checkpoint is not None:
            resolved = server_metadata.get(CHECKPOINT_PATH_HEADER)
            if not resolved:
                raise RuntimeError("Server did not acknowledge the client-selected checkpoint")
            self.server_checkpoint = resolved.decode("utf-8")
        self.config.rtc.validate_chunk_size(self.chunk_size)
        if not self.connected:
            self.connection_attempted = True
            self.robot.connect()
            self.connected = True
        logger.info(
            "RTC connected: server_address=%s server_policy_device=%s chunk=%s fps=%s rtc=%s",
            self.config.server_address,
            self.server_policy_device,
            self.chunk_size,
            self.config.fps,
            self.config.rtc,
        )
        if self.config.action_sg_window:
            logger.info("RTC action S-G requested_window=%d", self.config.action_sg_window)
        if self.config.rlt.enabled:
            logger.info("RLT mode: output_sg_window=%d; reference_source=raw RTC VLA", self.config.action_sg_window)
        if self.server_checkpoint is not None:
            logger.info("RTC server checkpoint: %s", self.server_checkpoint)

    def _delay_steps(self) -> int:
        delay = self.config.rtc.inference_delay
        if self.config.rtc.auto_delay and self.latencies:
            delay = math.ceil(float(np.quantile(self.latencies, 0.95)) * self.config.fps)
        if delay >= self.chunk_size:
            raise RuntimeError("End-to-end RTC latency exceeds the chunk duration; reduce fps or latency")
        return delay

    def _capture_observation(self):
        begin = time.monotonic_ns()
        observation = self.robot.get_observation()
        observation = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in observation.items()}
        observation["task"] = self.config.task
        return observation, begin, time.monotonic_ns()

    def _exchange_chunk(self, request: RTCObservation, begin: int, observation_ready: int):
        self.request_id += 1
        image_codec = self.config.rtc.observation_image_codec
        encode_started = time.monotonic_ns()
        request.observation = encode_observation_images(request.observation, image_codec)
        serialize_started = time.monotonic_ns()
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        send_started = time.monotonic_ns()
        self.stub.SendObservations(
            send_bytes_in_chunks(payload, services_pb2.Observation),
            metadata=self.metadata,
            timeout=self.config.rpc_timeout_s,
        )
        send_finished = time.monotonic_ns()
        get_request = services_pb2.Empty()
        get_started = time.monotonic_ns()
        response = self.stub.GetActions(
            get_request, metadata=self.metadata, timeout=self.config.rpc_timeout_s
        )
        get_finished = time.monotonic_ns()
        result = pickle.loads(response.data)  # nosec
        deserialized = time.monotonic_ns()
        if not isinstance(result, RTCActionChunk):
            raise ValueError("Server returned an incompatible RTC action payload")
        if result.request_id != request.request_id or result.start_step != request.start_step:
            raise ValueError("Server returned a mismatched RTC request ID or timeline origin")
        result.validate(len(self.robot.action_features))
        validate_rlt_stage(result, request, self.config.rlt, self.config.rtc.execution_horizon)
        if len(result.original_actions) != self.chunk_size:
            raise ValueError("Server returned an unexpected RTC chunk size")
        received = time.monotonic_ns()
        elapsed = (received - begin) / 1e9
        log_action_event(
            "rpc_timing",
            request_id=request.request_id,
            session_id=dict(self.metadata).get(SESSION_HEADER),
            observation_ms=(observation_ready - begin) / 1e6,
            request_setup_ms=(encode_started - observation_ready) / 1e6,
            image_codec=image_codec,
            image_encode_ms=(serialize_started - encode_started) / 1e6,
            serialize_ms=(send_started - serialize_started) / 1e6,
            observation_bytes=len(payload),
            send_observations_rpc_ms=(send_finished - send_started) / 1e6,
            rpc_handoff_ms=(get_started - send_finished) / 1e6,
            get_actions_rpc_ms=(get_finished - get_started) / 1e6,
            deserialize_response_ms=(deserialized - get_finished) / 1e6,
            response_validation_ms=(received - deserialized) / 1e6,
            total_ms=elapsed * 1000,
        )
        log_action_event(
            "action_chunk",
            request_id=result.request_id,
            start_step=result.start_step,
            received_monotonic_ns=received,
            estimated_delay=request.inference_delay,
            action_names=tuple(self.robot.action_features),
            original_actions=result.original_actions,
            processed_actions=result.processed_actions,
            **({"rlt_applied": result.rlt_applied, "rlt_action_mask": result.rlt_action_mask,
                "episode_step": request.episode_step,
                "rlt_reference_actions": result.rlt_reference_actions} if self.config.rlt.enabled else {}),
        )
        return result, elapsed

    def _update_rlt_phase(self, step: int):
        """Called under dispatch_lock; retain queued actions for RTC continuity."""
        if not self.config.rlt.enabled:
            return
        phase = self.config.rlt.active_at(step)
        if phase != self._rlt_phase:
            self._rlt_epoch += 1
            self._rlt_phase = phase
            self._phase_refresh = True
            self.refill_event.set()
            logger.info("RLT phase episode_step=%s enabled=%s interval=[%s,%s) epoch=%s queue=%s",
                        step, phase, self.config.rlt.start_step, self.config.rlt.end_step,
                        self._rlt_epoch, self.buffer.qsize())

    def request_chunk(self):
        if self.buffer.draining:
            return
        # Capture the origin before acquisition so latency includes sensing and both RPCs.
        with self._dispatch_lock:
            start_step, prefix, started = self.buffer.snapshot()
            self._update_rlt_phase(start_step)
            phase_epoch = self._rlt_epoch
            self._phase_refresh = False
        if self.config.recovery.enabled and started and prefix is None:
            self.stop_event.wait(1.0 / self.config.fps)
            return
        observation, begin, observation_ready = self._capture_observation()
        delay = self.config.rtc.inference_delay if prefix is None else self._delay_steps()
        request = RTCObservation(
            self.request_id, start_step, observation, prefix, delay, episode_step=start_step
        )
        result, elapsed = self._exchange_chunk(request, begin, observation_ready)
        if self.stop_event.is_set():
            return
        prefix_steps = 0 if prefix is None else len(prefix)
        effective_horizon = min(prefix_steps, self.config.rtc.execution_horizon) if started else None
        try:
            with self._dispatch_lock:
                if phase_epoch != self._rlt_epoch:
                    logger.info("Discarded RTC result request=%s from an earlier RLT phase", result.request_id)
                    self.refill_event.set()
                    return
                real_delay = self.buffer.merge(
                    result, prefix_horizon=effective_horizon, allow_late_splice=self.config.allow_late_splice,
                    drain_on_rejection=self.config.recovery.enabled,
                )
        except RTCContinuityError as exc:
            self._log_splice()
            if self.config.recovery.enabled:
                logger.warning(
                    "RTC continuation discarded request=%s queue=%s; waiting for queue exhaustion: %s",
                    request.request_id, self.buffer.qsize(), exc,
                )
                return
            raise
        except RuntimeError:
            self._log_splice()
            raise
        if real_delay is None:
            if self.buffer.motion_paused:
                logger.info(
                    "RTC inference-only result request=%s latency=%.3fs prefix_steps=%s",
                    request.request_id,
                    elapsed,
                    prefix_steps,
                )
            return
        self._log_splice()
        if self.config.action_sg_window:
            log_action_event(
                "filtered_action_chunk",
                request_id=result.request_id,
                first_chunk_index=real_delay,
                requested_window=self.config.action_sg_window,
                applied_window=self.buffer.applied_sg_window,
                action_names=tuple(self.robot.action_features),
                processed_actions=self.buffer.processed,
            )
        # First-call model warmup is not representative of steady-state latency.
        if started:
            self.latencies.append(elapsed)
        logger.info(
            "RTC request=%s latency=%.3fs estimated_delay=%s actual_delay=%s queue=%s",
            result.request_id,
            elapsed,
            delay,
            real_delay,
            self.buffer.qsize(),
        )
        logger.info(
            "RTC prefix request=%s prefix_steps=%s configured_horizon=%s effective_horizon=%s server_device=%s",
            result.request_id,
            prefix_steps,
            self.config.rtc.execution_horizon,
            min(prefix_steps, self.config.rtc.execution_horizon),
            self.server_policy_device,
        )
        if (
            self.config.allow_late_splice
            and effective_horizon is not None
            and real_delay >= effective_horizon
        ):
            logger.warning(
                "RTC late splice accepted request=%s actual_delay=%s effective_horizon=%s",
                result.request_id,
                real_delay,
                effective_horizon,
            )
        elif started and math.ceil(elapsed * self.config.fps) >= self.config.rtc.execution_horizon:
            logger.warning(
                "RTC measured latency reaches execution_horizon; review timing offline before further motion"
            )

    def _log_splice(self):
        splice = self.buffer.last_splice
        if splice is None:
            return
        delta = splice.overlap_delta
        index = int(np.argmax(np.abs(delta))) if delta is not None else None
        logger.info(
            "RTC splice request=%s actual_delay=%s effective_horizon=%s old_remaining=%s "
            "max_abs_overlap_delta=%s joint=%s signed_delta=%s (robot command units; same future slot)",
            splice.request_id,
            splice.actual_delay,
            splice.effective_horizon,
            splice.old_remaining,
            None if index is None else float(abs(delta[index])),
            None if index is None else tuple(self.robot.action_features)[index],
            None if index is None else float(delta[index]),
        )

    @staticmethod
    def _positions(values, names: tuple[str, ...]) -> np.ndarray | None:
        try:
            positions = np.array([float(values[name]) for name in names])
        except (KeyError, TypeError, ValueError):
            return None
        return positions if np.isfinite(positions).all() else None

    def _recovery_expired(self, captured_ns: int) -> bool:
        return (time.monotonic_ns() - captured_ns) / 1e9 > self.config.recovery.max_observation_age_s

    def _request_recovery(self):
        # Wait for an old command already reserved by control to finish before sensing.
        with self._dispatch_lock:
            start_step = self.buffer.paused_step()
            if self._pending_recovery is not None or start_step is None:
                return
            source_request_id = self._last_verified_request_id
            phase_epoch = self._rlt_epoch
        if source_request_id is None:
            logger.info("RTC recovery holding: last dispatched joint positions are unavailable or unverified")
            return

        observation, begin, observation_ready = self._capture_observation()
        positions = self._positions(observation, self._action_names)
        if positions is None or self._recovery_expired(begin) or self.stop_event.is_set():
            logger.info("RTC recovery holding: fresh positional observation is unavailable")
            return

        steps = self.config.recovery.prefix_steps
        # During hold the motion clock is stopped. Guide every row of the short
        # recovery prefix, including when the configured attention schedule is ZEROS.
        request = RTCObservation(
            self.request_id, start_step, observation, None, steps, recovery_prefix_steps=steps,
            episode_step=start_step,
        )
        logger.info(
            "RTC recovery request=%s source_request=%s prefix_steps=%s source=current_pose",
            request.request_id, source_request_id, steps,
        )
        result, _ = self._exchange_chunk(request, begin, observation_ready)
        if self.stop_event.is_set() or self._recovery_expired(begin):
            logger.info("RTC recovery holding: recovery result expired")
            return

        with self._dispatch_lock:
            if phase_epoch == self._rlt_epoch and self.buffer.motion_paused and not self.stop_event.is_set():
                self._pending_recovery = _RecoveryCandidate(result, source_request_id, steps, begin)

    def _commit_recovery(self) -> bool:
        candidate, self._pending_recovery = self._pending_recovery, None
        if candidate is None or self.stop_event.is_set():
            return False
        if self._recovery_expired(candidate.captured_ns):
            logger.info("RTC recovery holding: observation expired before dispatch")
            return False
        if not self.buffer.resume(candidate.chunk):
            logger.info("RTC recovery holding: request=%s no longer matches the paused queue", candidate.chunk.request_id)
            return False
        logger.info(
            "RTC resumed request=%s source_request=%s prefix_steps=%s queue=%s",
            candidate.chunk.request_id, candidate.source_request_id, candidate.prefix_steps, self.buffer.qsize(),
        )
        return True

    def _inference_loop(self):
        try:
            while not self.stop_event.is_set():
                self.refill_event.wait(timeout=0.1)
                self.refill_event.clear()
                if self.stop_event.is_set():
                    break
                if self.config.recovery.enabled and self.buffer.motion_paused:
                    self._request_recovery()
                    self.stop_event.wait(max(0.1, 1.0 / self.config.fps))
                    continue
                if self.buffer.draining:
                    continue
                if self._phase_refresh or self.buffer.qsize() <= self.config.rtc.queue_threshold:
                    self.request_chunk()
                    if self.buffer.motion_paused and not self.config.recovery.enabled:
                        # Pace diagnostic observations when there is no
                        # active command queue to drain.
                        self.stop_event.wait(
                            (self.chunk_size - self.config.rtc.queue_threshold) / self.config.fps
                        )
                    if not self.buffer.draining and self.buffer.qsize() <= self.config.rtc.queue_threshold:
                        self.refill_event.set()
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc
                logger.exception("RTC inference worker failed; stopping command loop")
                self.stop_event.set()

    def _dispatch(self, reservation: RTCActionReservation, skipped: int):
        command = {name: float(value) for name, value in zip(self._action_names, reservation.action, strict=True)}
        requested = command.copy()
        sent_at = time.monotonic_ns()
        sent = self.robot.send_action(command)
        if self.config.recovery.enabled:
            applied = self._positions(sent, self._joint_names)
            # Piper quantizes absolute position commands at 1/1000 of a command unit.
            self._last_verified_request_id = (
                reservation.request_id
                if applied is not None and np.allclose(applied, reservation.action[self._joint_columns], rtol=0, atol=0.001)
                else None
            )
        log_action_event(
            "sent_action",
            request_id=reservation.request_id,
            step=reservation.step,
            chunk_index=reservation.chunk_index,
            skipped_steps=skipped,
            sent_monotonic_ns=sent_at,
            requested_action=requested,
            sent_action=None if sent is None else dict(sent),
            rlt_applied=reservation.rlt_applied,
        )

    def control_loop(self):
        interval = 1.0 / self.config.fps
        deadline = time.monotonic()
        started_at = deadline
        waiting_for_recovery = False
        while not self.stop_event.is_set():
            if self.config.duration_s is not None and time.monotonic() - started_at >= self.config.duration_s:
                break
            if self.stop_event.wait(max(0, deadline - time.monotonic())):
                break
            with self._dispatch_lock:
                if self._commit_recovery():
                    deadline = time.monotonic()
                    waiting_for_recovery = False
                    skipped = 0
                else:
                    skipped = max(0, math.floor((time.monotonic() - deadline) / interval))
                step, _, _ = self.buffer.snapshot()
                self._update_rlt_phase(step + skipped)
                reservation = self.buffer.reserve(skipped_steps=skipped, pause_on_exhaustion=True)
                if reservation is not None:
                    self._dispatch(reservation, skipped)
            if reservation is None and self.buffer.motion_paused:
                if self.config.recovery.enabled:
                    if not waiting_for_recovery:
                        logger.warning("RTC motion paused; waiting for prefix-guided recovery")
                        waiting_for_recovery = True
                        self.refill_event.set()
                    deadline = time.monotonic() + interval
                    continue
                logger.warning("RTC action queue exhausted; pausing command output and continuing inference")
                self.refill_event.set()
                remaining = self.config.duration_s
                if remaining is not None:
                    remaining = max(0, remaining - (time.monotonic() - started_at))
                self.stop_event.wait(remaining)
                break
            if not self.buffer.draining and self.buffer.qsize() <= self.config.rtc.queue_threshold:
                self.refill_event.set()
            deadline += (skipped + 1) * interval

    def stop(self):
        self.stop_event.set()
        self.refill_event.set()
        if self.session_started:
            try:
                self.stub.Ready(
                    services_pb2.Empty(), metadata=(*self.metadata, ("rtc-command", "close")), timeout=2
                )
            except grpc.RpcError:
                logger.debug("RTC session release failed", exc_info=True)
            self.session_started = False
        self.channel.close()  # Cancel outstanding RPCs before joining the worker.
        if self.worker is not None:
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                logger.error(
                    "Observation worker has not exited; disconnecting robot to interrupt camera reads"
                )
        if self.connection_attempted:
            try:
                self.robot.disconnect()
            except Exception:
                logger.exception("Robot disconnection failed")
            finally:
                self.connected = False
                self.connection_attempted = False

    def run(self):
        try:
            self.start()
            self.worker = threading.Thread(target=self._inference_loop, name="rtc-inference", daemon=True)
            self.worker.start()
            self.control_loop()
        except KeyboardInterrupt:
            logger.info("Stopping RTC client")
        finally:
            self.stop()
        if self.error is not None:
            raise RuntimeError("RTC inference stopped; see the worker error above") from self.error


@draccus.wrap()
def main(config: RTCClientConfig):
    log_file = configure_logging("client", config.log_file)
    try:
        with log_actions_to_file(log_file):
            RTCRobotClient(config).run()
    except Exception:
        logger.exception("RTC client stopped with an error")
        raise


if __name__ == "__main__":
    main()
