"""Supply RTC actions to the existing human-in-loop recording loop.

The recording loop owns hardware reads, command dispatch and dataset writes.
This adapter receives its observations and runs only the two inference RPCs in
the background. Pausing or resetting invalidates every outstanding result.
"""

import logging
import math
import pickle  # nosec: trusted-network RTC transport
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np

from lerobot.async_inference.helpers import map_robot_keys_to_lerobot_features
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .action_buffer import RTCActionBuffer, RTCActionReservation, RTCContinuityError
from .configs import positive_finite
from .logging_utils import log_action_event
from .observation_codec import check_observation_image_codec, encode_observation_images
from .protocol import (
    CHECKPOINT_PATH_HEADER,
    CHECKPOINT_SELECTION_HEADER,
    CURRENT_POSE_RECOVERY_HEADER,
    POLICY_TYPE_HEADER,
    PROTOCOL_VERSION,
    RLT_HEADER,
    RLT_PHASE_HEADER,
    SESSION_HEADER,
    RTCActionChunk,
    RTCObservation,
    SessionInstructions,
    validate_rlt_stage,
)

if TYPE_CHECKING:
    from lerobot.robots import Robot
    from lerobot.scripts.recording_remote_policy import RemotePolicyRecordConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ObservationSnapshot:
    generation: int
    sequence: int
    captured_at: float
    start_step: int
    prefix: np.ndarray | None
    started: bool
    observation: dict[str, Any]
    recovering: bool = False
    episode_step: int | None = None


class RTCPolicyActionClient:
    """Nonblocking action source with exhaustion recovery and human takeover."""

    def __init__(self, cfg: "RemotePolicyRecordConfig", robot: "Robot", fps: float):
        cfg.__post_init__()
        if not cfg.enable or cfg.backend != "rtc":
            raise ValueError("RTCPolicyActionClient requires an enabled remote_policy.backend=rtc")
        positive_finite("dataset.fps", fps)
        self.cfg = cfg
        self.robot = robot
        self.fps = fps
        self.action_names = tuple(robot.action_features)
        self.joint_columns = tuple(
            index
            for index, name in enumerate(self.action_names)
            if name.endswith(".pos") and not name.endswith("gripper.pos")
        )
        if cfg.action_sg_window and not self.joint_columns:
            raise ValueError("action_sg_window requires joint position actions ending in .pos")
        if cfg.recovery.enabled and (
            not self.joint_columns
            or any(not name.endswith(".pos") or name not in robot.observation_features for name in self.action_names)
        ):
            raise ValueError("RTC recording recovery requires position feedback for every commanded position")
        self.channel = grpc.insecure_channel(
            cfg.server_address, options=grpc_channel_options(enable_retries=False)
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.metadata = ((SESSION_HEADER, uuid.uuid4().hex),)
        self.chunk_size = cfg.actions_per_chunk
        self.server_policy_device = None
        self.server_checkpoint = None
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._worker = None
        self._started = False
        self._session_started = False
        self._active = False
        self._generation = 0
        self._rlt_phase: bool | None = None
        self._phase_refresh = False
        self._observation_sequence = 0
        self._last_request_sequence = -1
        self._latest_observation = None
        self._pending_recovery: tuple[_ObservationSnapshot, RTCActionChunk] | None = None
        self._next_command_at = None
        self._last_reservation = None
        self._needs_setup = False
        self._session_dirty = False
        self._error = None
        self._request_id = 0
        self._latencies = deque(maxlen=cfg.rtc.latency_window)
        self._buffer = self._new_buffer()

    def _new_buffer(self):
        return RTCActionBuffer(sg_window=self.cfg.action_sg_window, joint_columns=self.joint_columns)

    @property
    def policy_id(self) -> str:
        if self.cfg.rlt.enabled:
            return f"{self.cfg.pretrained_name_or_path}::rlt={self.cfg.rlt.checkpoint}"
        return self.server_checkpoint or self.cfg.pretrained_name_or_path or self.cfg.policy_type

    @property
    def motion_paused(self) -> bool:
        with self._condition:
            return not self._active or self._error is not None or self._buffer.motion_paused

    @property
    def action_ready(self) -> bool:
        with self._condition:
            return (
                self._active
                and self._error is None
                and not self._buffer.motion_paused
                and bool(self._buffer.qsize())
            )

    @property
    def is_waiting(self) -> bool:
        with self._condition:
            return (
                self._active
                and self._error is None
                and not self._buffer.motion_paused
                and not (self._buffer.qsize())
            )

    @property
    def error(self) -> Exception | None:
        with self._condition:
            return self._error

    @property
    def last_reservation(self) -> RTCActionReservation | None:
        with self._condition:
            return self._last_reservation

    def _prepare_session(self, *, release_previous: bool = False) -> None:
        if release_previous:
            # The worker drains both RPCs before reaching this barrier. Clear
            # any pending request left by an earlier transport failure as well.
            self.stub.Ready(
                services_pb2.Empty(),
                metadata=(*self.metadata, ("rtc-command", "close")),
                timeout=self.cfg.connect_timeout_s,
            )
        _, call = self.stub.Ready.with_call(services_pb2.Empty(), timeout=self.cfg.connect_timeout_s)
        ready = dict(call.initial_metadata())
        if ready.get("rtc-protocol") != PROTOCOL_VERSION:
            raise RuntimeError("RTC recording requires lerobot.inference_time_rtc.policy_server")
        check_observation_image_codec(self.cfg.rtc.observation_image_codec, ready)
        if self.cfg.rlt.enabled and ready.get(RLT_HEADER) != "1":
            raise RuntimeError("Restart the updated RTC server to use direct VLA RLT without S-G")
        if self.cfg.rlt.enabled and ready.get(RLT_PHASE_HEADER) != "1":
            raise RuntimeError("Restart the updated RTC server to use RLT step intervals")
        if self.cfg.recovery.enabled and ready.get(CURRENT_POSE_RECOVERY_HEADER) != "1":
            raise RuntimeError("RTC rollout recovery requires an RTC server with current-pose recovery support")
        if ready.get(CHECKPOINT_SELECTION_HEADER) != "1":
            raise RuntimeError("This RTC server needs --checkpoint_root to accept a client-selected model")
        if ready.get(POLICY_TYPE_HEADER) != self.cfg.policy_type:
            raise ValueError(f"Expected server policy_type={self.cfg.policy_type}")
        instructions = SessionInstructions(
            lerobot_features=map_robot_keys_to_lerobot_features(self.robot),
            action_names=self.action_names,
            fps=self.fps,
            rtc=self.cfg.rtc,
            actions_per_chunk=self.cfg.actions_per_chunk,
            policy_device=self.cfg.policy_device,
            pretrained_name_or_path=self.cfg.pretrained_name_or_path,
            policy_type=self.cfg.policy_type,
            rlt=self.cfg.rlt if self.cfg.rlt.enabled else None,
            action_sg_window=self.cfg.action_sg_window,
        )
        _, call = self.stub.SendPolicyInstructions.with_call(
            services_pb2.PolicySetup(data=pickle.dumps(instructions, protocol=pickle.HIGHEST_PROTOCOL)),
            metadata=self.metadata,
            timeout=self.cfg.rpc_timeout_s,
        )
        self._session_started = True
        metadata = dict(call.initial_metadata())
        count = int(metadata["rtc-chunk-size"])
        if count != self.cfg.actions_per_chunk:
            raise ValueError("Server did not honor the requested actions_per_chunk")
        self.cfg.rtc.validate_chunk_size(count)
        device = metadata["rtc-policy-device"]
        if self.cfg.policy_device and device != self.cfg.policy_device:
            raise RuntimeError("Server did not honor the client-selected policy_device")
        checkpoint = metadata.get(CHECKPOINT_PATH_HEADER)
        if not checkpoint:
            raise RuntimeError("Server did not acknowledge the client-selected checkpoint")
        self.chunk_size = count
        self.server_policy_device = device
        self.server_checkpoint = checkpoint.decode("utf-8")

    def start(self) -> None:
        """Prepare the selected model before the caller connects any hardware."""
        if self._started or self._stop_event.is_set():
            raise RuntimeError("RTC recording client can only be started once")
        grpc.channel_ready_future(self.channel).result(timeout=self.cfg.connect_timeout_s)
        self._prepare_session()
        with self._condition:
            self._started = True
            self._active = True
        self._worker = threading.Thread(target=self._inference_loop, name="rtc-recording", daemon=True)
        self._worker.start()
        logger.info(
            "RTC recording connected server=%s checkpoint=%s device=%s chunk=%s fps=%s",
            self.cfg.server_address,
            self.server_checkpoint,
            self.server_policy_device,
            self.chunk_size,
            self.fps,
        )
        if self.cfg.rlt.enabled:
            logger.info("RLT mode: action S-G disabled; reference_source=RTC VLA")

    def _restart_generation(self, *, active: bool) -> None:
        with self._condition:
            self._generation += 1
            self._active = active
            self._needs_setup = self._needs_setup or self._session_dirty
            self._latest_observation = None
            self._pending_recovery = None
            self._next_command_at = None
            self._last_reservation = None
            self._rlt_phase = None
            self._phase_refresh = False
            self._buffer = self._new_buffer()
            self._latencies.clear()
            if active:
                self._error = None
            self._condition.notify_all()

    def pause(self) -> None:
        """Immediately invalidate queued and in-flight actions during takeover."""
        self._restart_generation(active=False)

    def reset(self) -> None:
        """Start a fresh timeline after the worker drains its old RPC pair."""
        self._restart_generation(active=True)

    def _recovery_expired(self, snapshot: _ObservationSnapshot) -> bool:
        return time.monotonic() - snapshot.captured_at > self.cfg.recovery.max_observation_age_s

    def _resume_pending_recovery(self, now: float) -> None:
        if self._pending_recovery is None:
            return
        snapshot, result = self._pending_recovery
        self._pending_recovery = None
        if self._recovery_expired(snapshot):
            logger.info("RTC recovery holding: observation expired before dispatch request=%s", result.request_id)
            return
        if self._buffer.resume(result):
            self._next_command_at = now
            logger.info(
                "RTC resumed request=%s prefix_steps=%s queue=%s generation=%s",
                result.request_id, self.cfg.recovery.prefix_steps, self._buffer.qsize(), self._generation,
            )

    def get_action(self, observation: dict, task: str | None, timestep: int) -> dict | None:
        """Submit one recorded observation and reserve at most one due action."""
        if not self._started:
            raise RuntimeError("Start the RTC recording client before requesting actions")
        captured_at = time.monotonic()
        with self._condition:
            self._last_reservation = None
            if not self._active or self._error is not None or self._stop_event.is_set():
                return None
            if self.cfg.rlt.enabled:
                phase = self.cfg.rlt.active_at(timestep)
                if phase != self._rlt_phase:
                    if self._rlt_phase is not None:
                        # Keep the command queue/prefix, but consume and discard
                        # an RPC started in the previous scheduled phase.
                        self._generation += 1
                        self._pending_recovery = None
                        self._phase_refresh = True
                    self._rlt_phase = phase
                    logger.info(
                        "RLT phase episode_step=%s enabled=%s interval=[%s,%s) generation=%s queue=%s",
                        timestep, phase, self.cfg.rlt.start_step, self.cfg.rlt.end_step,
                        self._generation, self._buffer.qsize(),
                    )
            self._resume_pending_recovery(captured_at)
            # Use an observation acquired after the pause was latched.
            recovering = self.cfg.recovery.enabled and self._buffer.motion_paused
            raw = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in observation.items()
            }
            if task is not None:
                raw["task"] = task
            start_step, prefix, started = self._buffer.snapshot()
            reservation = None
            skipped = 0
            if started and not self._buffer.motion_paused:
                if self._next_command_at is None:
                    self._next_command_at = captured_at
                if captured_at >= self._next_command_at:
                    skipped = max(0, math.floor((captured_at - self._next_command_at) * self.fps))
                    reservation = self._buffer.reserve(skipped_steps=skipped, pause_on_exhaustion=True)
                    self._next_command_at += (skipped + 1) / self.fps
                if self._buffer.motion_paused:
                    if self.cfg.recovery.enabled:
                        logger.warning("RTC recording queue exhausted; waiting for current-pose recovery")
                    else:
                        logger.warning("RTC recording queue exhausted; holding until human takeover and release")
            if self._buffer.motion_paused:
                start_step, prefix, started = self._buffer.snapshot()
                if recovering:
                    prefix = None
            else:
                start_step += skipped
                if prefix is not None and skipped:
                    prefix = prefix[skipped:]
            self._observation_sequence += 1
            self._latest_observation = _ObservationSnapshot(
                self._generation, self._observation_sequence, captured_at, start_step, prefix, started, raw,
                recovering=recovering, episode_step=timestep,
            )
            self._last_reservation = reservation
            self._condition.notify_all()
        if reservation is None:
            return None
        command = {
            name: float(value) for name, value in zip(self.action_names, reservation.action, strict=True)
        }
        log_action_event(
            "recording_action",
            generation=self._generation,
            recording_timestep=timestep,
            request_id=reservation.request_id,
            step=reservation.step,
            chunk_index=reservation.chunk_index,
            skipped_steps=skipped,
            requested_action=command,
            rlt_applied=reservation.rlt_applied,
        )
        return command

    def _delay_steps(self) -> int:
        delay = self.cfg.rtc.inference_delay
        if self.cfg.rtc.auto_delay and self._latencies:
            delay = math.ceil(float(np.quantile(self._latencies, 0.95)) * self.fps)
        if delay >= self.chunk_size:
            raise RuntimeError("End-to-end RTC latency exceeds the chunk duration")
        return delay

    def _request_chunk(self, snapshot: _ObservationSnapshot, delay: int):
        recovery_steps = self.cfg.recovery.prefix_steps if snapshot.recovering else 0
        request = RTCObservation(
            self._request_id, snapshot.start_step, snapshot.observation, snapshot.prefix, delay,
            recovery_prefix_steps=recovery_steps,
            episode_step=snapshot.episode_step,
        )
        self._request_id += 1
        if snapshot.recovering:
            logger.info(
                "RTC recovery request=%s prefix_steps=%s source=current_pose generation=%s",
                request.request_id, recovery_steps, snapshot.generation,
            )
        begin = time.monotonic_ns()
        image_codec = self.cfg.rtc.observation_image_codec
        request.observation = encode_observation_images(snapshot.observation, image_codec)
        serialize_started = time.monotonic_ns()
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        send_started = time.monotonic_ns()
        # Always complete this pair, even when a takeover invalidates its result.
        # This consumes the server's pending request before the next setup.
        self.stub.SendObservations(
            send_bytes_in_chunks(payload, services_pb2.Observation),
            metadata=self.metadata,
            timeout=self.cfg.rpc_timeout_s,
        )
        send_finished = time.monotonic_ns()
        get_started = time.monotonic_ns()
        response = self.stub.GetActions(
            services_pb2.Empty(), metadata=self.metadata, timeout=self.cfg.rpc_timeout_s
        )
        get_finished = time.monotonic_ns()
        result = pickle.loads(response.data)  # nosec
        received = time.monotonic_ns()
        if not isinstance(result, RTCActionChunk):
            raise ValueError("Server returned an incompatible RTC action payload")
        if result.request_id != request.request_id or result.start_step != snapshot.start_step:
            raise ValueError("Server returned a mismatched RTC request ID or timeline origin")
        result.validate(len(self.action_names))
        validate_rlt_stage(result, request, self.cfg.rlt, self.cfg.rtc.execution_horizon)
        if len(result.original_actions) != self.chunk_size:
            raise ValueError("Server returned an unexpected RTC chunk size")
        elapsed = time.monotonic() - snapshot.captured_at
        log_action_event(
            "rpc_timing",
            generation=snapshot.generation,
            request_id=request.request_id,
            observation_age_ms=max(0, begin / 1e6 - snapshot.captured_at * 1000),
            image_codec=image_codec,
            image_encode_ms=(serialize_started - begin) / 1e6,
            serialize_ms=(send_started - serialize_started) / 1e6,
            observation_bytes=len(payload),
            send_observations_rpc_ms=(send_finished - send_started) / 1e6,
            rpc_handoff_ms=(get_started - send_finished) / 1e6,
            get_actions_rpc_ms=(get_finished - get_started) / 1e6,
            deserialize_response_ms=(received - get_finished) / 1e6,
            total_ms=elapsed * 1000,
        )
        log_action_event(
            "action_chunk",
            generation=snapshot.generation,
            request_id=result.request_id,
            start_step=result.start_step,
            estimated_delay=delay,
            action_names=self.action_names,
            original_actions=result.original_actions,
            processed_actions=result.processed_actions,
            **({"rlt_applied": result.rlt_applied, "rlt_action_mask": result.rlt_action_mask,
                "episode_step": snapshot.episode_step,
                "rlt_reference_actions": result.rlt_reference_actions} if self.cfg.rlt.enabled else {}),
        )
        return result, elapsed

    def _work_available(self) -> bool:
        if self._stop_event.is_set():
            return True
        latest = self._latest_observation
        if not self._active or self._error is not None or latest is None:
            return False
        if self._needs_setup:
            return True
        if latest.sequence <= self._last_request_sequence or self._pending_recovery is not None:
            return False
        if self.cfg.recovery.enabled:
            if self._buffer.motion_paused:
                return latest.recovering
            if self._buffer.draining:
                return False
        return self._phase_refresh or self._buffer.qsize() <= self.cfg.rtc.queue_threshold

    def _recovery_observation_is_valid(self, snapshot: _ObservationSnapshot) -> bool:
        try:
            positions = [float(snapshot.observation[name]) for name in self.action_names]
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        return all(map(math.isfinite, positions)) and not self._recovery_expired(snapshot)

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait_for(self._work_available)
                if self._stop_event.is_set():
                    return
                generation = self._generation
                needs_setup = self._needs_setup
                snapshot = self._latest_observation
                self._last_request_sequence = snapshot.sequence
                self._phase_refresh = False
                self._session_dirty = True
            try:
                if needs_setup:
                    self._prepare_session(release_previous=True)
                    with self._condition:
                        if generation == self._generation:
                            self._needs_setup = False
                            self._session_dirty = False
                            self._last_request_sequence = -1
                    continue
                with self._condition:
                    delay = self.cfg.rtc.inference_delay if snapshot.prefix is None else self._delay_steps()
                if snapshot.recovering:
                    if not self._recovery_observation_is_valid(snapshot):
                        logger.info("RTC recovery holding: fresh positional observation is unavailable")
                        continue
                    delay = self.cfg.recovery.prefix_steps
                result, elapsed = self._request_chunk(snapshot, delay)
                with self._condition:
                    if generation != self._generation or not self._active or self._stop_event.is_set():
                        logger.info(
                            "Discarded RTC recording result request=%s from an earlier takeover or RLT phase",
                            result.request_id,
                        )
                        continue
                    if snapshot.recovering:
                        if self._recovery_expired(snapshot):
                            logger.info("RTC recovery holding: recovery result expired request=%s", result.request_id)
                        else:
                            self._pending_recovery = (snapshot, result)
                        continue
                    horizon = (
                        min(len(snapshot.prefix), self.cfg.rtc.execution_horizon)
                        if snapshot.started and snapshot.prefix is not None
                        else None
                    )
                    try:
                        real_delay = self._buffer.merge(
                            result, prefix_horizon=horizon, allow_late_splice=self.cfg.allow_late_splice,
                            drain_on_rejection=self.cfg.recovery.enabled,
                        )
                    except RTCContinuityError as exc:
                        if not self.cfg.recovery.enabled:
                            raise
                        logger.warning(
                            "RTC recording continuation discarded request=%s queue=%s; waiting for queue exhaustion: %s",
                            result.request_id, self._buffer.qsize(), exc,
                        )
                        continue
                    if real_delay is not None and snapshot.started:
                        self._latencies.append(elapsed)
                    logger.info(
                        "RTC recording request=%s latency=%.3fs estimated_delay=%s actual_delay=%s queue=%s motion_paused=%s",
                        result.request_id,
                        elapsed,
                        delay,
                        real_delay,
                        self._buffer.qsize(),
                        self._buffer.motion_paused,
                    )
            except Exception as exc:
                with self._condition:
                    if self._stop_event.is_set():
                        return
                    if generation != self._generation:
                        # A failed RPC pair may leave an uploaded observation
                        # pending. Phase changes retain the buffer, so unlike a
                        # takeover they did not already schedule a session reset.
                        self._needs_setup = True
                        self._phase_refresh = True
                        logger.debug("Discarded RTC error from an earlier takeover or RLT phase; resetting session",
                                     exc_info=True)
                        continue
                    self._error = exc
                    self._needs_setup = True
                    logger.exception(
                        "RTC recording inference failed; holding until human takeover and release"
                    )

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._active = False
            self._condition.notify_all()
        if self._session_started:
            try:
                self.stub.Ready(
                    services_pb2.Empty(), metadata=(*self.metadata, ("rtc-command", "close")), timeout=2
                )
            except grpc.RpcError:
                logger.debug("RTC recording session release failed", exc_info=True)
            self._session_started = False
        self.channel.close()
        if self._worker is not None:
            self._worker.join(timeout=5)
            if self._worker.is_alive():
                logger.error("RTC recording RPC worker did not stop after channel cancellation")
