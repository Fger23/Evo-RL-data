"""Serve a fixed or client-selected RTC checkpoint on the requested device."""

import copy
import gc
import logging
import pickle  # nosec: matches the trusted-network async_inference transport
import threading
import time
from concurrent import futures
from dataclasses import dataclass, replace
from pathlib import Path

import draccus
import grpc
import numpy as np
import torch

from lerobot.async_inference.helpers import raw_observation_to_observation
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, receive_bytes_in_chunks
from lerobot.utils.constants import ACTION

from .compile_warmup import CompileWarmupConfig, warmup
from .configs import RTCServerConfig, normalize_policy_device, positive_finite
from .logging_utils import configure_logging
from .modeling_rtc import InputGradientRTCProcessor
from .observation_codec import decode_observation_images, lossless_jpeg_available, lossless_png_available
from .protocol import (
    CHECKPOINT_PATH_HEADER,
    CHECKPOINT_SELECTION_HEADER,
    CURRENT_POSE_RECOVERY_HEADER,
    OBSERVATION_IMAGE_CODEC_HEADER,
    OBSERVATION_PNG_HEADER,
    POLICY_TYPE_HEADER,
    PROTOCOL_VERSION,
    SESSION_HEADER,
    RTCActionChunk,
    RTCObservation,
    SessionInstructions,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PendingObservation:
    observation: RTCObservation
    observation_bytes: int
    receive_ns: int
    decode_ns: int
    image_decode_ns: int
    ready_ns: int


def make_processors(config: RTCServerConfig, policy_config, device: str):
    overrides = {"device_processor": {"device": device}}
    if config.tokenizer_name_or_path:
        overrides["tokenizer_processor"] = {"tokenizer_name": config.tokenizer_name_or_path}
    return make_pre_post_processors(
        policy_config,
        pretrained_path=config.pretrained_name_or_path,
        preprocessor_overrides=overrides,
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )


def load_policy(config: RTCServerConfig):
    if config.pretrained_name_or_path is None:
        raise ValueError("A checkpoint is required to load the policy")
    device = config.policy_device or "cpu"
    policy_config = PreTrainedConfig.from_pretrained(
        config.pretrained_name_or_path, cli_overrides=[f"--device={device}"]
    )
    if policy_config.type != config.policy_type:
        raise ValueError(
            f"Checkpoint type {policy_config.type} does not match --policy_type={config.policy_type}"
        )
    policy_config.device = device
    policy_config.rtc_config = None
    # Keep checkpoint training options off. Runtime compilation is opt-in.
    for name in ("compile_model", "gradient_checkpointing"):
        if hasattr(policy_config, name):
            setattr(policy_config, name, False)
    if config.policy_type == "pi05":
        if config.policy_dtype is not None:
            policy_config.dtype = config.policy_dtype
        if config.compile_model:
            policy_config.compile_model = True
            policy_config.compile_mode = config.compile_mode
    policy = get_policy_class(config.policy_type).from_pretrained(
        config.pretrained_name_or_path, config=policy_config
    )
    policy.to(device).eval()
    policy.requires_grad_(False)
    preprocessor, postprocessor = make_processors(config, policy.config, device)
    return policy, preprocessor, postprocessor


class RTCPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, config: RTCServerConfig, *, components=None):
        self.config = config
        self.checkpoint_root = None
        if config.checkpoint_root is not None:
            self.checkpoint_root = Path(config.checkpoint_root).resolve(strict=True)
            if not self.checkpoint_root.is_dir():
                raise ValueError("checkpoint_root must be a directory on the RTC server")
        self.model_config: RTCServerConfig | None = None
        self.policy = self.preprocessor = self.postprocessor = None
        self.device = "cpu"
        self.chunk_size = self.action_dim = 0
        if config.pretrained_name_or_path is not None:
            logger.info(
                "Loading %s from %s on %s",
                config.policy_type,
                config.pretrained_name_or_path,
                config.policy_device or "cpu (client selects inference device)",
            )
            self._set_loaded_model(config, components or load_policy(config))
        elif components is not None:
            raise ValueError("Preloaded components require a default pretrained_name_or_path")
        self.default_chunk_size = self.chunk_size
        self.state_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.session_id = None
        self.instructions = None
        self.pending: _PendingObservation | None = None
        self.busy = False
        self.last_request_id = -1
        self.last_activity = time.monotonic()

    def _set_loaded_model(self, config: RTCServerConfig, components):
        policy, preprocessor, postprocessor = components
        self._check_rtc_support(policy)
        self.chunk_size = policy.config.chunk_size
        self.action_dim = policy.config.output_features[ACTION].shape[0]
        self.policy, self.preprocessor, self.postprocessor = policy, preprocessor, postprocessor
        self.model_config = config
        self.device = config.policy_device or "cpu"

    def _replace_loaded_model(self, config: RTCServerConfig, components):
        previous_device = self.device
        previous_policy = self.policy if previous_device.startswith("cuda:") else None
        self._set_loaded_model(config, components)
        if previous_policy is not None:
            # Do not reload old processors while releasing a previous checkpoint.
            try:
                previous_policy.to("cpu")
            finally:
                del previous_policy
                gc.collect()
                self._clear_cuda_cache(previous_device)

    @staticmethod
    def _check_rtc_support(policy):
        if not callable(getattr(policy, "init_rtc_processor", None)):
            raise ValueError("Policy does not implement inference-time RTC")

    def _resolve_checkpoint(self, requested: str | None) -> str:
        if requested is None:
            if self.config.pretrained_name_or_path is None:
                raise ValueError("Set --pretrained_name_or_path on the RTC client")
            return self.config.pretrained_name_or_path
        if self.checkpoint_root is None:
            raise ValueError("Client checkpoint selection requires --checkpoint_root on the RTC server")
        path = Path(requested)
        if not path.is_absolute():
            raise ValueError("Client checkpoint must be an absolute path on the RTC server")
        try:
            path = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Cannot resolve checkpoint on the RTC server: {requested}") from exc
        if not path.is_relative_to(self.checkpoint_root):
            raise ValueError(f"Checkpoint must be under server checkpoint_root {self.checkpoint_root}")
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError(f"Checkpoint must be a pretrained_model directory with config.json: {path}")
        return str(path)

    @staticmethod
    def _validate_model_inputs(instructions: SessionInstructions, policy) -> int:
        available = policy.config.chunk_size
        count = available if instructions.actions_per_chunk is None else instructions.actions_per_chunk
        if count > available or count <= 0:
            raise ValueError(f"actions_per_chunk must be in [1, {available}]")
        instructions.rtc.validate_chunk_size(count)
        action_dim = policy.config.output_features[ACTION].shape[0]
        if len(instructions.action_names) != action_dim:
            raise ValueError(
                f"Robot has {len(instructions.action_names)} actions; policy expects {action_dim}"
            )
        missing = set(policy.config.image_features) - instructions.lerobot_features.keys()
        if missing:
            raise ValueError(f"Robot is missing policy cameras: {sorted(missing)}")
        return count

    @staticmethod
    def _resolve_device(requested: str) -> str:
        device = normalize_policy_device(requested)
        if device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable on the server; choose a supported server device")
            index = int(device.split(":")[1])
            if index >= torch.cuda.device_count():
                raise ValueError(
                    f"Server GPU {device} is unavailable; visible GPU indices are 0..{torch.cuda.device_count() - 1}"
                )
        return device

    @staticmethod
    def _clear_cuda_cache(device: str):
        if device.startswith("cuda:") and torch.cuda.is_initialized():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    def _set_policy_device(self, device: str):
        if device == self.device:
            return
        previous = self.device
        # Processor device overrides must move with the model; reloading only
        # processor metadata/statistics does not reload the model weights.
        preprocessor, postprocessor = make_processors(self.model_config, self.policy.config, device)
        self.policy.reset()
        self.policy.config.rtc_config = None
        self.policy.init_rtc_processor()
        try:
            if previous.startswith("cuda:"):
                self.policy.to("cpu")
                self.preprocessor = self.postprocessor = None
                gc.collect()
                self._clear_cuda_cache(previous)
            self.policy.to(device)
        except Exception:
            # A failed CUDA move can leave different parameters on different
            # devices. Restore one CPU copy before accepting another session.
            self.policy.to("cpu")
            self.device = self.policy.config.device = "cpu"
            self.preprocessor, self.postprocessor = make_processors(
                self.model_config, self.policy.config, "cpu"
            )
            gc.collect()
            self._clear_cuda_cache(device)
            raise
        self.device = self.policy.config.device = device
        self.preprocessor, self.postprocessor = preprocessor, postprocessor
        logger.info("RTC model moved from %s to client-selected server device %s", previous, device)

    def _session(self, context):
        session_id = dict(context.invocation_metadata()).get(SESSION_HEADER)
        if not session_id:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "Use the inference_time_rtc client (session ID missing)"
            )
        return session_id

    def _check_owner(self, session_id, context):
        if self.session_id != session_id or self.instructions is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "RTC session is not initialized or has ended")
        self.last_activity = time.monotonic()

    def _configure_model(self, rtc):
        self.policy.config.rtc_config = copy.deepcopy(rtc)
        self.policy.init_rtc_processor()
        if isinstance(self.policy.rtc_processor, RTCProcessor):
            processor = InputGradientRTCProcessor(
                self.policy.config.rtc_config, prefix_weights_device=self.device
            )
            self.policy.rtc_processor = processor
            model = getattr(self.policy, "model", None)
            if model is not None:
                model.rtc_processor = processor
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def Ready(self, request, context):  # noqa: N802
        metadata = dict(context.invocation_metadata())
        if metadata.get("rtc-command") == "close":
            with self.state_lock:
                if self.session_id == metadata.get(SESSION_HEADER):
                    self.session_id = None
                    self.instructions = None
                    self.pending = None
        ready = (
            ("rtc-protocol", PROTOCOL_VERSION),
            ("rtc-chunk-size", str(self.default_chunk_size)),
            ("rtc-policy-device", self.device),
            (POLICY_TYPE_HEADER, self.config.policy_type),
            (CURRENT_POSE_RECOVERY_HEADER, "1"),
        )
        if self.checkpoint_root is not None:
            ready += ((CHECKPOINT_SELECTION_HEADER, "1"),)
        if lossless_jpeg_available(encoder=False):
            ready += ((OBSERVATION_IMAGE_CODEC_HEADER, "lossless_jpeg"),)
        if lossless_png_available():
            ready += ((OBSERVATION_PNG_HEADER, "1"),)
        context.send_initial_metadata(ready)
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        session_id = self._session(context)
        try:
            instructions = pickle.loads(request.data)  # nosec
            if (
                not isinstance(instructions, SessionInstructions)
                or instructions.protocol_version != PROTOCOL_VERSION
            ):
                raise ValueError("Incompatible client; both sides must use inference_time_rtc v3")
            positive_finite("fps", instructions.fps)
            image_codec = instructions.rtc.observation_image_codec
            if instructions.policy_type is not None and instructions.policy_type != self.config.policy_type:
                raise ValueError(f"Server policy_type is {self.config.policy_type}")
        except (ValueError, TypeError, AttributeError, pickle.UnpicklingError, EOFError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if image_codec == "lossless_jpeg" and not lossless_jpeg_available(encoder=False):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Server is missing the lossless JPEG decoder")
        if image_codec == "lossless_png" and not lossless_png_available():
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Server is missing the lossless PNG decoder")

        with self.model_lock, self.state_lock:
            lease_active = time.monotonic() - self.last_activity < self.config.session_timeout_s
            if self.session_id and self.session_id != session_id and lease_active:
                context.abort(grpc.StatusCode.ALREADY_EXISTS, "Another RTC client owns this server")
            if self.busy or (self.pending is not None and lease_active):
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "An RTC request is still in flight")
            if not context.is_active():
                context.abort(grpc.StatusCode.CANCELLED, "RTC setup was cancelled before model selection")
            requested = instructions.pretrained_name_or_path
            model_config = self.model_config
            components = None
            try:
                device = self._resolve_device(
                    instructions.policy_device or self.config.policy_device or "cuda:0"
                )
                checkpoint = self._resolve_checkpoint(requested)
                if model_config is None or checkpoint != model_config.pretrained_name_or_path:
                    # Keep the previous model available until the candidate
                    # loads and passes the handshake on CPU.
                    model_config = replace(
                        self.config, pretrained_name_or_path=checkpoint, policy_device="cpu"
                    )
                    logger.info(
                        "Loading client-selected %s checkpoint on CPU: %s",
                        self.config.policy_type,
                        checkpoint,
                    )
                    components = load_policy(model_config)
                    self._check_rtc_support(components[0])
                policy = self.policy if components is None else components[0]
                count = self._validate_model_inputs(instructions, policy)
            except (ValueError, TypeError) as exc:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except Exception as exc:
                logger.exception("Could not load the RTC checkpoint")
                context.abort(grpc.StatusCode.INTERNAL, f"Could not load the RTC checkpoint: {exc}")
            if not context.is_active():
                context.abort(grpc.StatusCode.CANCELLED, "RTC setup was cancelled during model selection")
            # Only the accepted owner may move the model. Old prefixes and
            # session instructions cannot survive a failed device switch.
            self.session_id = self.instructions = self.pending = None
            try:
                if components is not None:
                    self._replace_loaded_model(model_config, components)
                self._set_policy_device(device)
            except torch.cuda.OutOfMemoryError:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, f"Not enough memory on server {device}")
            except Exception as exc:
                logger.exception("Could not prepare the client-selected server device")
                context.abort(grpc.StatusCode.INTERNAL, f"Could not prepare server {device}: {exc}")
            if not context.is_active():
                context.abort(grpc.StatusCode.CANCELLED, "RTC setup was cancelled during device preparation")
            if self.config.warmup_on_connect:
                try:
                    warmup(
                        self,
                        CompileWarmupConfig(
                            checkpoint,
                            task=self.config.warmup_task,
                            rtc=instructions.rtc,
                            actions_per_chunk=count,
                            prefix_steps=[instructions.rtc.queue_threshold],
                            repeats=3,
                        ),
                        is_active=context.is_active,
                    )
                except Exception as exc:
                    if not context.is_active():
                        context.abort(grpc.StatusCode.CANCELLED, "RTC setup cancelled during warmup")
                    logger.exception("RTC inference warmup failed on %s", device)
                    context.abort(grpc.StatusCode.INTERNAL, f"RTC inference warmup failed: {exc}")
                if not context.is_active():
                    context.abort(grpc.StatusCode.CANCELLED, "RTC setup cancelled during warmup")
            else:
                self._configure_model(instructions.rtc)
            self.instructions = instructions
            self.session_id = session_id
            self.pending = None
            self.last_request_id = -1
            self.last_activity = time.monotonic()
            response = (("rtc-chunk-size", str(count)), ("rtc-policy-device", self.device))
            if requested is not None:
                response += ((CHECKPOINT_PATH_HEADER, checkpoint.encode("utf-8")),)
        context.send_initial_metadata(response)
        logger.info(
            "RTC client configured: device=%s, fps=%s, chunk=%s, rtc=%s",
            device,
            instructions.fps,
            count,
            instructions.rtc,
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        session_id = self._session(context)
        with self.state_lock:
            self._check_owner(session_id, context)
            instructions = self.instructions
            action_dim, chunk_size = self.action_dim, self.chunk_size
        try:
            receive_started_ns = time.monotonic_ns()
            payload = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, "[RTC]")
            received_ns = time.monotonic_ns()
            observation = pickle.loads(payload)  # nosec
            decoded_ns = time.monotonic_ns()
            if not isinstance(observation, RTCObservation):
                raise ValueError("Expected an RTCObservation")
            if observation.request_id < 0 or observation.start_step < 0 or observation.inference_delay < 0:
                raise ValueError("Request ID, start_step and inference_delay must be non-negative")
            prefix = observation.prev_chunk_left_over
            steps = observation.recovery_prefix_steps
            if type(steps) is not int or not 0 <= steps <= min(chunk_size, instructions.rtc.execution_horizon):
                raise ValueError("Invalid current-pose recovery prefix length")
            if steps and (prefix is not None or observation.inference_delay != steps):
                raise ValueError("Current-pose recovery guides every frame and does not take a previous chunk")
            if prefix is not None:
                if not isinstance(prefix, np.ndarray) or prefix.ndim != 2:
                    raise ValueError("RTC prefix must be a (T, A) numpy array")
                if prefix.shape[1] != action_dim or len(prefix) > chunk_size or not np.isfinite(prefix).all():
                    raise ValueError("RTC prefix has invalid shape or non-finite values")
            image_decode_ns = 0
            if instructions.rtc.observation_image_codec != "raw":
                image_decode_started_ns = time.monotonic_ns()
                observation.observation = decode_observation_images(observation.observation)
                image_decode_ns = time.monotonic_ns() - image_decode_started_ns
        except (ValueError, TypeError, AttributeError, pickle.UnpicklingError, EOFError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        with self.state_lock:
            self._check_owner(session_id, context)
            if self.instructions is not instructions:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION, "RTC setup changed during observation upload"
                )
            if self.pending is not None or self.busy:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Only one RTC request may be in flight")
            if observation.request_id <= self.last_request_id:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Stale or duplicate RTC request ID")
            self.last_request_id = observation.request_id
            self.pending = _PendingObservation(
                observation=observation,
                observation_bytes=len(payload),
                receive_ns=received_ns - receive_started_ns,
                decode_ns=decoded_ns - received_ns,
                image_decode_ns=image_decode_ns,
                ready_ns=time.monotonic_ns(),
            )
        return services_pb2.Empty()

    def _predict(self, request: RTCObservation, instructions: SessionInstructions) -> RTCActionChunk:
        observation = raw_observation_to_observation(
            request.observation, instructions.lerobot_features, self.policy.config.image_features
        )
        prefix = request.prev_chunk_left_over
        if request.recovery_prefix_steps:
            # Convert the hold target with the checkpoint's action preprocessing.
            # The observation state may use different ordering or statistics.
            positions = torch.tensor(
                [[float(request.observation[name]) for name in instructions.action_names]], dtype=torch.float32
            )
            if not torch.isfinite(positions).all():
                raise ValueError("Current-pose recovery needs finite joint and gripper positions")
            observation[ACTION] = positions.clone()
            observation = self.preprocessor(observation)
            target = observation.pop(ACTION, None)
            if not isinstance(target, torch.Tensor) or target.shape != positions.shape:
                raise ValueError("RTC checkpoint preprocessing did not return one normalized hold action")
            target = target.to(device=self.device, dtype=torch.float32)
            decoded = self.postprocessor(target.clone()).to(positions)
            if (
                not torch.isfinite(target).all()
                or decoded.shape != positions.shape
                or not torch.allclose(decoded, positions, rtol=0, atol=0.001)
            ):
                raise ValueError("RTC checkpoint could not encode the measured positions as a hold action")
            prefix = target.repeat(request.recovery_prefix_steps, 1)
        else:
            observation = self.preprocessor(observation)
        if prefix is not None and len(prefix):
            prefix = torch.as_tensor(prefix, device=self.device, dtype=torch.float32)
        else:
            prefix = None
        policy_inputs = {
            "prev_chunk_left_over": prefix,
            "inference_delay": request.inference_delay,
            "execution_horizon": instructions.rtc.execution_horizon,
        }
        processor = self.policy.rtc_processor
        if (
            prefix is not None
            and isinstance(self.policy.config, PI05Config)
            and isinstance(processor, InputGradientRTCProcessor)
            and not processor.is_debug_enabled()
        ):
            policy_inputs = {
                "prepared_rtc_prefix": processor.prepare_prefix(
                    prefix,
                    request.inference_delay,
                    instructions.rtc.execution_horizon,
                    chunk_size=self.chunk_size,
                    action_dim=self.policy.config.max_action_dim,
                )
            }
        # Do not use torch.inference_mode(): the RTC processor needs enable_grad().
        with torch.no_grad():
            actions = self.policy.predict_action_chunk(observation, **policy_inputs)
            count = (
                self.chunk_size if instructions.actions_per_chunk is None else instructions.actions_per_chunk
            )
            if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[1] < count:
                raise ValueError(f"Expected one complete action chunk, got {actions.shape}")
            actions = actions[:, :count, :]
            # Preserve model-space values even if a processor mutates its input.
            original = actions.detach().float().cpu().numpy().squeeze(0).copy()
            processed = torch.stack(
                [self.postprocessor(actions[:, i, :].clone()) for i in range(count)], dim=1
            )
            result = RTCActionChunk(
                request.request_id,
                request.start_step,
                original,
                processed.detach().float().cpu().numpy().squeeze(0).copy(),
            )
        result.validate(self.action_dim)
        return result

    def GetActions(self, request, context):  # noqa: N802
        get_actions_started_ns = time.monotonic_ns()
        session_id = self._session(context)
        with self.state_lock:
            self._check_owner(session_id, context)
            if self.pending is None or self.busy:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION, "Send one RTC observation before GetActions"
                )
            pending, self.pending = self.pending, None
            taken_ns = time.monotonic_ns()
            instructions = self.instructions
            self.busy = True
        try:
            pipeline_started_ns = time.monotonic_ns()
            with self.model_lock:
                if not context.is_active():
                    context.abort(grpc.StatusCode.CANCELLED, "RTC request was cancelled before inference")
                result = self._predict(pending.observation, instructions)
            with self.state_lock:
                self._check_owner(session_id, context)
            pipeline_finished_ns = time.monotonic_ns()
            payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            response = services_pb2.Actions(data=payload)
            encoded_ns = time.monotonic_ns()
            pipeline_ms = (pipeline_finished_ns - pipeline_started_ns) / 1e6
            logger.info(
                "RTC observation timing session=%s request=%s observation_bytes=%d "
                "receive_ms=%.3f decode_ms=%.3f image_codec=%s image_decode_ms=%.3f",
                session_id,
                result.request_id,
                pending.observation_bytes,
                pending.receive_ns / 1e6,
                pending.decode_ns / 1e6,
                instructions.rtc.observation_image_codec,
                pending.image_decode_ns / 1e6,
            )
            logger.info("RTC request=%s inference pipeline=%.3fs", result.request_id, pipeline_ms / 1000)
            logger.info(
                "RTC actions timing session=%s request=%s pending_wait_ms=%.3f "
                "pipeline_ms=%.3f encode_ms=%.3f get_actions_ms=%.3f action_bytes=%d",
                session_id,
                result.request_id,
                (taken_ns - pending.ready_ns) / 1e6,
                pipeline_ms,
                (encoded_ns - pipeline_finished_ns) / 1e6,
                (encoded_ns - get_actions_started_ns) / 1e6,
                len(payload),
            )
            return response
        except grpc.RpcError:
            raise
        except Exception as exc:
            logger.exception("RTC inference failed")
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        finally:
            with self.state_lock:
                self.busy = False
                self.last_activity = time.monotonic()


@draccus.wrap()
def serve(config: RTCServerConfig):
    configure_logging("server", config.log_file)
    try:
        service = RTCPolicyServer(config)
    except Exception:
        logger.exception("RTC server model initialization failed")
        raise
    # Warmup and later inference use the same worker and CUDA Graph state.
    workers = 1 if config.warmup_on_connect else 4
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers), options=grpc_channel_options(enable_retries=False)
    )
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    address = f"{config.host}:{config.port}"
    if not server.add_insecure_port(address):
        raise RuntimeError(f"Could not bind RTC server to {address}")
    server.start()
    if config.pretrained_name_or_path is None:
        logger.info("InferenceTimeRTC waiting for a client checkpoint; listening on %s", address)
    else:
        logger.info(
            "InferenceTimeRTC model loaded on %s; listening on %s (inference device and RTC options come from client)",
            service.device,
            address,
        )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown_event.set()
        server.stop(grace=2).wait()


if __name__ == "__main__":
    serve()
