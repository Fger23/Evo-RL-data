"""Feed the existing RTC VLA directly into an Evo-RLT head.

The actor receives the unfiltered VLA reference in model coordinates. Optional
S-G is applied once by the client after the actor and before command dispatch.
"""

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.processor import NormalizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import OBS_IMAGES

from ..configs import resolve_action_sg_window
from ..protocol import RTCActionChunk, RTCObservation, rlt_row_mask
from .networks import RLTHeadBundle, pool_tokens

logger = logging.getLogger(__name__)


def processor_fingerprint(checkpoint: str) -> str:
    root = Path(checkpoint)
    files = sorted(
        p for p in root.iterdir() if p.is_file() and ("processor" in p.name or p.name == "config.json")
    )
    if not any("preprocessor" in p.name and p.suffix == ".json" for p in files):
        raise ValueError("RLT needs the saved base policy_preprocessor.json and processor statistics")
    # Saved pipelines may refer to statistics with a custom filename.
    referenced = set(files)

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and Path(value).suffix in (".safetensors", ".pt", ".json"):
            candidate = (root / value).resolve()
            if candidate.is_relative_to(root.resolve()) and candidate.is_file():
                referenced.add(candidate)

    for path in files:
        if path.suffix == ".json":
            visit(json.loads(path.read_text(encoding="utf-8")))
    files = sorted(referenced)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def build_contract(service, instructions) -> dict:
    checkpoint = str(Path(service.model_config.pretrained_name_or_path).resolve())
    rtc = instructions.rtc
    contract = {
        "base_model_path": checkpoint,
        "action_names": list(instructions.action_names),
        "observation_state_names": list(
            instructions.lerobot_features.get("observation.state", {}).get("names", [])
        ),
        "image_features": list(service.policy.config.image_features),
        "sg_window": instructions.action_sg_window,
        "fps": float(instructions.fps),
        "normalization": {"processor_sha256": processor_fingerprint(checkpoint)},
        "rtc": {
            "execution_horizon": rtc.execution_horizon,
            "inference_delay": rtc.inference_delay,
            "auto_delay": rtc.auto_delay,
            "max_guidance_weight": rtc.max_guidance_weight,
            "prefix_attention_schedule": rtc.prefix_attention_schedule.value,
            "queue_threshold": rtc.queue_threshold,
        },
    }
    if instructions.action_sg_window:
        contract["sg_stage"] = "client_output"
    return contract


def normalize_actions(service, physical) -> torch.Tensor:
    """Use the checkpoint's action normalizer, without re-tokenizing images/text.

    This adapter supports native PI05's absolute-action processor pipeline. An
    additional action coordinate transform must be integrated explicitly.
    """
    allowed = {
        "RenameObservationsProcessorStep",
        "AddBatchDimensionProcessorStep",
        "NormalizerProcessorStep",
        "Pi05PrepareStateTokenizerProcessorStep",
        "TokenizerProcessorStep",
        "DeviceProcessorStep",
    }
    steps = service.preprocessor.steps
    if any(type(step).__name__ not in allowed for step in steps):
        raise ValueError("RLT only supports the native absolute-action PI05 processor pipeline")
    normalizers = [step for step in steps if isinstance(step, NormalizerProcessorStep)]
    if len(normalizers) != 1:
        raise ValueError("RLT requires exactly one checkpoint action normalizer")
    value = torch.as_tensor(physical, device=service.device, dtype=torch.float32).clone()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1 or not torch.isfinite(value).all():
        raise ValueError("Expected one finite physical action chunk [1,H,A]")
    normalized = normalizers[0]({TransitionKey.ACTION: value})[TransitionKey.ACTION]
    restored = service.postprocessor(normalized.clone()).to(value)
    if restored.shape != value.shape or not torch.allclose(restored, value, atol=0.001, rtol=1e-5):
        raise ValueError("The base checkpoint action processors do not round-trip physical targets")
    return normalized.detach().float()


def prepare_reference(service, request, instructions):
    """Return the VLA's normalized/physical reference, VLM tokens and proprio.

    Cache generation and deployment share this single RTC forward. Preserve
    model-space values directly, without S-G or a decode/re-encode round trip.
    """
    result, features, valid_mask, proprio = service._predict_vla(request, instructions, with_features=True)
    if valid_mask.shape != features.shape[:2]:
        raise ValueError("VLA features and prefix mask have incompatible shapes")
    normalized = (
        torch.as_tensor(result.original_actions, device=service.device, dtype=torch.float32)
        .unsqueeze(0)
        .clone()
    )
    physical = result.processed_actions.copy()
    # Match canonical Evo-RLT: ordered image+language final hidden states,
    # followed by adaptive pooling. Padding treatment is identical in cache.
    return normalized, physical, features.detach().float(), proprio.detach().float()


def load_head(service, instructions):
    instructions.rlt.__post_init__()
    if service.config.policy_type != "pi05":
        raise ValueError("RLT requires pi05")
    instructions.action_sg_window = resolve_action_sg_window(instructions.action_sg_window, rlt_enabled=True)
    if instructions.rtc.auto_delay:
        raise ValueError("This RLT training contract requires rtc.auto_delay=false")
    if not callable(getattr(service.policy, "predict_action_chunk_with_features", None)):
        raise ValueError("The VLA must implement predict_action_chunk_with_features")
    path = Path(instructions.rlt.checkpoint)
    if not path.is_absolute():
        raise ValueError("rlt.checkpoint must be an absolute path on the RTC server")
    path = path.resolve(strict=True)
    if service.checkpoint_root is None or not path.is_relative_to(service.checkpoint_root):
        raise ValueError("rlt.checkpoint must be under the server checkpoint_root")
    head = RLTHeadBundle.load(path, device=service.device)
    expected = build_contract(service, instructions)
    for key, value in expected.items():
        if head.contract.get(key) != value:
            raise ValueError(
                f"RLT checkpoint contract mismatch: {key}; rebuild cache/train for this configuration"
            )
    count = instructions.actions_per_chunk or service.chunk_size
    if head.config.chunk_size != count or head.config.action_dim != service.action_dim:
        raise ValueError("RLT chunk/action dimensions must match the complete RTC action chunk")
    state = service.policy.config.robot_state_feature
    if state is None or head.config.proprio_dim != state.shape[0]:
        raise ValueError("RLT proprio width must match the base checkpoint")
    backbone = service.policy.model.paligemma_with_expert.paligemma
    if head.config.token_dim != backbone.config.text_config.hidden_size:
        raise ValueError("RLT token width must match the VLA language backbone")
    logger.info("RLT loaded direct VLA: checkpoint=%s device=%s chunk=%d", path, service.device, count)
    return head


def protected_steps(request, instructions):
    if request.recovery_prefix_steps:
        return request.recovery_prefix_steps
    if request.prev_chunk_left_over is None:
        return 0
    return min(len(request.prev_chunk_left_over), instructions.rtc.execution_horizon)


def predict_chunk(service, request, instructions, *, warmup=False):
    if service.rlt_head is None:
        raise RuntimeError("RLT was requested without a validated trained checkpoint")
    count = instructions.actions_per_chunk or service.chunk_size
    mask = rlt_row_mask(instructions.rlt, request, count, instructions.rtc.execution_horizon)
    if warmup:
        # Exercise the actor/feature graph even when the selected interval is
        # shorter than the continuation prefix. These synthetic actions never leave warmup.
        mask = np.arange(count) >= protected_steps(request, instructions)
    if not mask.any():
        result = service._predict_vla(request, instructions)
        result.rlt_action_mask = mask
        logger.info("RLT bypass request=%s episode_step=%s interval=[%s,%s)",
                    request.request_id, request.episode_step,
                    instructions.rlt.start_step, instructions.rlt.end_step)
        return result
    started = time.perf_counter()
    reference, physical, tokens, proprio = prepare_reference(service, request, instructions)
    protected = protected_steps(request, instructions)
    head = service.rlt_head
    tokens = pool_tokens(tokens, head.config.token_pool_size)
    output = head.predict_chunk(
        tokens, proprio, reference, protected_steps=protected,
        action_mask=torch.as_tensor(mask, device=reference.device).unsqueeze(0),
    )
    with torch.no_grad():
        processed = service.postprocessor(output.clone()).detach().float().cpu().numpy()[0].copy()
    processed[~mask] = physical[~mask]
    # Prefixes for the next RTC request describe the actual RLT targets, not
    # the unmodified VLA's proposed plan.
    result = RTCActionChunk(
        request.request_id,
        request.start_step,
        output.detach().float().cpu().numpy()[0].copy(),
        processed,
        rlt_applied=True,
        applied_sg_window=0,
        rlt_reference_actions=physical.copy(),
        rlt_action_mask=mask,
    )
    result.validate(service.action_dim)
    logger.info(
        "RLT request=%s protected_steps=%d sg_window=%d total_pipeline_ms=%.2f max_physical_delta=%.4f "
        "episode_step=%s actor_rows=%d interval=[%s,%s)",
        request.request_id,
        protected,
        instructions.action_sg_window,
        (time.perf_counter() - started) * 1000,
        np.abs(processed - physical).max(),
        request.episode_step, int(mask.sum()), instructions.rlt.start_step, instructions.rlt.end_step,
    )
    return result


def warmup_head(service, instructions, *, is_active):
    """Warm both feature-returning graphs using synthetic observations only."""
    state_names = instructions.lerobot_features["observation.state"]["names"]
    observation = dict.fromkeys(state_names, 0.0)
    for key, feature in service.policy.config.image_features.items():
        channels, height, width = feature.shape
        observation[key.removeprefix(f"{OBS_IMAGES}.")] = np.zeros((height, width, channels), np.uint8)
    observation["task"] = service.config.warmup_task
    first = None
    for prefix_count in (0, instructions.rtc.queue_threshold):
        prefix = None if not prefix_count else first.original_actions[-prefix_count:]
        for index in range(3):
            if not is_active():
                raise TimeoutError("RLT feature warmup cancelled")
            request = RTCObservation(
                index, 0, observation, prefix, instructions.rtc.inference_delay,
                episode_step=instructions.rlt.start_step,
            )
            result = predict_chunk(service, request, instructions, warmup=True)
            if first is None:
                first = result
        logger.info("RLT feature-returning RTC warmup complete: prefix_steps=%d", prefix_count)
