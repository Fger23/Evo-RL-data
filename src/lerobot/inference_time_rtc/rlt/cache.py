"""Build offline RLT transitions with the deployed, unfiltered RTC VLA reference.

Only converted LeRobot datasets are accepted. This command loads a frozen model,
not a robot or a network server. Reference continuation is a causal, offline
counterfactual: leftovers from an earlier sampled reference are advanced along
the recorded observation timeline. It does not reconstruct network queue timing.
Critic targets always contain the recorded executed commands, never VLA samples.
VLA references stay unfiltered. The contract records optional client output S-G,
which training includes in actor/target Q evaluation rather than in VLA references.
Explicit per-frame RLT phase labels are required. Unmarked frames supply critic
and TD context, while token/actor learning is restricted by the marked masks.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

RLT_PHASE_KEY = "complementary_info.rlt_phase"
RLT_PHASE_ID_KEY = "complementary_info.rlt_phase_id"
PHASE_SCHEMA = {
    "version": 1,
    "active_key": RLT_PHASE_KEY,
    "interval_id_key": RLT_PHASE_ID_KEY,
    "outside_interval_id": -1,
    "anchor_sampling": "full_episode_stride_plus_marked_starts_and_td_successors",
    "token_sampling": "current_anchor_inside_a_marked_interval",
    "actor_mask": "valid_rows_in_the_anchors_interval",
    "td_scope": "full_episode",
}


def success_label(value: Any) -> bool:
    """Reject absent/ambiguous outcomes instead of assuming demonstrations succeed."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("episode_success must be a scalar success/failure label")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("success", "true", "1"):
            return True
        if normalized in ("failure", "false", "0"):
            return False
    raise ValueError(f"Missing/invalid episode_success {value!r}; label the episode explicitly")


def load_dataset_specs(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    specs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not specs:
        raise ValueError("--dataset-spec must be a nonempty JSON list")
    seen = set()
    for item in specs:
        if not isinstance(item, dict) or not item.get("repo_id") or not item.get("root"):
            raise ValueError("Each dataset specification requires repo_id and root")
        if item.get("source") not in ("demo", "rollout"):
            raise ValueError("Each dataset source must be demo or rollout")
        root = Path(item["root"])
        root = (root if root.is_absolute() else path.parent / root).resolve()
        if not (root / "meta" / "info.json").is_file():
            raise ValueError(
                f"{root} is not a converted LeRobot dataset; convert raw PKL with "
                "convert_pkl_dataset.py before building the RLT cache"
            )
        if str(root) in seen:
            raise ValueError(f"Duplicate dataset root: {root}")
        seen.add(str(root))
        item["root"] = str(root)
        if item.get("success_labels"):
            labels = Path(item["success_labels"])
            labels = labels if labels.is_absolute() else path.parent / labels
            item["success_labels"] = json.loads(labels.read_text(encoding="utf-8"))
            if not isinstance(item["success_labels"], dict):
                raise ValueError("success_labels must point to a JSON object mapping episode index to label")
    return specs


def _cpu_float(value: Any) -> torch.Tensor:
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).clone()


def _padded_actions(actions: torch.Tensor, start: int, horizon: int) -> tuple[torch.Tensor, int]:
    valid = actions[start : start + horizon]
    count = len(valid)
    if not count:
        raise ValueError("Cannot construct a transition from an empty executed-action interval")
    if count < horizon:
        valid = torch.cat((valid, valid[-1:].expand(horizon - count, -1)), dim=0)
    return valid.clone(), count


def phase_annotations(
    active: Any, interval_ids: Any, length: int
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]:
    """Validate explicit scalar phase columns and identify [start, stop) intervals."""
    if active is None or interval_ids is None:
        raise ValueError(
            f"RLT cache requires {RLT_PHASE_KEY} and {RLT_PHASE_ID_KEY}; "
            "mark collection intervals instead of training on the whole task"
        )

    def vector(value, name):
        try:
            if isinstance(value, torch.Tensor):
                result = value.detach().cpu()
            else:
                result = torch.stack([torch.as_tensor(row).reshape(()) for row in value])
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must contain one explicit scalar per frame") from exc
        if result.shape == (length, 1):
            result = result.squeeze(-1)
        if result.shape != (length,) or not torch.isfinite(result).all():
            raise ValueError(f"{name} must contain {length} finite scalar labels")
        return result

    active = vector(active, RLT_PHASE_KEY)
    ids = vector(interval_ids, RLT_PHASE_ID_KEY)
    if not ((active == 0) | (active == 1)).all():
        raise ValueError(f"{RLT_PHASE_KEY} must be boolean or explicit 0/1")
    if ids.dtype == torch.bool or ids.is_complex() or not torch.equal(ids, ids.round()):
        raise ValueError(f"{RLT_PHASE_ID_KEY} must contain integer interval IDs")
    active, ids = active.bool(), ids.long()
    if (ids[active] < 0).any() or (ids[~active] != -1).any():
        raise ValueError("Marked RLT frames require nonnegative phase IDs; outside frames require -1")
    if not active.any():
        raise ValueError("Every cached episode must contain at least one explicitly marked RLT interval")
    intervals = []
    seen = set()
    index = 0
    while index < length:
        if not active[index]:
            index += 1
            continue
        start, interval_id = index, int(ids[index])
        if interval_id in seen:
            raise ValueError("Each rlt_phase_id must describe one contiguous interval within its episode")
        seen.add(interval_id)
        while index < length and active[index] and int(ids[index]) == interval_id:
            index += 1
        intervals.append((start, index, interval_id))
    return active, ids, intervals


def _rlt_action_mask(
    active: torch.Tensor, interval_ids: torch.Tensor, start: int, horizon: int
) -> torch.Tensor:
    """Enable only valid rows in the marked interval containing this anchor."""
    mask = torch.zeros(horizon, dtype=torch.bool)
    if start >= len(active) or not active[start]:
        return mask
    visible = min(horizon, len(active) - start)
    mask[:visible] = active[start : start + visible] & (
        interval_ids[start : start + visible] == interval_ids[start]
    )
    return mask


def build_episode_transitions(
    actions: torch.Tensor,
    reference_at: Callable[[int, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    normalize_actions: Callable[[torch.Tensor], torch.Tensor],
    *,
    outcome: Any,
    rlt_phase: Any = None,
    rlt_phase_id: Any = None,
    chunk_size: int = 50,
    stride: int = 20,
    execution_horizon: int = 20,
) -> dict[str, torch.Tensor]:
    """Build episode-safe H-step TD targets with a separate causal RTC reference.

    ``reference_at(t, prefix)`` returns normalized reference [H,A], pooled VLA
    tokens [M,D], and normalized proprio [P]. Prefix rows are from the previous
    *sampled reference* that have not yet elapsed at t; dataset future commands
    and backwards replay of past executed commands are never used as RTC input.
    Critic anchors span the full episode. Additional anchors are relative to
    each marked interval's beginning, so even a one-frame interval is sampled.
    Every nonterminal TD successor is also a critic anchor. Token/actor learning
    uses the marked masks; an interval boundary is not an episode termination.
    Each next state is exactly t + n_steps, not t + reference stride.
    """
    actions = _cpu_float(actions)
    if actions.ndim != 2 or not len(actions) or not torch.isfinite(actions).all():
        raise ValueError("Executed commands must be a nonempty finite [T,A] tensor")
    if not 0 < execution_horizon < chunk_size or not 0 < stride <= chunk_size:
        raise ValueError("Require 0 < execution_horizon < chunk_size and 0 < stride <= chunk_size")
    active, interval_ids, intervals = phase_annotations(rlt_phase, rlt_phase_id, len(actions))
    primary_starts = set(range(0, len(actions), stride))
    primary_starts.update(position for start, stop, _ in intervals for position in range(start, stop, stride))
    # Close the H-step transition graph: bootstrap states must have their own
    # current critic samples, including after the final marked interval.
    starts = set(primary_starts)
    for start in primary_starts:
        successor = start + chunk_size
        while successor < len(actions):
            starts.add(successor)
            successor += chunk_size
    nodes: dict[int, dict[str, torch.Tensor]] = {}
    prior_reference = None
    prior_start = 0
    successful = success_label(outcome)
    for start in sorted(starts):
        prefix = None
        if prior_reference is not None:
            elapsed = start - prior_start
            if elapsed < chunk_size:
                prefix = prior_reference[elapsed:].clone()
        reference, tokens, proprio = reference_at(start, prefix)
        reference, tokens, proprio = map(_cpu_float, (reference, tokens, proprio))
        if reference.shape != (chunk_size, actions.shape[1]):
            raise ValueError("RTC reference must have the complete checkpoint [H,A] shape")
        if tokens.ndim != 2 or proprio.ndim != 1:
            raise ValueError("Reference features must be tokens [M,D] and proprio [P]")
        if not all(torch.isfinite(value).all() for value in (reference, tokens, proprio)):
            raise ValueError("RTC reference or observation features contain non-finite values")
        physical, count = _padded_actions(actions, start, chunk_size)
        normalized = _cpu_float(normalize_actions(physical))
        if normalized.ndim == 3 and normalized.shape[0] == 1:
            normalized = normalized.squeeze(0)
        if normalized.shape != reference.shape or not torch.isfinite(normalized).all():
            raise ValueError("Normalized executed commands must be finite [H,A]")
        nodes[start] = {
            "vla_tokens": tokens,
            "proprio": proprio,
            "reference_actions": reference,
            "executed_actions": normalized,
            "n_steps": torch.tensor(count, dtype=torch.long),
            "protected_steps": torch.tensor(
                min(execution_horizon, len(prefix), count) if prefix is not None else 0,
                dtype=torch.long,
            ),
            "action_mask": torch.arange(chunk_size) < count,
            "rlt_action_mask": _rlt_action_mask(active, interval_ids, start, chunk_size),
            "start_step": torch.tensor(start, dtype=torch.long),
            "rlt_phase_id": interval_ids[start].clone(),
        }
        # TD-only supplemental samples do not alter the primary RTC reference cadence.
        if start in primary_starts:
            prior_reference, prior_start = reference, start

    records: list[dict[str, torch.Tensor]] = []
    for start, node in nodes.items():
        count = int(node["n_steps"])
        terminal = start + count >= len(actions)
        next_node = node if terminal else nodes[start + count]
        for key in ("reference_actions", "vla_tokens", "proprio"):
            if next_node[key].shape != node[key].shape:
                raise ValueError("TD successor features must match the current feature shapes")
        rewards = torch.zeros(chunk_size, dtype=torch.float32)
        if terminal and successful:
            rewards[count - 1] = 1.0
        records.append(
            {
                **node,
                "next_vla_tokens": next_node["vla_tokens"],
                "next_proprio": next_node["proprio"],
                "next_reference_actions": next_node["reference_actions"],
                "rewards": rewards,
                "terminated": torch.tensor(terminal),
                "next_protected_steps": torch.tensor(0, dtype=torch.long) if terminal else next_node["protected_steps"],
                "next_action_mask": torch.zeros(chunk_size, dtype=torch.bool) if terminal else next_node["action_mask"],
                "next_rlt_action_mask": torch.zeros(chunk_size, dtype=torch.bool) if terminal else next_node["rlt_action_mask"],
            }
        )
    return {key: torch.stack([row[key] for row in records]) for key in records[0]}


def _feature_names(features: dict, key: str) -> tuple[str, ...]:
    feature = features.get(key, {})
    names = feature.get("names")
    if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"Dataset {key} requires ordered scalar feature names")
    if tuple(feature.get("shape", ())) != (len(names),) or len(set(names)) != len(names):
        raise ValueError(f"Dataset {key} names must uniquely match its vector width")
    return tuple(names)


def select_cache_features(features: dict, policy_image_features: dict) -> dict:
    """Use the frozen policy's cameras while retaining ordered state/action names.

    Converted captures can contain auxiliary cameras that the VLA never used.
    Passing those through the RTC raw-observation adapter would incorrectly try
    to look them up in the policy's image-feature map.
    """
    _feature_names(features, "observation.state")
    _feature_names(features, "action")
    selected = {key: features[key] for key in ("observation.state", "action")}
    for key in policy_image_features:
        if key not in features or features[key].get("dtype") not in ("image", "video"):
            raise ValueError(f"Dataset is missing the base checkpoint camera: {key}")
        selected[key] = features[key]
    return selected


def dataset_frame_to_raw(frame: dict, features: dict, task: str | None = None) -> dict:
    """Invert dataset vector/camera packing into the RTC client's raw observation."""
    names = _feature_names(features, "observation.state")
    state = _cpu_float(frame["observation.state"])
    if state.shape != (len(names),) or not torch.isfinite(state).all():
        raise ValueError("Recorded observation.state does not match its ordered feature names")
    raw = {name: float(value) for name, value in zip(names, state, strict=True)}
    for key in features:
        if not key.startswith("observation.images."):
            continue
        value = np.asarray(frame[key].detach().cpu() if isinstance(frame[key], torch.Tensor) else frame[key])
        if value.ndim != 3:
            raise ValueError(f"Recorded camera {key} must be CHW or HWC")
        if value.shape[0] == 3:
            value = value.transpose(1, 2, 0)
        if value.shape[-1] != 3 or not np.isfinite(value).all():
            raise ValueError(f"Recorded camera {key} must contain finite RGB pixels")
        if np.issubdtype(value.dtype, np.floating):
            if value.min() < 0 or value.max() > 1:
                raise ValueError(f"Floating camera {key} must be in [0,1]")
            value = np.rint(value * 255)
        raw[key.removeprefix("observation.images.")] = np.ascontiguousarray(value, dtype=np.uint8)
    selected_task = task if task is not None else frame.get("task")
    if not isinstance(selected_task, str) or not selected_task.strip():
        raise ValueError("Recorded observations need a task instruction (or --task)")
    raw["task"] = selected_task
    return raw


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Final RTC PI0.5 checkpoint directory")
    parser.add_argument(
        "--dataset-spec", required=True, help="JSON list of {repo_id,root,source,success_labels?}"
    )
    parser.add_argument("--output", required=True, help="Output transition .pt file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--sg-window",
        type=int,
        choices=(0, 5, 7),
        default=0,
        help="Client output S-G window after the actor; VLA references remain unfiltered",
    )
    parser.add_argument("--execution-horizon", type=int, default=20)
    parser.add_argument("--inference-delay", type=int, default=16)
    parser.add_argument("--queue-threshold", type=int, default=30)
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--prefix-attention-schedule", choices=("ZEROS", "ONES", "LINEAR", "EXP"), default="EXP"
    )
    parser.add_argument(
        "--stride", type=int, default=None, help="Reference cadence; default H - queue_threshold"
    )
    parser.add_argument("--token-pool-size", type=int, default=64)
    parser.add_argument("--task", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--tolerance-s", type=float, default=0.04)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser


def build_cache(args: argparse.Namespace) -> dict:
    if type(args.sg_window) is not int or args.sg_window not in (0, 5, 7):
        raise ValueError(
            "RLT output S-G window must be 0, 5, or 7"
        )
    # Imports stay here: inspecting cache tensors never loads camera/robot SDKs.
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.inference_time_rtc.configs import InferenceTimeRTCConfig, RTCServerConfig
    from lerobot.inference_time_rtc.policy_server import RTCPolicyServer
    from lerobot.inference_time_rtc.protocol import RTCObservation, SessionInstructions

    from .networks import RLTHeadConfig, pool_tokens
    from .runtime import build_contract, normalize_actions, prepare_reference

    specs = load_dataset_specs(args.dataset_spec)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new cache filename")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    if args.token_pool_size <= 0:
        raise ValueError("--token-pool-size must be positive")
    stride = args.chunk_size - args.queue_threshold if args.stride is None else args.stride
    if not 0 < stride <= args.chunk_size:
        raise ValueError("Reference stride must be in [1, chunk_size]")
    rtc = InferenceTimeRTCConfig(
        execution_horizon=args.execution_horizon,
        inference_delay=args.inference_delay,
        auto_delay=False,
        max_guidance_weight=args.max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule(args.prefix_attention_schedule),
        queue_threshold=args.queue_threshold,
    )
    rtc.validate_chunk_size(args.chunk_size)
    config = RTCServerConfig(
        pretrained_name_or_path=args.base_model,
        policy_type="pi05",
        policy_device=args.device,
        policy_dtype=args.dtype,
        compile_model=False,
    )
    service = RTCPolicyServer(config)
    if service.chunk_size != args.chunk_size:
        raise ValueError("RLT cache must use the complete checkpoint chunk_size")
    service._configure_model(rtc)
    all_episodes = []
    episode_info = []
    contract = None
    for spec in specs:
        dataset = LeRobotDataset(
            repo_id=spec["repo_id"],
            root=spec["root"],
            video_backend=args.video_backend,
            tolerance_s=args.tolerance_s,
        )
        if abs(float(dataset.fps) - args.fps) > 1e-6:
            raise ValueError(f"Dataset fps={dataset.fps} differs from requested fps={args.fps}")
        missing_phase = {RLT_PHASE_KEY, RLT_PHASE_ID_KEY} - dataset.features.keys()
        if missing_phase:
            raise ValueError(
                f"Dataset is missing explicit RLT interval labels {sorted(missing_phase)}; "
                "collect/annotate RLT intervals before building the cache"
            )
        action_names = _feature_names(dataset.features, "action")
        if len(action_names) != service.action_dim:
            raise ValueError("Dataset action vector width differs from the frozen RTC checkpoint")
        features = select_cache_features(dataset.features, service.policy.config.image_features)
        instructions = SessionInstructions(
            lerobot_features=features,
            action_names=action_names,
            fps=args.fps,
            rtc=rtc,
            actions_per_chunk=args.chunk_size,
            policy_device=args.device,
            action_sg_window=args.sg_window,
        )
        service._validate_model_inputs(instructions, service.policy)
        current_contract = build_contract(service, instructions)
        if contract is not None and contract != current_contract:
            raise ValueError("Dataset contracts differ (action order, normalization, RTC reference or fps)")
        contract = current_contract
        dataset._ensure_hf_dataset_loaded()
        for ep in dataset.meta.episodes:
            if args.max_episodes is not None and len(all_episodes) >= args.max_episodes:
                break
            ep_index = int(ep["episode_index"])
            first, stop = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            labels = spec.get("success_labels", {})
            outcome = success_label(labels.get(str(ep_index), ep.get("episode_success")))
            # Read scalar annotations and action vectors without decoding all video frames.
            rows = dataset.hf_dataset.select_columns(
                ["action", RLT_PHASE_KEY, RLT_PHASE_ID_KEY]
            )[first:stop]
            commands = torch.stack([_cpu_float(row) for row in rows["action"]])
            request_counter = 0

            def reference_at(
                start: int,
                prefix: torch.Tensor | None,
                *,
                _dataset=dataset,
                _first=first,
                _instructions=instructions,
                _features=features,
            ):
                nonlocal request_counter
                frame = _dataset[_first + start]
                raw = dataset_frame_to_raw(frame, _features, args.task)
                request = RTCObservation(
                    request_id=request_counter,
                    start_step=start,
                    observation=raw,
                    prev_chunk_left_over=None if prefix is None else prefix.numpy().copy(),
                    inference_delay=0 if prefix is None else args.inference_delay,
                )
                request_counter += 1
                ref, _physical, tokens, proprio = prepare_reference(service, request, _instructions)
                tokens = _cpu_float(tokens)
                if tokens.ndim == 3:
                    tokens = tokens.squeeze(0)
                # Same canonical pooling rule used by the RLT head; idempotent if already pooled.
                if tokens.shape[0] > args.token_pool_size:
                    tokens = pool_tokens(tokens.unsqueeze(0), args.token_pool_size).squeeze(0)
                return _cpu_float(ref).squeeze(0), tokens, _cpu_float(proprio).squeeze(0)

            tensors = build_episode_transitions(
                commands,
                reference_at,
                lambda value: normalize_actions(service, value),
                outcome=outcome,
                rlt_phase=rows[RLT_PHASE_KEY],
                rlt_phase_id=rows[RLT_PHASE_ID_KEY],
                chunk_size=args.chunk_size,
                stride=stride,
                execution_horizon=args.execution_horizon,
            )
            tensors["episode_id"] = torch.full(
                (len(tensors["n_steps"]),), len(all_episodes), dtype=torch.long
            )
            all_episodes.append(tensors)
            episode_info.append(
                {
                    "repo_id": spec["repo_id"],
                    "root": spec["root"],
                    "episode_index": ep_index,
                    "source": spec["source"],
                    "episode_success": outcome,
                    "rlt_interval_count": int(tensors["rlt_phase_id"][tensors["rlt_phase_id"] >= 0].unique().numel()),
                }
            )
            print(
                f"Cached {spec['repo_id']} episode={ep_index} transitions={len(tensors['n_steps'])}",
                flush=True,
            )
    if not all_episodes:
        raise ValueError("No nonempty labeled episodes were found")
    tensors = {key: torch.cat([episode[key] for episode in all_episodes]) for key in all_episodes[0]}
    heads = RLTHeadConfig(
        token_dim=tensors["vla_tokens"].shape[-1],
        token_pool_size=args.token_pool_size,
        proprio_dim=tensors["proprio"].shape[-1],
        action_dim=tensors["reference_actions"].shape[-1],
        chunk_size=args.chunk_size,
    )
    payload = {
        "metadata": {
            "head_config": asdict(heads),
            "contract": contract,
            "episodes": episode_info,
            "phase_schema": dict(PHASE_SCHEMA),
            "reference_stride": stride,
            "reference_semantics": "causal offline RTC counterfactual; not network timing replay",
        },
        "tensors": tensors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"Saved {len(tensors['n_steps'])} RLT transitions to {output}", flush=True)
    return payload


def main() -> None:
    build_cache(make_parser().parse_args())


if __name__ == "__main__":
    main()
