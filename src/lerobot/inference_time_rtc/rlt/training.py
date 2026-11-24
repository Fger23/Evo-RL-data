# SPDX-License-Identifier: Apache-2.0
# Adapted from MINT-SJTU/Evo-RLT (Apache-2.0), core/{algorithm,losses}.py.
# https://github.com/MINT-SJTU/Evo-RLT
# RTC changes: normalized raw VLA references, masks, protected prefixes, safe offline
# caches, explicit terminal handling, and gamma**actual_steps for truncated chunks.
"""Offline RL-token reconstruction followed by chunk actor/twin-Q training.

Caches are tensor-only .pt dictionaries with ``metadata.head_config``,
``metadata.contract`` and ``tensors``. References are the unfiltered RTC VLA
chunks. Optional client output S-G is included in actor/target Q evaluation.
All actions and proprioception are already normalized by
the VLA processors; images and robot control are never used here. Critic samples
cover the full episode; token reconstruction selects only marked-start samples.
Row masks restrict the actor/BC objective to those intervals while full-episode
rewards and TD successors remain intact.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional

from ..action_filter import filter_joint_targets_tensor
from .networks import RLTHeadBundle, RLTHeadConfig, protect_prefix, validate_contract


def validate_cache(
    payload: dict, *, require_transitions: bool
) -> tuple[RLTHeadConfig, dict, dict[str, Tensor]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("Cache requires metadata and tensors dictionaries")
    metadata, tensors = payload["metadata"], payload.get("tensors")
    config = RLTHeadConfig(**metadata["head_config"])
    contract = metadata["contract"]
    validate_contract(contract, config)
    if not isinstance(tensors, dict) or "vla_tokens" not in tensors:
        raise ValueError("Cache tensors must include vla_tokens")
    tokens = tensors["vla_tokens"]
    if not isinstance(tokens, Tensor) or tokens.ndim != 3 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
        raise ValueError("vla_tokens must have nonempty [N,M,D] shape")
    if tokens.shape[-1] != config.token_dim or not torch.isfinite(tokens).all():
        raise ValueError("VLA token dimension or finite-value check failed")
    if min(tokens.shape[1], config.token_pool_size or tokens.shape[1]) < config.num_rl_tokens:
        raise ValueError("Token reconstruction requires token_count >= num_rl_tokens")
    count, horizon, actions = tokens.shape[0], config.chunk_size, config.action_dim
    marked = tensors.get("rlt_action_mask")
    if not isinstance(marked, Tensor) or marked.shape != (count, horizon) or marked.dtype != torch.bool:
        raise ValueError("Cache requires boolean rlt_action_mask [N,H]; rebuild marked-interval caches")
    if not marked[:, 0].any():
        raise ValueError("Cache requires at least one marked-start sample for RL-token training")
    if not require_transitions:
        return config, contract, tensors
    expected = {
        "proprio": (count, config.proprio_dim),
        "reference_actions": (count, horizon, actions),
        "executed_actions": (count, horizon, actions),
        "next_vla_tokens": tuple(tokens.shape),
        "next_proprio": (count, config.proprio_dim),
        "next_reference_actions": (count, horizon, actions),
        "rewards": (count, horizon),
        "terminated": (count,),
        "n_steps": (count,),
        "protected_steps": (count,),
        "next_protected_steps": (count,),
        "action_mask": (count, horizon),
        "next_action_mask": (count, horizon),
        "next_rlt_action_mask": (count, horizon),
    }
    for key, shape in expected.items():
        value = tensors.get(key)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Cache tensor {key} must have shape {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Cache tensor {key} contains nonfinite values")
    for key in ("n_steps", "protected_steps", "next_protected_steps"):
        if tensors[key].dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f"{key} must be an integer tensor")
    for key in ("action_mask", "next_action_mask", "next_rlt_action_mask", "terminated"):
        if tensors[key].dtype != torch.bool:
            raise ValueError(f"{key} must be boolean")
    steps = tensors["n_steps"]
    if (steps < 1).any() or (steps > horizon).any():
        raise ValueError("n_steps must count 1..chunk_size actually executed actions")
    valid = torch.arange(horizon, device=steps.device)[None, :] < steps[:, None]
    if not torch.equal(tensors["action_mask"], valid):
        raise ValueError("action_mask must contain exactly the first n_steps valid rows")
    for key in ("protected_steps", "next_protected_steps"):
        if (tensors[key] < 0).any() or (tensors[key] > horizon).any():
            raise ValueError(f"{key} must lie in [0, chunk_size]")
    if (~tensors["next_action_mask"].any(dim=1) & ~tensors["terminated"]).any():
        raise ValueError("Nonterminal transitions require a valid next action mask")
    if (marked & ~tensors["action_mask"]).any() or (
        tensors["next_rlt_action_mask"] & ~tensors["next_action_mask"]
    ).any():
        raise ValueError("RLT interval masks cannot include padded action rows")
    if (tensors["rewards"][~valid] != 0).any():
        raise ValueError("Rewards beyond n_steps must be padded with zeros")
    return config, contract, tensors


def load_cache(path: str | Path, *, require_transitions: bool = False):
    return validate_cache(
        torch.load(path, map_location="cpu", weights_only=True), require_transitions=require_transitions
    )


def _batch(tensors: dict[str, Tensor], indices: Tensor, device: torch.device) -> dict[str, Tensor]:
    return {
        key: value[indices].to(
            device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype
        )
        for key, value in tensors.items()
        if isinstance(value, Tensor) and value.ndim and value.shape[0] == tensors["vla_tokens"].shape[0]
    }


def _check_training_args(steps: int, batch_size: int, learning_rate: float) -> None:
    if steps < 1 or batch_size < 1 or not 0 < learning_rate < float("inf"):
        raise ValueError("Training steps, batch size, and learning rate must be positive")


def _log_progress(metric: dict, log_every: int, total_steps: int) -> None:
    if log_every > 0 and (
        metric["step"] == 1 or metric["step"] == total_steps or metric["step"] % log_every == 0
    ):
        print(json.dumps(metric, allow_nan=False), flush=True)


def _save_progress(bundle: RLTHeadBundle, path: str | Path | None, every: int, step: int) -> None:
    if path is not None and every > 0 and step % every == 0:
        bundle.save(path)
        print(json.dumps({"checkpoint": str(path), "saved_step": step}), flush=True)


def train_token(
    bundle: RLTHeadBundle,
    tensors: dict[str, Tensor],
    *,
    steps: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    grad_clip: float = 1.0,
    log_every: int = 0,
    checkpoint_path: str | Path | None = None,
    save_every: int = 0,
) -> list[dict]:
    """Train only the reconstruction bottleneck; invalidate any previously fitted actor."""
    _check_training_args(steps, batch_size, learning_rate)
    validate_cache(
        {"metadata": {"head_config": asdict(bundle.config), "contract": bundle.contract}, "tensors": tensors},
        require_transitions=False,
    )
    optimizer = torch.optim.Adam(bundle.rl_token.parameters(), lr=learning_rate)
    marked_starts = tensors["rlt_action_mask"][:, 0].nonzero(as_tuple=False).flatten()
    device = next(bundle.parameters()).device
    bundle.rl_token.requires_grad_(True).train()
    bundle.training_updates["critic"] = bundle.training_updates["actor"] = 0
    metrics = []
    for step in range(steps):
        rows = marked_starts[torch.randint(len(marked_starts), (batch_size,))]
        tokens = tensors["vla_tokens"][rows].to(device=device, dtype=torch.float32)
        loss = bundle.rl_token.reconstruction_loss(tokens)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite RL-token reconstruction loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.rl_token.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        bundle.training_updates["token"] += 1
        metrics.append({"step": step + 1, "reconstruction_loss": float(loss.detach())})
        _log_progress(metrics[-1], log_every, steps)
        _save_progress(bundle, checkpoint_path, save_every, step + 1)
    bundle.eval()
    return metrics


def discounted_chunk_return(rewards: Tensor, n_steps: Tensor, gamma: float) -> Tensor:
    """Ignore padded rewards explicitly, including for short terminal chunks."""
    indices = torch.arange(rewards.shape[1], device=rewards.device)
    valid = indices[None, :] < n_steps[:, None]
    discount = torch.as_tensor(gamma, device=rewards.device).pow(indices)
    return (rewards * valid * discount).sum(dim=1, keepdim=True)


def td_target(rewards: Tensor, n_steps: Tensor, terminated: Tensor, next_q: Tensor, gamma: float) -> Tensor:
    reward = discounted_chunk_return(rewards, n_steps, gamma)
    discount = torch.as_tensor(gamma, device=rewards.device).pow(n_steps[:, None])
    # torch.where also blocks a terminal transition's irrelevant next-Q value.
    return reward + torch.where(terminated[:, None], torch.zeros_like(next_q), discount * next_q)


def _actor_chunk(
    bundle: RLTHeadBundle,
    state: Tensor,
    reference: Tensor,
    protected: Tensor,
    *,
    training: bool,
    clip: bool = True,
    action_mask: Tensor | None = None,
) -> Tensor:
    mean, _ = bundle.actor(state, reference.flatten(1), training=training)
    mean = mean.reshape_as(reference)
    if clip:
        mean = mean.clamp(-1.0, 1.0)
    return protect_prefix(mean, reference, protected, action_mask)


def train_actor_critic(
    bundle: RLTHeadBundle,
    tensors: dict[str, Tensor],
    *,
    steps: int = 10000,
    batch_size: int = 64,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    gamma: float = 0.99,
    beta: float = 1.0,
    tau: float = 0.005,
    actor_update_interval: int = 2,
    grad_clip: float = 1.0,
    log_every: int = 0,
    checkpoint_path: str | Path | None = None,
    save_every: int = 0,
) -> list[dict]:
    """Canonical clipped double-Q objective plus raw-VLA-reference BC regularization.

    The critic observes actual executed targets (including human actions), never
    substituted VLA predictions. Before output S-G, actor/target/BC preserve RTC
    prefix rows and use VLA actions outside marked intervals. Invalid padded rows
    contribute neither Q input nor BC error; episode reward is never truncated
    at an interval boundary.
    """
    _check_training_args(steps, batch_size, actor_lr)
    _check_training_args(steps, batch_size, critic_lr)
    if not bundle.training_updates["token"]:
        raise ValueError("Train/load the RL-token reconstruction checkpoint before actor-critic training")
    if not 0 <= gamma <= 1 or not 0 <= beta < float("inf") or not 0 < tau <= 1 or actor_update_interval < 1:
        raise ValueError("Invalid gamma, BC beta, Polyak tau, or actor update interval")
    validate_cache(
        {"metadata": {"head_config": asdict(bundle.config), "contract": bundle.contract}, "tensors": tensors},
        require_transitions=True,
    )
    free = torch.arange(bundle.config.chunk_size)[None, :] >= tensors["protected_steps"][:, None]
    if not (free & tensors["action_mask"] & tensors["rlt_action_mask"]).any():
        raise ValueError(
            "Cache contains no marked, unprotected valid actions for actor training; "
            "enlarge marked intervals or check RTC prefix protection"
        )
    device = next(bundle.parameters()).device
    sg_window = bundle.contract["sg_window"]
    joint_columns = tuple(
        index for index, name in enumerate(bundle.contract["action_names"])
        if name.endswith(".pos") and not name.endswith("gripper.pos")
    )
    bundle.rl_token.requires_grad_(False).eval()
    bundle.actor.requires_grad_(True).train()
    bundle.critic.requires_grad_(True).train()
    target_critic = copy.deepcopy(bundle.critic).requires_grad_(False).eval()
    actor_optimizer = torch.optim.Adam(bundle.actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(bundle.critic.parameters(), lr=critic_lr)
    bundle.training_metadata.update(
        gamma=gamma,
        beta=beta,
        tau=tau,
        actor_update_interval=actor_update_interval,
        optimizer_resumed=False,
        target_critic_resumed=False,
    )
    metrics = []
    for step in range(steps):
        batch = _batch(tensors, torch.randint(tensors["vla_tokens"].shape[0], (batch_size,)), device)
        with torch.no_grad():
            state = torch.cat((bundle.rl_token.encode(batch["vla_tokens"]), batch["proprio"]), dim=-1)
            next_state = torch.cat(
                (bundle.rl_token.encode(batch["next_vla_tokens"]), batch["next_proprio"]), dim=-1
            )
            next_actions = _actor_chunk(
                bundle,
                next_state,
                batch["next_reference_actions"],
                batch["next_protected_steps"],
                training=False,
                action_mask=batch["next_rlt_action_mask"],
            )
            next_actions = filter_joint_targets_tensor(
                next_actions, joint_columns, sg_window, batch["next_action_mask"]
            ) * batch["next_action_mask"][:, :, None]
            next_q = target_critic.min_q(next_state, next_actions.flatten(1)).clamp(-100.0, 100.0)
            target = td_target(batch["rewards"], batch["n_steps"], batch["terminated"], next_q, gamma)
        executed = batch["executed_actions"] * batch["action_mask"][:, :, None]
        q1, q2 = bundle.critic(state, executed.flatten(1))
        critic_loss = functional.mse_loss(q1, target) + functional.mse_loss(q2, target)
        if not torch.isfinite(critic_loss):
            raise RuntimeError("Nonfinite RLT twin-Q loss")
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.critic.parameters(), grad_clip, error_if_nonfinite=True)
        critic_optimizer.step()
        bundle.training_updates["critic"] += 1
        row = {"step": step + 1, "critic_loss": float(critic_loss.detach())}
        actor_rows = (
            batch["rlt_action_mask"]
            & batch["action_mask"]
            & (
                torch.arange(bundle.config.chunk_size, device=device)[None, :]
                >= batch["protected_steps"][:, None]
            )
        )
        actor_samples = actor_rows.any(dim=1)
        if step % actor_update_interval == 0 and actor_samples.any():
            bundle.critic.requires_grad_(False)
            actor_state = state[actor_samples]
            actor_reference = batch["reference_actions"][actor_samples]
            actor_protected = batch["protected_steps"][actor_samples]
            actor_marked = batch["rlt_action_mask"][actor_samples]
            actor_valid = batch["action_mask"][actor_samples]
            proposed = _actor_chunk(
                bundle,
                actor_state,
                actor_reference,
                actor_protected,
                training=True,
                clip=False,
                action_mask=actor_marked,
            )
            proposed = proposed * actor_valid[:, :, None]
            # Bound the actor first, then apply the output filter before Q. BC
            # uses the raw mean so out-of-range outputs keep a restoring gradient.
            bounded = (
                protect_prefix(
                    proposed.clamp(-1.0, 1.0),
                    actor_reference,
                    actor_protected,
                    actor_marked,
                )
                * actor_valid[:, :, None]
            )
            bounded = filter_joint_targets_tensor(bounded, joint_columns, sg_window, actor_valid)
            actor_q = bundle.critic.min_q(actor_state, bounded.flatten(1))
            # Sum over chunk/action dimensions, as in Evo-RLT's canonical beta convention.
            reference = actor_reference * actor_valid[:, :, None]
            bc_loss = (proposed - reference).square().sum(dim=(1, 2)).mean()
            actor_loss = -actor_q.mean() + beta * bc_loss
            if not torch.isfinite(actor_loss):
                raise RuntimeError("Nonfinite RLT actor loss")
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(bundle.actor.parameters(), grad_clip, error_if_nonfinite=True)
            actor_optimizer.step()
            bundle.critic.requires_grad_(True)
            bundle.training_updates["actor"] += 1
            row.update(actor_loss=float(actor_loss.detach()), bc_loss=float(bc_loss.detach()))
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target_critic.parameters(), bundle.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, tau)
        metrics.append(row)
        _log_progress(row, log_every, steps)
        _save_progress(bundle, checkpoint_path, save_every, step + 1)
    bundle.eval()
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("token", "ac"):
        command = commands.add_parser(name)
        command.add_argument("--cache", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--device", default="cpu")
        command.add_argument("--steps", default=1000 if name == "token" else 10000, type=int)
        command.add_argument("--batch-size", default=16 if name == "token" else 64, type=int)
        command.add_argument("--seed", default=0, type=int)
        command.add_argument("--log-every", default=100, type=int)
        command.add_argument(
            "--save-every",
            default=1000,
            type=int,
            help="Atomically replace --output every N updates; 0 saves only at completion",
        )
        if name == "token":
            command.add_argument("--learning-rate", default=1e-4, type=float)
        else:
            command.add_argument(
                "--token-checkpoint",
                required=True,
                type=Path,
                help="Trained token or actor checkpoint; weights continue, optimizer/target restart",
            )
            command.add_argument("--actor-lr", default=3e-4, type=float)
            command.add_argument("--critic-lr", default=3e-4, type=float)
            command.add_argument("--gamma", default=0.99, type=float)
            command.add_argument(
                "--beta",
                default=1.0,
                type=float,
                help="Canonical summed squared-distance penalty toward the unfiltered RTC VLA reference",
            )
            command.add_argument("--tau", default=0.005, type=float)
            command.add_argument("--actor-update-interval", default=2, type=int)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    config, contract, tensors = load_cache(args.cache, require_transitions=args.command == "ac")
    if args.command == "token":
        bundle = RLTHeadBundle(config, contract).to(args.device)
        bundle.training_metadata.update(cache=str(args.cache.resolve()), seed=args.seed)
        train_token(
            bundle,
            tensors,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            log_every=args.log_every,
            checkpoint_path=args.output,
            save_every=args.save_every,
        )
    else:
        bundle = RLTHeadBundle.load(args.token_checkpoint, device=args.device, require_actor=False)
        if bundle.config != config or bundle.contract != contract:
            raise ValueError("Token checkpoint config/contract does not match this transition cache")
        bundle.training_metadata.update(
            cache=str(args.cache.resolve()),
            seed=args.seed,
            initialized_from=str(args.token_checkpoint.resolve()),
        )
        train_actor_critic(
            bundle,
            tensors,
            steps=args.steps,
            batch_size=args.batch_size,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            beta=args.beta,
            tau=args.tau,
            actor_update_interval=args.actor_update_interval,
            log_every=args.log_every,
            checkpoint_path=args.output,
            save_every=args.save_every,
        )
    bundle.training_metadata.update(cache=str(args.cache.resolve()), seed=args.seed)
    bundle.save(args.output)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "stage": bundle.trained_stage,
                "training_updates": bundle.training_updates,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
