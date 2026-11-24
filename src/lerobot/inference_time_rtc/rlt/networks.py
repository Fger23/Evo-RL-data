# SPDX-License-Identifier: Apache-2.0
# Adapted from MINT-SJTU/Evo-RLT (Apache-2.0), src/evo_rlt/core/{rl_token,actor,critic}.py.
# https://github.com/MINT-SJTU/Evo-RLT
# RTC adaptations: full unfiltered VLA chunks, protected prefixes, explicit deployment
# contract, shape checks, and tensor-only checkpoints. No VLA weights are included.
"""Torch-only RL-token bottleneck, absolute chunk actor, and twin Q heads."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

CHECKPOINT_FORMAT = "lerobot.rtc.evo_rlt.v2"


@dataclass(frozen=True)
class RLTHeadConfig:
    token_dim: int = 2048
    token_pool_size: int = 64
    num_rl_tokens: int = 1
    token_heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    token_ff_dim: int = 0  # 0 selects 4 * token_dim.
    chunk_size: int = 50
    action_dim: int = 14
    proprio_dim: int = 14
    actor_hidden_dim: int = 256
    actor_layers: int = 2
    critic_hidden_dim: int = 256
    critic_layers: int = 2
    actor_ref_dropout: float = 0.5
    actor_fixed_std: float = 0.05

    def __post_init__(self):
        positive = (
            "token_dim",
            "num_rl_tokens",
            "token_heads",
            "enc_layers",
            "dec_layers",
            "chunk_size",
            "action_dim",
            "proprio_dim",
            "actor_hidden_dim",
            "actor_layers",
            "critic_hidden_dim",
            "critic_layers",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) <= 0 for name in positive):
            raise ValueError("RLT dimensions and layer counts must be positive integers")
        if self.token_dim % self.token_heads or self.token_pool_size < 0 or self.token_ff_dim < 0:
            raise ValueError("token_dim must be divisible by token_heads; pool/FF sizes must be nonnegative")
        if not 0 <= self.actor_ref_dropout < 1 or not 0 <= self.actor_fixed_std < math.inf:
            raise ValueError("Invalid actor reference dropout or Gaussian standard deviation")


def validate_contract(contract: dict, config: RLTHeadConfig) -> None:
    """Require enough information to reject a different VLA/processor at deployment."""
    required = {"base_model_path", "action_names", "sg_window", "fps", "normalization", "rtc"}
    if not isinstance(contract, dict) or required - contract.keys():
        raise ValueError(f"RLT contract requires {sorted(required)}")
    names = contract["action_names"]
    if not isinstance(names, list) or len(names) != config.action_dim or len(set(names)) != len(names):
        raise ValueError("RLT action_names must give each action dimension once, in execution order")
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("RLT action_names must be nonempty strings")
    if not isinstance(contract["base_model_path"], str) or not contract["base_model_path"]:
        raise ValueError("RLT base_model_path is required")
    if type(contract["sg_window"]) is not int or contract["sg_window"] not in (0, 5, 7):
        raise ValueError("RLT sg_window must be 0, 5, or 7")
    if contract["sg_window"] and contract.get("sg_stage") != "client_output":
        raise ValueError(
            "RLT S-G contract requires sg_stage=client_output; "
            "rebuild the cache and retrain old S-G-reference heads"
        )
    if not isinstance(contract["fps"], (int, float)) or not 0 < contract["fps"] < math.inf:
        raise ValueError("RLT fps must be finite and positive")
    normalization = contract["normalization"]
    if not isinstance(normalization, dict) or not normalization.get("processor_sha256"):
        raise ValueError("RLT normalization requires processor_sha256 of the original VLA processors")
    if not isinstance(contract["rtc"], dict):
        raise ValueError("RLT rtc contract must be a dictionary")
    # Only JSON data belong here: torch.load(weights_only=True) must never need user classes.
    json.dumps(contract, allow_nan=False)


def pool_tokens(tokens: Tensor, pool_size: int) -> Tensor:
    if pool_size and tokens.shape[1] > pool_size:
        return functional.adaptive_avg_pool1d(tokens.transpose(1, 2), pool_size).transpose(1, 2)
    return tokens


class RLTokenModule(nn.Module):
    """Append learned RL tokens; reconstruct detached VLA tokens with a causal decoder."""

    def __init__(self, config: RLTHeadConfig):
        super().__init__()
        self.config = config
        self.rl_token_embed = nn.Parameter(torch.randn(1, config.num_rl_tokens, config.token_dim) * 0.02)
        common = {
            "d_model": config.token_dim,
            "nhead": config.token_heads,
            "dim_feedforward": config.token_ff_dim or 4 * config.token_dim,
            "batch_first": True,
            "norm_first": True,
        }
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**common), num_layers=config.enc_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(**common), num_layers=config.dec_layers
        )
        self.out_proj = nn.Linear(config.token_dim, config.token_dim)

    def encode_multi(self, vla_tokens: Tensor) -> Tensor:
        if vla_tokens.ndim != 3 or vla_tokens.shape[-1] != self.config.token_dim:
            raise ValueError("Expected VLA tokens [batch, token_count, token_dim]")
        tokens = pool_tokens(vla_tokens, self.config.token_pool_size)
        learned = self.rl_token_embed.expand(tokens.shape[0], -1, -1)
        return self.encoder(torch.cat((tokens, learned), dim=1))[:, -self.config.num_rl_tokens :]

    def encode(self, vla_tokens: Tensor) -> Tensor:
        return self.encode_multi(vla_tokens).mean(dim=1)

    def reconstruction_loss(self, vla_tokens: Tensor) -> Tensor:
        target = pool_tokens(vla_tokens.detach(), self.config.token_pool_size)
        memory = self.encode_multi(target)
        count = target.shape[1]
        if count < self.config.num_rl_tokens:
            raise ValueError("Reconstruction needs at least num_rl_tokens input tokens")
        teacher = torch.cat((memory, target[:, : count - self.config.num_rl_tokens]), dim=1)
        causal = torch.triu(torch.ones(count, count, device=target.device, dtype=torch.bool), diagonal=1)
        reconstruction = self.out_proj(self.decoder(teacher, memory, tgt_mask=causal))
        return functional.mse_loss(reconstruction, target)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> nn.Sequential:
    modules = []
    for _ in range(layers):
        modules.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        input_dim = hidden_dim
    modules.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*modules)


class ChunkActor(nn.Module):
    """Canonical absolute action head; the reference is an input, not an additive residual."""

    def __init__(self, config: RLTHeadConfig):
        super().__init__()
        self.config = config
        flat_dim = config.chunk_size * config.action_dim
        self.net = _mlp(
            config.token_dim + config.proprio_dim + flat_dim,
            config.actor_hidden_dim,
            flat_dim,
            config.actor_layers,
        )

    def forward(self, state_vec: Tensor, ref_chunk_flat: Tensor, training: bool = False):
        if training and self.config.actor_ref_dropout:
            keep = torch.rand(ref_chunk_flat.shape[0], 1, device=ref_chunk_flat.device)
            ref_chunk_flat = ref_chunk_flat * (keep >= self.config.actor_ref_dropout).to(ref_chunk_flat.dtype)
        mean = self.net(torch.cat((state_vec, ref_chunk_flat), dim=-1))
        return mean, torch.full_like(mean, self.config.actor_fixed_std)

    def sample(self, state_vec: Tensor, ref_chunk_flat: Tensor, training: bool = False):
        mean, std = self(state_vec, ref_chunk_flat, training=training)
        return mean + std * torch.randn_like(mean), mean


class TwinCritic(nn.Module):
    def __init__(self, config: RLTHeadConfig):
        super().__init__()
        inputs = config.token_dim + config.proprio_dim + config.chunk_size * config.action_dim
        self.q1 = _mlp(inputs, config.critic_hidden_dim, 1, config.critic_layers)
        self.q2 = _mlp(inputs, config.critic_hidden_dim, 1, config.critic_layers)

    def forward(self, state_vec: Tensor, action_flat: Tensor):
        inputs = torch.cat((state_vec, action_flat), dim=-1)
        return self.q1(inputs), self.q2(inputs)

    def min_q(self, state_vec: Tensor, action_flat: Tensor):
        return torch.minimum(*self(state_vec, action_flat))


def protect_prefix(
    proposed: Tensor,
    reference: Tensor,
    protected_steps: int | Tensor = 0,
    action_mask: Tensor | None = None,
) -> Tensor:
    """Keep VLA rows outside the marked interval and inside the RTC-protected prefix."""
    count = torch.as_tensor(protected_steps, device=proposed.device)
    if count.ndim == 0:
        count = count.expand(proposed.shape[0])
    if count.shape != (proposed.shape[0],) or (count < 0).any() or (count > proposed.shape[1]).any():
        raise ValueError("protected_steps must be in [0, chunk_size] for every batch row")
    mask = torch.arange(proposed.shape[1], device=proposed.device)[None, :, None] < count[:, None, None]
    if action_mask is not None:
        if not isinstance(action_mask, Tensor) or action_mask.dtype != torch.bool:
            raise ValueError("RLT action_mask must be a boolean tensor")
        if action_mask.shape == (proposed.shape[1],):
            action_mask = action_mask[None, :].expand(proposed.shape[0], -1)
        if action_mask.shape != proposed.shape[:2]:
            raise ValueError("RLT action_mask must have shape [batch, chunk_size] or [chunk_size]")
        mask = mask | ~action_mask.to(device=proposed.device)[:, :, None]
    return torch.where(mask, reference, proposed)


class RLTHeadBundle(nn.Module):
    """A deployable RL-token encoder and actor, plus critic weights for training."""

    def __init__(self, config: RLTHeadConfig, contract: dict):
        super().__init__()
        validate_contract(contract, config)
        self.config = config
        self.contract = copy.deepcopy(contract)
        self.rl_token = RLTokenModule(config)
        self.actor = ChunkActor(config)
        self.critic = TwinCritic(config)
        self.training_updates = {"token": 0, "critic": 0, "actor": 0}
        self.training_metadata: dict = {}

    @property
    def trained_stage(self) -> str:
        if self.training_updates["actor"] and self.training_updates["critic"]:
            return "actor_critic"
        return "token" if self.training_updates["token"] else "untrained"

    @torch.inference_mode()
    def predict_chunk(
        self,
        tokens: Tensor,
        normalized_proprio: Tensor,
        normalized_vla_reference: Tensor,
        protected_steps: int | Tensor = 0,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        """Correct marked rows only; return the exact reference outside the interval."""
        if self.trained_stage != "actor_critic":
            raise RuntimeError(
                "RLT inference requires a trained actor-critic checkpoint; random heads are forbidden"
            )
        expected = (tokens.shape[0], self.config.chunk_size, self.config.action_dim)
        if normalized_vla_reference.shape != expected:
            raise ValueError(f"RLT reference must be a complete chunk of shape {expected}")
        if normalized_proprio.shape != (tokens.shape[0], self.config.proprio_dim):
            raise ValueError("RLT proprio shape does not match the training contract")
        for value in (tokens, normalized_proprio, normalized_vla_reference):
            if not torch.isfinite(value).all():
                raise ValueError("RLT inputs contain nonfinite values")
        self.eval()
        state = torch.cat((self.rl_token.encode(tokens), normalized_proprio), dim=-1)
        mean, _ = self.actor(state, normalized_vla_reference.flatten(1))
        proposed = mean.reshape(expected).clamp(-1.0, 1.0)
        result = protect_prefix(proposed, normalized_vla_reference, protected_steps, action_mask)
        if not torch.isfinite(result).all():
            raise RuntimeError("Trained RLT actor produced nonfinite actions")
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "config": asdict(self.config),
            "contract": copy.deepcopy(self.contract),
            "stage": self.trained_stage,
            "training_updates": dict(self.training_updates),
            "training_metadata": self.training_metadata,
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
        }
        handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(handle)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu", require_actor: bool = True):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                "Unsupported RTC Evo-RLT checkpoint format; rebuild marked-interval caches "
                "and retrain a v2 head"
            )
        config = RLTHeadConfig(**payload["config"])
        result = cls(config, payload["contract"])
        result.load_state_dict(payload["state_dict"], strict=True)
        updates = payload.get("training_updates", {})
        if set(updates) != {"token", "critic", "actor"} or any(
            type(v) is not int or v < 0 for v in updates.values()
        ):
            raise ValueError("Invalid RLT training update counters")
        result.training_updates = updates
        result.training_metadata = payload.get("training_metadata", {})
        if payload.get("stage") != result.trained_stage:
            raise ValueError("RLT checkpoint stage and update counters disagree")
        if require_actor and (result.trained_stage != "actor_critic" or not updates["token"]):
            raise ValueError("Checkpoint has no trained RL-token encoder and actor-critic")
        if any(not torch.isfinite(v).all() for v in result.state_dict().values()):
            raise ValueError("RLT checkpoint contains nonfinite weights")
        return result.to(device).eval()
