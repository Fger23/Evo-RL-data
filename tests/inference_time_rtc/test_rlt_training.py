"""Small CPU tests of actual RLT learning and safe checkpoint deployment, without a robot."""

from dataclasses import asdict

import pytest
import torch

from lerobot.inference_time_rtc.rlt.networks import RLTHeadBundle, RLTHeadConfig, protect_prefix
from lerobot.inference_time_rtc.rlt.training import (
    _actor_chunk,
    load_cache,
    main,
    td_target,
    train_actor_critic,
    train_token,
    validate_cache,
)


@pytest.fixture
def tiny_cache():
    torch.manual_seed(7)
    config = RLTHeadConfig(
        token_dim=8,
        token_heads=2,
        token_pool_size=4,
        enc_layers=1,
        dec_layers=1,
        token_ff_dim=16,
        chunk_size=4,
        action_dim=2,
        proprio_dim=2,
        actor_hidden_dim=12,
        actor_layers=1,
        critic_hidden_dim=12,
        critic_layers=1,
    )
    contract = {
        "base_model_path": "/models/pi05",
        "action_names": ["joint1", "joint2"],
        "sg_window": 0,
        "fps": 25.0,
        "normalization": {"processor_sha256": "a" * 64},
        "rtc": {"execution_horizon": 2, "inference_delay": 1},
    }
    count = 3
    tensors = {
        "vla_tokens": torch.randn(count, 5, 8),
        "proprio": torch.randn(count, 2),
        "reference_actions": torch.rand(count, 4, 2) * 2 - 1,
        "executed_actions": torch.rand(count, 4, 2) * 2 - 1,
        "next_vla_tokens": torch.randn(count, 5, 8),
        "next_proprio": torch.randn(count, 2),
        "next_reference_actions": torch.rand(count, 4, 2) * 2 - 1,
        "rewards": torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        "terminated": torch.tensor([False, True, True]),
        "n_steps": torch.tensor([4, 2, 4]),
        "protected_steps": torch.tensor([0, 2, 2]),
        "next_protected_steps": torch.tensor([2, 0, 0]),
        "action_mask": torch.tensor([[True] * 4, [True, True, False, False], [True] * 4]),
        "next_action_mask": torch.tensor([[True] * 4, [False] * 4, [False] * 4]),
        "rlt_action_mask": torch.tensor([[True, True, False, False], [True, True, False, False], [True] * 4]),
        "next_rlt_action_mask": torch.zeros(count, 4, dtype=torch.bool),
    }
    return {"metadata": {"head_config": asdict(config), "contract": contract}, "tensors": tensors}


def test_reconstruction_actor_critic_update_and_roundtrip(tiny_cache, tmp_path):
    config, contract, tensors = validate_cache(tiny_cache, require_transitions=True)
    bundle = RLTHeadBundle(config, contract)
    with pytest.raises(RuntimeError, match="random heads"):
        bundle.predict_chunk(tensors["vla_tokens"], tensors["proprio"], tensors["reference_actions"])
    token_before = bundle.rl_token.rl_token_embed.detach().clone()
    token_metrics = train_token(bundle, tensors, steps=1, batch_size=3)
    assert torch.isfinite(torch.tensor(token_metrics[0]["reconstruction_loss"]))
    assert not torch.equal(token_before, bundle.rl_token.rl_token_embed)
    token_path = tmp_path / "token.pt"
    bundle.save(token_path)
    with pytest.raises(ValueError, match="no trained"):
        RLTHeadBundle.load(token_path)
    bundle = RLTHeadBundle.load(token_path, require_actor=False)
    encoder_before = bundle.rl_token.rl_token_embed.detach().clone()
    actor_before = next(bundle.actor.parameters()).detach().clone()
    critic_before = next(bundle.critic.parameters()).detach().clone()
    metrics = train_actor_critic(bundle, tensors, steps=1, batch_size=8)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics[0].values())
    assert torch.equal(encoder_before, bundle.rl_token.rl_token_embed)
    assert not torch.equal(actor_before, next(bundle.actor.parameters()))
    assert not torch.equal(critic_before, next(bundle.critic.parameters()))
    predicted = bundle.predict_chunk(
        tensors["vla_tokens"], tensors["proprio"], tensors["reference_actions"], tensors["protected_steps"]
    )
    assert predicted.shape == (3, 4, 2) and predicted.isfinite().all()
    assert (predicted.abs() <= 1).all()
    assert torch.equal(predicted[1:, :2], tensors["reference_actions"][1:, :2])
    actor_path = tmp_path / "actor.pt"
    bundle.save(actor_path)
    restored = RLTHeadBundle.load(actor_path)
    assert restored.contract == contract
    assert restored.training_updates == {"token": 1, "critic": 1, "actor": 1}
    torch.testing.assert_close(
        restored.predict_chunk(
            tensors["vla_tokens"],
            tensors["proprio"],
            tensors["reference_actions"],
            tensors["protected_steps"],
        ),
        predicted,
    )


def test_td_target_uses_actual_steps_terminal_and_ignores_padded_rewards():
    rewards = torch.tensor([[1.0, 2.0, 999.0, 999.0], [0.0, 1.0, 999.0, 999.0]])
    result = td_target(
        rewards, torch.tensor([2, 2]), torch.tensor([False, True]), torch.tensor([[8.0], [100.0]]), 0.5
    )
    # Nonterminal: 1+.5*2 + (.5**2)*8; terminal: 0+.5*1.
    torch.testing.assert_close(result, torch.tensor([[4.0], [0.5]]))


def test_protected_prefix_has_no_actor_gradient():
    proposed = torch.randn(2, 4, 2, requires_grad=True)
    reference = torch.randn(2, 4, 2)
    result = protect_prefix(proposed, reference, torch.tensor([1, 3]))
    result.sum().backward()
    assert proposed.grad[0, :1].eq(0).all() and proposed.grad[1, :3].eq(0).all()
    assert proposed.grad[0, 1:].eq(1).all() and proposed.grad[1, 3:].eq(1).all()


def test_interval_mask_and_protected_prefix_both_have_zero_actor_gradient():
    proposed = torch.randn(1, 4, 2, requires_grad=True)
    reference = torch.full((1, 4, 2), 2.0)
    result = protect_prefix(proposed, reference, 1, torch.tensor([True, True, False, False]))
    torch.testing.assert_close(result[:, (0, 2, 3)], reference[:, (0, 2, 3)])
    result.sum().backward()
    assert proposed.grad[:, (0, 2, 3)].eq(0).all()
    assert proposed.grad[:, 1].eq(1).all()
    with pytest.raises(ValueError, match="boolean"):
        protect_prefix(proposed, reference, 0, torch.ones(4))


def test_predict_chunk_preserves_exact_vla_rows_outside_interval(tiny_cache):
    config, contract, tensors = validate_cache(tiny_cache, require_transitions=True)
    bundle = RLTHeadBundle(config, contract)
    train_token(bundle, tensors, steps=1, batch_size=2)
    train_actor_critic(bundle, tensors, steps=1, batch_size=8)
    reference = tensors["reference_actions"].clone()
    reference[:, 2:] = 1.5  # VLA reference rows must not be clipped by the actor.
    result = bundle.predict_chunk(
        tensors["vla_tokens"],
        tensors["proprio"],
        reference,
        protected_steps=1,
        action_mask=torch.tensor([True, True, False, False]),
    )
    assert torch.equal(result[:, (0, 2, 3)], reference[:, (0, 2, 3)])
    assert result[:, 1].abs().le(1).all()


def test_actor_bc_updates_only_marked_action_outputs(tiny_cache):
    config, contract, tensors = validate_cache(tiny_cache, require_transitions=True)
    tensors["rlt_action_mask"][:] = False
    tensors["rlt_action_mask"][:, 0] = True
    tensors["protected_steps"][:] = 0
    bundle = RLTHeadBundle(config, contract)
    train_token(bundle, tensors, steps=1, batch_size=2)
    before = bundle.actor.net[-1].bias.detach().clone().reshape(4, 2)
    train_actor_critic(bundle, tensors, steps=1, batch_size=16)
    after = bundle.actor.net[-1].bias.detach().reshape(4, 2)
    assert not torch.equal(after[0], before[0])
    assert torch.equal(after[1:], before[1:])
    assert bundle.actor.net[-1].bias.grad.reshape(4, 2)[1:].eq(0).all()


def test_unmarked_td_successor_uses_vla_reference_not_actor(tiny_cache):
    config, contract, tensors = validate_cache(tiny_cache, require_transitions=True)
    bundle = RLTHeadBundle(config, contract)
    state = torch.randn(3, config.token_dim + config.proprio_dim)
    reference = tensors["next_reference_actions"]
    result = _actor_chunk(
        bundle,
        state,
        reference,
        tensors["next_protected_steps"],
        training=False,
        action_mask=tensors["next_rlt_action_mask"],
    )
    assert torch.equal(result, reference)


def test_token_training_selects_only_marked_starts_from_full_episode_cache(tiny_cache, monkeypatch):
    config = RLTHeadConfig(**tiny_cache["metadata"]["head_config"])
    bundle = RLTHeadBundle(config, tiny_cache["metadata"]["contract"])
    tiny_cache["tensors"]["rlt_action_mask"][0] = False
    tiny_cache["tensors"]["vla_tokens"][0].fill_(123.0)
    observed = []
    reconstruction = bundle.rl_token.reconstruction_loss

    def capture(tokens):
        observed.append(tokens.clone())
        return reconstruction(tokens)

    monkeypatch.setattr(bundle.rl_token, "reconstruction_loss", capture)
    validate_cache(tiny_cache, require_transitions=True)
    train_token(bundle, tiny_cache["tensors"], steps=2, batch_size=16)
    assert len(observed) == 2
    assert all(not value.eq(123.0).all(dim=(1, 2)).any() for value in observed)


def test_cache_with_no_marked_starts_is_rejected(tiny_cache):
    config = RLTHeadConfig(**tiny_cache["metadata"]["head_config"])
    bundle = RLTHeadBundle(config, tiny_cache["metadata"]["contract"])
    tiny_cache["tensors"]["rlt_action_mask"][:] = False
    with pytest.raises(ValueError, match="at least one marked-start"):
        train_token(bundle, tiny_cache["tensors"], steps=1)


def test_unmarked_batch_still_fits_critic_but_skips_actor_update(tiny_cache, monkeypatch):
    config, contract, tensors = validate_cache(tiny_cache, require_transitions=True)
    bundle = RLTHeadBundle(config, contract)
    train_token(bundle, tensors, steps=1, batch_size=2)
    tensors["rlt_action_mask"][0] = False
    before_critic = next(bundle.critic.parameters()).detach().clone()
    before_actor = next(bundle.actor.parameters()).detach().clone()
    monkeypatch.setattr(torch, "randint", lambda high, size: torch.zeros(size, dtype=torch.long))
    metrics = train_actor_critic(bundle, tensors, steps=1, batch_size=4)
    assert not torch.equal(next(bundle.critic.parameters()), before_critic)
    assert torch.equal(next(bundle.actor.parameters()), before_actor)
    assert bundle.training_updates["critic"] == 1 and bundle.training_updates["actor"] == 0
    assert "actor_loss" not in metrics[0]


def test_all_marked_rows_protected_rejects_actor_training_before_any_update(tiny_cache):
    config, contract, tensors = validate_cache(tiny_cache, require_transitions=True)
    tensors["rlt_action_mask"][:, 2:] = False
    tensors["protected_steps"][:] = 2
    bundle = RLTHeadBundle(config, contract)
    # Token reconstruction still learns from marked observations.
    train_token(bundle, tensors, steps=1, batch_size=2)
    critic_before = next(bundle.critic.parameters()).detach().clone()
    with pytest.raises(ValueError, match="enlarge marked intervals or check RTC prefix"):
        train_actor_critic(bundle, tensors, steps=1)
    assert bundle.training_updates == {"token": 1, "critic": 0, "actor": 0}
    assert torch.equal(next(bundle.critic.parameters()), critic_before)


def test_cache_rejects_interval_mask_on_padding(tiny_cache):
    tiny_cache["tensors"]["rlt_action_mask"][1, 3] = True
    with pytest.raises(ValueError, match="padded action rows"):
        validate_cache(tiny_cache, require_transitions=True)


def test_old_unbounded_head_requires_marked_interval_v2_training(tiny_cache, tmp_path):
    config, contract, _tensors = validate_cache(tiny_cache, require_transitions=True)
    path = tmp_path / "unbounded_v1.pt"
    RLTHeadBundle(config, contract).save(path)
    payload = torch.load(path, weights_only=True)
    payload["format"] = "lerobot.rtc.evo_rlt.v1"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="marked-interval.*v2"):
        RLTHeadBundle.load(path, require_actor=False)


def test_cache_rejects_invalid_action_mask_and_raw_shapes(tiny_cache):
    tiny_cache["tensors"]["action_mask"][1, 3] = True
    with pytest.raises(ValueError, match="first n_steps"):
        validate_cache(tiny_cache, require_transitions=True)


@pytest.mark.parametrize("require_transitions", [False, True])
def test_old_filtered_cache_requires_rebuilding_before_training(tiny_cache, tmp_path, require_transitions):
    tiny_cache["metadata"]["contract"]["sg_window"] = 7
    path = tmp_path / "old_filtered_cache.pt"
    torch.save(tiny_cache, path)
    with pytest.raises(ValueError, match="sg_window=0.*rebuild the cache and retrain"):
        load_cache(path, require_transitions=require_transitions)


def test_old_filtered_head_requires_retraining_before_loading(tiny_cache, tmp_path):
    config, contract, _tensors = validate_cache(tiny_cache, require_transitions=True)
    path = tmp_path / "old_filtered_head.pt"
    RLTHeadBundle(config, contract).save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["contract"]["sg_window"] = 7
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="sg_window=0.*rebuild the cache and retrain"):
        RLTHeadBundle.load(path, require_actor=False)


def test_bc_restores_out_of_bounds_actor_mean(tiny_cache):
    config, contract, tensors = validate_cache(tiny_cache, require_transitions=True)
    bundle = RLTHeadBundle(config, contract)
    train_token(bundle, tensors, steps=1, batch_size=2)
    with torch.no_grad():
        for parameter in bundle.actor.parameters():
            parameter.zero_()
        bundle.actor.net[-1].bias.fill_(3.0)
    before = bundle.actor.net[-1].bias.detach().clone()
    train_actor_critic(bundle, tensors, steps=1, batch_size=32, beta=1.0)
    # Deployable Q inputs are clamped at +1, but raw-mean BC must still correct +3.
    assert (bundle.actor.net[-1].bias < before).any()


def test_training_cli_token_then_ac(tiny_cache, tmp_path, capsys):
    cache = tmp_path / "cache.pt"
    token = tmp_path / "token.pt"
    actor = tmp_path / "actor.pt"
    torch.save(tiny_cache, cache)
    assert load_cache(cache)[0].chunk_size == 4
    main(
        [
            "token",
            "--cache",
            str(cache),
            "--output",
            str(token),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--save-every",
            "1",
        ]
    )
    main(
        [
            "ac",
            "--cache",
            str(cache),
            "--token-checkpoint",
            str(token),
            "--output",
            str(actor),
            "--steps",
            "1",
            "--batch-size",
            "3",
            "--save-every",
            "1",
        ]
    )
    assert RLTHeadBundle.load(actor).trained_stage == "actor_critic"
    output = capsys.readouterr().out
    assert '"saved_step": 1' in output and '"reconstruction_loss"' in output
