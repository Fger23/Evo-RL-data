"""Exercise checkpoint loading without constructing the full PI05 network."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class TinyPolicy(nn.Module):
    from_pretrained = classmethod(PI05Policy.from_pretrained.__func__)

    def __init__(self, config, **kwargs):
        super().__init__()
        assert config.device == "cpu"
        self.config = config
        self.model = nn.Module()
        self.model.linear = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        self.model.norm = nn.LayerNorm(2, bias=False, dtype=torch.float32)
        self.loaded = False
        self.moves = []

    def _fix_pytorch_state_dict_keys(self, state_dict, config):
        return state_dict

    def load_state_dict(self, state_dict, **kwargs):
        assert all(value.device.type == "cpu" for value in state_dict.values())
        result = super().load_state_dict(state_dict, **kwargs)
        self.loaded = True
        return result

    def to(self, device):
        # Record a requested CUDA transfer without requiring CUDA in the test.
        assert self.loaded
        self.moves.append(device)
        return self


def test_bf16_checkpoint_stays_on_cpu_until_loaded(tmp_path, monkeypatch):
    weights = {
        "linear.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "norm.weight": torch.tensor([5.0, 6.0]),
    }
    checkpoint = tmp_path / "model.safetensors"
    save_file(weights, checkpoint)
    resolver_calls = []

    def resolve(path, filename, **kwargs):
        resolver_calls.append((path, filename, kwargs))
        return str(checkpoint)

    monkeypatch.setattr("transformers.utils.cached_file", resolve)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = SimpleNamespace(device="cuda:0", dtype="bfloat16")
    policy = TinyPolicy.from_pretrained(
        tmp_path, config=config, local_files_only=True, revision="test-revision"
    )

    assert policy.model.linear.weight.dtype == torch.bfloat16
    assert policy.model.norm.weight.dtype == torch.float32
    torch.testing.assert_close(policy.model.linear.weight.float(), weights["linear.weight"])
    torch.testing.assert_close(policy.model.norm.weight, weights["norm.weight"])
    assert policy.moves == ["cuda:0"]
    assert policy.config.device == "cuda:0"
    assert config.device == "cuda:0"
    assert resolver_calls[0][2]["local_files_only"] is True
    assert resolver_calls[0][2]["revision"] == "test-revision"


@pytest.mark.parametrize("failure", ["missing_file", "invalid_keys"])
def test_checkpoint_errors_abort_loading(tmp_path, monkeypatch, failure):
    checkpoint = tmp_path / "model.safetensors"
    if failure == "invalid_keys":
        save_file({"unexpected.weight": torch.ones(2)}, checkpoint)
    monkeypatch.setattr("transformers.utils.cached_file", lambda *args, **kwargs: str(checkpoint))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="Failed to load PI05 pretrained weights"):
        TinyPolicy.from_pretrained(tmp_path, config=SimpleNamespace(device="cpu", dtype="bfloat16"))
