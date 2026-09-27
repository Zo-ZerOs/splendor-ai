import torch

from splendor_ai.agents.network import ActorCritic
from splendor_ai.training.checkpoint import build_network_from_checkpoint, load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint_round_trip(tmp_path):
    network = ActorCritic(obs_dim=10, action_dim=5, hidden_sizes=(16, 16))
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    path = str(tmp_path / "ckpt.pt")

    save_checkpoint(path, network, optimizer, update=42, total_games=7, total_steps=123, extra={"note": "hi"})

    payload = load_checkpoint(path)
    assert payload["meta"]["update"] == 42
    assert payload["meta"]["total_games"] == 7
    assert payload["meta"]["total_steps"] == 123
    assert payload["extra"]["note"] == "hi"
    assert payload["obs_dim"] == 10
    assert payload["action_dim"] == 5

    restored = build_network_from_checkpoint(payload)
    obs = torch.randn(1, 10)
    mask = torch.ones(1, 5, dtype=torch.int8)
    with torch.no_grad():
        logits_a, value_a = network(obs)
        logits_b, value_b = restored(obs)
    assert torch.allclose(logits_a, logits_b)
    assert torch.allclose(value_a, value_b)


def test_checkpoint_directory_created_if_missing(tmp_path):
    network = ActorCritic(obs_dim=4, action_dim=3, hidden_sizes=(8,))
    path = str(tmp_path / "nested" / "dir" / "ckpt.pt")
    save_checkpoint(path, network, None, update=1)
    payload = load_checkpoint(path)
    assert payload["meta"]["update"] == 1
    assert payload["optimizer_state_dict"] is None
