"""Checkpoint save/load for the shared self-play network + optimizer."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch

from splendor_ai.agents.network import ActorCritic


@dataclass
class CheckpointMeta:
    update: int
    total_games: int
    total_steps: int
    hidden_sizes: tuple


def save_checkpoint(
    path: str,
    network: ActorCritic,
    optimizer: Optional[torch.optim.Optimizer],
    update: int,
    total_games: int = 0,
    total_steps: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    hidden_sizes = tuple(m.out_features for m in network.trunk if isinstance(m, torch.nn.Linear))
    payload = {
        "network_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "obs_dim": network.obs_dim,
        "action_dim": network.action_dim,
        "hidden_sizes": hidden_sizes,
        "meta": {
            "update": update,
            "total_games": total_games,
            "total_steps": total_steps,
        },
        "extra": extra or {},
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def build_network_from_checkpoint(payload: Dict[str, Any], hidden_sizes=None) -> ActorCritic:
    hidden_sizes = hidden_sizes or payload.get("hidden_sizes", (256, 256))
    network = ActorCritic(obs_dim=payload["obs_dim"], action_dim=payload["action_dim"], hidden_sizes=hidden_sizes)
    network.load_state_dict(payload["network_state_dict"])
    return network
