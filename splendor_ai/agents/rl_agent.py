"""Wraps a trained ActorCritic checkpoint as an Agent, for use by the
terminal CLI (human vs 3 AI) and by evaluate.py's tournaments.
"""
from __future__ import annotations

from typing import Optional

import torch

from splendor_ai.agents.base import Agent
from splendor_ai.agents.network import ActorCritic
from splendor_ai.config import GameConfig
from splendor_ai.data.loader import DataBundle, load_all
from splendor_ai.env.action_mask import compute_action_mask
from splendor_ai.env.action_space import decode_action
from splendor_ai.env.observation import encode_observation
from splendor_ai.game.state import GameState
from splendor_ai.training.checkpoint import build_network_from_checkpoint, load_checkpoint


class RLAgent(Agent):
    def __init__(
        self,
        network: ActorCritic,
        bundle: Optional[DataBundle] = None,
        device: str = "cpu",
        deterministic: bool = True,
        seat_name: str = "rl",
    ):
        self.network = network.to(device)
        self.network.eval()
        self.bundle = bundle if bundle is not None else load_all()
        self.device = device
        self.deterministic = deterministic
        self.name = seat_name
        self._last_value = 0.0

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        bundle: Optional[DataBundle] = None,
        device: str = "cpu",
        deterministic: bool = True,
        seat_name: str = "rl",
    ) -> "RLAgent":
        payload = load_checkpoint(checkpoint_path, map_location=device)
        network = build_network_from_checkpoint(payload)
        return cls(network, bundle=bundle, device=device, deterministic=deterministic, seat_name=seat_name)

    def act(self, state: GameState, config: GameConfig):
        obs = encode_observation(state, config, self.bundle)
        mask = compute_action_mask(state, config)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, _log_prob_t, value_t = self.network.act(obs_t, mask_t, deterministic=self.deterministic)
        self._last_value = float(value_t.item())
        index = int(action_t.item())
        return decode_action(index, state)

    def explain(self, state: GameState, config: GameConfig, action) -> str:
        return f"{self.name} (learned policy, est. value={self._last_value:+.2f}) chose {action.kind()}"
