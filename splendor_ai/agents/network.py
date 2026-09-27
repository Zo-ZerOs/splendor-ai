"""Actor-critic network shared by all 4 self-play seats (parameter
sharing). Small MLP trunk with a policy head (logits over the fixed
149-action space) and a value head. Illegal-action masking is applied by
the caller (self-play collector / RLAgent), not inside the network, so
the network stays a pure function of the observation.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from splendor_ai.env.action_space import ACTION_SPACE_SIZE
from splendor_ai.env.observation import OBSERVATION_DIM

NEG_INF = -1e9


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBSERVATION_DIM, action_dim: int = ACTION_SPACE_SIZE, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def masked_logits(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(obs)
        return logits.masked_fill(mask == 0, NEG_INF)

    def act(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
        """Returns (action, log_prob, value) for a batch of observations."""
        logits, value = self.forward(obs)
        logits = logits.masked_fill(mask == 0, NEG_INF)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        """Returns (log_prob, entropy, value) for PPO's update phase."""
        logits, value = self.forward(obs)
        logits = logits.masked_fill(mask == 0, NEG_INF)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value
