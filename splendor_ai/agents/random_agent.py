"""Baseline agent: selects uniformly at random among all legal actions.

Used as (a) an evaluation floor for RL training, and (b) an opponent for
manual/headless playtesting.
"""
from __future__ import annotations

import random

from splendor_ai.agents.base import Agent
from splendor_ai.config import GameConfig
from splendor_ai.game.rules import legal_actions
from splendor_ai.game.state import GameState


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, state: GameState, config: GameConfig):
        actions = legal_actions(state, config)
        if not actions:
            raise RuntimeError("RandomAgent asked to act with no legal actions available")
        return self.rng.choice(actions)

    def explain(self, state: GameState, config: GameConfig, action) -> str:
        return f"{self.name} picked uniformly at random ({action.kind()})"
