"""Headless game-loop utilities shared by tests, evaluation, and the
terminal CLI's non-interactive paths. Never imports Pygame or any RL
framework -- pure engine + agents.
"""
from __future__ import annotations

from typing import List, Optional

from splendor_ai.agents.base import Agent
from splendor_ai.config import GameConfig
from splendor_ai.data.loader import DataBundle
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.state import GameState

DEFAULT_MAX_TURNS = 2000


def play_game(
    bundle: DataBundle,
    config: GameConfig,
    agents: List[Agent],
    seed: Optional[int] = None,
    starting_player: int = 0,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> GameState:
    """Play one full headless game to completion (or until max_turns
    individual actions have been applied, as a safety valve) and return
    the final GameState."""
    if len(agents) != config.num_players:
        raise ValueError(f"expected {config.num_players} agents, got {len(agents)}")

    state = setup_game(bundle, config, seed=seed, starting_player=starting_player)
    steps = 0
    while not state.game_over and steps < max_turns:
        agent = agents[state.current_player]
        action = agent.act(state, config)
        apply_action(state, config, action)
        steps += 1

    return state
