"""Common interface all baseline and RL agents implement.

An agent only ever needs to decide the CURRENT player's next move --
main action, token return, or noble choice are all just "the next
Action" from the engine's point of view (see splendor_ai.game.engine).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from splendor_ai.config import GameConfig
from splendor_ai.game.state import GameState


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def act(self, state: GameState, config: GameConfig):
        """Return one legal Action for state.current_player."""
        raise NotImplementedError

    def explain(self, state: GameState, config: GameConfig, action) -> str:
        """Short human-readable rationale, printed by the terminal CLI
        after an AI turn. Subclasses should override for anything more
        informative than the generic fallback."""
        return f"{self.name} chose {action.kind()}"
