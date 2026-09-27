"""Legal-action mask over the fixed RL action space."""
from __future__ import annotations

from typing import Optional, Set

import numpy as np

from splendor_ai.config import GameConfig
from splendor_ai.env.action_space import ACTION_SPACE_SIZE, encode_action
from splendor_ai.game.rules import legal_actions
from splendor_ai.game.state import GameState


def compute_action_mask(state: GameState, config: GameConfig, excluded_indices: Optional[Set[int]] = None) -> np.ndarray:
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    for action in legal_actions(state, config):
        mask[encode_action(action, state)] = 1

    if excluded_indices:
        reduced = mask.copy()
        for idx in excluded_indices:
            reduced[idx] = 0
        if reduced.sum() > 0:
            return reduced
        # Excluding these would leave NO legal action at all -- fall back
        # to the unreduced mask rather than return an invalid empty one
        # (a categorical action distribution needs at least one option).

    return mask
