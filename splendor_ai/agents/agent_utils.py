"""Small mechanical helpers shared by the hand-crafted strategy agents
(heuristic_agent, no_noble_agent, reserve_rush_agent, bonus_engine_agent).

Only pure bookkeeping lives here (resolving a card reference, computing a
token shortfall, generic return/noble tie-breaks) -- never the actual
STRATEGY logic that makes each of these agents behaviorally distinct.
"""
from __future__ import annotations

from typing import Dict, Optional, Set

from splendor_ai.data.schemas import Card, Color
from splendor_ai.game.actions import BuyCardAction, CardSource
from splendor_ai.game.player import PlayerState
from splendor_ai.game.state import GameState


def resolve_card(state: GameState, player: PlayerState, action: BuyCardAction) -> Card:
    if action.source == CardSource.VISIBLE:
        return state.board.visible[action.tier][action.slot]
    return player.reserved_cards[action.reserved_index]


def shortfall(card: Card, player: PlayerState) -> Dict[Color, int]:
    eff = player.effective_cost(card)
    return {c: max(0, eff[c] - player.tokens.get(c, 0)) for c in eff}


def pick_action_toward(pick_actions, needed_colors: Set[Color]):
    """Best legal pick that covers the most of `needed_colors`, or None if
    none of them cover any (caller should fall back to a default pick)."""
    if not pick_actions:
        return None
    best = max(pick_actions, key=lambda a: (len(set(a.colors) & needed_colors), len(set(a.colors))))
    if len(set(best.colors) & needed_colors) > 0:
        return best
    return None


def default_return_choice(actions, needed_colors: Set[Color]):
    def penalty(action) -> int:
        score = 0
        for c in action.colors:
            if c == Color.GOLD:
                score += 100
            elif c in needed_colors:
                score += 10
        return score

    return min(actions, key=penalty)


def default_noble_choice(actions):
    """All base-game nobles are worth equal points, so which one doesn't
    matter for agents that don't specifically strategize around nobles --
    just take the first legally offered."""
    return actions[0]
