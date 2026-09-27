"""Single-minded strategy: identify the single visible card with the best
"worth" ratio (points / total cost), reserve it as the top priority, then
dedicate every subsequent turn to affording and buying exactly that card
-- ignoring other opportunities along the way. Once bought (or stolen by
an opponent), immediately re-targets the next-best visible card.

Built to test whether a hyper-focused single-target strategy beats a
broader "buy whatever's best right now" approach.
"""
from __future__ import annotations

from typing import Optional

from splendor_ai.agents.agent_utils import (
    default_noble_choice,
    default_return_choice,
    pick_action_toward,
    resolve_card,
    shortfall,
)
from splendor_ai.agents.base import Agent
from splendor_ai.config import GameConfig
from splendor_ai.data.schemas import Card
from splendor_ai.game.actions import PassAction, ReserveSource
from splendor_ai.game.player import PlayerState
from splendor_ai.game.rules import (
    legal_buy_actions,
    legal_noble_choice_actions,
    legal_pick_actions,
    legal_reserve_actions,
    legal_return_actions,
)
from splendor_ai.game.state import GameState, TurnPhase


class ReserveRushAgent(Agent):
    name = "reserve_rush"

    def __init__(self):
        self.target_card_id: Optional[str] = None

    def act(self, state: GameState, config: GameConfig):
        player = state.player(state.current_player)
        if state.phase == TurnPhase.TOKEN_RETURN:
            return self._choose_return(state, config, player)
        if state.phase == TurnPhase.NOBLE_CHOICE:
            return default_noble_choice(legal_noble_choice_actions(state))
        return self._choose_main(state, config, player)

    def _worth(self, card: Card) -> float:
        total_cost = sum(card.cost.values())
        return card.points / total_cost if total_cost > 0 else 0.0

    def _current_target(self, state: GameState, player: PlayerState) -> Optional[Card]:
        if self.target_card_id is None:
            return None
        for card in player.reserved_cards:
            if card.id == self.target_card_id:
                return card
        for tier in (1, 2, 3):
            for card in state.board.visible[tier]:
                if card is not None and card.id == self.target_card_id:
                    return card
        return None  # bought by someone else, or otherwise gone -- re-target

    def _pick_new_target(self, state: GameState) -> Optional[Card]:
        best, best_worth = None, -1.0
        for tier in (1, 2, 3):
            for card in state.board.visible[tier]:
                if card is None:
                    continue
                w = self._worth(card)
                if w > best_worth:
                    best_worth, best = w, card
        return best

    def _choose_main(self, state: GameState, config: GameConfig, player: PlayerState):
        target = self._current_target(state, player)
        if target is None:
            target = self._pick_new_target(state)
            self.target_card_id = target.id if target is not None else None

        buy_actions = legal_buy_actions(state, player)
        if target is not None:
            for a in buy_actions:
                if resolve_card(state, player, a).id == target.id:
                    self.target_card_id = None  # bought -- re-target next turn
                    return a

            already_reserved = any(c.id == target.id for c in player.reserved_cards)
            if not already_reserved:
                for ra in legal_reserve_actions(state, player, config):
                    if ra.source == ReserveSource.VISIBLE and state.board.visible[ra.tier][ra.slot].id == target.id:
                        return ra

        # Can't act toward the target this turn (already reserved, or
        # unaffordable) -- buy whatever else is available rather than
        # wasting the turn. Not gated on `target is None`: a stalled
        # target must never block otherwise-legal progress.
        if buy_actions:
            return max(buy_actions, key=lambda a: resolve_card(state, player, a).points)

        pick_actions = legal_pick_actions(state, config)
        if target is not None and pick_actions:
            needed = {c for c, n in shortfall(target, player).items() if n > 0}
            chosen = pick_action_toward(pick_actions, needed)
            if chosen is not None:
                return chosen

        if pick_actions:
            return max(pick_actions, key=lambda a: len(set(a.colors)))

        reserve_actions = legal_reserve_actions(state, player, config)
        if reserve_actions:
            return reserve_actions[0]
        return PassAction()

    def _choose_return(self, state: GameState, config: GameConfig, player: PlayerState):
        actions = legal_return_actions(state, player, config)
        target = self._current_target(state, player)
        needed = {c for c, n in shortfall(target, player).items() if n > 0} if target is not None else set()
        return default_return_choice(actions, needed)

    def explain(self, state: GameState, config: GameConfig, action) -> str:
        target = f" (target={self.target_card_id})" if self.target_card_id else ""
        return f"{self.name}{target} chose {action.kind()}"
