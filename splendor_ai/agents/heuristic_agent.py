"""A hand-crafted greedy baseline, stronger than random, used to sanity
check that RL training is actually learning something rather than just
beating a uniform-random opponent.

Priorities (in order): buy the best affordable scoring/bonus card ->
reserve a strong card we can't afford yet if we're far short and slots
remain -> pick tokens targeted at whatever card we're building toward ->
fall back to any legal pick/reserve -> pass only if nothing else is legal.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from splendor_ai.agents.base import Agent
from splendor_ai.config import GameConfig
from splendor_ai.data.schemas import Card, Color
from splendor_ai.game.actions import BuyCardAction, CardSource, PassAction, ReserveSource
from splendor_ai.game.player import PlayerState
from splendor_ai.game.rules import (
    legal_buy_actions,
    legal_noble_choice_actions,
    legal_pick_actions,
    legal_reserve_actions,
    legal_return_actions,
)
from splendor_ai.game.state import GameState, TurnPhase


class HeuristicAgent(Agent):
    name = "heuristic"

    def act(self, state: GameState, config: GameConfig):
        player = state.player(state.current_player)
        if state.phase == TurnPhase.TOKEN_RETURN:
            return self._choose_return(state, config, player)
        if state.phase == TurnPhase.NOBLE_CHOICE:
            return self._choose_noble(state)
        return self._choose_main(state, config, player)

    # ---------------- scoring helpers ----------------

    def _card_value(self, card: Card, player: PlayerState, board) -> float:
        value = card.points * 3.0
        demand = sum(1 for t in board.visible for c in board.visible[t] if c is not None and c.cost_for(card.bonus) > 0)
        value += 0.1 * demand
        for noble in board.nobles:
            req = noble.requirement_for(card.bonus)
            have = player.bonus_for(card.bonus)
            if 0 < have < req:
                value += 1.0
        return value

    def _shortfall(self, card: Card, player: PlayerState) -> Dict[Color, int]:
        eff = player.effective_cost(card)
        return {c: max(0, eff[c] - player.tokens.get(c, 0)) for c in eff}

    def _resolve_card(self, state: GameState, player: PlayerState, action: BuyCardAction) -> Card:
        if action.source == CardSource.VISIBLE:
            return state.board.visible[action.tier][action.slot]
        return player.reserved_cards[action.reserved_index]

    def _find_target_card(self, state: GameState, player: PlayerState) -> Optional[Card]:
        best, best_score = None, float("-inf")
        for tier in (1, 2, 3):
            for card in state.board.visible[tier]:
                if card is None:
                    continue
                shortfall_total = sum(self._shortfall(card, player).values())
                score = self._card_value(card, player, state.board) - 0.3 * shortfall_total
                if score > best_score:
                    best_score, best = score, card
        return best

    # ---------------- decisions ----------------

    def _choose_main(self, state: GameState, config: GameConfig, player: PlayerState):
        buy_actions = legal_buy_actions(state, player)
        if buy_actions:
            return max(buy_actions, key=lambda a: self._card_value(self._resolve_card(state, player, a), player, state.board))

        target = self._find_target_card(state, player)
        reserve_actions = legal_reserve_actions(state, player, config)

        if target is not None and reserve_actions:
            shortfall_total = sum(self._shortfall(target, player).values())
            if target.points >= 1 and shortfall_total >= 3:
                for ra in reserve_actions:
                    if ra.source == ReserveSource.VISIBLE and state.board.visible[ra.tier][ra.slot] is target:
                        return ra

        pick_actions = legal_pick_actions(state, config)
        if pick_actions and target is not None:
            shortfall = self._shortfall(target, player)
            needed_colors = {c for c, n in shortfall.items() if n > 0}
            best_pick = max(pick_actions, key=lambda a: (len(set(a.colors) & needed_colors), len(set(a.colors))))
            if len(set(best_pick.colors) & needed_colors) > 0:
                return best_pick

        if pick_actions:
            return max(pick_actions, key=lambda a: len(set(a.colors)))

        if reserve_actions:
            return reserve_actions[0]

        return PassAction()

    def _choose_return(self, state: GameState, config: GameConfig, player: PlayerState):
        actions = legal_return_actions(state, player, config)
        target = self._find_target_card(state, player)
        needed = set(self._shortfall(target, player)) if target is None else {c for c, n in self._shortfall(target, player).items() if n > 0}

        def penalty(action) -> int:
            score = 0
            for c in action.colors:
                if c == Color.GOLD:
                    score += 100
                elif c in needed:
                    score += 10
            return score

        return min(actions, key=penalty)

    def _choose_noble(self, state: GameState):
        actions = legal_noble_choice_actions(state)

        def noble_points(action) -> int:
            noble = next(n for n in state.board.nobles if n.id == action.noble_id)
            return noble.points

        return max(actions, key=noble_points)

    def explain(self, state: GameState, config: GameConfig, action) -> str:
        return f"{self.name} greedily chose {action.kind()}"
