"""A greedy card-value strategy that NEVER factors noble progress into its
decisions -- otherwise identical in spirit to HeuristicAgent (buy the best
affordable card, reserve when far short, target tokens toward whatever
card it's building). Built to compare against HeuristicAgent/others and
answer "how much does chasing nobles actually help?"
"""
from __future__ import annotations

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


class NoNobleAgent(Agent):
    name = "no_noble"

    def act(self, state: GameState, config: GameConfig):
        player = state.player(state.current_player)
        if state.phase == TurnPhase.TOKEN_RETURN:
            return self._choose_return(state, config, player)
        if state.phase == TurnPhase.NOBLE_CHOICE:
            return default_noble_choice(legal_noble_choice_actions(state))
        return self._choose_main(state, config, player)

    def _card_value(self, card: Card, player: PlayerState, board) -> float:
        """Same shape as HeuristicAgent._card_value MINUS the noble term --
        this is the entire point of this agent."""
        value = card.points * 3.0
        demand = sum(1 for t in board.visible for c in board.visible[t] if c is not None and c.cost_for(card.bonus) > 0)
        value += 0.1 * demand
        return value

    def _find_target_card(self, state: GameState, player: PlayerState):
        best, best_score = None, float("-inf")
        for tier in (1, 2, 3):
            for card in state.board.visible[tier]:
                if card is None:
                    continue
                shortfall_total = sum(shortfall(card, player).values())
                score = self._card_value(card, player, state.board) - 0.3 * shortfall_total
                if score > best_score:
                    best_score, best = score, card
        return best

    def _choose_main(self, state: GameState, config: GameConfig, player: PlayerState):
        buy_actions = legal_buy_actions(state, player)
        if buy_actions:
            return max(buy_actions, key=lambda a: self._card_value(resolve_card(state, player, a), player, state.board))

        target = self._find_target_card(state, player)
        reserve_actions = legal_reserve_actions(state, player, config)

        if target is not None and reserve_actions:
            shortfall_total = sum(shortfall(target, player).values())
            if target.points >= 1 and shortfall_total >= 3:
                for ra in reserve_actions:
                    if ra.source == ReserveSource.VISIBLE and state.board.visible[ra.tier][ra.slot] is target:
                        return ra

        pick_actions = legal_pick_actions(state, config)
        if pick_actions and target is not None:
            needed = {c for c, n in shortfall(target, player).items() if n > 0}
            chosen = pick_action_toward(pick_actions, needed)
            if chosen is not None:
                return chosen

        if pick_actions:
            return max(pick_actions, key=lambda a: len(set(a.colors)))
        if reserve_actions:
            return reserve_actions[0]
        return PassAction()

    def _choose_return(self, state: GameState, config: GameConfig, player: PlayerState):
        actions = legal_return_actions(state, player, config)
        target = self._find_target_card(state, player)
        needed = {c for c, n in shortfall(target, player).items() if n > 0} if target is not None else set()
        return default_return_choice(actions, needed)

    def explain(self, state: GameState, config: GameConfig, action) -> str:
        return f"{self.name} (ignores nobles) chose {action.kind()}"
