"""Adversarial "spoiler" strategy: specifically counters ONE named
opponent seat rather than playing its own independent game plan.

Every turn, it checks whether the target is currently able to afford a
visible card, or is close to it (shortfall <= BLOCK_SHORTFALL_THRESHOLD):
  1. If reservable, RESERVE that card first -- this permanently denies it
     (once reserved by us, the target can never buy it from the board;
     they'd have to catch us reserving something else instead).
  2. Otherwise, PICK TOKENS the target needs for that card -- denying bank
     supply so the target can't complete the purchase as fast.
Only when no such threat is detected does it fall back to playing
productively for itself (buy its own best affordable card, otherwise
generic pick/reserve).

This reads the target's exact holdings and reserved cards directly from
GameState, which is legitimate here: hidden-information rules only
constrain what's exposed to the RL OBSERVATION ENCODER (env/observation.py)
for policy training, not what a hand-crafted heuristic operating directly
on the engine's state can see -- exactly like every other agent in this
project (HeuristicAgent, NoNobleAgent, etc.) already does.
"""
from __future__ import annotations

from typing import Optional, Tuple

from splendor_ai.agents.agent_utils import (
    default_noble_choice,
    pick_action_toward,
    resolve_card,
    shortfall,
)
from splendor_ai.agents.base import Agent
from splendor_ai.config import GameConfig
from splendor_ai.data.schemas import Card, Color
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


class BlockerAgent(Agent):
    name = "blocker"
    BLOCK_SHORTFALL_THRESHOLD = 2  # "about to buy" = within this many missing tokens

    def __init__(self, target_seat: int):
        self.target_seat = target_seat

    def act(self, state: GameState, config: GameConfig):
        player = state.player(state.current_player)
        if state.phase == TurnPhase.TOKEN_RETURN:
            return self._choose_return(state, config, player)
        if state.phase == TurnPhase.NOBLE_CHOICE:
            return default_noble_choice(legal_noble_choice_actions(state))
        return self._choose_main(state, config, player)

    def _target(self, state: GameState) -> PlayerState:
        return state.player(self.target_seat)

    def _target_threat_card(self, state: GameState) -> Tuple[Optional[Card], Optional[int]]:
        """The visible card the target is closest to affording (lowest
        shortfall), and that shortfall amount."""
        target = self._target(state)
        best, best_shortfall = None, None
        for tier in (1, 2, 3):
            for card in state.board.visible[tier]:
                if card is None:
                    continue
                sf = sum(shortfall(card, target).values())
                if best is None or sf < best_shortfall:
                    best, best_shortfall = card, sf
        return best, best_shortfall

    def _own_best_buy(self, state: GameState, player: PlayerState):
        buy_actions = legal_buy_actions(state, player)
        if not buy_actions:
            return None
        return max(buy_actions, key=lambda a: resolve_card(state, player, a).points)

    def _choose_main(self, state: GameState, config: GameConfig, player: PlayerState):
        threat_card, threat_shortfall = self._target_threat_card(state)
        is_threat = threat_card is not None and threat_shortfall is not None and threat_shortfall <= self.BLOCK_SHORTFALL_THRESHOLD

        if is_threat:
            reserve_actions = legal_reserve_actions(state, player, config)
            for ra in reserve_actions:
                if ra.source == ReserveSource.VISIBLE and state.board.visible[ra.tier][ra.slot] is threat_card:
                    return ra  # deny it permanently

            pick_actions = legal_pick_actions(state, config)
            needed = {c for c, n in shortfall(threat_card, self._target(state)).items() if n > 0}
            chosen = pick_action_toward(pick_actions, needed)
            if chosen is not None:
                return chosen  # deny bank supply even if we can't reserve it

        # No active threat (or nothing we can do about it): play for ourselves.
        own_buy = self._own_best_buy(state, player)
        if own_buy is not None:
            return own_buy

        pick_actions = legal_pick_actions(state, config)
        if pick_actions:
            return max(pick_actions, key=lambda a: len(set(a.colors)))

        reserve_actions = legal_reserve_actions(state, player, config)
        if reserve_actions:
            return reserve_actions[0]
        return PassAction()

    def _choose_return(self, state: GameState, config: GameConfig, player: PlayerState):
        actions = legal_return_actions(state, player, config)
        threat_card, _ = self._target_threat_card(state)
        # Keep whatever the target needs (deny it), return everything else.
        target_needed = {c for c, n in shortfall(threat_card, self._target(state)).items() if n > 0} if threat_card is not None else set()

        def penalty(action) -> int:
            score = 0
            for c in action.colors:
                if c == Color.GOLD:
                    score += 100
                elif c in target_needed:
                    score += 50  # strongly avoid giving back denial tokens
            return score

        return min(actions, key=penalty)

    def explain(self, state: GameState, config: GameConfig, action) -> str:
        return f"{self.name} (targeting seat {self.target_seat + 1}) chose {action.kind()}"
