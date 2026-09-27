"""Engine-building strategy: while permanent bonuses are still sparse
(bootstrap phase), prioritize buying the cheapest affordable card in the
color it has the LEAST bonus coverage in, to diversify discounts fast.
Once enough bonuses are built up, switches to buying the highest-value
affordable card overall -- the "payoff" phase where those bonuses make
otherwise-expensive cards cheap.

Built to test the classic Splendor "small cards for bonuses, then big
cards" engine-building approach against more direct strategies.
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


class BonusEngineAgent(Agent):
    name = "bonus_engine"
    BONUS_THRESHOLD = 3  # total permanent bonuses before switching to payoff mode

    def act(self, state: GameState, config: GameConfig):
        player = state.player(state.current_player)
        if state.phase == TurnPhase.TOKEN_RETURN:
            return self._choose_return(state, config, player)
        if state.phase == TurnPhase.NOBLE_CHOICE:
            return default_noble_choice(legal_noble_choice_actions(state))
        return self._choose_main(state, config, player)

    def _in_bootstrap(self, player: PlayerState) -> bool:
        return sum(player.bonuses.values()) < self.BONUS_THRESHOLD

    def _find_target(self, state: GameState, player: PlayerState) -> Optional[Card]:
        candidates = [c for tier in (1, 2, 3) for c in state.board.visible[tier] if c is not None]
        if not candidates:
            return None
        if self._in_bootstrap(player):
            tier1 = [c for c in candidates if c.tier == 1]
            pool = tier1 or candidates
            return min(pool, key=lambda c: (player.bonuses.get(c.bonus, 0), sum(c.cost.values())))
        return max(candidates, key=lambda c: c.points - 0.3 * sum(shortfall(c, player).values()))

    def _choose_main(self, state: GameState, config: GameConfig, player: PlayerState):
        buy_actions = legal_buy_actions(state, player)
        if buy_actions:
            if self._in_bootstrap(player):
                tier1_buys = [a for a in buy_actions if resolve_card(state, player, a).tier == 1]
                if tier1_buys:
                    return min(tier1_buys, key=lambda a: player.bonuses.get(resolve_card(state, player, a).bonus, 0))
            return max(buy_actions, key=lambda a: resolve_card(state, player, a).points)

        target = self._find_target(state, player)
        reserve_actions = legal_reserve_actions(state, player, config)
        if target is not None and reserve_actions and not self._in_bootstrap(player):
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
        target = self._find_target(state, player)
        needed = {c for c, n in shortfall(target, player).items() if n > 0} if target is not None else set()
        return default_return_choice(actions, needed)

    def explain(self, state: GameState, config: GameConfig, action) -> str:
        player = state.player(state.current_player)
        mode = "bootstrap" if self._in_bootstrap(player) else "payoff"
        return f"{self.name} ({mode}) chose {action.kind()}"
