"""Reserve-then-bridge strategy: identify a good-value tier-2 or tier-3
card (cost/point ratio around 2-2.5 for tier-2, 1.75-2 for tier-3) that is
"reachable" -- some visible tier-1 card's permanent bonus matches one of
the target's cost colors, so buying that tier-1 "bridge" card later
discounts the target for free (permanent bonuses reduce effective cost
1-for-1 in that color, see PlayerState.effective_cost). Reserves the
target early (denial + a free gold token if the bank has one), then works
toward completing the combo: buy the bridge card when affordable, and
otherwise pick 2 tokens of a single color the target (or the bridge)
still needs, rather than spreading across colors.

Built to test the specific "reserve a big card banking on a bonus you
don't have yet" tempo idea against strategies that only ever look at
what's affordable right now.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from splendor_ai.agents.agent_utils import (
    default_noble_choice,
    default_return_choice,
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

# "Good value" floors from the strategy's own cost/point language, expressed
# as points/cost (the inverse, since that's what's convenient to rank by):
# tier-2 cost/point <= 2.5, tier-3 cost/point <= 2.0.
MIN_WORTH_TIER2 = 1.0 / 2.5
MIN_WORTH_TIER3 = 1.0 / 2.0


class SynergyReserveAgent(Agent):
    name = "synergy_reserve"

    def __init__(self):
        self.target_card_id: Optional[str] = None
        self.bridge_card_id: Optional[str] = None

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

    def _bridge_for(self, state: GameState, target: Card) -> Optional[Card]:
        """A visible tier-1 card whose bonus matches one of `target`'s cost
        colors -- buying it later discounts `target` for free."""
        needed_colors = {c for c, n in target.cost.items() if n > 0}
        candidates = [c for c in state.board.visible[1] if c is not None and c.bonus in needed_colors]
        if not candidates:
            return None
        return min(candidates, key=lambda c: sum(c.cost.values()))  # cheapest bridge first

    def _find_synergy_target(self, state: GameState) -> Tuple[Optional[Card], Optional[Card]]:
        best, best_bridge, best_worth = None, None, -1.0
        for tier, floor in ((2, MIN_WORTH_TIER2), (3, MIN_WORTH_TIER3)):
            for card in state.board.visible[tier]:
                if card is None or self._worth(card) < floor:
                    continue
                bridge = self._bridge_for(state, card)
                if bridge is None:
                    continue
                w = self._worth(card)
                if w > best_worth:
                    best, best_bridge, best_worth = card, bridge, w
        return best, best_bridge

    def _current_target(self, player: PlayerState) -> Optional[Card]:
        if self.target_card_id is None:
            return None
        for card in player.reserved_cards:
            if card.id == self.target_card_id:
                return card
        return None  # bought or gone -- re-target

    def _current_bridge(self, state: GameState) -> Optional[Card]:
        if self.bridge_card_id is None:
            return None
        for card in state.board.visible[1]:
            if card is not None and card.id == self.bridge_card_id:
                return card
        return None  # bought (by us or someone else) or gone

    def _pick_double_toward(self, pick_actions, needed: Dict[Color, int]):
        """Prefer taking 2 tokens of a single color still needed at least 2
        of, over spreading across colors -- the point of the combo is
        stacking one color fast enough to afford the bridge card and the
        reservation together, not a balanced spread."""
        doubles = [a for a in pick_actions if len(a.colors) == 2 and a.colors[0] == a.colors[1]]
        eligible = [a for a in doubles if needed.get(a.colors[0], 0) >= 2]
        if not eligible:
            return None
        return max(eligible, key=lambda a: needed[a.colors[0]])

    def _choose_main(self, state: GameState, config: GameConfig, player: PlayerState):
        target = self._current_target(player)

        buy_actions = legal_buy_actions(state, player)
        if target is not None:
            for a in buy_actions:
                if resolve_card(state, player, a).id == target.id:
                    self.target_card_id = None
                    self.bridge_card_id = None
                    return a  # combo complete

        if target is None:
            candidate, bridge = self._find_synergy_target(state)
            if candidate is not None:
                for ra in legal_reserve_actions(state, player, config):
                    if ra.source == ReserveSource.VISIBLE and state.board.visible[ra.tier][ra.slot].id == candidate.id:
                        self.target_card_id, self.bridge_card_id = candidate.id, bridge.id
                        return ra
                # Couldn't actually reserve it (e.g. already 3 reserved) -- play normally below.

        bridge = self._current_bridge(state) if target is not None else None
        if bridge is not None:
            for a in buy_actions:
                if resolve_card(state, player, a).id == bridge.id:
                    return a  # bank the discount before it's stolen

        if target is not None:
            needed_target = shortfall(target, player)
            needed_bridge = shortfall(bridge, player) if bridge is not None else {}
            combined = {c: needed_target.get(c, 0) + needed_bridge.get(c, 0) for c in set(needed_target) | set(needed_bridge)}
            pick_actions = legal_pick_actions(state, config)
            double = self._pick_double_toward(pick_actions, combined)
            if double is not None:
                return double
            chosen = pick_action_toward(pick_actions, {c for c, n in combined.items() if n > 0})
            if chosen is not None:
                return chosen

        if buy_actions:
            return max(buy_actions, key=lambda a: resolve_card(state, player, a).points)

        pick_actions = legal_pick_actions(state, config)
        if pick_actions:
            return max(pick_actions, key=lambda a: len(set(a.colors)))

        reserve_actions = legal_reserve_actions(state, player, config)
        if reserve_actions:
            return reserve_actions[0]
        return PassAction()

    def _choose_return(self, state: GameState, config: GameConfig, player: PlayerState):
        actions = legal_return_actions(state, player, config)
        target = self._current_target(player)
        needed = {c for c, n in shortfall(target, player).items() if n > 0} if target is not None else set()
        return default_return_choice(actions, needed)

    def explain(self, state: GameState, config: GameConfig, action) -> str:
        info = f" (target={self.target_card_id}, bridge={self.bridge_card_id})" if self.target_card_id else ""
        return f"{self.name}{info} chose {action.kind()}"
