"""Per-player state and pure helper queries used by rules/engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from splendor_ai.data.schemas import Card, Color, Noble, NORMAL_COLORS


@dataclass
class PlayerState:
    seat: int
    name: str
    tokens: Dict[Color, int] = field(default_factory=lambda: {c: 0 for c in Color})
    bonuses: Dict[Color, int] = field(default_factory=lambda: {c: 0 for c in NORMAL_COLORS})
    purchased_cards: List[Card] = field(default_factory=list)
    reserved_cards: List[Card] = field(default_factory=list)
    # Parallel to reserved_cards: "visible" or "blind", recording how each
    # was obtained. This is NOT secret information in physical Splendor --
    # everyone at the table sees a card reserved from a visible slot, only
    # a blind (deck-top) reservation is genuinely hidden -- so the UI uses
    # this to show visible-sourced opponent reservations face up while
    # keeping blind ones face down. Never exposed to RL observations.
    reserved_card_sources: List[str] = field(default_factory=list)
    nobles: List[Noble] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens.values())

    @property
    def prestige(self) -> int:
        return sum(c.points for c in self.purchased_cards) + sum(n.points for n in self.nobles)

    @property
    def purchased_card_count(self) -> int:
        return len(self.purchased_cards)

    def bonus_for(self, color: Color) -> int:
        return self.bonuses.get(color, 0)

    def effective_cost(self, card: Card) -> Dict[Color, int]:
        """Cost remaining per color after permanent bonuses discount it."""
        return {color: max(0, card.cost_for(color) - self.bonus_for(color)) for color in NORMAL_COLORS}

    def gold_needed_for(self, card: Card) -> int:
        remaining = self.effective_cost(card)
        shortfall = 0
        for color in NORMAL_COLORS:
            shortfall += max(0, remaining[color] - self.tokens.get(color, 0))
        return shortfall

    def can_afford(self, card: Card) -> bool:
        return self.gold_needed_for(card) <= self.tokens.get(Color.GOLD, 0)

    def payment_for(self, card: Card) -> Dict[Color, int]:
        """Colored-tokens-first payment breakdown (colored + GOLD keys)."""
        remaining = self.effective_cost(card)
        payment: Dict[Color, int] = {c: 0 for c in Color}
        gold_needed = 0
        for color in NORMAL_COLORS:
            pay_colored = min(remaining[color], self.tokens.get(color, 0))
            payment[color] = pay_colored
            gold_needed += remaining[color] - pay_colored
        payment[Color.GOLD] = gold_needed
        return payment

    def qualifies_for_noble(self, noble: Noble) -> bool:
        return all(self.bonuses.get(color, 0) >= req for color, req in noble.requirements.items())

    def clone(self) -> "PlayerState":
        return PlayerState(
            seat=self.seat,
            name=self.name,
            tokens=dict(self.tokens),
            bonuses=dict(self.bonuses),
            purchased_cards=list(self.purchased_cards),
            reserved_cards=list(self.reserved_cards),
            reserved_card_sources=list(self.reserved_card_sources),
            nobles=list(self.nobles),
        )


def new_player(seat: int, name: str) -> PlayerState:
    return PlayerState(seat=seat, name=name)
