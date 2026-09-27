"""Fixed-size numeric observation, always encoded from the CURRENT
player's point of view (seat 0 in the encoded vector is always "me").

Hidden-information rules enforced here:
  - Opponents' reserved cards are exposed only as a COUNT, never as
    identities or properties.
  - Deck order/contents are exposed only as a remaining-count per tier.
  - Everything else (own reserved cards, visible cards, bank, nobles) is
    fully observable, matching physical Splendor where all of that sits
    face-up on the table.

Visible/reserved cards are encoded by PROPERTY (tier one-hot, bonus
one-hot, points, cost vector), never by card ID, since which card
occupies a given slot changes every game.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from splendor_ai.config import GameConfig
from splendor_ai.data.loader import DataBundle
from splendor_ai.data.schemas import ALL_COLORS, Card, NORMAL_COLORS
from splendor_ai.game.player import PlayerState
from splendor_ai.game.state import GameState

MAX_RESERVED = 3
NUM_OPPONENTS = 3
TIERS = (1, 2, 3)
SLOTS_PER_TIER = 4
MAX_NOBLES = 5

TOKEN_NORM = 10.0
BONUS_NORM = 10.0
PRESTIGE_NORM = 15.0
PURCHASED_COUNT_NORM = 20.0
POINTS_NORM = 5.0
COST_NORM = 7.0
REQUIREMENT_NORM = 4.0
ROUND_NORM = 60.0

VISIBLE_CARD_DIM = 1 + len(NORMAL_COLORS) + 1 + len(NORMAL_COLORS)  # exists + bonus one-hot + points + cost = 12
RESERVED_CARD_DIM = 1 + len(TIERS) + len(NORMAL_COLORS) + 1 + len(NORMAL_COLORS)  # exists + tier one-hot + bonus one-hot + points + cost = 15
NOBLE_DIM = 1 + len(NORMAL_COLORS) + 1  # exists + requirements + points = 7

OWN_DIM = len(ALL_COLORS) + len(NORMAL_COLORS) + 1 + 1 + MAX_RESERVED * RESERVED_CARD_DIM
OPPONENT_DIM = len(ALL_COLORS) + len(NORMAL_COLORS) + 1 + 1 + 1  # tokens + bonuses + prestige + purchased_count + reserved_count
BANK_DIM = len(ALL_COLORS)
VISIBLE_DIM = len(TIERS) * SLOTS_PER_TIER * VISIBLE_CARD_DIM
DECK_DIM = len(TIERS)
NOBLES_DIM = MAX_NOBLES * NOBLE_DIM
SEAT_DIM = 4  # num_players, fixed for this project
MISC_DIM = 2  # round_norm, final_round_started flag

OBSERVATION_DIM = OWN_DIM + NUM_OPPONENTS * OPPONENT_DIM + BANK_DIM + VISIBLE_DIM + DECK_DIM + NOBLES_DIM + SEAT_DIM + MISC_DIM


def _encode_card(card: Optional[Card], include_tier: bool) -> List[float]:
    if card is None:
        dim = RESERVED_CARD_DIM if include_tier else VISIBLE_CARD_DIM
        return [0.0] * dim
    out = [1.0]
    if include_tier:
        out.extend(1.0 if card.tier == t else 0.0 for t in TIERS)
    out.extend(1.0 if card.bonus == c else 0.0 for c in NORMAL_COLORS)
    out.append(card.points / POINTS_NORM)
    out.extend(card.cost_for(c) / COST_NORM for c in NORMAL_COLORS)
    return out


def _encode_own(player: PlayerState) -> List[float]:
    out = [player.tokens.get(c, 0) / TOKEN_NORM for c in ALL_COLORS]
    out.extend(player.bonuses.get(c, 0) / BONUS_NORM for c in NORMAL_COLORS)
    out.append(player.prestige / PRESTIGE_NORM)
    out.append(player.purchased_card_count / PURCHASED_COUNT_NORM)
    for i in range(MAX_RESERVED):
        card = player.reserved_cards[i] if i < len(player.reserved_cards) else None
        out.extend(_encode_card(card, include_tier=True))
    return out


def _encode_opponent(player: PlayerState) -> List[float]:
    out = [player.tokens.get(c, 0) / TOKEN_NORM for c in ALL_COLORS]
    out.extend(player.bonuses.get(c, 0) / BONUS_NORM for c in NORMAL_COLORS)
    out.append(player.prestige / PRESTIGE_NORM)
    out.append(player.purchased_card_count / PURCHASED_COUNT_NORM)
    out.append(len(player.reserved_cards) / MAX_RESERVED)
    return out


def encode_observation(state: GameState, config: GameConfig, bundle: DataBundle) -> np.ndarray:
    seat = state.current_player
    features: List[float] = []

    features.extend(_encode_own(state.player(seat)))

    for offset in range(1, config.num_players):
        opp_seat = (seat + offset) % config.num_players
        features.extend(_encode_opponent(state.player(opp_seat)))
    # Pad if fewer than NUM_OPPONENTS (shouldn't happen for a fixed 4p game).
    while len(features) < OWN_DIM + NUM_OPPONENTS * OPPONENT_DIM:
        features.extend([0.0] * OPPONENT_DIM)

    features.extend(state.board.bank.get(c, 0) / TOKEN_NORM for c in ALL_COLORS)

    for tier in TIERS:
        for slot in range(SLOTS_PER_TIER):
            card = state.board.visible[tier][slot]
            features.extend(_encode_card(card, include_tier=False))

    tier_sizes = {t: len(bundle.cards_by_tier.get(t, [])) or 1 for t in TIERS}
    features.extend(state.board.deck_remaining(t) / tier_sizes[t] for t in TIERS)

    for i in range(MAX_NOBLES):
        if i < len(state.board.nobles):
            noble = state.board.nobles[i]
            row = [1.0]
            row.extend(noble.requirement_for(c) / REQUIREMENT_NORM for c in NORMAL_COLORS)
            row.append(noble.points / POINTS_NORM)
        else:
            row = [0.0] * NOBLE_DIM
        features.extend(row)

    features.extend(1.0 if s == seat else 0.0 for s in range(SEAT_DIM))

    features.append(state.round_number / ROUND_NORM)
    features.append(1.0 if state.final_round_started else 0.0)

    arr = np.asarray(features, dtype=np.float32)
    assert arr.shape[0] == OBSERVATION_DIM, f"observation dim mismatch: got {arr.shape[0]}, expected {OBSERVATION_DIM}"
    return arr
