"""The fixed, documented RL action-index mapping.

This is a flat `gymnasium.spaces.Discrete(ACTION_SPACE_SIZE)` space built
entirely from BOARD POSITIONS and token-color combinations -- never from
card or noble identities -- so the mapping is stable even though which
card sits in "visible tier-2 slot 3" changes every game. It sits ON TOP
of the engine's structured actions (splendor_ai.game.actions); the engine
itself never sees an action index.

Index layout (documented, do not reorder without a strong reason -- it
changes the meaning of every previously trained checkpoint's policy
head):

    [  0 -  29]  PICK_TOKENS   (30 specs: 5 singles, 5 same-color pairs,
                                 10 distinct pairs, 10 distinct triples --
                                 see PICK_SPECS)
    [ 30 -  41]  BUY_VISIBLE   (12 = 3 tiers x 4 slots)
    [ 42 -  44]  BUY_RESERVED  (3 = reserved slot 0/1/2)
    [ 45 -  56]  RESERVE_VISIBLE (12 = 3 tiers x 4 slots)
    [ 57 -  59]  RESERVE_BLIND (3 = one per tier)
    [ 60      ]  PASS          (only ever legal with zero other options)
    [ 61 - 143]  RETURN_TOKENS (83 = all multisets of size 1..3 over the
                                 6 colors incl. gold -- the only sizes an
                                 over-the-limit turn can ever produce)
    [144 - 148]  SELECT_NOBLE  (5 = index into the *currently pending*
                                 eligible-noble list, i.e. a position, not
                                 a noble identity)

Total size: 149.
"""
from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

from splendor_ai.data.schemas import ALL_COLORS, Color, NORMAL_COLORS
from splendor_ai.game.actions import (
    BuyCardAction,
    CardSource,
    PassAction,
    PickTokensAction,
    ReserveCardAction,
    ReserveSource,
    ReturnTokensAction,
    SelectNobleAction,
    canonicalize_action,
)
from splendor_ai.game.state import GameState

# Canonical color orderings -- MUST match how splendor_ai.game.rules
# canonicalizes PickTokensAction/ReturnTokensAction color tuples (sorted
# by Color.value) so encode/decode round-trips exactly.
_SORTED_NORMAL_COLORS: List[Color] = sorted(NORMAL_COLORS, key=lambda c: c.value)
_SORTED_ALL_COLORS: List[Color] = sorted(ALL_COLORS, key=lambda c: c.value)

TIERS: Tuple[int, ...] = (1, 2, 3)
SLOTS_PER_TIER = 4
MAX_RESERVED_CARDS = 3
MAX_NOBLES_IN_PLAY = 5  # matches configs/game.yaml nobles.count_in_play

# ---- Pick-token specs ----
PICK_SPECS: List[Tuple[Color, ...]] = (
    [(c,) for c in _SORTED_NORMAL_COLORS]  # 5 singles
    + [(c, c) for c in _SORTED_NORMAL_COLORS]  # 5 same-color pairs
    + list(itertools.combinations(_SORTED_NORMAL_COLORS, 2))  # 10 distinct pairs
    + list(itertools.combinations(_SORTED_NORMAL_COLORS, 3))  # 10 distinct triples
)

# ---- Buy / reserve specs (board positions) ----
BUY_VISIBLE_SPECS: List[Tuple[int, int]] = [(t, s) for t in TIERS for s in range(SLOTS_PER_TIER)]
BUY_RESERVED_SPECS: List[int] = list(range(MAX_RESERVED_CARDS))
RESERVE_VISIBLE_SPECS: List[Tuple[int, int]] = [(t, s) for t in TIERS for s in range(SLOTS_PER_TIER)]
RESERVE_BLIND_SPECS: List[int] = list(TIERS)

# ---- Return-token specs: every multiset of size 1..3 over all 6 colors ----
RETURN_SPECS: List[Tuple[Color, ...]] = []
for _k in (1, 2, 3):
    RETURN_SPECS.extend(itertools.combinations_with_replacement(_SORTED_ALL_COLORS, _k))

# ---- Fixed offsets ----
PICK_START = 0
BUY_VISIBLE_START = PICK_START + len(PICK_SPECS)
BUY_RESERVED_START = BUY_VISIBLE_START + len(BUY_VISIBLE_SPECS)
RESERVE_VISIBLE_START = BUY_RESERVED_START + len(BUY_RESERVED_SPECS)
RESERVE_BLIND_START = RESERVE_VISIBLE_START + len(RESERVE_VISIBLE_SPECS)
PASS_INDEX = RESERVE_BLIND_START + len(RESERVE_BLIND_SPECS)
RETURN_START = PASS_INDEX + 1
NOBLE_START = RETURN_START + len(RETURN_SPECS)
ACTION_SPACE_SIZE = NOBLE_START + MAX_NOBLES_IN_PLAY

_PICK_INDEX = {spec: i for i, spec in enumerate(PICK_SPECS)}
_BUY_VISIBLE_INDEX = {spec: i for i, spec in enumerate(BUY_VISIBLE_SPECS)}
_RESERVE_VISIBLE_INDEX = {spec: i for i, spec in enumerate(RESERVE_VISIBLE_SPECS)}
_RESERVE_BLIND_INDEX = {spec: i for i, spec in enumerate(RESERVE_BLIND_SPECS)}
_RETURN_INDEX = {spec: i for i, spec in enumerate(RETURN_SPECS)}


class ActionEncodingError(Exception):
    pass


def encode_action(action, state: Optional[GameState] = None) -> int:
    """Structured Action -> fixed index. `state` is only required to
    encode SelectNobleAction (its index depends on the currently pending
    eligible-noble list, not on noble identity).

    Colors on Pick/Return actions are canonicalized before lookup, so this
    works regardless of what order they're listed in -- important because
    a human-typed command like "P white blue green" is NOT necessarily in
    the same order legal_actions()/PICK_SPECS canonically use, but must
    still map to exactly the same index as the AI's version of that move.
    """
    action = canonicalize_action(action)

    if isinstance(action, PickTokensAction):
        return PICK_START + _PICK_INDEX[tuple(action.colors)]

    if isinstance(action, BuyCardAction):
        if action.source == CardSource.VISIBLE:
            return BUY_VISIBLE_START + _BUY_VISIBLE_INDEX[(action.tier, action.slot)]
        return BUY_RESERVED_START + action.reserved_index

    if isinstance(action, ReserveCardAction):
        if action.source == ReserveSource.VISIBLE:
            return RESERVE_VISIBLE_START + _RESERVE_VISIBLE_INDEX[(action.tier, action.slot)]
        return RESERVE_BLIND_START + _RESERVE_BLIND_INDEX[action.tier]

    if isinstance(action, PassAction):
        return PASS_INDEX

    if isinstance(action, ReturnTokensAction):
        return RETURN_START + _RETURN_INDEX[tuple(action.colors)]

    if isinstance(action, SelectNobleAction):
        if state is None:
            raise ActionEncodingError("encoding a SelectNobleAction requires `state` (pending_eligible_nobles)")
        idx = state.pending_eligible_nobles.index(action.noble_id)
        return NOBLE_START + idx

    raise ActionEncodingError(f"unrecognized action type: {type(action)!r}")


def decode_action(index: int, state: Optional[GameState] = None):
    """Fixed index -> structured Action. `state` is only required to
    decode a SELECT_NOBLE index into an actual noble_id."""
    if not (0 <= index < ACTION_SPACE_SIZE):
        raise ActionEncodingError(f"action index {index} out of range [0, {ACTION_SPACE_SIZE})")

    if PICK_START <= index < BUY_VISIBLE_START:
        return PickTokensAction(colors=PICK_SPECS[index - PICK_START])

    if BUY_VISIBLE_START <= index < BUY_RESERVED_START:
        tier, slot = BUY_VISIBLE_SPECS[index - BUY_VISIBLE_START]
        return BuyCardAction(source=CardSource.VISIBLE, tier=tier, slot=slot)

    if BUY_RESERVED_START <= index < RESERVE_VISIBLE_START:
        return BuyCardAction(source=CardSource.RESERVED, reserved_index=BUY_RESERVED_SPECS[index - BUY_RESERVED_START])

    if RESERVE_VISIBLE_START <= index < RESERVE_BLIND_START:
        tier, slot = RESERVE_VISIBLE_SPECS[index - RESERVE_VISIBLE_START]
        return ReserveCardAction(source=ReserveSource.VISIBLE, tier=tier, slot=slot)

    if RESERVE_BLIND_START <= index < PASS_INDEX:
        return ReserveCardAction(source=ReserveSource.BLIND, tier=RESERVE_BLIND_SPECS[index - RESERVE_BLIND_START])

    if index == PASS_INDEX:
        return PassAction()

    if RETURN_START <= index < NOBLE_START:
        return ReturnTokensAction(colors=RETURN_SPECS[index - RETURN_START])

    # NOBLE_START <= index < ACTION_SPACE_SIZE
    noble_slot = index - NOBLE_START
    if state is None:
        raise ActionEncodingError("decoding a SELECT_NOBLE index requires `state` (pending_eligible_nobles)")
    if noble_slot >= len(state.pending_eligible_nobles):
        raise ActionEncodingError(f"noble slot {noble_slot} has no pending eligible noble in this state")
    return SelectNobleAction(noble_id=state.pending_eligible_nobles[noble_slot])
