"""Pure rule functions: legal-action generation, noble eligibility, and
winner determination. These functions never mutate the GameState they are
given -- state mutation is the engine's job (splendor_ai.game.engine).
"""
from __future__ import annotations

import itertools
from typing import List, Union

from splendor_ai.config import GameConfig
from splendor_ai.data.schemas import Color, NORMAL_COLORS, Noble
from splendor_ai.game.actions import (
    BuyCardAction,
    CardSource,
    PassAction,
    PickTokensAction,
    ReserveCardAction,
    ReserveSource,
    ReturnTokensAction,
    SelectNobleAction,
)
from splendor_ai.game.player import PlayerState
from splendor_ai.game.state import Board, GameState, TurnPhase

Action = Union[PickTokensAction, BuyCardAction, ReserveCardAction, ReturnTokensAction, SelectNobleAction, PassAction]


def legal_pick_actions(state: GameState, config: GameConfig) -> List[PickTokensAction]:
    board = state.board
    available_colors = [c for c in NORMAL_COLORS if board.bank[c] > 0]
    actions: List[PickTokensAction] = []

    if len(available_colors) >= 3:
        for combo in itertools.combinations(available_colors, 3):
            actions.append(PickTokensAction(colors=tuple(sorted(combo, key=lambda c: c.value))))
    elif len(available_colors) > 0:
        # Official rule: if fewer than 3 colors remain in the bank, a
        # player may take one of each remaining color instead.
        actions.append(PickTokensAction(colors=tuple(sorted(available_colors, key=lambda c: c.value))))

    for c in NORMAL_COLORS:
        if board.bank[c] >= config.two_same_color_min_bank:
            actions.append(PickTokensAction(colors=(c, c)))

    return actions


def legal_buy_actions(state: GameState, player: PlayerState) -> List[BuyCardAction]:
    board = state.board
    actions: List[BuyCardAction] = []

    for tier in sorted(board.visible.keys()):
        for slot, card in enumerate(board.visible[tier]):
            if card is not None and player.can_afford(card):
                actions.append(BuyCardAction(source=CardSource.VISIBLE, tier=tier, slot=slot))

    for idx, card in enumerate(player.reserved_cards):
        if player.can_afford(card):
            actions.append(BuyCardAction(source=CardSource.RESERVED, reserved_index=idx))

    return actions


def legal_reserve_actions(state: GameState, player: PlayerState, config: GameConfig) -> List[ReserveCardAction]:
    if len(player.reserved_cards) >= config.max_reserved_cards:
        return []

    board = state.board
    actions: List[ReserveCardAction] = []
    for tier in sorted(board.visible.keys()):
        for slot, card in enumerate(board.visible[tier]):
            if card is not None:
                actions.append(ReserveCardAction(source=ReserveSource.VISIBLE, tier=tier, slot=slot))
        if board.deck_remaining(tier) > 0:
            actions.append(ReserveCardAction(source=ReserveSource.BLIND, tier=tier))
    return actions


def legal_return_actions(state: GameState, player: PlayerState, config: GameConfig) -> List[ReturnTokensAction]:
    excess = player.total_tokens - config.max_tokens_held
    if excess <= 0:
        return []

    pool: List[Color] = []
    for color, count in player.tokens.items():
        pool.extend([color] * count)

    combos = set()
    for combo in itertools.combinations(range(len(pool)), excess):
        colors = tuple(sorted((pool[i] for i in combo), key=lambda c: c.value))
        combos.add(colors)

    # Iterating a `set` directly would order results by Python's per-process
    # hash seed, breaking the engine's "deterministic given a seed" guarantee
    # (e.g. two runs of the same game seed could tie-break differently
    # anywhere a caller does `min`/`max` over this list). Sort explicitly by
    # color value so the order is reproducible across processes.
    ordered = sorted(combos, key=lambda colors: tuple(c.value for c in colors))
    return [ReturnTokensAction(colors=c) for c in ordered]


def legal_noble_choice_actions(state: GameState) -> List[SelectNobleAction]:
    return [SelectNobleAction(noble_id=nid) for nid in state.pending_eligible_nobles]


def legal_actions(state: GameState, config: GameConfig) -> List[Action]:
    if state.game_over or state.phase == TurnPhase.GAME_OVER:
        return []

    player = state.player(state.current_player)
    if state.phase == TurnPhase.MAIN_ACTION:
        actions: List[Action] = []
        actions.extend(legal_pick_actions(state, config))
        actions.extend(legal_buy_actions(state, player))
        actions.extend(legal_reserve_actions(state, player, config))
        if not actions:
            # Official-rules fallback: no P/B/R action is available at all
            # (e.g. bank has no normal-color tokens left, nothing is
            # affordable, and 3 cards are already reserved).
            return [PassAction()]
        return actions
    if state.phase == TurnPhase.TOKEN_RETURN:
        return legal_return_actions(state, player, config)
    if state.phase == TurnPhase.NOBLE_CHOICE:
        return legal_noble_choice_actions(state)
    return []


def eligible_nobles(player: PlayerState, board: Board) -> List[Noble]:
    return [n for n in board.nobles if player.qualifies_for_noble(n)]


def compute_ranking(state: GameState) -> List[List[int]]:
    """Return seats grouped into ranked tiers, best first. Each inner list
    is a group of seats tied at that rank (ties reported explicitly)."""
    seats = list(range(state.num_players))

    def key(seat: int):
        p = state.player(seat)
        return (-p.prestige, p.purchased_card_count)

    seats.sort(key=key)
    groups: List[List[int]] = []
    for seat in seats:
        if groups and key(seat) == key(groups[-1][0]):
            groups[-1].append(seat)
        else:
            groups.append([seat])
    return groups


def compute_winners(state: GameState) -> List[int]:
    ranking = compute_ranking(state)
    return ranking[0] if ranking else []
