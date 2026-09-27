"""The Splendor game engine: the single source of truth for legality and
state transitions. Independent of Gymnasium/PettingZoo and Pygame -- the
RL environment and the terminal/graphical UI both sit on top of this.

A turn is a small state machine driven by `state.phase`:

    MAIN_ACTION --(pick/buy/reserve)--> [token check] --(over limit?)--> TOKEN_RETURN
        \\                                                                    |
         \\--(at/under limit)--> [noble check] <-----------------------------/
                |
                +-- 0 eligible nobles  -> finalize turn -> next player's MAIN_ACTION
                +-- 1 eligible noble   -> auto-grant     -> finalize turn
                +-- 2+ eligible nobles -> NOBLE_CHOICE -> (player picks) -> finalize turn

`apply_action` fully validates before mutating anything -- an illegal
action raises IllegalActionError and the state is left untouched.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Union

from splendor_ai.config import GameConfig
from splendor_ai.data.loader import DataBundle
from splendor_ai.data.schemas import Color, NORMAL_COLORS
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
from splendor_ai.game.player import PlayerState, new_player
from splendor_ai.game.rules import compute_winners, eligible_nobles, legal_actions
from splendor_ai.game.state import Board, GameState, HistoryEntry, TurnPhase

Action = Union[PickTokensAction, BuyCardAction, ReserveCardAction, ReturnTokensAction, SelectNobleAction, PassAction]


class IllegalActionError(Exception):
    def __init__(self, action: Action, phase: TurnPhase, seat: int):
        self.action = action
        self.phase = phase
        self.seat = seat
        super().__init__(f"Illegal action for seat {seat} in phase {phase.value}: {action!r}")


# Kept as a module-level alias (some tests/tools import it by this name):
# the canonicalization logic itself lives in splendor_ai.game.actions so
# both the engine and splendor_ai.env.action_space use the exact same
# normalization.
_canonicalize = canonicalize_action


def setup_game(bundle: DataBundle, config: GameConfig, seed: Optional[int] = None, starting_player: int = 0) -> GameState:
    """Deterministic 4-player setup (given the same seed)."""
    rng = random.Random(seed)

    decks = {}
    for tier in config.tiers:
        cards = list(bundle.cards_by_tier.get(tier, []))
        rng.shuffle(cards)
        decks[tier] = cards  # end of list == top of deck

    visible = {}
    for tier in config.tiers:
        slots = []
        for _ in range(config.visible_slots_per_tier):
            slots.append(decks[tier].pop() if decks[tier] else None)
        visible[tier] = slots

    bank = {c: config.tokens_per_normal_color for c in NORMAL_COLORS}
    bank[Color.GOLD] = config.gold_tokens

    all_nobles = list(bundle.nobles)
    rng.shuffle(all_nobles)
    chosen_nobles = all_nobles[: config.nobles_count_in_play]

    players: List[PlayerState] = [new_player(i, f"Player {i + 1}") for i in range(config.num_players)]
    board = Board(bank=bank, decks=decks, visible=visible, nobles=chosen_nobles)

    return GameState(
        board=board,
        players=players,
        current_player=starting_player,
        phase=TurnPhase.MAIN_ACTION,
        starting_player=starting_player,
        rng=rng,
    )


def is_legal(state: GameState, config: GameConfig, action: Action) -> bool:
    return _canonicalize(action) in legal_actions(state, config)


def get_legal_actions(state: GameState, config: GameConfig) -> List[Action]:
    return legal_actions(state, config)


def _refill_slot(board: Board, tier: int, slot: int) -> None:
    board.visible[tier][slot] = board.decks[tier].pop() if board.decks[tier] else None


def _apply_pick_tokens(state: GameState, action: PickTokensAction) -> str:
    player = state.player(state.current_player)
    for color in action.colors:
        state.board.bank[color] -= 1
        player.tokens[color] += 1
    return f"picked tokens {[c.value for c in action.colors]}"


def _apply_buy_card(state: GameState, action: BuyCardAction) -> Tuple[str, Dict[Color, int]]:
    player = state.player(state.current_player)
    board = state.board

    if action.source == CardSource.VISIBLE:
        card = board.visible[action.tier][action.slot]
    else:
        card = player.reserved_cards[action.reserved_index]

    payment = player.payment_for(card)
    for color, amount in payment.items():
        player.tokens[color] -= amount
        board.bank[color] += amount

    player.bonuses[card.bonus] = player.bonuses.get(card.bonus, 0) + 1
    player.purchased_cards.append(card)

    if action.source == CardSource.VISIBLE:
        _refill_slot(board, action.tier, action.slot)
        desc = f"bought visible T{action.tier}-{action.slot + 1} ({card.id})"
    else:
        player.reserved_cards.pop(action.reserved_index)
        if action.reserved_index < len(player.reserved_card_sources):
            player.reserved_card_sources.pop(action.reserved_index)
        desc = f"bought reserved card ({card.id})"
    return desc, payment


def _apply_reserve_card(state: GameState, action: ReserveCardAction) -> str:
    player = state.player(state.current_player)
    board = state.board

    if action.source == ReserveSource.VISIBLE:
        card = board.visible[action.tier][action.slot]
        player.reserved_cards.append(card)
        player.reserved_card_sources.append("visible")
        _refill_slot(board, action.tier, action.slot)
        desc = f"reserved visible T{action.tier}-{action.slot + 1} ({card.id})"
    else:
        card = board.decks[action.tier].pop()
        player.reserved_cards.append(card)
        player.reserved_card_sources.append("blind")
        desc = f"reserved face-down top of tier {action.tier} ({card.id})"

    if board.bank[Color.GOLD] > 0:
        board.bank[Color.GOLD] -= 1
        player.tokens[Color.GOLD] += 1
        desc += " and received 1 gold"

    return desc


def _apply_return_tokens(state: GameState, action: ReturnTokensAction) -> str:
    player = state.player(state.current_player)
    for color in action.colors:
        player.tokens[color] -= 1
        state.board.bank[color] += 1
    return f"returned tokens {[c.value for c in action.colors]}"


def _apply_select_noble(state: GameState, action: SelectNobleAction) -> str:
    player = state.player(state.current_player)
    board = state.board
    noble = next(n for n in board.nobles if n.id == action.noble_id)
    board.nobles.remove(noble)
    player.nobles.append(noble)
    state.pending_eligible_nobles = []
    return f"was visited by noble {noble.id}"


def _advance_after_main_action(state: GameState, config: GameConfig) -> None:
    player = state.player(state.current_player)
    if player.total_tokens > config.max_tokens_held:
        state.phase = TurnPhase.TOKEN_RETURN
    else:
        _check_nobles_and_advance(state, config)


def _check_nobles_and_advance(state: GameState, config: GameConfig) -> None:
    player = state.player(state.current_player)
    elig = eligible_nobles(player, state.board)
    if len(elig) == 0:
        _finalize_turn(state, config)
    elif len(elig) == 1:
        noble = elig[0]
        state.board.nobles.remove(noble)
        player.nobles.append(noble)
        state.history.append(HistoryEntry(state.turns_completed, player.seat, "noble", f"was visited by noble {noble.id}", action=SelectNobleAction(noble_id=noble.id)))
        _finalize_turn(state, config)
    else:
        state.phase = TurnPhase.NOBLE_CHOICE
        state.pending_eligible_nobles = [n.id for n in elig]


def _finalize_turn(state: GameState, config: GameConfig) -> None:
    player = state.player(state.current_player)
    state.turns_completed += 1

    if state.consecutive_passes >= state.num_players:
        # Every player passed in a row: nothing in the state can ever
        # change again (no tokens can be picked, reserved, or spent), so
        # the game can never progress further. This is a real, if very
        # rare, structural deadlock reachable under the official rules
        # (e.g. all players simultaneously hit the 3-reserved-card limit
        # while the bank has run dry of normal-color tokens and no one's
        # holdings match any visible card). Score it as final immediately
        # rather than looping forever.
        state.game_over = True
        state.phase = TurnPhase.GAME_OVER
        state.winners = compute_winners(state)
        return

    if state.final_round_end_at is None and player.prestige >= config.winning_score:
        n = state.num_players
        state.final_round_end_at = ((state.turns_completed + n - 1) // n) * n

    if state.final_round_end_at is not None and state.turns_completed >= state.final_round_end_at:
        state.game_over = True
        state.phase = TurnPhase.GAME_OVER
        state.winners = compute_winners(state)
        return

    state.current_player = (state.current_player + 1) % state.num_players
    state.phase = TurnPhase.MAIN_ACTION


def apply_action(state: GameState, config: GameConfig, action: Action) -> GameState:
    """Validate and apply `action` to `state` in place. Raises
    IllegalActionError (without mutating state) if the action is not
    currently legal."""
    canonical = _canonicalize(action)
    if canonical not in legal_actions(state, config):
        raise IllegalActionError(action, state.phase, state.current_player)

    seat = state.current_player
    phase = state.phase

    if phase == TurnPhase.MAIN_ACTION:
        payment = None
        if isinstance(canonical, PickTokensAction):
            desc = _apply_pick_tokens(state, canonical)
            kind = "pick_tokens"
        elif isinstance(canonical, BuyCardAction):
            desc, payment = _apply_buy_card(state, canonical)
            kind = "buy_card"
        elif isinstance(canonical, ReserveCardAction):
            desc = _apply_reserve_card(state, canonical)
            kind = "reserve_card"
        elif isinstance(canonical, PassAction):
            desc = "passed (no legal Pick/Buy/Reserve action available)"
            kind = "pass"
        else:
            raise IllegalActionError(action, phase, seat)
        state.consecutive_passes = state.consecutive_passes + 1 if kind == "pass" else 0
        state.history.append(HistoryEntry(state.turns_completed, seat, kind, desc, action=canonical, payment=payment))
        _advance_after_main_action(state, config)

    elif phase == TurnPhase.TOKEN_RETURN:
        if not isinstance(canonical, ReturnTokensAction):
            raise IllegalActionError(action, phase, seat)
        desc = _apply_return_tokens(state, canonical)
        state.history.append(HistoryEntry(state.turns_completed, seat, "return_tokens", desc, action=canonical))
        _check_nobles_and_advance(state, config)

    elif phase == TurnPhase.NOBLE_CHOICE:
        if not isinstance(canonical, SelectNobleAction):
            raise IllegalActionError(action, phase, seat)
        desc = _apply_select_noble(state, canonical)
        state.history.append(HistoryEntry(state.turns_completed, seat, "select_noble", desc, action=canonical))
        _finalize_turn(state, config)

    else:
        raise IllegalActionError(action, phase, seat)

    return state
