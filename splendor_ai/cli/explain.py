"""Human-readable explanations for WHY an action was rejected.

Critically, this module NEVER decides legality. It is only ever called
AFTER splendor_ai.game.engine.apply_action has already raised
IllegalActionError -- it just turns that rejection into a specific reason
for the terminal player, reusing the exact same state/player/rules
helpers the engine itself uses (never re-implementing any rule). Nothing
here can make an illegal action succeed or a legal one fail; it's a pure
diagnostic layer.
"""
from __future__ import annotations

from typing import Optional

from splendor_ai.config import GameConfig
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
)
from splendor_ai.game.state import GameState, TurnPhase

_MAIN_ACTION_TYPES = (PickTokensAction, BuyCardAction, ReserveCardAction, PassAction)


def explain_wrong_phase(action, state: GameState, config: GameConfig) -> Optional[str]:
    """If `action`'s TYPE can never apply in the CURRENT phase (a pending
    token return or noble choice must be resolved first, or vice versa),
    return a specific explanation. Otherwise None -- the action's type
    matches the phase, so any illegality is about the action's content,
    not its timing (see explain_illegal_reason)."""
    phase = state.phase

    if phase == TurnPhase.TOKEN_RETURN and isinstance(action, _MAIN_ACTION_TYPES):
        excess = state.player(state.current_player).total_tokens - config.max_tokens_held
        return (
            f"You have a pending token return to complete first "
            f"({excess} token(s) over the {config.max_tokens_held}-token limit). "
            f"Use: RETURN <color> [<color> ...]"
        )
    if phase == TurnPhase.NOBLE_CHOICE and isinstance(action, _MAIN_ACTION_TYPES):
        return "You have a pending noble choice to complete first. Use: NOBLE <n>"
    if phase == TurnPhase.MAIN_ACTION and isinstance(action, (ReturnTokensAction, SelectNobleAction)):
        return "There is no pending token return or noble choice right now. Use P, B, or R."
    if phase == TurnPhase.TOKEN_RETURN and isinstance(action, SelectNobleAction):
        return "You have a pending token return to complete first. Use: RETURN <color> [<color> ...]"
    if phase == TurnPhase.NOBLE_CHOICE and isinstance(action, ReturnTokensAction):
        return "You have a pending noble choice to complete first. Use: NOBLE <n>"
    return None


def explain_illegal_reason(action, state: GameState, config: GameConfig) -> str:
    """Best-effort specific reason an already-rejected action is illegal."""
    wrong_phase = explain_wrong_phase(action, state, config)
    if wrong_phase:
        return wrong_phase

    player = state.player(state.current_player)
    board = state.board

    if isinstance(action, PickTokensAction):
        colors = action.colors
        available = [c for c in NORMAL_COLORS if board.bank.get(c, 0) > 0]

        if Color.GOLD in colors:
            return "Gold cannot be taken with a Pick action -- you only receive gold by reserving a card."

        if len(colors) == 3:
            zero = [c for c in colors if board.bank.get(c, 0) == 0]
            if zero:
                names = ", ".join(c.value for c in zero)
                return f"There are no {names} token(s) left in the bank."
            if len(available) < 3:
                names = ", ".join(c.value for c in available)
                return (
                    f"Only {len(available)} color(s) remain in the bank ({names}); "
                    f"you may only take one of each remaining color, e.g. 'P {' '.join(c.value for c in available)}'."
                )
            return "That token pick is not currently legal."

        if len(colors) == 2 and colors[0] == colors[1]:
            c = colors[0]
            have = board.bank.get(c, 0)
            return (
                f"You cannot take two {c.value} tokens because only {have} {c.value} token(s) "
                f"are in the bank. At least {config.two_same_color_min_bank} are required."
            )

        # 1 color, or 2 DIFFERENT colors: only ever legal as "take one of
        # each remaining color" when the bank has fewer than 3 colors left.
        if len(available) >= 3:
            return (
                f"Taking {len(colors)} different color(s) at once is only allowed when fewer than 3 "
                f"normal colors remain in the bank (currently {len(available)} remain). "
                f"Take 3 different colors (e.g. 'P white blue green') or 2 of the same color (e.g. 'P red red')."
            )
        if set(colors) != set(available):
            names = ", ".join(c.value for c in available)
            return f"With only {len(available)} color(s) left in the bank, you must take exactly one of each: {names}."
        return "That token pick is not currently legal."

    if isinstance(action, BuyCardAction):
        if action.source == CardSource.VISIBLE:
            card = board.visible.get(action.tier, [None] * 4)[action.slot] if 0 <= action.slot < len(board.visible.get(action.tier, [])) else None
            if card is None:
                return f"There is no card in tier {action.tier} slot {action.slot + 1}."
        else:
            if action.reserved_index >= len(player.reserved_cards):
                return f"You don't have a reserved card #{action.reserved_index + 1} (you have {len(player.reserved_cards)})."
            card = player.reserved_cards[action.reserved_index]
        remaining = player.effective_cost(card)
        shortfall = {c: max(0, remaining[c] - player.tokens.get(c, 0)) for c in remaining}
        needed_gold = player.gold_needed_for(card)
        have_gold = player.tokens.get(Color.GOLD, 0)
        if needed_gold > have_gold:
            parts = ", ".join(f"{n} more {c.value}" for c, n in shortfall.items() if n > 0)
            return (
                f"You cannot afford this card: after using your colored tokens you still need "
                f"{needed_gold} gold to cover ({parts}), but you only have {have_gold} gold."
            )
        return "That purchase is not currently legal."

    if isinstance(action, ReserveCardAction):
        if len(player.reserved_cards) >= config.max_reserved_cards:
            return f"You already have {config.max_reserved_cards} reserved cards (the maximum)."
        if action.source == ReserveSource.VISIBLE:
            slots = board.visible.get(action.tier, [])
            card = slots[action.slot] if 0 <= action.slot < len(slots) else None
            if card is None:
                return f"There is no card in tier {action.tier} slot {action.slot + 1} to reserve."
        else:
            if board.deck_remaining(action.tier) == 0:
                return f"The tier {action.tier} deck is empty -- there is nothing to reserve face-down."
        return "That reservation is not currently legal."

    if isinstance(action, ReturnTokensAction):
        excess = player.total_tokens - config.max_tokens_held
        given = len(action.colors)
        if excess <= 0:
            return "You are not currently over the token limit -- no return is needed."
        if given != excess:
            return f"You must return exactly {excess} token(s); you specified {given}."
        for c in set(action.colors):
            need = action.colors.count(c)
            have = player.tokens.get(c, 0)
            if need > have:
                return f"You only hold {have} {c.value} token(s), but tried to return {need}."
        return "That token return is not currently legal."

    if isinstance(action, SelectNobleAction):
        if action.noble_id not in state.pending_eligible_nobles:
            return "That noble is not currently available to choose."
        return "That noble choice is not currently legal."

    if isinstance(action, PassAction):
        return "Passing is only allowed when no Pick, Buy, or Reserve action is legal -- type 'legal' to see your options."

    return "That action is not currently legal."
