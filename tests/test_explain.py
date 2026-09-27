"""Unit tests for splendor_ai.cli.explain -- verifies each specific
explanation text without going through the full terminal loop. explain.py
never decides legality itself, so every scenario here first confirms the
action really is illegal (apply_action raises) before checking the
explanation text."""
import pytest

from splendor_ai.cli.explain import explain_illegal_reason
from splendor_ai.data.schemas import Color
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
from splendor_ai.game.engine import IllegalActionError, apply_action, setup_game
from splendor_ai.game.state import TurnPhase


def _assert_illegal_and_explain(state, config, action) -> str:
    with pytest.raises(IllegalActionError):
        apply_action(state, config, action)
    return explain_illegal_reason(action, state, config)


def test_explain_gold_cannot_be_picked(bundle, config):
    state = setup_game(bundle, config, seed=1)
    action = PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GOLD))
    reason = _assert_illegal_and_explain(state, config, action)
    assert "Gold cannot be taken with a Pick action" in reason


def test_explain_two_same_color_insufficient_bank(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.bank[Color.RED] = 3
    reason = _assert_illegal_and_explain(state, config, PickTokensAction(colors=(Color.RED, Color.RED)))
    assert "only 3 red token(s)" in reason
    assert "At least 4 are required" in reason


def test_explain_three_distinct_zero_in_bank(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.bank[Color.BLACK] = 0
    reason = _assert_illegal_and_explain(state, config, PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.BLACK)))
    assert "no black token(s) left" in reason


def test_explain_two_different_while_3plus_available(bundle, config):
    state = setup_game(bundle, config, seed=1)
    reason = _assert_illegal_and_explain(state, config, PickTokensAction(colors=(Color.RED, Color.BLUE)))
    assert "only allowed when fewer than 3" in reason


def test_explain_wrong_subset_when_fewer_than_3_remain(bundle, config):
    state = setup_game(bundle, config, seed=1)
    for c in (Color.GREEN, Color.RED, Color.BLACK):
        state.board.bank[c] = 0
    # Only white and blue remain -- but ask for white and green (green is empty).
    reason = _assert_illegal_and_explain(state, config, PickTokensAction(colors=(Color.WHITE, Color.GREEN)))
    assert "must take exactly one of each: black, blue, red" not in reason  # sanity: not the wrong set
    assert "white, blue" in reason or "blue, white" in reason


def test_explain_buy_visible_empty_slot(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.visible[1][0] = None
    state.board.decks[1] = []
    reason = _assert_illegal_and_explain(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))
    assert "no card in tier 1 slot 1" in reason


def test_explain_buy_reserved_out_of_range(bundle, config):
    state = setup_game(bundle, config, seed=1)
    reason = _assert_illegal_and_explain(state, config, BuyCardAction(source=CardSource.RESERVED, reserved_index=0))
    assert "don't have a reserved card #1" in reason


def test_explain_buy_cannot_afford(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_036"]  # cost blue:4
    state.board.visible[1][0] = card
    reason = _assert_illegal_and_explain(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))
    assert "cannot afford this card" in reason
    assert "gold" in reason


def test_explain_reserve_at_limit(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_001"]
    state.player(0).reserved_cards = [card, card, card]
    reason = _assert_illegal_and_explain(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))
    assert "already have 3 reserved cards" in reason


def test_explain_reserve_blind_empty_deck(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.decks[3] = []
    reason = _assert_illegal_and_explain(state, config, ReserveCardAction(source=ReserveSource.BLIND, tier=3))
    assert "tier 3 deck is empty" in reason


def test_explain_return_wrong_count(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.player(0).tokens[Color.WHITE] = 4
    state.player(0).tokens[Color.BLUE] = 4
    state.player(0).tokens[Color.GREEN] = 3  # 11 total, excess=1
    from splendor_ai.game.engine import _advance_after_main_action

    _advance_after_main_action(state, config)
    assert state.phase == TurnPhase.TOKEN_RETURN

    reason = _assert_illegal_and_explain(state, config, ReturnTokensAction(colors=(Color.WHITE, Color.BLUE)))
    assert "must return exactly 1 token(s); you specified 2" in reason


def test_explain_return_more_than_held(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.player(0).tokens[Color.WHITE] = 11  # trivially over the limit
    from splendor_ai.game.engine import _advance_after_main_action

    _advance_after_main_action(state, config)
    assert state.phase == TurnPhase.TOKEN_RETURN

    reason = _assert_illegal_and_explain(state, config, ReturnTokensAction(colors=(Color.BLACK,)))
    assert "You only hold 0 black token(s)" in reason


def test_explain_noble_not_pending(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.phase = TurnPhase.NOBLE_CHOICE
    state.pending_eligible_nobles = ["noble_001"]
    reason = _assert_illegal_and_explain(state, config, SelectNobleAction(noble_id="noble_999"))
    assert "not currently available to choose" in reason


def test_explain_pass_when_other_actions_exist(bundle, config):
    state = setup_game(bundle, config, seed=1)
    reason = _assert_illegal_and_explain(state, config, PassAction())
    assert "only allowed when no Pick, Buy, or Reserve action is legal" in reason


def test_explain_wrong_phase_main_action_during_token_return(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.player(0).tokens[Color.WHITE] = 11
    from splendor_ai.game.engine import _advance_after_main_action

    _advance_after_main_action(state, config)
    assert state.phase == TurnPhase.TOKEN_RETURN

    reason = _assert_illegal_and_explain(state, config, PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN)))
    assert "pending token return to complete first" in reason


def test_explain_wrong_phase_return_during_main_action(bundle, config):
    state = setup_game(bundle, config, seed=1)
    reason = _assert_illegal_and_explain(state, config, ReturnTokensAction(colors=(Color.WHITE,)))
    assert "no pending token return or noble choice" in reason
