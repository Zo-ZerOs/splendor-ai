import pytest

from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import PickTokensAction, ReturnTokensAction
from splendor_ai.game.engine import IllegalActionError, apply_action, setup_game
from splendor_ai.game.rules import legal_return_actions
from splendor_ai.game.state import TurnPhase


def _give_tokens(state, seat, color, n):
    state.player(seat).tokens[color] += n
    state.board.bank[color] -= n


def test_exceeding_ten_triggers_token_return_phase(bundle, config):
    state = setup_game(bundle, config, seed=1)
    _give_tokens(state, 0, Color.WHITE, 4)
    _give_tokens(state, 0, Color.BLUE, 4)
    # 8 tokens so far; picking 3 more distinct colors -> 11 total, over the limit.
    apply_action(state, config, PickTokensAction(colors=(Color.GREEN, Color.RED, Color.BLACK)))
    assert state.phase == TurnPhase.TOKEN_RETURN
    assert state.current_player == 0
    assert state.player(0).total_tokens == 11


def test_cannot_take_other_actions_while_token_return_pending(bundle, config):
    state = setup_game(bundle, config, seed=1)
    _give_tokens(state, 0, Color.WHITE, 4)
    _give_tokens(state, 0, Color.BLUE, 4)
    apply_action(state, config, PickTokensAction(colors=(Color.GREEN, Color.RED, Color.BLACK)))
    with pytest.raises(IllegalActionError):
        apply_action(state, config, PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN)))


def test_return_tokens_resolves_and_advances_turn(bundle, config):
    state = setup_game(bundle, config, seed=1)
    _give_tokens(state, 0, Color.WHITE, 4)
    _give_tokens(state, 0, Color.BLUE, 4)
    apply_action(state, config, PickTokensAction(colors=(Color.GREEN, Color.RED, Color.BLACK)))
    legal_returns = legal_return_actions(state, state.player(0), config)
    assert all(len(a.colors) == 1 for a in legal_returns)  # excess == 1

    apply_action(state, config, legal_returns[0])
    assert state.player(0).total_tokens == 10
    assert state.phase == TurnPhase.MAIN_ACTION
    assert state.current_player == 1


def test_return_actions_are_exhaustive_over_held_colors(bundle, config):
    state = setup_game(bundle, config, seed=1)
    _give_tokens(state, 0, Color.WHITE, 3)
    _give_tokens(state, 0, Color.BLUE, 3)
    _give_tokens(state, 0, Color.GREEN, 3)
    # 9 tokens; pick 2 more (same color, white has enough bank) -> 11 total, excess 1
    apply_action(state, config, PickTokensAction(colors=(Color.RED, Color.RED)))
    returns = legal_return_actions(state, state.player(0), config)
    returned_colors = {a.colors[0] for a in returns}
    assert returned_colors == {Color.WHITE, Color.BLUE, Color.GREEN, Color.RED}


def test_no_token_return_needed_when_at_limit(bundle, config):
    state = setup_game(bundle, config, seed=1)
    _give_tokens(state, 0, Color.WHITE, 4)
    _give_tokens(state, 0, Color.BLUE, 3)
    # 7 tokens so far; pick 3 more distinct -> exactly 10.
    apply_action(state, config, PickTokensAction(colors=(Color.GREEN, Color.RED, Color.BLACK)))
    assert state.player(0).total_tokens == 10
    assert state.phase == TurnPhase.MAIN_ACTION
    assert state.current_player == 1
