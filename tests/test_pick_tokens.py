import pytest

from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import PickTokensAction
from splendor_ai.game.engine import IllegalActionError, apply_action, setup_game
from splendor_ai.game.rules import legal_pick_actions


def test_three_distinct_colors_is_legal(bundle, config):
    state = setup_game(bundle, config, seed=1)
    action = PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN))
    apply_action(state, config, action)
    player = state.player(0)
    assert player.tokens[Color.WHITE] == 1
    assert player.tokens[Color.BLUE] == 1
    assert player.tokens[Color.GREEN] == 1
    assert state.board.bank[Color.WHITE] == 6


def test_two_same_color_requires_four_in_bank(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.bank[Color.WHITE] = 3
    action = PickTokensAction(colors=(Color.WHITE, Color.WHITE))
    with pytest.raises(IllegalActionError):
        apply_action(state, config, action)


def test_two_same_color_legal_when_four_or_more(bundle, config):
    state = setup_game(bundle, config, seed=1)
    action = PickTokensAction(colors=(Color.WHITE, Color.WHITE))
    apply_action(state, config, action)
    assert state.player(0).tokens[Color.WHITE] == 2
    assert state.board.bank[Color.WHITE] == 5


def test_gold_cannot_be_picked_via_pick_action(bundle, config):
    state = setup_game(bundle, config, seed=1)
    action = PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GOLD))
    with pytest.raises(IllegalActionError):
        apply_action(state, config, action)


def test_pick_action_order_is_irrelevant(bundle, config):
    state = setup_game(bundle, config, seed=1)
    action = PickTokensAction(colors=(Color.GREEN, Color.WHITE, Color.BLUE))
    apply_action(state, config, action)  # should not raise despite unsorted order
    assert state.player(0).total_tokens == 3


def test_fewer_than_three_colors_available(bundle, config):
    state = setup_game(bundle, config, seed=1)
    # Deplete all but 2 normal colors from the bank.
    state.board.bank[Color.WHITE] = 0
    state.board.bank[Color.BLUE] = 0
    state.board.bank[Color.GREEN] = 0
    actions = legal_pick_actions(state, config)
    three_distinct = [a for a in actions if len(set(a.colors)) == 3]
    assert three_distinct == []
    remaining_pick = [a for a in actions if len(a.colors) == 2 and a.colors[0] != a.colors[1]]
    assert len(remaining_pick) == 1
    assert set(remaining_pick[0].colors) == {Color.RED, Color.BLACK}


def test_no_normal_colors_left_means_no_pick_actions(bundle, config):
    state = setup_game(bundle, config, seed=1)
    for c in (Color.WHITE, Color.BLUE, Color.GREEN, Color.RED, Color.BLACK):
        state.board.bank[c] = 0
    assert legal_pick_actions(state, config) == []
