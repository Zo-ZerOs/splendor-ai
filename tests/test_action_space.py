import random

import pytest

from splendor_ai.env.action_space import (
    ACTION_SPACE_SIZE,
    NOBLE_START,
    decode_action,
    encode_action,
)
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.rules import legal_actions


def test_action_space_size_matches_documentation():
    assert ACTION_SPACE_SIZE == 149


def test_encode_decode_round_trip_for_all_non_noble_indices(bundle, config):
    for idx in range(NOBLE_START):
        action = decode_action(idx)
        assert encode_action(action) == idx


def test_encode_decode_round_trip_for_every_legal_action_in_random_games(bundle, config):
    rng = random.Random(0)
    for seed in range(20):
        state = setup_game(bundle, config, seed=seed)
        turns = 0
        while not state.game_over and turns < 300:
            actions = legal_actions(state, config)
            for action in actions:
                idx = encode_action(action, state)
                assert 0 <= idx < ACTION_SPACE_SIZE
                decoded = decode_action(idx, state)
                assert decoded == action
            action = rng.choice(actions)
            apply_action(state, config, action)
            turns += 1


def test_decode_noble_requires_state():
    with pytest.raises(Exception):
        decode_action(NOBLE_START)


def test_decode_out_of_range_raises():
    with pytest.raises(Exception):
        decode_action(-1)
    with pytest.raises(Exception):
        decode_action(ACTION_SPACE_SIZE)


def test_pick_action_index_independent_of_input_color_order(bundle, config):
    from splendor_ai.data.schemas import Color
    from splendor_ai.game.actions import PickTokensAction

    a1 = PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN))
    a2 = PickTokensAction(colors=(Color.GREEN, Color.WHITE, Color.BLUE))
    # encode_action expects canonical (sorted) order, as produced by rules.py;
    # verify that canonical form is what round-trips.
    from splendor_ai.game.engine import _canonicalize

    assert encode_action(_canonicalize(a1)) == encode_action(_canonicalize(a2))
