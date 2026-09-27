import numpy as np

from splendor_ai.env.observation import OBSERVATION_DIM, encode_observation
from splendor_ai.game.engine import setup_game
from splendor_ai.game.actions import ReserveCardAction, ReserveSource
from splendor_ai.game.engine import apply_action


def test_observation_has_fixed_shape(bundle, config):
    state = setup_game(bundle, config, seed=1)
    obs = encode_observation(state, config, bundle)
    assert obs.shape == (OBSERVATION_DIM,)
    assert obs.dtype == np.float32


def test_observation_shape_is_seat_independent(bundle, config):
    state = setup_game(bundle, config, seed=1)
    for seat in range(config.num_players):
        state.current_player = seat
        obs = encode_observation(state, config, bundle)
        assert obs.shape == (OBSERVATION_DIM,)


def test_opponent_reserved_card_identity_is_not_leaked(bundle, config):
    """Changing WHICH card an opponent has reserved must not change the
    current player's observation -- only the reserved COUNT may appear."""
    state = setup_game(bundle, config, seed=1)
    state.current_player = 0
    opponent = state.player(1)

    card_a = bundle.card_by_id["t1_001"]
    card_b = bundle.card_by_id["t3_020"]  # very different cost/points/bonus

    opponent.reserved_cards = [card_a]
    obs_a = encode_observation(state, config, bundle)

    opponent.reserved_cards = [card_b]
    obs_b = encode_observation(state, config, bundle)

    assert np.array_equal(obs_a, obs_b)


def test_own_reserved_card_identity_is_visible(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.current_player = 0
    me = state.player(0)

    card_a = bundle.card_by_id["t1_001"]
    card_b = bundle.card_by_id["t3_020"]

    me.reserved_cards = [card_a]
    obs_a = encode_observation(state, config, bundle)

    me.reserved_cards = [card_b]
    obs_b = encode_observation(state, config, bundle)

    assert not np.array_equal(obs_a, obs_b)


def test_deck_order_not_encoded_only_count(bundle, config):
    state = setup_game(bundle, config, seed=1)
    obs_before = encode_observation(state, config, bundle)

    # Shuffle the tier-1 deck order without changing its size.
    import random

    rng = random.Random(99)
    rng.shuffle(state.board.decks[1])

    obs_after = encode_observation(state, config, bundle)
    assert np.array_equal(obs_before, obs_after)


def test_deck_count_changes_are_reflected(bundle, config):
    state = setup_game(bundle, config, seed=1)
    obs_before = encode_observation(state, config, bundle)
    state.board.decks[1].pop()
    obs_after = encode_observation(state, config, bundle)
    assert not np.array_equal(obs_before, obs_after)


def test_visible_card_properties_are_encoded_not_ids(bundle, config):
    state = setup_game(bundle, config, seed=1)
    original = state.board.visible[1][0]
    replacement = bundle.card_by_id["t1_001"] if original.id != "t1_001" else bundle.card_by_id["t1_002"]

    obs_before = encode_observation(state, config, bundle)
    state.board.visible[1][0] = replacement
    obs_after = encode_observation(state, config, bundle)

    if original.cost == replacement.cost and original.bonus == replacement.bonus and original.points == replacement.points:
        assert np.array_equal(obs_before, obs_after)
    else:
        assert not np.array_equal(obs_before, obs_after)


def test_current_player_rotation_reindexes_seats(bundle, config):
    """The observation is always from 'my' point of view -- rotating
    current_player should reorder which seat's stats appear first."""
    state = setup_game(bundle, config, seed=1)
    state.player(0).tokens[list(state.player(0).tokens.keys())[0]] = 3
    state.current_player = 0
    obs0 = encode_observation(state, config, bundle)
    state.current_player = 1
    obs1 = encode_observation(state, config, bundle)
    assert not np.array_equal(obs0, obs1)
