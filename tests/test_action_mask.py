import random

from splendor_ai.env.action_mask import compute_action_mask
from splendor_ai.env.action_space import (
    ACTION_SPACE_SIZE,
    RESERVE_BLIND_SPECS,
    RESERVE_BLIND_START,
    decode_action,
    encode_action,
)
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.rules import legal_actions


def test_mask_shape_and_dtype(bundle, config):
    state = setup_game(bundle, config, seed=1)
    mask = compute_action_mask(state, config)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype.kind in ("i", "u")


def test_mask_matches_legal_actions_exactly(bundle, config):
    rng = random.Random(1)
    state = setup_game(bundle, config, seed=1)
    for _ in range(50):
        actions = legal_actions(state, config)
        mask = compute_action_mask(state, config)
        expected_indices = {encode_action(a, state) for a in actions}
        actual_indices = set(int(i) for i in mask.nonzero()[0])
        assert actual_indices == expected_indices
        assert mask.sum() == len(actions)  # no duplicate encodings
        action = rng.choice(actions)
        apply_action(state, config, action)
        if state.game_over:
            break


def test_mask_always_has_at_least_one_legal_action_until_game_over(bundle, config):
    rng = random.Random(2)
    state = setup_game(bundle, config, seed=2)
    steps = 0
    while not state.game_over and steps < 500:
        mask = compute_action_mask(state, config)
        assert mask.sum() >= 1
        idx = rng.choice([i for i in range(ACTION_SPACE_SIZE) if mask[i]])
        action = decode_action(idx, state)
        apply_action(state, config, action)
        steps += 1


_BLIND_RESERVE_INDICES = {RESERVE_BLIND_START + i for i in range(len(RESERVE_BLIND_SPECS))}


def test_excluded_indices_are_masked_out_when_otherwise_legal(bundle, config):
    """A blind-reserve action that IS otherwise legal must disappear from
    the mask once excluded (used to train without R D1/D2/D3, see
    TrainingRunConfig.disable_blind_reserve_during_training)."""
    state = setup_game(bundle, config, seed=1)  # turn 1: reserving is always legal (0 reserved cards)
    mask_normal = compute_action_mask(state, config)
    assert any(mask_normal[i] for i in _BLIND_RESERVE_INDICES), "test assumes blind reserve is legal turn 1"

    mask_excluded = compute_action_mask(state, config, excluded_indices=_BLIND_RESERVE_INDICES)
    assert not any(mask_excluded[i] for i in _BLIND_RESERVE_INDICES)
    # Every other legal action must be unaffected.
    for i in range(ACTION_SPACE_SIZE):
        if i not in _BLIND_RESERVE_INDICES:
            assert mask_excluded[i] == mask_normal[i]


def test_excluded_indices_fall_back_when_they_are_the_only_legal_actions(bundle, config):
    """If excluding indices would leave an all-zero mask, the exclusion
    must be skipped entirely rather than returning an invalid empty mask
    -- a categorical action distribution needs at least one option."""
    state = setup_game(bundle, config, seed=1)
    legal = legal_actions(state, config)
    all_indices = {encode_action(a, state) for a in legal}

    # Exclude every currently-legal action, not just blind reserves --
    # simulates the (real, if rare) case where the excluded set happens
    # to be everything that's legal right now.
    mask = compute_action_mask(state, config, excluded_indices=all_indices)
    assert mask.sum() >= 1
    assert set(int(i) for i in mask.nonzero()[0]) == all_indices


def test_excluded_indices_none_or_empty_behaves_like_no_exclusion(bundle, config):
    state = setup_game(bundle, config, seed=1)
    baseline = compute_action_mask(state, config)
    assert (compute_action_mask(state, config, excluded_indices=None) == baseline).all()
    assert (compute_action_mask(state, config, excluded_indices=set()) == baseline).all()
