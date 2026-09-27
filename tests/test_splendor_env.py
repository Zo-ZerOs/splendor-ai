import random

import numpy as np
import pytest

from splendor_ai.config import RewardConfig
from splendor_ai.env.action_space import ACTION_SPACE_SIZE, encode_action
from splendor_ai.env.observation import OBSERVATION_DIM
from splendor_ai.env.splendor_env import SplendorEnv
from splendor_ai.game.actions import PickTokensAction
from splendor_ai.data.schemas import Color


def _masked_random_action(obs, rng):
    legal = [i for i in range(ACTION_SPACE_SIZE) if obs["action_mask"][i]]
    return rng.choice(legal)


def test_reset_returns_valid_observation(bundle, config):
    env = SplendorEnv(bundle=bundle, config=config, seed=1)
    obs, info = env.reset(seed=1)
    assert obs["observation"].shape == (OBSERVATION_DIM,)
    assert obs["action_mask"].shape == (ACTION_SPACE_SIZE,)
    assert obs["action_mask"].sum() >= 1
    assert "seat" in info


def test_masked_random_rollout_terminates(bundle, config):
    env = SplendorEnv(bundle=bundle, config=config, seed=5, max_steps=3000)
    rng = random.Random(5)
    obs, info = env.reset(seed=5)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = _masked_random_action(obs, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert steps < 3000
    assert terminated
    assert "terminal_scores" in info
    assert set(info["terminal_scores"].keys()) == set(range(config.num_players))


def test_terminal_scores_sum_reflects_win_loss(bundle, config):
    env = SplendorEnv(bundle=bundle, config=config, seed=7, max_steps=3000)
    rng = random.Random(7)
    obs, info = env.reset(seed=7)
    terminated = truncated = False
    last_info = info
    while not (terminated or truncated):
        action = _masked_random_action(obs, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info

    scores = last_info["terminal_scores"]
    winners = [s for s, v in scores.items() if v == env.reward_config.win]
    losers = [s for s, v in scores.items() if v == env.reward_config.loss]
    ties = [s for s, v in scores.items() if v == env.reward_config.tie]
    assert len(winners) + len(losers) + len(ties) == config.num_players
    assert not (winners and ties)  # can't have both a solo winner and a tie group


def test_illegal_action_raises(bundle, config):
    env = SplendorEnv(bundle=bundle, config=config, seed=1)
    obs, info = env.reset(seed=1)
    illegal_indices = [i for i in range(ACTION_SPACE_SIZE) if not obs["action_mask"][i]]
    assert illegal_indices
    with pytest.raises(RuntimeError):
        env.step(illegal_indices[0])


def test_step_before_reset_raises(bundle, config):
    env = SplendorEnv(bundle=bundle, config=config, seed=1)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_observation_space_and_action_space_declared(bundle, config):
    env = SplendorEnv(bundle=bundle, config=config, seed=1)
    assert env.action_space.n == ACTION_SPACE_SIZE
    assert env.observation_space["observation"].shape == (OBSERVATION_DIM,)


def test_reset_is_reproducible_with_same_seed(bundle, config):
    env1 = SplendorEnv(bundle=bundle, config=config, seed=123, randomize_starting_player=False)
    env2 = SplendorEnv(bundle=bundle, config=config, seed=123, randomize_starting_player=False)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    assert np.array_equal(obs1["observation"], obs2["observation"])


def test_step_cost_is_deducted_from_a_non_scoring_action(bundle, config):
    """A PICK action (never grants prestige or a bonus) must lose exactly
    `step_cost_coef`, confirming the per-turn efficiency pressure applies
    to every turn, not just buys."""
    reward_cfg = RewardConfig(step_cost_coef=0.01)
    env = SplendorEnv(bundle=bundle, config=config, seed=1, reward_config=reward_cfg)
    obs, info = env.reset(seed=1)

    legal_colors = [c for c in (Color.WHITE, Color.BLUE, Color.GREEN, Color.RED, Color.BLACK) if env.state.board.bank.get(c, 0) > 0]
    action = PickTokensAction(colors=tuple(legal_colors[:3]))
    idx = encode_action(action, env.state)
    assert obs["action_mask"][idx] == 1, "test assumes this pick is legal on turn 1 of a fresh game"

    _, reward, _, _, _ = env.step(idx)
    assert reward == pytest.approx(-0.01)


def test_step_cost_dataclass_default_is_zero():
    """The RewardConfig FIELD default (not configs/training.yaml's current
    tuned value, which is expected to change) must stay 0.0 -- so a
    config file that predates this option, or a bare `RewardConfig()`,
    reproduces the old reward shape exactly."""
    assert RewardConfig().step_cost_coef == 0.0


def _env_with_winner(bundle, config, reward_cfg, seat=0, buys=0, turns=0):
    env = SplendorEnv(bundle=bundle, config=config, seed=1, reward_config=reward_cfg)
    env.reset(seed=1)
    filler_card = bundle.cards[0]
    env.state.player(seat).purchased_cards = [filler_card] * buys
    return env


def test_efficiency_gated_win_over_buy_cap_counts_as_loss(bundle, config):
    reward_cfg = RewardConfig(max_buys_for_win=13, max_turns_for_win=26)
    env = _env_with_winner(bundle, config, reward_cfg, seat=0, buys=14)
    score = env._efficiency_gated_win_score(seat=0, turns=20)  # well under the turn cap
    assert score == reward_cfg.loss


def test_efficiency_gated_win_over_turn_cap_counts_as_loss(bundle, config):
    reward_cfg = RewardConfig(max_buys_for_win=13, max_turns_for_win=26)
    env = _env_with_winner(bundle, config, reward_cfg, seat=0, buys=8)  # well under the buy cap
    score = env._efficiency_gated_win_score(seat=0, turns=27)
    assert score == reward_cfg.loss


def test_efficiency_gated_win_within_caps_gives_win_plus_bonus(bundle, config):
    reward_cfg = RewardConfig(
        max_buys_for_win=13, max_turns_for_win=26,
        buy_efficiency_bonus_coef=0.02, turn_efficiency_bonus_coef=0.01,
    )
    env = _env_with_winner(bundle, config, reward_cfg, seat=0, buys=8)  # 5 under the buy cap
    score = env._efficiency_gated_win_score(seat=0, turns=20)  # 6 under the turn cap
    expected = reward_cfg.win + 0.02 * (13 - 8) + 0.01 * (26 - 20)
    assert score == pytest.approx(expected)


def test_efficiency_gated_win_exactly_at_caps_gives_plain_win(bundle, config):
    reward_cfg = RewardConfig(max_buys_for_win=13, max_turns_for_win=26, buy_efficiency_bonus_coef=0.02, turn_efficiency_bonus_coef=0.01)
    env = _env_with_winner(bundle, config, reward_cfg, seat=0, buys=13)
    score = env._efficiency_gated_win_score(seat=0, turns=26)
    assert score == reward_cfg.win  # zero bonus exactly at the cap, but still a genuine win


def test_efficiency_gating_disabled_by_default_ignores_buy_and_turn_counts(bundle, config):
    """With both caps left at None (the default), a 'win' scores as a
    plain win regardless of how many buys/turns it took -- this mechanism
    must be fully opt-in."""
    reward_cfg = RewardConfig()
    env = _env_with_winner(bundle, config, reward_cfg, seat=0, buys=999)
    score = env._efficiency_gated_win_score(seat=0, turns=999)
    assert score == reward_cfg.win


def test_efficiency_gating_end_to_end_via_real_game(bundle, config):
    """A real completed game where the winner necessarily needed more than
    1 buy to reach 15 points must show up as `loss` in terminal_scores
    when max_buys_for_win=1, proving the gating wires through step()'s
    actual terminal-score computation, not just the helper in isolation."""
    reward_cfg = RewardConfig(max_buys_for_win=1)
    env = SplendorEnv(bundle=bundle, config=config, seed=7, reward_config=reward_cfg, max_steps=3000)
    rng = random.Random(7)
    obs, info = env.reset(seed=7)
    terminated = truncated = False
    last_info = info
    while not (terminated or truncated):
        action = _masked_random_action(obs, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info

    scores = last_info["terminal_scores"]
    winners = [s for s, v in scores.items() if v == reward_cfg.win]
    # No solo winner can have scored a plain `win` here -- winning takes
    # far more than 1 purchased card, so every winner must show as `loss`.
    assert not winners


def test_excluded_action_indices_are_masked_out_of_every_observation(bundle, config):
    from splendor_ai.env.action_space import RESERVE_BLIND_SPECS, RESERVE_BLIND_START

    blind_reserve_indices = {RESERVE_BLIND_START + i for i in range(len(RESERVE_BLIND_SPECS))}
    env = SplendorEnv(bundle=bundle, config=config, seed=1, excluded_action_indices=blind_reserve_indices)
    obs, info = env.reset(seed=1)

    assert not any(obs["action_mask"][i] for i in blind_reserve_indices)
    assert env.action_masks()[list(blind_reserve_indices)].sum() == 0


def test_default_env_never_excludes_any_action(bundle, config):
    env = SplendorEnv(bundle=bundle, config=config, seed=1)
    assert env.excluded_action_indices is None
