"""A tiny, fast end-to-end PPO training run -- proves the whole pipeline
(rollout collection -> GAE -> PPO update -> checkpointing -> periodic
eval -> CSV logging) works, without claiming anything about training
QUALITY (that requires a much longer run; see README)."""
import csv
import os

import yaml

from splendor_ai.training.checkpoint import load_checkpoint
from splendor_ai.training.train import train


def _write_tiny_training_config(tmp_path) -> str:
    cfg = {
        "reward": {"win": 1.0, "loss": -1.0, "tie": 0.0, "prestige_point_coef": 0.02, "bonus_gained_coef": 0.01},
        "ppo": {
            "learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
            "value_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5,
            "ppo_epochs": 1, "minibatch_size": 64, "rollout_steps": 128,
        },
        "network": {"hidden_sizes": [32, 32]},
        "self_play": {
            "num_parallel_envs": 2, "historical_pool_size": 2,
            "historical_snapshot_interval_updates": 2, "historical_opponent_prob": 0.5,
            "heuristic_opponent_prob": 0.5,
        },
        "training": {
            "total_updates": 3, "device": "cpu", "seed": 0,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "checkpoint_interval_updates": 2, "eval_interval_updates": 2, "eval_games": 4,
            "log_dir": str(tmp_path / "runs"),
        },
    }
    path = str(tmp_path / "training.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _write_tiny_training_config_with_zero_noble_points(tmp_path) -> str:
    cfg = {
        "reward": {"win": 1.0, "loss": -1.0, "tie": 0.0, "prestige_point_coef": 0.02, "bonus_gained_coef": 0.01},
        "ppo": {
            "learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
            "value_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5,
            "ppo_epochs": 1, "minibatch_size": 64, "rollout_steps": 128,
        },
        "network": {"hidden_sizes": [32, 32]},
        "self_play": {
            "num_parallel_envs": 2, "historical_pool_size": 2,
            "historical_snapshot_interval_updates": 2, "historical_opponent_prob": 0.0,
            "heuristic_opponent_prob": 0.0,
        },
        "training": {
            "total_updates": 3, "device": "cpu", "seed": 0,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "checkpoint_interval_updates": 2, "eval_interval_updates": 2, "eval_games": 4,
            "log_dir": str(tmp_path / "runs"),
            "zero_noble_points": True,
        },
    }
    path = str(tmp_path / "training.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_short_training_run_end_to_end(tmp_path):
    config_path = _write_tiny_training_config(tmp_path)

    ckpt_dir = train(training_config_path=config_path, run_name="pytest_smoke")

    assert os.path.isdir(ckpt_dir)
    saved = [f for f in os.listdir(ckpt_dir) if f.startswith("update_")]
    assert saved, "no checkpoints were saved"

    latest_path = os.path.join(str(tmp_path / "checkpoints"), "latest.pt")
    assert os.path.isfile(latest_path)
    payload = load_checkpoint(latest_path)
    assert payload["meta"]["update"] == 3
    assert payload["meta"]["total_games"] > 0

    log_path = os.path.join(str(tmp_path / "runs"), "pytest_smoke", "training_log.csv")
    assert os.path.isfile(log_path)
    with open(log_path) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "update"
    assert len(rows) == 4  # header + 3 updates


def test_short_training_run_with_zero_noble_points_end_to_end(tmp_path):
    """The full pipeline must also work with zero_noble_points on --
    proves the wiring in train() (train_bundle for envs, real bundle for
    eval) doesn't crash across a real rollout/PPO-update/checkpoint cycle."""
    config_path = _write_tiny_training_config_with_zero_noble_points(tmp_path)

    ckpt_dir = train(training_config_path=config_path, run_name="pytest_smoke_zero_nobles")

    assert os.path.isdir(ckpt_dir)
    saved = [f for f in os.listdir(ckpt_dir) if f.startswith("update_")]
    assert saved, "no checkpoints were saved"
    payload = load_checkpoint(os.path.join(str(tmp_path / "checkpoints"), "latest.pt"))
    assert payload["meta"]["update"] == 3


def _write_tiny_training_config_with_blind_reserve_disabled(tmp_path) -> str:
    cfg = {
        "reward": {"win": 1.0, "loss": -1.0, "tie": 0.0, "prestige_point_coef": 0.02, "bonus_gained_coef": 0.01},
        "ppo": {
            "learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
            "value_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5,
            "ppo_epochs": 1, "minibatch_size": 64, "rollout_steps": 128,
        },
        "network": {"hidden_sizes": [32, 32]},
        "self_play": {
            "num_parallel_envs": 2, "historical_pool_size": 2,
            "historical_snapshot_interval_updates": 2, "historical_opponent_prob": 0.0,
            "heuristic_opponent_prob": 0.0,
        },
        "training": {
            "total_updates": 3, "device": "cpu", "seed": 0,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "checkpoint_interval_updates": 2, "eval_interval_updates": 2, "eval_games": 4,
            "log_dir": str(tmp_path / "runs"),
            "disable_blind_reserve_during_training": True,
        },
    }
    path = str(tmp_path / "training.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_short_training_run_with_blind_reserve_disabled_end_to_end(tmp_path):
    """The full pipeline must also work with R D1/D2/D3 masked out of
    training -- proves the excluded_action_indices wiring in train()
    doesn't crash across a real rollout/PPO-update/checkpoint cycle."""
    config_path = _write_tiny_training_config_with_blind_reserve_disabled(tmp_path)

    ckpt_dir = train(training_config_path=config_path, run_name="pytest_smoke_no_blind_reserve")

    assert os.path.isdir(ckpt_dir)
    saved = [f for f in os.listdir(ckpt_dir) if f.startswith("update_")]
    assert saved, "no checkpoints were saved"
    payload = load_checkpoint(os.path.join(str(tmp_path / "checkpoints"), "latest.pt"))
    assert payload["meta"]["update"] == 3


def test_resume_continues_from_saved_update(tmp_path):
    config_path = _write_tiny_training_config(tmp_path)
    train(training_config_path=config_path, run_name="pytest_smoke_a")

    latest_path = os.path.join(str(tmp_path / "checkpoints"), "latest.pt")
    ckpt_dir = train(training_config_path=config_path, resume_path=latest_path, updates=5, run_name="pytest_smoke_b")

    payload = load_checkpoint(os.path.join(str(tmp_path / "checkpoints"), "latest.pt"))
    assert payload["meta"]["update"] == 5


def test_heuristic_seat_sampling_never_overlaps_historical_seats():
    import random

    from splendor_ai.config import SelfPlayConfig
    from splendor_ai.training.train import _heuristic_agent_factories, _sample_heuristic_seat_agents

    rng = random.Random(0)
    factories = _heuristic_agent_factories(rng, num_players=4)
    cfg = SelfPlayConfig(heuristic_opponent_prob=1.0)  # always sample, to exercise the path deterministically

    for historical_seats in ([], [0], [0, 1], [0, 1, 2]):
        available = [s for s in range(4) if s not in historical_seats]
        seat_agent_factories = _sample_heuristic_seat_agents(rng, cfg, available, factories)
        assert set(seat_agent_factories).isdisjoint(historical_seats)
        assert set(seat_agent_factories).issubset(available)
        # Regression: historical + heuristic together must never claim
        # every seat -- that leaves nothing for the network to learn from
        # this update (an all-zero-loss, wasted PPO step).
        assert len(historical_seats) + len(seat_agent_factories) < 4


def test_heuristic_seat_sampling_leaves_no_seats_when_only_one_is_available():
    """If historical opponents already claimed 3 of 4 seats, heuristic
    sampling must claim NONE of the last one, however high its own
    probability is -- there must always be a learnable seat left."""
    import random

    from splendor_ai.config import SelfPlayConfig
    from splendor_ai.training.train import _heuristic_agent_factories, _sample_heuristic_seat_agents

    rng = random.Random(0)
    factories = _heuristic_agent_factories(rng, num_players=4)
    cfg = SelfPlayConfig(heuristic_opponent_prob=1.0)

    for _ in range(50):
        assert _sample_heuristic_seat_agents(rng, cfg, available_seats=[3], factories=factories) == {}


def test_heuristic_seat_sampling_produces_working_agent_factories():
    import random

    from splendor_ai.agents.base import Agent
    from splendor_ai.training.train import _heuristic_agent_factories

    rng = random.Random(1)
    factories = _heuristic_agent_factories(rng, num_players=4)
    for factory in factories:
        agent = factory(2)  # seat 2 -- BlockerAgent's factory must avoid targeting itself
        assert isinstance(agent, Agent)
        if hasattr(agent, "target_seat"):
            assert agent.target_seat != 2


def test_zero_noble_points_sets_every_noble_to_zero_without_touching_the_original(bundle):
    from splendor_ai.training.train import _zero_noble_points

    zeroed = _zero_noble_points(bundle)

    assert all(n.points == 0 for n in zeroed.nobles)
    assert all(n.points == 0 for n in zeroed.noble_by_id.values())
    assert len(zeroed.nobles) == len(bundle.nobles)
    assert {n.id for n in zeroed.nobles} == {n.id for n in bundle.nobles}
    # The original bundle (frozen dataclasses) must be completely untouched.
    assert any(n.points > 0 for n in bundle.nobles), "test assumes real nobles are worth points"
    # Everything else about the bundle stays the exact same object.
    assert zeroed.cards is bundle.cards
    assert zeroed.card_by_id is bundle.card_by_id
    assert zeroed.coins is bundle.coins


def test_zero_noble_points_env_awards_no_prestige_for_a_claimed_noble(bundle, config):
    """Integration check: a player who satisfies a noble's requirements in
    a zero_noble_points env must gain 0 prestige from it -- the point
    VALUE is what changed, not whether the noble still gets claimed."""
    from splendor_ai.env.splendor_env import SplendorEnv
    from splendor_ai.training.train import _zero_noble_points

    zeroed_bundle = _zero_noble_points(bundle)
    env = SplendorEnv(bundle=zeroed_bundle, config=config, seed=1)
    env.reset(seed=1)

    noble = env.state.board.nobles[0]
    player = env.state.player(env.state.current_player)
    for color, amount in noble.requirements.items():
        player.bonuses[color] = amount

    from splendor_ai.env.action_space import encode_action
    from splendor_ai.game.rules import legal_pick_actions

    action = legal_pick_actions(env.state, config)[0]
    idx = encode_action(action, env.state)
    env.step(idx)

    assert noble in player.nobles or any(n.id == noble.id for n in player.nobles)
    assert player.prestige == 0  # claimed the noble, but it's worth 0 points here
