import numpy as np
import torch

from splendor_ai.agents.network import ActorCritic
from splendor_ai.agents.reserve_rush_agent import ReserveRushAgent
from splendor_ai.env.splendor_env import SplendorEnv
from splendor_ai.training.self_play import RolloutStats, collect_rollout, compute_gae


def test_collect_rollout_patches_terminal_reward_for_every_seat(bundle, config):
    torch.manual_seed(0)
    network = ActorCritic()
    envs = [SplendorEnv(bundle=bundle, config=config, seed=s, max_steps=3000) for s in range(2)]
    current_obs = [env.reset(seed=s)[0] for s, env in enumerate(envs)]

    stats = RolloutStats()
    buffers = collect_rollout(envs, current_obs, network, torch.device("cpu"), num_steps=2500, stats=stats)

    assert stats.episodes_completed >= 1

    # For each completed game, every seat's LAST decision in that game must
    # be marked done, and the 4 seats' terminal rewards must be consistent
    # with exactly one winner (or a tie group), never a mix of "winner" and
    # "tie" reward magnitudes in the same game.
    for env_buffers in buffers:
        num_terminal_segments = [len([1 for d in buf.dones if d]) for buf in env_buffers]
        # Every seat should see the same number of finished games this rollout.
        assert len(set(num_terminal_segments)) == 1

        num_games = num_terminal_segments[0]
        for game_idx in range(num_games):
            rewards_at_end = {}
            for seat, buf in enumerate(env_buffers):
                done_positions = [i for i, d in enumerate(buf.dones) if d]
                idx = done_positions[game_idx]
                assert buf.dones[idx] is True
                rewards_at_end[seat] = buf.rewards[idx]

            winners = [s for s, r in rewards_at_end.items() if r > 0.5]
            losers = [s for s, r in rewards_at_end.items() if r < -0.5]
            ties = [s for s, r in rewards_at_end.items() if -0.5 <= r <= 0.5]
            assert len(winners) + len(losers) + len(ties) == 4
            assert not (winners and ties)
            assert len(winners) <= 1


def test_compute_gae_matches_hand_computation():
    rewards = [1.0, 0.0, -1.0]
    values = [0.5, 0.5, 0.5]
    dones = [False, False, True]
    gamma, lam = 1.0, 1.0

    advantages, returns = compute_gae(rewards, values, dones, last_value=0.0, gamma=gamma, gae_lambda=lam)

    # delta_2 = -1 + 1*0*0 - 0.5 = -1.5 ; adv_2 = -1.5
    # delta_1 = 0 + 1*0.5*1 - 0.5 = 0.0 ; adv_1 = 0.0 + 1*1*1*(-1.5) = -1.5
    # delta_0 = 1 + 1*0.5*1 - 0.5 = 1.0 ; adv_0 = 1.0 + 1*1*1*(-1.5) = -0.5
    expected_adv = np.array([-0.5, -1.5, -1.5], dtype=np.float32)
    np.testing.assert_allclose(advantages, expected_adv, atol=1e-5)
    np.testing.assert_allclose(returns, expected_adv + np.array(values), atol=1e-5)


def test_collect_rollout_continues_seamlessly_across_calls(bundle, config):
    torch.manual_seed(1)
    network = ActorCritic()
    envs = [SplendorEnv(bundle=bundle, config=config, seed=0, max_steps=3000)]
    current_obs = [envs[0].reset(seed=0)[0]]

    buffers1 = collect_rollout(envs, current_obs, network, torch.device("cpu"), num_steps=50)
    buffers2 = collect_rollout(envs, current_obs, network, torch.device("cpu"), num_steps=50)

    total1 = sum(len(buf) for buf in buffers1[0])
    total2 = sum(len(buf) for buf in buffers2[0])
    assert total1 == 50
    assert total2 == 50


def test_historical_opponent_seats_are_not_recorded(bundle, config):
    torch.manual_seed(2)
    network = ActorCritic()
    historical = ActorCritic()  # a separate, frozen "old" policy
    envs = [SplendorEnv(bundle=bundle, config=config, seed=0, max_steps=3000)]
    current_obs = [envs[0].reset(seed=0)[0]]

    buffers = collect_rollout(
        envs, current_obs, network, torch.device("cpu"), num_steps=200,
        seat_networks={1: historical, 2: historical},
    )

    # Seats 1 and 2 are historical-controlled: nothing should be recorded for them.
    assert len(buffers[0][1]) == 0
    assert len(buffers[0][2]) == 0
    # Seats 0 and 3 use the trainable network and should have recorded transitions.
    assert len(buffers[0][0]) > 0
    assert len(buffers[0][3]) > 0


def test_heuristic_opponent_seats_are_not_recorded(bundle, config):
    torch.manual_seed(3)
    network = ActorCritic()
    envs = [SplendorEnv(bundle=bundle, config=config, seed=0, max_steps=3000)]
    current_obs = [envs[0].reset(seed=0)[0]]

    buffers = collect_rollout(
        envs, current_obs, network, torch.device("cpu"), num_steps=200,
        seat_agent_factories={1: ReserveRushAgent, 2: ReserveRushAgent},
    )

    # Seats 1 and 2 are heuristic-controlled: nothing should be recorded for them.
    assert len(buffers[0][1]) == 0
    assert len(buffers[0][2]) == 0
    # Seats 0 and 3 use the trainable network and should have recorded transitions.
    assert len(buffers[0][0]) > 0
    assert len(buffers[0][3]) > 0


def test_heuristic_and_historical_opponents_can_be_mixed_in_the_same_rollout(bundle, config):
    """Historical-network seats and heuristic-agent seats must be able to
    coexist (disjoint seats) in one collect_rollout call without the
    network-batching path choking on the heuristic-controlled envs."""
    torch.manual_seed(4)
    network = ActorCritic()
    historical = ActorCritic()
    envs = [SplendorEnv(bundle=bundle, config=config, seed=s, max_steps=3000) for s in range(3)]
    current_obs = [env.reset(seed=s)[0] for s, env in enumerate(envs)]

    stats = RolloutStats()
    buffers = collect_rollout(
        envs, current_obs, network, torch.device("cpu"), num_steps=300, stats=stats,
        seat_networks={1: historical}, seat_agent_factories={2: ReserveRushAgent},
    )

    assert stats.episodes_completed >= 1
    for env_buffers in buffers:
        assert len(env_buffers[1]) == 0  # historical
        assert len(env_buffers[2]) == 0  # heuristic
        assert len(env_buffers[0]) > 0   # learnable
        assert len(env_buffers[3]) > 0   # learnable


def test_heuristic_agent_gets_a_fresh_instance_per_episode(bundle, config):
    """A heuristic agent that carries per-game state (like
    ReserveRushAgent's target_card_id) must not leak that state from one
    finished game into the next -- collect_rollout must build a NEW
    instance per (env, seat) whenever that env resets."""
    seen_instances = []

    class TrackingAgent(ReserveRushAgent):
        def __init__(self):
            super().__init__()
            seen_instances.append(self)

    torch.manual_seed(5)
    network = ActorCritic()
    envs = [SplendorEnv(bundle=bundle, config=config, seed=0, max_steps=60)]  # short games -> multiple episodes
    current_obs = [envs[0].reset(seed=0)[0]]

    collect_rollout(
        envs, current_obs, network, torch.device("cpu"), num_steps=400,
        seat_agent_factories={1: TrackingAgent},
    )

    # max_steps=60 forces several env resets (terminated or truncated --
    # either way collect_rollout must rebuild the heuristic instance) over
    # a 400-step collection window.
    assert len(seen_instances) >= 2
    assert len(set(id(a) for a in seen_instances)) == len(seen_instances)  # all genuinely distinct objects
