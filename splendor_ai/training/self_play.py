"""Self-play rollout collection for PPO.

The tricky part of turn-based multi-agent self-play with ONE shared
policy: a single SplendorEnv episode interleaves all 4 seats' decisions
chronologically. GAE must NEVER bootstrap a seat's value estimate off the
very next buffer entry, because that next entry usually belongs to a
DIFFERENT seat's turn -- its value has nothing to do with the acting
seat's continuation value. So each env keeps FOUR separate per-seat
trajectory buffers; a seat's buffer only ever contains that seat's own
consecutive decisions (with other seats' turns "in between" treated as
ordinary environment dynamics, exactly as GAE expects).

The other subtlety: `SplendorEnv.step()` only tells the mover their own
terminal reward when the episode ends -- the other three seats' last
decisions happened at earlier timesteps, before the outcome was known.
`collect_rollout` retroactively patches those seats' last stored
transition (reward += their terminal score, done=True) using
`info["terminal_scores"]`, which the env always populates on the step
that ends the game.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from splendor_ai.agents.base import Agent
from splendor_ai.agents.network import ActorCritic
from splendor_ai.env.action_space import encode_action
from splendor_ai.env.splendor_env import SplendorEnv


@dataclass
class TrajectoryBuffer:
    observations: List[np.ndarray] = field(default_factory=list)
    action_masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(self, obs, mask, action, log_prob, value, reward, done) -> int:
        idx = len(self.rewards)
        self.observations.append(obs)
        self.action_masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        return idx

    def patch_reward(self, index: int, extra_reward: float, done: bool = True) -> None:
        self.rewards[index] += extra_reward
        self.dones[index] = done

    def __len__(self) -> int:
        return len(self.rewards)


@dataclass
class RolloutStats:
    episodes_completed: int = 0
    episode_lengths_turns: List[int] = field(default_factory=list)  # turns-per-player in each finished game
    wins_by_seat: Dict[int, int] = field(default_factory=dict)
    games_by_seat: Dict[int, int] = field(default_factory=dict)
    final_prestige: List[float] = field(default_factory=list)


def collect_rollout(
    envs: List[SplendorEnv],
    current_obs: List[dict],
    network: ActorCritic,
    device: torch.device,
    num_steps: int,
    stats: Optional[RolloutStats] = None,
    seat_networks: Optional[Dict[int, ActorCritic]] = None,
    seat_agent_factories: Optional[Dict[int, Callable[[], Agent]]] = None,
) -> List[List[TrajectoryBuffer]]:
    """Collect `num_steps` env-steps from each of len(envs) environments.

    Returns `buffers[k][seat]` -- one TrajectoryBuffer per (env, seat)
    pair, each containing only that seat's own chronological decisions.

    `current_obs` is mutated in place so consecutive calls continue
    seamlessly across PPO update boundaries.

    `seat_networks`: optional {seat_index: frozen ActorCritic}, constant
    for this call, used for historical-opponent sampling -- seats named
    here act using that frozen network and their transitions are NOT
    recorded (nothing to learn from a policy we're not updating).

    `seat_agent_factories`: optional {seat_index: zero-arg factory ->
    Agent}, constant for this call, used for heuristic-opponent sampling
    (must not overlap with `seat_networks`' keys). Each of the K envs gets
    its OWN agent instance for a given seat (heuristic agents like
    ReserveRushAgent carry per-game state such as a reservation target,
    so instances can never be shared across concurrently-running games);
    a fresh instance replaces it whenever that env resets. Acts directly
    on `env.state`/`env.config` -- Action objects are re-encoded to the
    flat discrete index via `encode_action` so they flow through
    `env.step()` exactly like a network-produced action. These seats'
    transitions are likewise NOT recorded.

    Every other seat uses `network` (the one being trained) and is
    recorded normally. Terminal-reward patching still works correctly
    regardless of which seat ends the game.
    """
    K = len(envs)
    num_players = envs[0].config.num_players
    seat_networks = seat_networks or {}
    seat_agent_factories = seat_agent_factories or {}
    buffers: List[List[TrajectoryBuffer]] = [[TrajectoryBuffer() for _ in range(num_players)] for _ in range(K)]
    last_index_this_episode: List[Dict[int, int]] = [dict() for _ in range(K)]
    heuristic_agents: List[Dict[int, Agent]] = [
        {seat: factory() for seat, factory in seat_agent_factories.items()} for _ in range(K)
    ]

    for _ in range(num_steps):
        seat_of_env = [envs[k].state.current_player for k in range(K)]
        heuristic_ks = [k for k in range(K) if seat_of_env[k] in seat_agent_factories]
        heuristic_k_set = set(heuristic_ks)

        # Group the remaining (non-heuristic) envs by which network should
        # act this step (usually all one group -- the common case with no
        # historical opponents either).
        groups: Dict[int, Tuple[ActorCritic, List[int]]] = {}
        for k in range(K):
            if k in heuristic_k_set:
                continue
            net = seat_networks.get(seat_of_env[k], network)
            groups.setdefault(id(net), (net, []))[1].append(k)

        actions_by_k: Dict[int, int] = {}
        log_probs_by_k: Dict[int, float] = {}
        values_by_k: Dict[int, float] = {}
        for net, k_list in groups.values():
            obs_arr = np.stack([current_obs[k]["observation"] for k in k_list])
            mask_arr = np.stack([current_obs[k]["action_mask"] for k in k_list])
            obs_t = torch.from_numpy(obs_arr).to(device)
            mask_t = torch.from_numpy(mask_arr).to(device)
            with torch.no_grad():
                a_t, lp_t, v_t = net.act(obs_t, mask_t, deterministic=False)
            a_np, lp_np, v_np = a_t.cpu().numpy(), lp_t.cpu().numpy(), v_t.cpu().numpy()
            for i, k in enumerate(k_list):
                actions_by_k[k] = int(a_np[i])
                log_probs_by_k[k] = float(lp_np[i])
                values_by_k[k] = float(v_np[i])

        for k in heuristic_ks:
            seat = seat_of_env[k]
            action_obj = heuristic_agents[k][seat].act(envs[k].state, envs[k].config)
            actions_by_k[k] = encode_action(action_obj, envs[k].state)

        for k, env in enumerate(envs):
            seat = seat_of_env[k]
            obs_k = current_obs[k]
            action_k = actions_by_k[k]

            next_obs, reward, terminated, truncated, info = env.step(action_k)
            done = bool(terminated or truncated)

            is_learnable_seat = seat not in seat_networks and seat not in seat_agent_factories
            if is_learnable_seat:
                idx = buffers[k][seat].add(
                    obs_k["observation"], obs_k["action_mask"], action_k,
                    log_probs_by_k[k], values_by_k[k], float(reward), done,
                )
                last_index_this_episode[k][seat] = idx

            if terminated and "terminal_scores" in info:
                for other_seat, score in info["terminal_scores"].items():
                    if other_seat == seat:
                        continue  # mover's own terminal score is already in `reward` above
                    if other_seat in last_index_this_episode[k]:
                        buffers[k][other_seat].patch_reward(last_index_this_episode[k][other_seat], score, done=True)

                if stats is not None:
                    scores = info["terminal_scores"]
                    top = max(scores.values())
                    winners = [s for s, sc in scores.items() if sc == top]
                    stats.episodes_completed += 1
                    stats.episode_lengths_turns.append(env.state.turns_completed // env.state.num_players)
                    for s in range(env.state.num_players):
                        stats.games_by_seat[s] = stats.games_by_seat.get(s, 0) + 1
                        if s in winners and len(winners) == 1:
                            stats.wins_by_seat[s] = stats.wins_by_seat.get(s, 0) + 1
                    stats.final_prestige.append(float(np.mean([p.prestige for p in env.state.players])))

            if done:
                next_obs, _reset_info = env.reset()
                last_index_this_episode[k] = {}
                if seat_agent_factories:
                    heuristic_agents[k] = {seat_: factory() for seat_, factory in seat_agent_factories.items()}

            current_obs[k] = next_obs

    return buffers


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard GAE over ONE seat's own chronological sub-trajectory.
    `last_value` bootstraps the (possibly truncated, non-terminal) tail."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else last_value
        next_non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    values_arr = np.asarray(values, dtype=np.float32)
    returns = advantages + values_arr
    return advantages, returns
