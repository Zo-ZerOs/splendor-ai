"""PPO self-play training with legal-action masking.

    python -m splendor_ai.training.train --config configs/training.yaml
    python -m splendor_ai.training.train --resume checkpoints/latest.pt
    python -m splendor_ai.training.train --updates 200 --num-envs 4 --device cpu

Never launches Pygame. All 4 seats share one policy (parameter sharing);
self-play optionally mixes in frozen historical-checkpoint opponents for
some seats each update round to reduce policy cycling.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from splendor_ai.agents.blocker_agent import BlockerAgent
from splendor_ai.agents.bonus_engine_agent import BonusEngineAgent
from splendor_ai.agents.heuristic_agent import HeuristicAgent
from splendor_ai.agents.network import ActorCritic
from splendor_ai.agents.no_noble_agent import NoNobleAgent
from splendor_ai.agents.random_agent import RandomAgent
from splendor_ai.agents.reserve_rush_agent import ReserveRushAgent
from splendor_ai.agents.synergy_reserve_agent import SynergyReserveAgent
from splendor_ai.config import (
    DEFAULT_TRAINING_CONFIG_PATH,
    GameConfig,
    PPOConfig,
    RewardConfig,
    SelfPlayConfig,
    TrainingRunConfig,
)
from splendor_ai.data.loader import DataBundle
from splendor_ai.data.loader import load_all
from splendor_ai.env.action_space import RESERVE_BLIND_SPECS, RESERVE_BLIND_START
from splendor_ai.env.splendor_env import SplendorEnv
from splendor_ai.training.checkpoint import load_checkpoint, save_checkpoint
from splendor_ai.training.evaluate import evaluate_matchup
from splendor_ai.training.self_play import RolloutStats, collect_rollout, compute_gae

try:
    from torch.utils.tensorboard import SummaryWriter

    _TENSORBOARD_AVAILABLE = True
except Exception:  # noqa: BLE001 -- broken protobuf/tensorboard installs must not block training
    SummaryWriter = None
    _TENSORBOARD_AVAILABLE = False


def flatten_for_training(all_buffers, gamma: float, gae_lambda: float):
    obs, masks, actions, old_log_probs, advantages, returns = [], [], [], [], [], []
    for env_buffers in all_buffers:
        for buf in env_buffers:
            if len(buf) == 0:
                continue
            last_value = 0.0 if buf.dones[-1] else buf.values[-1]
            adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, last_value, gamma, gae_lambda)
            obs.extend(buf.observations)
            masks.extend(buf.action_masks)
            actions.extend(buf.actions)
            old_log_probs.extend(buf.log_probs)
            advantages.extend(adv.tolist())
            returns.extend(ret.tolist())
    return (
        np.asarray(obs, dtype=np.float32),
        np.asarray(masks, dtype=np.int8),
        np.asarray(actions, dtype=np.int64),
        np.asarray(old_log_probs, dtype=np.float32),
        np.asarray(advantages, dtype=np.float32),
        np.asarray(returns, dtype=np.float32),
    )


def ppo_update(network: ActorCritic, optimizer: torch.optim.Optimizer, data, ppo_cfg: PPOConfig, device: torch.device) -> Dict[str, float]:
    obs, masks, actions, old_log_probs, advantages, returns = data
    obs_t = torch.from_numpy(obs).to(device)
    masks_t = torch.from_numpy(masks).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    old_log_probs_t = torch.from_numpy(old_log_probs).to(device)
    returns_t = torch.from_numpy(returns).to(device)
    advantages_t = torch.from_numpy(advantages).to(device)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    n = obs_t.shape[0]
    idxs = np.arange(n)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    batches = 0

    for _ in range(ppo_cfg.ppo_epochs):
        np.random.shuffle(idxs)
        for start in range(0, n, ppo_cfg.minibatch_size):
            mb = torch.from_numpy(idxs[start : start + ppo_cfg.minibatch_size]).long().to(device)

            log_probs, entropy, values = network.evaluate_actions(obs_t[mb], masks_t[mb], actions_t[mb])
            ratio = torch.exp(log_probs - old_log_probs_t[mb])
            mb_adv = advantages_t[mb]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - ppo_cfg.clip_range, 1 + ppo_cfg.clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, returns_t[mb])
            entropy_mean = entropy.mean()
            loss = policy_loss + ppo_cfg.value_coef * value_loss - ppo_cfg.entropy_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), ppo_cfg.max_grad_norm)
            optimizer.step()

            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy_mean.item()
            batches += 1

    b = max(1, batches)
    return {k: v / b for k, v in totals.items()}


def _zero_noble_points(bundle: DataBundle) -> DataBundle:
    """A copy of `bundle` where every noble is worth 0 points. Nobles are
    still claimed normally (bonus requirements, board removal, the
    noble-choice sub-decision when eligible for several) -- only the
    point VALUE changes, so a policy trained against this bundle can't
    lean on free noble points and must earn all 15 points via card
    purchases. See TrainingRunConfig.zero_noble_points; only ever used
    for TRAINING environments, never for evaluation or human play."""
    zeroed_nobles = [dataclasses.replace(n, points=0) for n in bundle.nobles]
    zeroed_by_id = {n.id: n for n in zeroed_nobles}
    return dataclasses.replace(bundle, nobles=zeroed_nobles, noble_by_id=zeroed_by_id)


def _sample_historical_seat_networks(
    historical_pool: List[dict], rng: random.Random, cfg: SelfPlayConfig, num_players: int, hidden_sizes, device
) -> Dict[int, ActorCritic]:
    if not historical_pool or rng.random() >= cfg.historical_opponent_prob:
        return {}
    num_historical_seats = rng.randint(1, num_players - 1)
    seats = rng.sample(range(num_players), num_historical_seats)
    seat_networks = {}
    for s in seats:
        state_dict = rng.choice(historical_pool)
        net = ActorCritic(hidden_sizes=hidden_sizes).to(device)
        net.load_state_dict(state_dict)
        net.eval()
        seat_networks[s] = net
    return seat_networks


def _heuristic_agent_factories(rng: random.Random, num_players: int) -> List[Callable[[int], object]]:
    """Zero-hyperparameter strategy roster used to diversify training
    opponents beyond pure self-play (see `_sample_heuristic_seat_agents`).
    Each factory takes the seat it will occupy (only BlockerAgent needs
    it, to pick a DIFFERENT seat to target)."""
    return [
        lambda seat: HeuristicAgent(),
        lambda seat: ReserveRushAgent(),
        lambda seat: BonusEngineAgent(),
        lambda seat: NoNobleAgent(),
        lambda seat: SynergyReserveAgent(),
        lambda seat: BlockerAgent(target_seat=rng.choice([s for s in range(num_players) if s != seat])),
    ]


def _sample_heuristic_seat_agents(
    rng: random.Random, cfg: SelfPlayConfig, available_seats: List[int], factories: List[Callable[[int], object]]
) -> Dict[int, Callable[[], object]]:
    """Picks a random subset of `available_seats` (those NOT already
    claimed by a historical-network opponent this update) to be played by
    a random heuristic strategy instead of the learning policy. Returns
    {seat: zero-arg factory} rather than instances directly, since
    `collect_rollout` needs a FRESH instance per parallel env (these
    strategies carry per-game state).

    Always leaves at least one seat learnable: claiming every seat would
    hand this update's whole rollout to non-learning agents, producing an
    empty PPO batch (a wasted, all-zero-loss update)."""
    max_claimable = len(available_seats) - 1
    if max_claimable <= 0 or rng.random() >= cfg.heuristic_opponent_prob:
        return {}
    num_heuristic_seats = rng.randint(1, max_claimable)
    seats = rng.sample(available_seats, num_heuristic_seats)
    seat_agent_factories = {}
    for s in seats:
        factory = rng.choice(factories)
        seat_agent_factories[s] = lambda f=factory, seat=s: f(seat)
    return seat_agent_factories


def train(
    updates: Optional[int] = None,
    num_envs: Optional[int] = None,
    device_str: Optional[str] = None,
    seed: Optional[int] = None,
    resume_path: Optional[str] = None,
    run_name: Optional[str] = None,
    training_config_path: str = DEFAULT_TRAINING_CONFIG_PATH,
    game_config_path: Optional[str] = None,
) -> str:
    """Runs PPO self-play training. Returns the run's checkpoint directory."""
    try:
        sys.stdout.reconfigure(line_buffering=True)  # keep progress visible when stdout is redirected to a file
    except (AttributeError, ValueError):
        pass

    bundle = load_all()
    game_config = GameConfig.load(game_config_path) if game_config_path else GameConfig.load()
    reward_config = RewardConfig.load(training_config_path)
    ppo_cfg = PPOConfig.load(training_config_path)
    self_play_cfg = SelfPlayConfig.load(training_config_path)
    run_cfg = TrainingRunConfig.load(training_config_path)

    device = torch.device(device_str or run_cfg.device)
    n_envs = num_envs or self_play_cfg.num_parallel_envs
    total_updates = updates if updates is not None else run_cfg.total_updates
    run_seed = seed if seed is not None else run_cfg.seed

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    network = ActorCritic(hidden_sizes=ppo_cfg.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=ppo_cfg.learning_rate)

    start_update = 0
    total_games = 0
    total_steps = 0
    historical_pool: List[dict] = []

    if resume_path:
        payload = load_checkpoint(resume_path, map_location=str(device))
        network.load_state_dict(payload["network_state_dict"])
        if payload.get("optimizer_state_dict"):
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_update = payload["meta"]["update"]
        total_games = payload["meta"].get("total_games", 0)
        total_steps = payload["meta"].get("total_steps", 0)
        historical_pool = payload.get("extra", {}).get("historical_pool", [])
        print(f"Resumed from {resume_path} at update {start_update} (total_games={total_games}, total_steps={total_steps})")

    train_bundle = _zero_noble_points(bundle) if run_cfg.zero_noble_points else bundle
    if run_cfg.zero_noble_points:
        print("NOTE: training with zero_noble_points -- nobles are still claimed normally but worth 0 points; evaluation below still uses the real bundle.")

    excluded_action_indices = None
    if run_cfg.disable_blind_reserve_during_training:
        excluded_action_indices = {RESERVE_BLIND_START + i for i in range(len(RESERVE_BLIND_SPECS))}
        print("NOTE: training with disable_blind_reserve_during_training -- R D1/D2/D3 are masked out; evaluation below still allows them.")

    envs = [
        SplendorEnv(bundle=train_bundle, config=game_config, reward_config=reward_config, seed=run_seed + i, excluded_action_indices=excluded_action_indices)
        for i in range(n_envs)
    ]
    current_obs = [env.reset(seed=run_seed * 1000 + i)[0] for i, env in enumerate(envs)]

    run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(run_cfg.checkpoint_dir, run_name)
    log_dir = os.path.join(run_cfg.log_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir) if _TENSORBOARD_AVAILABLE else None
    if not _TENSORBOARD_AVAILABLE:
        print("NOTE: TensorBoard unavailable in this environment (import failed) -- logging to CSV only.")

    csv_path = os.path.join(log_dir, "training_log.csv")
    csv_is_new = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if csv_is_new:
        csv_writer.writerow(
            [
                "update", "total_games", "total_steps", "episodes_this_update",
                "avg_game_length_turns", "avg_final_prestige",
                "win_rate_seat0", "win_rate_seat1", "win_rate_seat2", "win_rate_seat3",
                "policy_loss", "value_loss", "entropy",
                "eval_win_rate_vs_random", "eval_win_rate_vs_heuristic", "eval_win_rate_vs_synergy_reserve",
            ]
        )

    rng = random.Random(run_seed + 999)
    heuristic_factories = _heuristic_agent_factories(rng, game_config.num_players)

    try:
        for update in range(start_update, total_updates):
            seat_networks = _sample_historical_seat_networks(
                historical_pool, rng, self_play_cfg, game_config.num_players, ppo_cfg.hidden_sizes, device
            )
            available_seats = [s for s in range(game_config.num_players) if s not in seat_networks]
            seat_agent_factories = _sample_heuristic_seat_agents(rng, self_play_cfg, available_seats, heuristic_factories)

            stats = RolloutStats()
            buffers = collect_rollout(
                envs, current_obs, network, device, ppo_cfg.rollout_steps, stats=stats,
                seat_networks=seat_networks or None, seat_agent_factories=seat_agent_factories or None,
            )

            data = flatten_for_training(buffers, ppo_cfg.gamma, ppo_cfg.gae_lambda)
            total_steps += data[0].shape[0]
            total_games += stats.episodes_completed

            losses = ppo_update(network, optimizer, data, ppo_cfg, device)

            win_rates = {
                s: (stats.wins_by_seat.get(s, 0) / stats.games_by_seat[s] if stats.games_by_seat.get(s) else 0.0)
                for s in range(game_config.num_players)
            }
            avg_len = float(np.mean(stats.episode_lengths_turns)) if stats.episode_lengths_turns else 0.0
            avg_prestige = float(np.mean(stats.final_prestige)) if stats.final_prestige else 0.0

            print(
                f"[update {update + 1}/{total_updates}] episodes={stats.episodes_completed} total_games={total_games} "
                f"avg_len={avg_len:.1f} avg_prestige={avg_prestige:.2f} "
                f"policy_loss={losses['policy_loss']:.4f} value_loss={losses['value_loss']:.4f} entropy={losses['entropy']:.4f}"
            )

            if writer is not None:
                writer.add_scalar("rollout/episodes", stats.episodes_completed, update)
                writer.add_scalar("rollout/avg_game_length_turns", avg_len, update)
                writer.add_scalar("rollout/avg_final_prestige", avg_prestige, update)
                for s, r in win_rates.items():
                    writer.add_scalar(f"rollout/win_rate_seat{s}", r, update)
                writer.add_scalar("loss/policy", losses["policy_loss"], update)
                writer.add_scalar("loss/value", losses["value_loss"], update)
                writer.add_scalar("loss/entropy", losses["entropy"], update)

            eval_vs_random = eval_vs_heuristic = eval_vs_synergy_reserve = None
            if (update + 1) % run_cfg.eval_interval_updates == 0:
                from splendor_ai.agents.rl_agent import RLAgent

                network.eval()
                eval_agent = RLAgent(network, bundle=bundle, device=str(device), deterministic=True, seat_name="training-snapshot")
                rand_res = evaluate_matchup(bundle, game_config, lambda: eval_agent, lambda: RandomAgent(), run_cfg.eval_games, base_seed=update)
                heur_res = evaluate_matchup(bundle, game_config, lambda: eval_agent, lambda: HeuristicAgent(), run_cfg.eval_games, base_seed=update + 5_000_000)
                synergy_res = evaluate_matchup(bundle, game_config, lambda: eval_agent, lambda: SynergyReserveAgent(), run_cfg.eval_games, base_seed=update + 9_000_000)
                network.train()
                eval_vs_random, eval_vs_heuristic, eval_vs_synergy_reserve = rand_res.win_rate, heur_res.win_rate, synergy_res.win_rate
                print(
                    f"  eval @ update {update + 1}: vs_random={eval_vs_random:.2f} "
                    f"vs_heuristic={eval_vs_heuristic:.2f} vs_synergy_reserve={eval_vs_synergy_reserve:.2f}"
                )
                if writer is not None:
                    writer.add_scalar("eval/win_rate_vs_random", eval_vs_random, update)
                    writer.add_scalar("eval/win_rate_vs_heuristic", eval_vs_heuristic, update)
                    writer.add_scalar("eval/win_rate_vs_synergy_reserve", eval_vs_synergy_reserve, update)

            csv_writer.writerow(
                [
                    update + 1, total_games, total_steps, stats.episodes_completed, avg_len, avg_prestige,
                    win_rates[0], win_rates[1], win_rates[2], win_rates[3],
                    losses["policy_loss"], losses["value_loss"], losses["entropy"],
                    eval_vs_random if eval_vs_random is not None else "", eval_vs_heuristic if eval_vs_heuristic is not None else "",
                    eval_vs_synergy_reserve if eval_vs_synergy_reserve is not None else "",
                ]
            )
            csv_file.flush()

            if (update + 1) % self_play_cfg.historical_snapshot_interval_updates == 0:
                historical_pool.append({k: v.detach().cpu().clone() for k, v in network.state_dict().items()})
                if len(historical_pool) > self_play_cfg.historical_pool_size:
                    historical_pool.pop(0)

            if (update + 1) % run_cfg.checkpoint_interval_updates == 0 or (update + 1) == total_updates:
                ckpt_path = os.path.join(ckpt_dir, f"update_{update + 1}.pt")
                save_checkpoint(ckpt_path, network, optimizer, update + 1, total_games, total_steps, extra={"historical_pool": historical_pool})
                save_checkpoint(os.path.join(run_cfg.checkpoint_dir, "latest.pt"), network, optimizer, update + 1, total_games, total_steps, extra={"historical_pool": historical_pool, "run_name": run_name})
                print(f"  saved checkpoint: {ckpt_path}")
    finally:
        csv_file.close()
        if writer is not None:
            writer.close()

    return ckpt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Splendor PPO self-play.")
    parser.add_argument("--config", type=str, default=DEFAULT_TRAINING_CONFIG_PATH, help="Path to training.yaml")
    parser.add_argument("--game-config", type=str, default=None, help="Path to game.yaml (default: configs/game.yaml)")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--updates", type=int, default=None, help="Override total_updates from training.yaml")
    parser.add_argument("--num-envs", type=int, default=None, help="Override self_play.num_parallel_envs")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    train(
        updates=args.updates,
        num_envs=args.num_envs,
        device_str=args.device,
        seed=args.seed,
        resume_path=args.resume,
        run_name=args.run_name,
        training_config_path=args.config,
        game_config_path=args.game_config,
    )


if __name__ == "__main__":
    main()
