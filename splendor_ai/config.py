"""Loader for configs/game.yaml -- the base-game rule constants.

Kept separate from splendor_ai.data (which loads card/noble/coin content)
because this is project configuration we own, not user-prepared game data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import yaml

DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_GAME_CONFIG_PATH = os.path.join(DEFAULT_ROOT, "configs", "game.yaml")


@dataclass(frozen=True)
class GameConfig:
    num_players: int = 4
    tokens_per_normal_color: int = 7
    gold_tokens: int = 5
    visible_slots_per_tier: int = 4
    tiers: tuple = (1, 2, 3)
    nobles_count_in_play: int = 5
    max_reserved_cards: int = 3
    max_tokens_held: int = 10
    two_same_color_min_bank: int = 4
    winning_score: int = 15
    default_human_seat: int = 0

    @staticmethod
    def load(path: str = DEFAULT_GAME_CONFIG_PATH) -> "GameConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return GameConfig(
            num_players=raw["num_players"],
            tokens_per_normal_color=raw["bank"]["tokens_per_normal_color"],
            gold_tokens=raw["bank"]["gold_tokens"],
            visible_slots_per_tier=raw["board"]["visible_slots_per_tier"],
            tiers=tuple(raw["board"]["tiers"]),
            nobles_count_in_play=raw["nobles"]["count_in_play"],
            max_reserved_cards=raw["player"]["max_reserved_cards"],
            max_tokens_held=raw["player"]["max_tokens_held"],
            two_same_color_min_bank=raw["rules"]["two_same_color_min_bank"],
            winning_score=raw["rules"]["winning_score"],
            default_human_seat=raw.get("default_human_seat", 0),
        )


DEFAULT_GAME_CONFIG = GameConfig()

DEFAULT_TRAINING_CONFIG_PATH = os.path.join(DEFAULT_ROOT, "configs", "training.yaml")


@dataclass(frozen=True)
class RewardConfig:
    win: float = 1.0
    loss: float = -1.0
    tie: float = 0.0
    prestige_point_coef: float = 0.02
    bonus_gained_coef: float = 0.01
    # Small negative shaping applied to EVERY one of a seat's own turns
    # (pick, buy, or reserve alike), independent of prestige/bonus gained.
    # Pro Splendor play finishes in fewer turns than intermediate play by
    # taking fewer, bigger buys instead of many small ones -- the
    # prestige/bonus shaping above already rewards a bigger buy more than
    # a smaller one, but says nothing about how many turns it took to get
    # there. This adds that missing pressure directly. Kept far below the
    # terminal win/loss magnitude (see prestige_point_coef/bonus_gained_coef
    # for the same reasoning). Defaults to 0.0 for backward compatibility.
    step_cost_coef: float = 0.0
    # Efficiency-gated win: if set, a seat that reaches 15 points (a
    # win in game terms) using MORE buys than `max_buys_for_win` or MORE
    # turns than `max_turns_for_win` has its terminal reward converted to
    # `loss` instead of `win` -- winning too slowly/wastefully is scored
    # as a failure, not a win. A win that DOES meet both caps gets `win`
    # plus a small bonus for each buy/turn UNDER the cap, so a more
    # efficient qualifying win is preferred over a merely-qualifying one.
    # None (either cap, the default) disables this whole mechanism for
    # backward compatibility -- ordinary win/loss/tie scoring only.
    max_buys_for_win: Optional[int] = None
    max_turns_for_win: Optional[int] = None
    buy_efficiency_bonus_coef: float = 0.0
    turn_efficiency_bonus_coef: float = 0.0

    @staticmethod
    def load(path: str = DEFAULT_TRAINING_CONFIG_PATH) -> "RewardConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        r = raw["reward"]
        return RewardConfig(
            win=r["win"],
            loss=r["loss"],
            tie=r["tie"],
            prestige_point_coef=r["prestige_point_coef"],
            bonus_gained_coef=r["bonus_gained_coef"],
            step_cost_coef=r.get("step_cost_coef", 0.0),
            max_buys_for_win=r.get("max_buys_for_win"),
            max_turns_for_win=r.get("max_turns_for_win"),
            buy_efficiency_bonus_coef=r.get("buy_efficiency_bonus_coef", 0.0),
            turn_efficiency_bonus_coef=r.get("turn_efficiency_bonus_coef", 0.0),
        )


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 4096
    hidden_sizes: tuple = (256, 256)

    @staticmethod
    def load(path: str = DEFAULT_TRAINING_CONFIG_PATH) -> "PPOConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        p = raw["ppo"]
        n = raw["network"]
        return PPOConfig(
            learning_rate=p["learning_rate"],
            gamma=p["gamma"],
            gae_lambda=p["gae_lambda"],
            clip_range=p["clip_range"],
            value_coef=p["value_coef"],
            entropy_coef=p["entropy_coef"],
            max_grad_norm=p["max_grad_norm"],
            ppo_epochs=p["ppo_epochs"],
            minibatch_size=p["minibatch_size"],
            rollout_steps=p["rollout_steps"],
            hidden_sizes=tuple(n["hidden_sizes"]),
        )


@dataclass(frozen=True)
class SelfPlayConfig:
    num_parallel_envs: int = 8
    historical_pool_size: int = 10
    historical_snapshot_interval_updates: int = 25
    historical_opponent_prob: float = 0.3
    # Probability an opponent seat (not already claimed by a historical
    # snapshot) is instead played by a hand-crafted heuristic strategy this
    # update -- pure self-play never has to counter (or discover) tactics
    # it didn't stumble onto itself, so this seeds the population with
    # known strategies (reserve-rushing, bonus-engine building, blocking,
    # etc.) it must learn to beat and can absorb from. Defaults to 0.0
    # (off) for backward compatibility with existing configs.
    heuristic_opponent_prob: float = 0.0

    @staticmethod
    def load(path: str = DEFAULT_TRAINING_CONFIG_PATH) -> "SelfPlayConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        s = raw["self_play"]
        return SelfPlayConfig(
            num_parallel_envs=s["num_parallel_envs"],
            historical_pool_size=s["historical_pool_size"],
            historical_snapshot_interval_updates=s["historical_snapshot_interval_updates"],
            historical_opponent_prob=s["historical_opponent_prob"],
            heuristic_opponent_prob=s.get("heuristic_opponent_prob", 0.0),
        )


@dataclass(frozen=True)
class TrainingRunConfig:
    total_updates: int = 2000
    device: str = "cpu"
    seed: int = 0
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval_updates: int = 25
    eval_interval_updates: int = 25
    eval_games: int = 100
    log_dir: str = "runs"
    # When true, TRAINING environments use a copy of the data bundle where
    # every noble is worth 0 points (nobles are still claimed normally
    # when bonus requirements are met -- only their point value changes),
    # so the policy can't rely on free noble points and must earn all 15
    # points via card purchases. Evaluation/human play always use the
    # real bundle regardless -- this only ever affects what the policy is
    # TRAINED against, never what counts as winning when it's measured or
    # played against. Defaults to False for backward compatibility.
    zero_noble_points: bool = False
    # When true, TRAINING environments mask out R D1/D2/D3 (blind/
    # face-down tier reservations) entirely -- the policy is never even
    # OFFERED that action, on the theory that strong human play almost
    # never uses it, so it shouldn't be a crutch the policy leans on.
    # Evaluation/human play are unaffected -- the action still exists
    # normally there. Defaults to False for backward compatibility.
    disable_blind_reserve_during_training: bool = False

    @staticmethod
    def load(path: str = DEFAULT_TRAINING_CONFIG_PATH) -> "TrainingRunConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        t = raw["training"]
        return TrainingRunConfig(
            total_updates=t["total_updates"],
            device=t["device"],
            seed=t["seed"],
            checkpoint_dir=t["checkpoint_dir"],
            checkpoint_interval_updates=t["checkpoint_interval_updates"],
            eval_interval_updates=t["eval_interval_updates"],
            eval_games=t["eval_games"],
            log_dir=t["log_dir"],
            zero_noble_points=t.get("zero_noble_points", False),
            disable_blind_reserve_during_training=t.get("disable_blind_reserve_during_training", False),
        )
