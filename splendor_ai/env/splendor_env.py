"""Gymnasium environment wrapping the engine for PPO self-play training.

Design note -- turn-based multi-agent via a single-agent Gym API:
Splendor is 4-player and turn-based. Rather than pull in a full
multi-agent framework, `step()` always acts for "whichever seat is
currently to move" (`info["seat"]`) and returns the observation for
whichever seat moves next (`info["next_seat"]`). Since training uses one
shared policy across all 4 seats (parameter sharing, see
splendor_ai.training), the self-play loop just routes each transition to
the trajectory buffer for `info["seat"]` -- there is no cross-seat
credit-assignment problem because a single network is trained on all of
them together.

Reward has three parts:
  1. A small per-step shaping reward for the seat that just acted, from
     prestige gained and permanent bonuses gained this step.
  2. A small per-step cost applied to every one of a seat's own turns
     regardless of what it gained, so the policy is pressured toward
     winning in fewer turns (see RewardConfig.step_cost_coef) rather than
     only differentiating a big buy from a small one.
  3. A terminal reward folded into the reward of the LAST step any given
     seat takes. Because the final round guarantees equal turns, only the
     very last mover gets `terminated=True` from Gymnasium's point of
     view -- but `info["terminal_scores"]` (a seat -> score dict) is
     always populated when the game ends, so the self-play loop can patch
     the final stored transition of every OTHER seat's trajectory too.
     For the winning seat specifically, this terminal score can be
     EFFICIENCY-GATED (see RewardConfig.max_buys_for_win/
     max_turns_for_win, and SplendorEnv._efficiency_gated_win_score):
     reaching 15 points using too many buys or turns scores as a `loss`
     instead of a `win` -- winning too slowly/wastefully doesn't count.

Never imports Pygame. Training must never launch the graphical viewer.
"""
from __future__ import annotations

import random
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from splendor_ai.config import GameConfig, RewardConfig
from splendor_ai.data.loader import DataBundle, load_all
from splendor_ai.env.action_mask import compute_action_mask
from splendor_ai.env.action_space import ACTION_SPACE_SIZE, decode_action
from splendor_ai.env.observation import OBSERVATION_DIM, encode_observation
from splendor_ai.game.engine import IllegalActionError, apply_action, setup_game

DEFAULT_MAX_STEPS = 4000


class SplendorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bundle: Optional[DataBundle] = None,
        config: Optional[GameConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        randomize_starting_player: bool = True,
        excluded_action_indices: Optional[set] = None,
    ):
        super().__init__()
        self.bundle = bundle if bundle is not None else load_all()
        self.config = config if config is not None else GameConfig.load()
        self.reward_config = reward_config if reward_config is not None else RewardConfig.load()
        self.max_steps = max_steps
        self.randomize_starting_player = randomize_starting_player
        # Masked out of EVERY action_mask this env produces, on top of the
        # normal legal-action mask (see TrainingRunConfig.
        # disable_blind_reserve_during_training) -- e.g. to train without
        # ever letting the policy pick R D1/D2/D3. Never used for
        # evaluation/human play, only specific training environments.
        self.excluded_action_indices = excluded_action_indices
        self._rng = random.Random(seed)

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-2.0, high=3.0, shape=(OBSERVATION_DIM,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8),
            }
        )

        self.state = None
        self._steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        game_seed = seed if seed is not None else self._rng.randrange(2**31)
        starting_player = self._rng.randrange(self.config.num_players) if self.randomize_starting_player else 0
        self.state = setup_game(self.bundle, self.config, seed=game_seed, starting_player=starting_player)
        self._steps = 0
        return self._obs(), {"seat": self.state.current_player}

    def _obs(self) -> dict:
        return {
            "observation": encode_observation(self.state, self.config, self.bundle),
            "action_mask": compute_action_mask(self.state, self.config, self.excluded_action_indices),
        }

    def action_masks(self) -> np.ndarray:
        """Convenience accessor some maskable-PPO integrations expect."""
        return compute_action_mask(self.state, self.config, self.excluded_action_indices)

    def step(self, action_index: int):
        if self.state is None or self.state.game_over:
            raise RuntimeError("step() called before reset() or after the episode ended")

        mover = self.state.current_player
        pre_player = self.state.player(mover)
        pre_prestige = pre_player.prestige
        pre_bonus_count = sum(pre_player.bonuses.values())

        action = decode_action(int(action_index), self.state)
        try:
            apply_action(self.state, self.config, action)
        except IllegalActionError as e:
            raise RuntimeError(
                f"SplendorEnv.step() received an illegal action ({e}). "
                "Callers must sample only indices where action_mask == 1."
            ) from e

        self._steps += 1

        post_player = self.state.player(mover)
        prestige_gain = post_player.prestige - pre_prestige
        bonus_gain = max(0, sum(post_player.bonuses.values()) - pre_bonus_count)
        reward = (
            prestige_gain * self.reward_config.prestige_point_coef
            + bonus_gain * self.reward_config.bonus_gained_coef
            - self.reward_config.step_cost_coef
        )

        info = {"seat": mover, "next_seat": self.state.current_player}
        terminated = bool(self.state.game_over)

        if terminated:
            winners = self.state.winners
            tie = len(winners) > 1
            per_player_turns = self.state.turns_completed // self.state.num_players
            terminal_scores = {}
            for seat in range(self.config.num_players):
                if seat in winners:
                    terminal_scores[seat] = (
                        self.reward_config.tie if tie else self._efficiency_gated_win_score(seat, per_player_turns)
                    )
                else:
                    terminal_scores[seat] = self.reward_config.loss
            info["terminal_scores"] = terminal_scores
            reward += terminal_scores[mover]

        truncated = (not terminated) and self._steps >= self.max_steps

        return self._obs(), reward, terminated, truncated, info

    def _efficiency_gated_win_score(self, seat: int, turns: int) -> float:
        """Score for a seat that reached 15 points (a win in game terms).
        See RewardConfig.max_buys_for_win/max_turns_for_win: if either cap
        is set and this seat exceeded it, the win is scored as `loss`
        instead -- reaching 15 points too slowly/wastefully doesn't count.
        Otherwise `win` plus a small bonus per buy/turn under each cap."""
        cfg = self.reward_config
        buys = self.state.player(seat).purchased_card_count

        over_buy_cap = cfg.max_buys_for_win is not None and buys > cfg.max_buys_for_win
        over_turn_cap = cfg.max_turns_for_win is not None and turns > cfg.max_turns_for_win
        if over_buy_cap or over_turn_cap:
            return cfg.loss

        bonus = 0.0
        if cfg.max_buys_for_win is not None:
            bonus += cfg.buy_efficiency_bonus_coef * (cfg.max_buys_for_win - buys)
        if cfg.max_turns_for_win is not None:
            bonus += cfg.turn_efficiency_bonus_coef * (cfg.max_turns_for_win - turns)
        return cfg.win + bonus
