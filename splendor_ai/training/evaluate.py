"""Evaluation tournaments: baseline measurement (Phase 6 prerequisite for
picking Medium/Expert thresholds) and test-agent-vs-opponent matchups with
seat rotation, used for win-rate-vs-random, win-rate-vs-heuristic, and
checkpoint-vs-checkpoint comparisons.

    python -m splendor_ai.training.evaluate --baseline --opponent random --games 1000
    python -m splendor_ai.training.evaluate --baseline --opponent heuristic --games 1000
    python -m splendor_ai.training.evaluate --checkpoint checkpoints/expert/best.pt --opponent random --games 1000
    python -m splendor_ai.training.evaluate --checkpoint checkpoints/expert/best.pt --opponent heuristic --games 1000
    python -m splendor_ai.training.evaluate --checkpoint checkpoints/medium/best.pt --opponent checkpoint --opponent-checkpoint checkpoints/expert/best.pt --games 1000

Never claims an agent is "well trained" -- it only reports measured
numbers over the requested number of games.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

from splendor_ai.agents.base import Agent
from splendor_ai.agents.heuristic_agent import HeuristicAgent
from splendor_ai.agents.random_agent import RandomAgent
from splendor_ai.config import GameConfig
from splendor_ai.data.loader import DataBundle, load_all
from splendor_ai.game.simulate import play_game

AgentFactory = Callable[[], Agent]


@dataclass
class MatchupResult:
    games: int
    win_rate: float
    tie_rate: float
    loss_rate: float
    avg_prestige: float
    avg_purchased_cards: float
    avg_turns_per_player: float
    win_rate_by_seat: Dict[int, float]
    games_by_seat: Dict[int, int]
    # Isolates the test agent's OWN turn/buy count in games it actually
    # WON (None if it never won) -- `avg_purchased_cards`/
    # `avg_turns_per_player` above average in every game including losses,
    # which dilutes exactly the number that distinguishes efficient play:
    # pro Splendor players win in fewer turns than intermediate ones by
    # taking fewer, bigger buys rather than many small ones (see README).
    winner_avg_turns: Optional[float] = None
    winner_avg_buys: Optional[float] = None


@dataclass
class BaselineResult:
    games: int
    avg_turns_per_player: float
    avg_prestige: float
    avg_purchased_cards: float
    win_rate_by_seat: Dict[int, float]


def evaluate_matchup(
    bundle: DataBundle,
    config: GameConfig,
    test_agent_factory: AgentFactory,
    opponent_agent_factory: AgentFactory,
    num_games: int,
    base_seed: int = 0,
    rotate_seats: bool = True,
    max_turns: int = 3000,
) -> MatchupResult:
    """`test_agent_factory` occupies one seat per game (rotating through
    all seats if `rotate_seats`); every other seat uses
    `opponent_agent_factory`. Returns metrics from the test agent's point
    of view."""
    wins = ties = losses = 0
    prestige_sum = 0.0
    cards_sum = 0.0
    turns_sum = 0
    winner_turns_sum = 0
    winner_buys_sum = 0
    wins_by_seat = {s: 0 for s in range(config.num_players)}
    games_by_seat = {s: 0 for s in range(config.num_players)}

    for i in range(num_games):
        test_seat = (i % config.num_players) if rotate_seats else 0
        agents = [test_agent_factory() if seat == test_seat else opponent_agent_factory() for seat in range(config.num_players)]
        state = play_game(bundle, config, agents, seed=base_seed + i, max_turns=max_turns)

        game_turns = state.turns_completed // state.num_players
        turns_sum += game_turns
        p = state.player(test_seat)
        prestige_sum += p.prestige
        cards_sum += p.purchased_card_count
        games_by_seat[test_seat] += 1

        winners = state.winners
        if test_seat in winners:
            if len(winners) == 1:
                wins += 1
                wins_by_seat[test_seat] += 1
                winner_turns_sum += game_turns
                winner_buys_sum += p.purchased_card_count
            else:
                ties += 1
        else:
            losses += 1

    n = num_games
    win_rate_by_seat = {s: (wins_by_seat[s] / games_by_seat[s] if games_by_seat[s] else 0.0) for s in range(config.num_players)}
    return MatchupResult(
        games=n,
        win_rate=wins / n,
        tie_rate=ties / n,
        loss_rate=losses / n,
        avg_prestige=prestige_sum / n,
        avg_purchased_cards=cards_sum / n,
        avg_turns_per_player=turns_sum / n,
        win_rate_by_seat=win_rate_by_seat,
        games_by_seat=games_by_seat,
        winner_avg_turns=(winner_turns_sum / wins) if wins else None,
        winner_avg_buys=(winner_buys_sum / wins) if wins else None,
    )


def measure_baseline(
    bundle: DataBundle,
    config: GameConfig,
    agent_factory: AgentFactory,
    num_games: int,
    base_seed: int = 0,
    max_turns: int = 3000,
) -> BaselineResult:
    """All 4 seats use the same kind of agent -- establishes a game-length
    / prestige baseline (e.g. pure-random or pure-heuristic self-play)
    BEFORE picking Medium/Expert thresholds from measured numbers rather
    than guessed ones."""
    turns_sum = 0
    prestige_sum = 0.0
    cards_sum = 0.0
    wins_by_seat = {s: 0 for s in range(config.num_players)}

    for i in range(num_games):
        agents = [agent_factory() for _ in range(config.num_players)]
        state = play_game(bundle, config, agents, seed=base_seed + i, max_turns=max_turns)
        turns_sum += state.turns_completed // state.num_players
        prestige_sum += sum(p.prestige for p in state.players) / state.num_players
        cards_sum += sum(p.purchased_card_count for p in state.players) / state.num_players
        if len(state.winners) == 1:
            wins_by_seat[state.winners[0]] += 1

    return BaselineResult(
        games=num_games,
        avg_turns_per_player=turns_sum / num_games,
        avg_prestige=prestige_sum / num_games,
        avg_purchased_cards=cards_sum / num_games,
        win_rate_by_seat={s: wins_by_seat[s] / num_games for s in range(config.num_players)},
    )


def _agent_factory_from_name(name: str, checkpoint_path: Optional[str], bundle: DataBundle, device: str) -> AgentFactory:
    if name == "random":
        return lambda: RandomAgent()
    if name == "heuristic":
        return lambda: HeuristicAgent()
    if name == "checkpoint":
        if not checkpoint_path:
            raise SystemExit("--opponent checkpoint requires --opponent-checkpoint")
        from splendor_ai.agents.rl_agent import RLAgent

        agent = RLAgent.load(checkpoint_path, bundle=bundle, device=device, deterministic=True)
        return lambda: agent
    raise SystemExit(f"unknown agent kind '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Splendor agents over many headless games.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for the agent under test (omit with --baseline).")
    parser.add_argument("--opponent", choices=["random", "heuristic", "checkpoint"], default="random")
    parser.add_argument("--opponent-checkpoint", type=str, default=None)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", action="store_true", help="Measure a same-agent-in-all-seats baseline instead of a matchup.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    bundle = load_all()
    config = GameConfig.load()

    if args.baseline:
        factory = _agent_factory_from_name(args.opponent if args.opponent != "checkpoint" else "random", None, bundle, args.device)
        result = measure_baseline(bundle, config, factory, args.games, base_seed=args.seed)
        print(json.dumps(asdict(result), indent=2))
        return

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --baseline is set")

    from splendor_ai.agents.rl_agent import RLAgent

    test_agent = RLAgent.load(args.checkpoint, bundle=bundle, device=args.device, deterministic=True)
    opponent_factory = _agent_factory_from_name(args.opponent, args.opponent_checkpoint, bundle, args.device)
    result = evaluate_matchup(bundle, config, lambda: test_agent, opponent_factory, args.games, base_seed=args.seed)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
