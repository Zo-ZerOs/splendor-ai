"""Round-robin-style comparison of N DISTINCT named agents (not "one test
agent vs identical opponents" like evaluate.evaluate_matchup) all sitting
at the same table together. Rotates which seat each named agent occupies
across games so no agent is stuck disadvantaged by always going last.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from splendor_ai.agents.base import Agent
from splendor_ai.config import GameConfig
from splendor_ai.data.loader import DataBundle
from splendor_ai.game.simulate import play_game

AgentFactory = Callable[[], Agent]


@dataclass
class TournamentResult:
    games: int
    win_rate: float
    tie_rate: float
    loss_rate: float
    avg_prestige: float
    avg_purchased_cards: float


def run_tournament(
    bundle: DataBundle,
    config: GameConfig,
    named_agent_factories: Dict[str, AgentFactory],
    num_games: int,
    base_seed: int = 0,
    max_turns: int = 3000,
) -> Dict[str, TournamentResult]:
    names = list(named_agent_factories.keys())
    n = len(names)
    if n != config.num_players:
        raise ValueError(f"expected exactly {config.num_players} named agents, got {n}")

    totals = {name: {"wins": 0, "ties": 0, "losses": 0, "prestige": 0.0, "cards": 0.0, "games": 0} for name in names}

    for i in range(num_games):
        rotation = i % n
        seat_names = [names[(seat + rotation) % n] for seat in range(n)]
        agents = [named_agent_factories[name]() for name in seat_names]

        state = play_game(bundle, config, agents, seed=base_seed + i, max_turns=max_turns)
        winners = set(state.winners or [])

        for seat, name in enumerate(seat_names):
            p = state.player(seat)
            t = totals[name]
            t["games"] += 1
            t["prestige"] += p.prestige
            t["cards"] += p.purchased_card_count
            if seat in winners:
                if len(winners) == 1:
                    t["wins"] += 1
                else:
                    t["ties"] += 1
            else:
                t["losses"] += 1

    results = {}
    for name, t in totals.items():
        g = t["games"]
        results[name] = TournamentResult(
            games=g,
            win_rate=t["wins"] / g,
            tie_rate=t["ties"] / g,
            loss_rate=t["losses"] / g,
            avg_prestige=t["prestige"] / g,
            avg_purchased_cards=t["cards"] / g,
        )
    return results
