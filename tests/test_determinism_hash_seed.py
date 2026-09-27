"""Regression test for a real bug: legal_return_actions() used to
deduplicate token-return combos via a bare `set()` and return them in set
iteration order, which depends on Python's per-process hash randomization
(PYTHONHASHSEED). Any deterministic tie-break over that list (e.g.
HeuristicAgent's `min(..., key=penalty)`) could therefore diverge between
processes for the IDENTICAL game seed, silently breaking the engine's
"deterministic given a seed" guarantee. Fixed by sorting explicitly by
color value. This test runs the same scenario in two subprocesses forced
to different hash seeds and checks the results match.
"""
import subprocess
import sys

_SCRIPT = """
from splendor_ai.data.schemas import Color
from splendor_ai.config import GameConfig
from splendor_ai.data.loader import load_all
from splendor_ai.game.engine import setup_game
from splendor_ai.game.rules import legal_return_actions

bundle = load_all()
config = GameConfig.load()
state = setup_game(bundle, config, seed=1)
player = state.player(0)
player.tokens[Color.WHITE] = 3
player.tokens[Color.BLUE] = 3
player.tokens[Color.GREEN] = 3
player.tokens[Color.GOLD] = 2  # 11 tokens total, excess = 1

actions = legal_return_actions(state, player, config)
print([tuple(c.value for c in a.colors) for a in actions])
"""


def _run_with_hash_seed(seed: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=r"C:\myFile\KAIST\Splendor",
        env={"PYTHONHASHSEED": seed, "PATH": __import__("os").environ.get("PATH", "")},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_legal_return_actions_order_is_independent_of_hash_seed():
    out1 = _run_with_hash_seed("1")
    out2 = _run_with_hash_seed("2")
    out3 = _run_with_hash_seed("42")
    assert out1 == out2 == out3


def test_heuristic_vs_heuristic_games_are_reproducible_across_hash_seeds():
    script = """
from splendor_ai.config import GameConfig
from splendor_ai.data.loader import load_all
from splendor_ai.agents.heuristic_agent import HeuristicAgent
from splendor_ai.game.simulate import play_game

bundle = load_all()
config = GameConfig.load()
agents = [HeuristicAgent() for _ in range(4)]
state = play_game(bundle, config, agents, seed=5, max_turns=3000)
print(state.turns_completed, state.game_over, [p.prestige for p in state.players])
"""
    import os

    def run(seed):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=r"C:\myFile\KAIST\Splendor",
            env={"PYTHONHASHSEED": seed, "PATH": os.environ.get("PATH", "")},
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    out1 = run("7")
    out2 = run("99")
    assert out1 == out2
    assert "True" in out1  # game actually finished
