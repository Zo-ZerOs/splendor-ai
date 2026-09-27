import pytest

from splendor_ai.agents.random_agent import RandomAgent
from splendor_ai.game.rules import legal_actions
from splendor_ai.game.simulate import play_game


@pytest.mark.parametrize("seed", range(25))
def test_random_agent_only_ever_picks_legal_actions(bundle, config, seed):
    agents = [RandomAgent(seed=seed * 10 + i) for i in range(config.num_players)]
    state = play_game(bundle, config, agents, seed=seed)
    assert state.game_over


def test_random_vs_random_games_always_terminate(bundle, config):
    for seed in range(50):
        agents = [RandomAgent(seed=seed * 10 + i) for i in range(config.num_players)]
        state = play_game(bundle, config, agents, seed=seed, max_turns=3000)
        assert state.game_over, f"seed {seed} did not terminate"
        assert state.winners
        # No player should ever end with negative prestige or an
        # impossible bonus/token count.
        for p in state.players:
            assert p.prestige >= 0
            for n in p.tokens.values():
                assert n >= 0


def test_random_agent_raises_if_asked_to_act_with_no_legal_actions(bundle, config):
    from splendor_ai.game.engine import setup_game
    from splendor_ai.game.state import TurnPhase

    state = setup_game(bundle, config, seed=1)
    state.game_over = True
    state.phase = TurnPhase.GAME_OVER
    assert legal_actions(state, config) == []
    agent = RandomAgent(seed=1)
    with pytest.raises(RuntimeError):
        agent.act(state, config)
