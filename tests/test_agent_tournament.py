from splendor_ai.agents.random_agent import RandomAgent
from splendor_ai.training.agent_tournament import run_tournament


def test_tournament_seat_rotation_and_totals(bundle, config):
    factories = {f"agent{i}": (lambda i=i: RandomAgent(seed=i)) for i in range(config.num_players)}
    results = run_tournament(bundle, config, factories, num_games=8, base_seed=0)

    assert set(results.keys()) == set(factories.keys())
    for name, r in results.items():
        assert r.games == 8
        assert 0.0 <= r.win_rate <= 1.0
        assert abs((r.win_rate + r.tie_rate + r.loss_rate) - 1.0) < 1e-9

    total_wins = sum(r.win_rate * r.games for r in results.values())
    assert total_wins > 0  # someone won across 8 games


def test_tournament_rejects_wrong_agent_count(bundle, config):
    import pytest

    factories = {"only_one": lambda: RandomAgent()}
    with pytest.raises(ValueError):
        run_tournament(bundle, config, factories, num_games=1)
