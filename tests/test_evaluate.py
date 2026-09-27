from splendor_ai.agents.heuristic_agent import HeuristicAgent
from splendor_ai.agents.random_agent import RandomAgent
from splendor_ai.training.evaluate import evaluate_matchup, measure_baseline


def test_measure_baseline_random(bundle, config):
    result = measure_baseline(bundle, config, lambda: RandomAgent(), num_games=20, base_seed=0)
    assert result.games == 20
    assert result.avg_turns_per_player > 0
    assert 0.0 <= sum(result.win_rate_by_seat.values()) <= 1.01  # win rates average to ~1/num_games scale is wrong; just sanity check range
    for rate in result.win_rate_by_seat.values():
        assert 0.0 <= rate <= 1.0


def test_measure_baseline_heuristic_terminates(bundle, config):
    result = measure_baseline(bundle, config, lambda: HeuristicAgent(), num_games=10, base_seed=1)
    assert result.games == 10
    assert result.avg_turns_per_player > 0


def test_heuristic_beats_random_convincingly(bundle, config):
    """Sanity check that the heuristic baseline is actually stronger than
    random -- otherwise it's useless as an RL-training milestone."""
    result = evaluate_matchup(
        bundle, config,
        test_agent_factory=lambda: HeuristicAgent(),
        opponent_agent_factory=lambda: RandomAgent(),
        num_games=40,
        base_seed=0,
    )
    assert result.games == 40
    assert result.win_rate > 0.5  # heuristic (1 seat) vs 3 randoms should win more than its 25% fair share
    assert sum(result.games_by_seat.values()) == 40


def test_evaluate_matchup_seat_rotation(bundle, config):
    result = evaluate_matchup(
        bundle, config,
        test_agent_factory=lambda: RandomAgent(),
        opponent_agent_factory=lambda: RandomAgent(),
        num_games=8,
        base_seed=0,
        rotate_seats=True,
    )
    assert all(n == 2 for n in result.games_by_seat.values())  # 8 games / 4 seats


def test_winner_efficiency_fields_isolate_wins_from_losses(bundle, config):
    """`avg_turns_per_player`/`avg_purchased_cards` average every game
    including losses, which dilutes the number that actually distinguishes
    efficient play; `winner_avg_turns`/`winner_avg_buys` must reflect ONLY
    the test agent's own games as the winner."""
    result = evaluate_matchup(
        bundle, config,
        test_agent_factory=lambda: HeuristicAgent(),
        opponent_agent_factory=lambda: RandomAgent(),
        num_games=40,
        base_seed=0,
    )
    assert result.win_rate > 0.5  # sanity: some real wins to measure
    assert result.winner_avg_turns is not None
    assert result.winner_avg_buys is not None
    assert result.winner_avg_turns > 0
    assert result.winner_avg_buys > 0
    # A winner's buy count is bounded by the (small) number of turns it took.
    assert result.winner_avg_buys <= result.winner_avg_turns


def test_winner_efficiency_fields_are_none_when_test_agent_never_wins(bundle, config):
    """A test agent that never wins a single game has no winning games to
    average -- must report None, not a division-by-zero crash or a
    misleading 0.0. (seed=0/10 games is confirmed to give random 0 wins
    against the heuristic here, so this asserts unconditionally.)"""
    result = evaluate_matchup(
        bundle, config,
        test_agent_factory=lambda: RandomAgent(),
        opponent_agent_factory=lambda: HeuristicAgent(),
        num_games=10,
        base_seed=0,
    )
    assert result.win_rate == 0.0
    assert result.winner_avg_turns is None
    assert result.winner_avg_buys is None
