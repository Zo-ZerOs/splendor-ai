from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import PickTokensAction
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.rules import compute_ranking, compute_winners
from splendor_ai.game.state import TurnPhase


def _trivial_pick(state, config):
    """A pick action that never triggers a token-return or noble phase."""
    apply_action(state, config, PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN)))
    while state.phase != TurnPhase.MAIN_ACTION and not state.game_over:
        # shouldn't happen with fresh token counts, but stay robust
        break


def test_reaching_winning_score_triggers_final_round_not_immediate_end(bundle, config):
    state = setup_game(bundle, config, seed=1)
    five_pt_card = bundle.card_by_id["t3_016"]
    state.player(0).purchased_cards = [five_pt_card, five_pt_card, five_pt_card]  # prestige 15

    _trivial_pick(state, config)  # player 0's turn ends, triggers final round

    assert state.final_round_started
    assert not state.game_over
    assert state.current_player == 1
    assert state.final_round_end_at == 4


def test_all_players_get_equal_turns_before_game_ends(bundle, config):
    state = setup_game(bundle, config, seed=1)
    five_pt_card = bundle.card_by_id["t3_016"]
    state.player(0).purchased_cards = [five_pt_card, five_pt_card, five_pt_card]

    _trivial_pick(state, config)  # player 0 -> triggers, turns_completed=1
    assert not state.game_over
    _trivial_pick(state, config)  # player 1, turns_completed=2
    assert not state.game_over
    _trivial_pick(state, config)  # player 2, turns_completed=3
    assert not state.game_over
    _trivial_pick(state, config)  # player 3, turns_completed=4 -> round complete
    assert state.game_over
    assert state.phase == TurnPhase.GAME_OVER
    assert state.turns_completed == 4


def test_winner_is_highest_prestige(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.player(0).purchased_cards = [bundle.card_by_id["t3_016"]] * 3  # 15
    state.player(1).purchased_cards = [bundle.card_by_id["t3_016"]] * 2  # 10
    winners = compute_winners(state)
    assert winners == [0]


def test_tiebreak_by_fewer_purchased_cards(bundle, config):
    state = setup_game(bundle, config, seed=1)
    five = bundle.card_by_id["t3_016"]
    zero = bundle.card_by_id["t1_001"]  # 0 points
    # Both players end with 15 prestige, but player 1 used more cards to get there.
    state.player(0).purchased_cards = [five, five, five]  # 15 pts, 3 cards
    state.player(1).purchased_cards = [five, five, five, zero, zero]  # 15 pts, 5 cards

    winners = compute_winners(state)
    assert winners == [0]


def test_full_tie_is_reported_as_shared(bundle, config):
    state = setup_game(bundle, config, seed=1)
    five = bundle.card_by_id["t3_016"]
    state.player(0).purchased_cards = [five, five, five]
    state.player(1).purchased_cards = [five, five, five]

    winners = compute_winners(state)
    assert set(winners) == {0, 1}


def test_ranking_orders_all_players(bundle, config):
    state = setup_game(bundle, config, seed=1)
    five = bundle.card_by_id["t3_016"]
    state.player(0).purchased_cards = [five, five, five]  # 15
    state.player(1).purchased_cards = [five, five]  # 10
    state.player(2).purchased_cards = [five]  # 5
    # player 3 stays at 0

    ranking = compute_ranking(state)
    assert ranking == [[0], [1], [2], [3]]
