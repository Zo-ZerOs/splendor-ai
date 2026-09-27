from splendor_ai.data.schemas import Color, NORMAL_COLORS
from splendor_ai.game.engine import setup_game
from splendor_ai.game.state import TurnPhase


def test_four_player_setup_bank(bundle, config):
    state = setup_game(bundle, config, seed=1)
    assert len(state.players) == 4
    for c in NORMAL_COLORS:
        assert state.board.bank[c] == 7
    assert state.board.bank[Color.GOLD] == 5


def test_four_player_setup_visible_cards(bundle, config):
    state = setup_game(bundle, config, seed=1)
    for tier in (1, 2, 3):
        assert len(state.board.visible[tier]) == 4
        assert all(card is not None for card in state.board.visible[tier])
    assert state.board.deck_remaining(1) == 40 - 4
    assert state.board.deck_remaining(2) == 30 - 4
    assert state.board.deck_remaining(3) == 20 - 4


def test_four_player_setup_nobles(bundle, config):
    state = setup_game(bundle, config, seed=1)
    assert len(state.board.nobles) == 5
    assert len(set(n.id for n in state.board.nobles)) == 5


def test_players_start_empty(bundle, config):
    state = setup_game(bundle, config, seed=1)
    for p in state.players:
        assert p.total_tokens == 0
        assert p.purchased_cards == []
        assert p.reserved_cards == []
        assert p.nobles == []
        assert p.prestige == 0

    assert state.current_player == 0
    assert state.phase == TurnPhase.MAIN_ACTION
    assert state.turns_completed == 0
    assert state.final_round_end_at is None
    assert not state.game_over


def test_seed_reproducibility(bundle, config):
    s1 = setup_game(bundle, config, seed=42)
    s2 = setup_game(bundle, config, seed=42)
    d1, d2 = s1.to_dict(), s2.to_dict()
    del d1["rng_state"], d2["rng_state"]
    assert d1 == d2


def test_different_seeds_produce_different_layouts(bundle, config):
    s1 = setup_game(bundle, config, seed=1)
    s2 = setup_game(bundle, config, seed=2)
    ids1 = [c.id for c in s1.board.visible[1]]
    ids2 = [c.id for c in s2.board.visible[1]]
    assert ids1 != ids2
