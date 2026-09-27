import pytest

from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import ReserveCardAction, ReserveSource
from splendor_ai.game.engine import IllegalActionError, apply_action, setup_game
from splendor_ai.game.state import TurnPhase


def test_reserve_visible_card_grants_gold_and_refills(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = state.board.visible[1][0]
    deck_before = state.board.deck_remaining(1)
    gold_before = state.board.bank[Color.GOLD]

    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))

    player = state.player(0)
    assert card in player.reserved_cards
    assert player.tokens[Color.GOLD] == 1
    assert state.board.bank[Color.GOLD] == gold_before - 1
    assert state.board.visible[1][0] is not None
    assert state.board.visible[1][0] is not card
    assert state.board.deck_remaining(1) == deck_before - 1


def test_reserve_blind_top_card(bundle, config):
    state = setup_game(bundle, config, seed=1)
    expected_card = state.board.decks[2][-1]
    deck_before = state.board.deck_remaining(2)

    apply_action(state, config, ReserveCardAction(source=ReserveSource.BLIND, tier=2))

    player = state.player(0)
    assert player.reserved_cards == [expected_card]
    assert state.board.deck_remaining(2) == deck_before - 1


def test_reserve_without_gold_remains_legal(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.bank[Color.GOLD] = 0

    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))
    player = state.player(0)
    assert len(player.reserved_cards) == 1
    assert player.tokens[Color.GOLD] == 0


def test_max_three_reserved_cards(bundle, config):
    state = setup_game(bundle, config, seed=1)
    for tier, slot in [(1, 0), (1, 1), (1, 2)]:
        state.board.bank[Color.GOLD] = 5  # keep gold available throughout
        apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=tier, slot=slot))
        state.current_player = 0  # stay on player 0 for this test
        state.phase = TurnPhase.MAIN_ACTION

    assert len(state.player(0).reserved_cards) == 3
    with pytest.raises(IllegalActionError):
        apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))


def test_cannot_reserve_from_empty_slot(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.visible[1][0] = None
    state.board.decks[1] = []
    with pytest.raises(IllegalActionError):
        apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))


def test_cannot_reserve_blind_from_empty_deck(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.decks[3] = []
    with pytest.raises(IllegalActionError):
        apply_action(state, config, ReserveCardAction(source=ReserveSource.BLIND, tier=3))
