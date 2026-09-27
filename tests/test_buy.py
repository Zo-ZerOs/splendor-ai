import pytest

from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import BuyCardAction, CardSource
from splendor_ai.game.engine import IllegalActionError, apply_action, setup_game


def _place_visible(state, tier, slot, card):
    state.board.visible[tier][slot] = card


def test_buy_visible_card_pays_and_grants_bonus(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_036"]  # bonus=black, points=1, cost blue:4
    _place_visible(state, 1, 0, card)
    state.player(0).tokens[Color.BLUE] = 4

    apply_action(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))

    player = state.player(0)
    assert player.tokens[Color.BLUE] == 0
    assert state.board.bank[Color.BLUE] == 7 + 4
    assert player.bonuses[Color.BLACK] == 1
    assert player.prestige == 1
    assert card in player.purchased_cards


def test_buy_refills_visible_slot_from_deck(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_036"]
    _place_visible(state, 1, 0, card)
    state.player(0).tokens[Color.BLUE] = 4
    deck_before = state.board.deck_remaining(1)

    apply_action(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))

    assert state.board.visible[1][0] is not None
    assert state.board.visible[1][0] is not card
    assert state.board.deck_remaining(1) == deck_before - 1


def test_buy_slot_becomes_empty_when_deck_exhausted(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_036"]
    _place_visible(state, 1, 0, card)
    state.player(0).tokens[Color.BLUE] = 4
    state.board.decks[1] = []  # empty the deck

    apply_action(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))
    assert state.board.visible[1][0] is None


def test_permanent_bonus_discounts_cost(bundle, config):
    state = setup_game(bundle, config, seed=1)
    # t1_001: bonus black, cost green:1 red:3 black:1
    card = bundle.card_by_id["t1_001"]
    _place_visible(state, 1, 0, card)
    player = state.player(0)
    player.bonuses[Color.RED] = 3  # fully discounts the red cost
    player.tokens[Color.GREEN] = 1
    player.tokens[Color.BLACK] = 1

    apply_action(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))
    assert player.tokens[Color.GREEN] == 0
    assert player.tokens[Color.BLACK] == 0


def test_gold_pays_for_shortfall(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_036"]  # cost blue:4
    _place_visible(state, 1, 0, card)
    player = state.player(0)
    player.tokens[Color.BLUE] = 2
    player.tokens[Color.GOLD] = 2

    apply_action(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))
    assert player.tokens[Color.BLUE] == 0
    assert player.tokens[Color.GOLD] == 0
    assert state.board.bank[Color.GOLD] == config.gold_tokens + 2


def test_colored_tokens_used_before_gold(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_036"]  # cost blue:4
    _place_visible(state, 1, 0, card)
    player = state.player(0)
    player.tokens[Color.BLUE] = 4
    player.tokens[Color.GOLD] = 3  # should remain untouched

    apply_action(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))
    assert player.tokens[Color.BLUE] == 0
    assert player.tokens[Color.GOLD] == 3


def test_cannot_buy_unaffordable_card(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_036"]  # cost blue:4
    _place_visible(state, 1, 0, card)
    with pytest.raises(IllegalActionError):
        apply_action(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))


def test_buy_reserved_card(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_036"]
    player = state.player(0)
    player.reserved_cards.append(card)
    player.tokens[Color.BLUE] = 4

    apply_action(state, config, BuyCardAction(source=CardSource.RESERVED, reserved_index=0))
    assert player.reserved_cards == []
    assert card in player.purchased_cards
