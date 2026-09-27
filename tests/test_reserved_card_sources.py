"""Tracks HOW each reserved card was obtained (visible vs. blind), used by
the Pygame viewer to decide face-up/down display for opponents: a card
reserved from a visible slot was seen by everyone at the table (not
secret in physical Splendor), a blind (deck-top) reservation genuinely
is secret. This is a display concern only -- never exposed to RL
observations."""
from splendor_ai.game.actions import BuyCardAction, CardSource, ReserveCardAction, ReserveSource
from splendor_ai.game.engine import apply_action, setup_game


def test_visible_reservation_recorded_as_visible_source(bundle, config):
    state = setup_game(bundle, config, seed=1)
    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))
    assert state.player(0).reserved_card_sources == ["visible"]


def test_blind_reservation_recorded_as_blind_source(bundle, config):
    state = setup_game(bundle, config, seed=1)
    apply_action(state, config, ReserveCardAction(source=ReserveSource.BLIND, tier=2))
    assert state.player(0).reserved_card_sources == ["blind"]


def test_sources_stay_aligned_with_reserved_cards_across_multiple_reserves(bundle, config):
    state = setup_game(bundle, config, seed=1)
    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))
    state.current_player = 0
    from splendor_ai.game.state import TurnPhase

    state.phase = TurnPhase.MAIN_ACTION
    apply_action(state, config, ReserveCardAction(source=ReserveSource.BLIND, tier=3))
    player = state.player(0)
    assert len(player.reserved_cards) == len(player.reserved_card_sources) == 2
    assert player.reserved_card_sources == ["visible", "blind"]


def test_buying_a_reserved_card_removes_its_source_entry(bundle, config):
    from splendor_ai.data.schemas import Color
    from splendor_ai.game.state import TurnPhase

    state = setup_game(bundle, config, seed=1)
    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))
    state.current_player = 0
    state.phase = TurnPhase.MAIN_ACTION
    apply_action(state, config, ReserveCardAction(source=ReserveSource.BLIND, tier=3))

    player = state.player(0)
    card_to_buy = player.reserved_cards[0]
    remaining = player.effective_cost(card_to_buy)
    for color in remaining:
        player.tokens[color] = remaining[color]

    state.current_player = 0
    state.phase = TurnPhase.MAIN_ACTION
    apply_action(state, config, BuyCardAction(source=CardSource.RESERVED, reserved_index=0))

    assert len(player.reserved_cards) == len(player.reserved_card_sources) == 1
    assert player.reserved_card_sources == ["blind"]  # the visible-sourced one was bought and removed


def test_clone_preserves_reserved_card_sources(bundle, config):
    state = setup_game(bundle, config, seed=1)
    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))
    clone = state.clone()
    assert clone.player(0).reserved_card_sources == ["visible"]


def test_serialization_round_trip_preserves_reserved_card_sources(bundle, config):
    from splendor_ai.game.state import GameState

    state = setup_game(bundle, config, seed=1)
    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))
    data = state.to_dict()
    restored = GameState.from_dict(data, bundle)
    assert restored.player(0).reserved_card_sources == ["visible"]
