"""Renders the read-only viewer across every state the game can be in --
normal play, pending token-return, pending noble-choice, final round, and
game-over/tie -- to prove none of them crash the draw loop, plus a
missing-image fallback check. Runs under SDL's dummy video driver so it
works in a headless test environment; the drawing code path is identical
to a real window (only pygame.display.set_mode's backend differs).
"""
import dataclasses
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from splendor_ai.data.schemas import Color
from splendor_ai.game.engine import _advance_after_main_action, apply_action, setup_game
from splendor_ai.game.state import TurnPhase
from splendor_ai.ui.asset_loader import AssetLoader
from splendor_ai.ui.pygame_view import SplendorView


@pytest.fixture(scope="module", autouse=True)
def _pygame_display():
    pygame.init()
    pygame.display.set_mode((100, 100))
    yield
    pygame.quit()


def _view(bundle, config):
    return SplendorView(bundle, config, human_seat=0, resolution=(1000, 700))


def _render_once(view, state):
    """Exercise the exact draw path _run() uses, without spinning up the
    background thread (so failures raise directly instead of being
    swallowed by the thread's own try/except)."""
    screen = pygame.display.get_surface()
    fonts = {
        "small": pygame.font.SysFont("arial", 14),
        "med": pygame.font.SysFont("arial", 17),
        "big": pygame.font.SysFont("arial", 30, bold=True),
        "deck": pygame.font.SysFont("arial", 22, bold=True),
        "coin": pygame.font.SysFont("arial", 19, bold=True),
    }
    view._draw(screen, state, fonts)


def test_render_normal_main_action_state(bundle, config):
    state = setup_game(bundle, config, seed=1)
    _render_once(_view(bundle, config), state)


def test_render_pending_token_return_state(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.player(0).tokens[Color.WHITE] = 11
    _advance_after_main_action(state, config)
    assert state.phase == TurnPhase.TOKEN_RETURN
    _render_once(_view(bundle, config), state)


def test_render_pending_noble_choice_state(bundle, config):
    state = setup_game(bundle, config, seed=1)
    noble_a = bundle.noble_by_id["noble_001"]
    noble_b = bundle.noble_by_id["noble_005"]
    state.board.nobles = [noble_a, noble_b]
    state.player(0).bonuses[Color.WHITE] = 3
    state.player(0).bonuses[Color.BLUE] = 3
    state.player(0).bonuses[Color.GREEN] = 3
    state.player(0).bonuses[Color.BLACK] = 3
    _advance_after_main_action(state, config)
    assert state.phase == TurnPhase.NOBLE_CHOICE
    _render_once(_view(bundle, config), state)


def test_render_final_round_state(bundle, config):
    state = setup_game(bundle, config, seed=1)
    five = bundle.card_by_id["t3_016"]
    state.player(0).purchased_cards = [five, five, five]  # 15 prestige -- triggers final round
    from splendor_ai.data.schemas import Color as _C
    from splendor_ai.game.actions import PickTokensAction

    apply_action(state, config, PickTokensAction(colors=(_C.WHITE, _C.BLUE, _C.GREEN)))
    assert state.final_round_started
    _render_once(_view(bundle, config), state)


def test_render_game_over_solo_winner(bundle, config):
    state = setup_game(bundle, config, seed=1)
    five = bundle.card_by_id["t3_016"]
    state.player(0).purchased_cards = [five, five, five]
    state.game_over = True
    state.phase = TurnPhase.GAME_OVER
    from splendor_ai.game.rules import compute_winners

    state.winners = compute_winners(state)
    _render_once(_view(bundle, config), state)


def test_render_game_over_tie(bundle, config):
    state = setup_game(bundle, config, seed=1)
    five = bundle.card_by_id["t3_016"]
    state.player(0).purchased_cards = [five, five, five]
    state.player(1).purchased_cards = [five, five, five]
    state.game_over = True
    state.phase = TurnPhase.GAME_OVER
    from splendor_ai.game.rules import compute_winners

    state.winners = compute_winners(state)
    assert len(state.winners) > 1
    _render_once(_view(bundle, config), state)


def test_render_with_reserved_cards_face_up_and_face_down(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_001"]
    state.player(0).reserved_cards = [card, card]  # human -- face up
    state.player(1).reserved_cards = [card]  # opponent -- face down
    _render_once(_view(bundle, config), state)


def test_render_empty_visible_slots_and_empty_decks(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.visible[1][0] = None
    state.board.decks[1] = []
    state.board.decks[2] = []
    state.board.decks[3] = []
    _render_once(_view(bundle, config), state)


def test_render_no_nobles_remaining(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.nobles = []
    _render_once(_view(bundle, config), state)


def test_render_survives_a_full_random_game(bundle, config):
    """Render after every single action of a full random game -- broadest
    possible smoke coverage of real board configurations."""
    import random

    from splendor_ai.game.rules import legal_actions

    view = _view(bundle, config)
    rng = random.Random(3)
    state = setup_game(bundle, config, seed=3)
    steps = 0
    while not state.game_over and steps < 1000:
        action = rng.choice(legal_actions(state, config))
        apply_action(state, config, action)
        _render_once(view, state)
        steps += 1
    assert state.game_over


def test_missing_card_image_falls_back_to_placeholder_without_crashing(bundle, config, caplog):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_001"]
    broken_card = dataclasses.replace(card, image_abs="C:/does/not/exist.png")
    state.board.visible[1][0] = broken_card

    view = _view(bundle, config)
    with caplog.at_level("WARNING"):
        _render_once(view, state)  # must not raise
    assert any("Could not load image" in r.message for r in caplog.records)


def test_missing_noble_image_falls_back_to_placeholder(bundle, config):
    noble = bundle.noble_by_id["noble_001"]
    broken_noble = dataclasses.replace(noble, image_abs="C:/does/not/exist.png")
    loader = AssetLoader()
    surf = loader.noble_image(broken_noble, (70, 70))
    assert surf.get_size() == (70, 70)  # placeholder still returns a correctly-sized surface


def test_card_tracker_skips_bought_and_outlines_visible_cards(bundle, config):
    state = setup_game(bundle, config, seed=1, starting_player=0)
    bought_card = bundle.card_by_id["t1_001"]
    state.player(0).purchased_cards.append(bought_card)
    visible_card = state.board.visible[1][0]

    view = _view(bundle, config)
    bought_ids = {c.id for player in state.players for c in player.purchased_cards}
    visible_ids = {c.id for slots in state.board.visible.values() for c in slots if c is not None}
    assert bought_card.id in bought_ids
    assert visible_card.id in visible_ids
    _render_once(view, state)  # must not raise


def test_card_tracker_shows_waiting_message_on_ai_turn(bundle, config):
    state = setup_game(bundle, config, seed=1, starting_player=1)
    view = _view(bundle, config)
    _render_once(view, state)  # must not raise


def test_count_player_actions_counts_only_pick_buy_reserve(bundle, config):
    from splendor_ai.data.schemas import Color as _Color
    from splendor_ai.game.actions import BuyCardAction, CardSource, PickTokensAction, ReserveCardAction, ReserveSource
    from splendor_ai.ui.pygame_view import _count_player_actions

    state = setup_game(bundle, config, seed=1)
    apply_action(state, config, PickTokensAction(colors=(_Color.WHITE, _Color.BLUE, _Color.GREEN)))
    state.player(0).tokens[_Color.WHITE] = 4  # afford t1_037 (cost white:4) via a fresh pick/buy sequence
    card = bundle.card_by_id["t1_037"]
    state.board.visible[1][0] = card
    from splendor_ai.game.state import TurnPhase

    state.current_player = 0
    state.phase = TurnPhase.MAIN_ACTION
    apply_action(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))
    state.current_player = 0
    state.phase = TurnPhase.MAIN_ACTION
    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=2, slot=0))

    counts = _count_player_actions(state, 0)
    assert counts == {"pick_tokens": 1, "buy_card": 1, "reserve_card": 1}
    assert _count_player_actions(state, 1) == {"pick_tokens": 0, "buy_card": 0, "reserve_card": 0}


def test_reserved_card_face_up_down_logic(bundle, config, monkeypatch):
    """Opponents' visible-sourced reservations render face up (not secret
    in physical Splendor), blind-sourced ones render face down; the
    human's own reservations always render face up regardless of source."""
    state = setup_game(bundle, config, seed=1)
    card_a, card_b = bundle.card_by_id["t1_001"], bundle.card_by_id["t1_002"]

    state.player(0).reserved_cards = [card_a, card_b]  # human: visible, blind
    state.player(0).reserved_card_sources = ["visible", "blind"]
    state.player(1).reserved_cards = [card_a, card_b]  # opponent: visible, blind
    state.player(1).reserved_card_sources = ["visible", "blind"]

    view = _view(bundle, config)
    fonts = {
        "small": pygame.font.SysFont("arial", 14), "med": pygame.font.SysFont("arial", 17),
        "big": pygame.font.SysFont("arial", 30, bold=True), "deck": pygame.font.SysFont("arial", 22, bold=True),
        "coin": pygame.font.SysFont("arial", 19, bold=True),
    }
    screen = pygame.display.get_surface()
    layout = view.layout

    calls = []
    monkeypatch.setattr(view.assets, "card_image", lambda card, size: calls.append(("up", card.id)) or pygame.Surface(size))
    monkeypatch.setattr(view.assets, "card_back", lambda size: calls.append(("down",)) or pygame.Surface(size))

    calls.clear()
    view._draw_player_panel(screen, state, 0, layout.player_panel_rects[0], fonts)
    assert calls == [("up", card_a.id), ("up", card_b.id)]  # human: both face up regardless of source

    calls.clear()
    view._draw_player_panel(screen, state, 1, layout.player_panel_rects[1], fonts)
    assert calls == [("up", card_a.id), ("down",)]  # opponent: visible face up, blind face down


def test_view_closed_flag_starts_false():
    from splendor_ai.data.loader import load_all
    from splendor_ai.config import GameConfig

    bundle = load_all()
    config = GameConfig.load()
    view = SplendorView(bundle, config, human_seat=0, resolution=(400, 300))
    assert view.closed is False
