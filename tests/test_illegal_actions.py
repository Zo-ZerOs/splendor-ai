import pytest

from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import BuyCardAction, CardSource, PickTokensAction, ReserveCardAction, ReserveSource
from splendor_ai.game.engine import IllegalActionError, apply_action, setup_game


def test_illegal_action_does_not_mutate_state(bundle, config):
    state = setup_game(bundle, config, seed=1)
    before = state.to_dict()
    before.pop("rng_state")

    with pytest.raises(IllegalActionError):
        # bank has only 7 white tokens < the 4 needed isn't the issue; use an
        # affordability failure instead: no card at this slot has been reserved.
        apply_action(state, config, BuyCardAction(source=CardSource.RESERVED, reserved_index=0))

    after = state.to_dict()
    after.pop("rng_state")
    assert before == after


def test_illegal_pick_two_same_color_does_not_mutate_state(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.board.bank[Color.WHITE] = 2
    before = state.to_dict()
    before.pop("rng_state")

    with pytest.raises(IllegalActionError):
        apply_action(state, config, PickTokensAction(colors=(Color.WHITE, Color.WHITE)))

    after = state.to_dict()
    after.pop("rng_state")
    assert before == after


def test_illegal_reserve_beyond_limit_does_not_mutate_state(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_001"]
    state.player(0).reserved_cards = [card, card, card]
    before = state.to_dict()
    before.pop("rng_state")

    with pytest.raises(IllegalActionError):
        apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))

    after = state.to_dict()
    after.pop("rng_state")
    assert before == after


def test_illegal_action_error_reports_seat_and_phase(bundle, config):
    state = setup_game(bundle, config, seed=1)
    try:
        apply_action(state, config, BuyCardAction(source=CardSource.RESERVED, reserved_index=0))
        assert False, "expected IllegalActionError"
    except IllegalActionError as e:
        assert e.seat == 0
        assert e.phase == state.phase
