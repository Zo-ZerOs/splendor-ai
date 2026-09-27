"""Round-trip tests for splendor_ai.game.actions.action_to_dict/from_dict,
which back GameState.history so a full game's exact action sequence
survives save/load and can be re-rendered as terminal-command syntax
later (see splendor_ai.cli.human_input.format_action)."""
from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import (
    BuyCardAction,
    CardSource,
    PassAction,
    PickTokensAction,
    ReserveCardAction,
    ReserveSource,
    ReturnTokensAction,
    SelectNobleAction,
    action_from_dict,
    action_to_dict,
)


def _round_trip(action):
    return action_from_dict(action_to_dict(action))


def test_pick_tokens_round_trip():
    a = PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN))
    assert _round_trip(a) == a


def test_buy_visible_round_trip():
    a = BuyCardAction(source=CardSource.VISIBLE, tier=2, slot=3)
    assert _round_trip(a) == a


def test_buy_reserved_round_trip():
    a = BuyCardAction(source=CardSource.RESERVED, reserved_index=1)
    assert _round_trip(a) == a


def test_reserve_visible_round_trip():
    a = ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0)
    assert _round_trip(a) == a


def test_reserve_blind_round_trip():
    a = ReserveCardAction(source=ReserveSource.BLIND, tier=3)
    assert _round_trip(a) == a


def test_return_tokens_round_trip():
    a = ReturnTokensAction(colors=(Color.GOLD, Color.RED))
    assert _round_trip(a) == a


def test_select_noble_round_trip():
    a = SelectNobleAction(noble_id="noble_003")
    assert _round_trip(a) == a


def test_pass_round_trip():
    a = PassAction()
    assert _round_trip(a) == a


def test_dict_is_json_safe():
    import json

    a = PickTokensAction(colors=(Color.WHITE, Color.GOLD))
    d = action_to_dict(a)
    json.dumps(d)  # must not raise
