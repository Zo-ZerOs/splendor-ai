import pytest

from splendor_ai.cli.human_input import format_action, parse_command
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
)
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.state import TurnPhase


def test_parse_pick_three(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("P white blue green", state, config)
    assert parsed.kind == "action"
    assert parsed.action == PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN))


def test_parse_pick_three_case_insensitive(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("p WHITE Blue GrEeN", state, config)
    assert parsed.kind == "action"


def test_parse_pick_two_same(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("P red red", state, config)
    assert parsed.action == PickTokensAction(colors=(Color.RED, Color.RED))


def test_parse_pick_two_different_parses_but_is_illegal_when_3plus_colors_remain(bundle, config):
    """'P red blue' is syntactically a valid PickTokensAction shape --
    it's only illegal right now because 3+ normal colors remain in the
    bank (it becomes legal once fewer than 3 remain). Legality is the
    engine's call, not the parser's -- see explain.py / test_explain.py."""
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("P red blue", state, config)
    assert parsed.kind == "action"
    assert parsed.action == PickTokensAction(colors=(Color.RED, Color.BLUE))

    from splendor_ai.game.engine import IllegalActionError

    with pytest.raises(IllegalActionError):
        apply_action(state, config, parsed.action)


def test_parse_pick_two_different_is_legal_when_fewer_than_3_colors_remain(bundle, config):
    state = setup_game(bundle, config, seed=1)
    for c in (Color.GREEN, Color.RED, Color.BLACK):
        state.board.bank[c] = 0
    parsed = parse_command("P white blue", state, config)
    assert parsed.kind == "action"
    apply_action(state, config, parsed.action)  # must not raise
    assert state.player(0).tokens[Color.WHITE] == 1
    assert state.player(0).tokens[Color.BLUE] == 1


def test_parse_pick_with_gold_is_error(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("P white blue gold", state, config)
    assert parsed.kind == "error"


def test_parse_buy_visible(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("B T1-1", state, config)
    assert parsed.action == BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0)


def test_parse_buy_reserved(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("B R2", state, config)
    assert parsed.action == BuyCardAction(source=CardSource.RESERVED, reserved_index=1)


def test_parse_reserve_visible(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("R T2-3", state, config)
    assert parsed.action == ReserveCardAction(source=ReserveSource.VISIBLE, tier=2, slot=2)


def test_parse_reserve_blind(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("R D3", state, config)
    assert parsed.action == ReserveCardAction(source=ReserveSource.BLIND, tier=3)


def test_parse_invalid_syntax_gives_helpful_message(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("B T9-9", state, config)
    assert parsed.kind == "error"
    assert parsed.error


def test_parse_unknown_command(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("XYZZY", state, config)
    assert parsed.kind == "error"


def test_meta_commands(bundle, config):
    state = setup_game(bundle, config, seed=1)
    for cmd, kind in [("help", "help"), ("STATE", "state"), ("Legal", "legal"), ("history", "history"), ("quit", "quit")]:
        assert parse_command(cmd, state, config).kind == kind


def test_parse_return_tokens(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("RETURN white gold", state, config)
    assert parsed.action == ReturnTokensAction(colors=(Color.WHITE, Color.GOLD))


def test_parse_noble_choice(bundle, config):
    state = setup_game(bundle, config, seed=1)
    state.pending_eligible_nobles = ["noble_003", "noble_007"]
    parsed = parse_command("NOBLE 2", state, config)
    assert parsed.action == SelectNobleAction(noble_id="noble_007")


def test_parse_pass(bundle, config):
    state = setup_game(bundle, config, seed=1)
    parsed = parse_command("PASS", state, config)
    assert parsed.action == PassAction()


def test_format_action_round_trips_through_parser(bundle, config):
    state = setup_game(bundle, config, seed=1)
    actions = [
        PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN)),
        BuyCardAction(source=CardSource.VISIBLE, tier=2, slot=3),
        ReserveCardAction(source=ReserveSource.BLIND, tier=1),
    ]
    for action in actions:
        cmd_str = format_action(action, state)
        reparsed = parse_command(cmd_str, state, config)
        assert reparsed.action == action, f"round-trip failed for {action}"
