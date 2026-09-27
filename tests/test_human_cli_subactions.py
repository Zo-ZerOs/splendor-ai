"""End-to-end scripted tests driving the REAL splendor_ai.cli.play.run()
loop (input() monkeypatched) through both required sub-actions -- token
return and noble choice -- to verify the terminal experience described in
the task: clear illegal-action explanations, phase-aware rejection of
unrelated commands, retry-until-valid, no state mutation or turn
consumption on invalid input, and control only passing to AI seats once
every pending sub-decision is resolved.

The initial GameState is hand-constructed (via monkeypatching
`play.setup_game`) so the sub-action scenarios are deterministic instead
of hoping random dealing produces them naturally.
"""
import builtins

from splendor_ai.cli import play
from splendor_ai.cli.human_input import parse_command
from splendor_ai.data.schemas import Color
from splendor_ai.game.engine import apply_action, setup_game as real_setup_game
from splendor_ai.game.state import TurnPhase


def _scripted_input(commands):
    it = iter(commands)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError()

    return fake_input


def _build_forced_scenario_state(bundle, config):
    """Player 0 (human) is set up so that picking white/blue/green pushes
    them from 8 to 11 tokens (a required 1-token return), and their
    permanent bonuses already qualify for two nobles at once (a required
    noble choice) as soon as the return resolves."""
    state = real_setup_game(bundle, config, seed=1, starting_player=0)
    player0 = state.player(0)
    player0.tokens[Color.WHITE] = 4
    player0.tokens[Color.BLUE] = 4
    player0.bonuses[Color.WHITE] = 3
    player0.bonuses[Color.BLUE] = 3
    player0.bonuses[Color.GREEN] = 3
    player0.bonuses[Color.BLACK] = 3

    noble_a = bundle.noble_by_id["noble_001"]  # white:3 blue:3 black:3
    noble_b = bundle.noble_by_id["noble_005"]  # white:3 blue:3 green:3
    state.board.nobles = [noble_a, noble_b]
    return state, noble_a, noble_b


def test_full_scripted_token_return_then_noble_choice(monkeypatch, capsys, bundle, config):
    state, noble_a, noble_b = _build_forced_scenario_state(bundle, config)
    monkeypatch.setattr(play, "setup_game", lambda *a, **k: state)

    commands = [
        "P white blue green",     # main action -> 11 tokens, triggers TOKEN_RETURN
        "RETURN white white",     # INVALID: must return exactly 1, not 2
        "P red red",              # INVALID: wrong phase (pending token return)
        "RETURN white",           # VALID: resolves to exactly 10 tokens
        "NOBLE 9",                 # INVALID: out of range
        "R D3",                    # INVALID: wrong phase (pending noble choice)
        "NOBLE 1",                 # VALID: selects noble_a
        "quit",                    # end the test once it's the human's turn again
    ]
    monkeypatch.setattr(builtins, "input", _scripted_input(commands))

    play.run(seed=1, human_seat=0, checkpoint_path=None, render_enabled=False, resolution=(800, 600))
    out = capsys.readouterr().out

    # --- token-return sub-action ---
    assert "You now have 11 tokens" in out
    assert "You must return exactly 1 token" in out
    assert "you specified 2" in out  # explanation for "RETURN white white"
    assert "pending token return to complete first" in out  # explanation for "P red red"

    # --- noble-choice sub-action ---
    assert "You qualify for multiple nobles" in out
    assert "noble index out of range" in out.lower() or "NOBLE index out of range" in out
    assert "pending noble choice to complete first" in out  # explanation for "R D3"

    # --- turn accounting / no illegal mutation ---
    assert "Your turn has not been consumed." in out

    player0 = state.player(0)
    assert player0.total_tokens == 10  # exactly at the limit after the valid return
    assert player0.nobles == [noble_a]
    assert noble_b in state.board.nobles  # not granted -- only one noble per turn

    # AI seats must not move until BOTH the token-return and noble-choice
    # sub-actions are fully resolved -- confirmed by the noble prompt
    # appearing in the transcript strictly before any AI seat's move.
    assert out.index("You qualify for multiple nobles") < out.index("Player 2 (AI):")


def test_invalid_return_does_not_mutate_state_or_advance(monkeypatch, capsys, bundle, config):
    state, _noble_a, _noble_b = _build_forced_scenario_state(bundle, config)
    monkeypatch.setattr(play, "setup_game", lambda *a, **k: state)

    commands = [
        "P white blue green",   # -> TOKEN_RETURN, excess=1
        "RETURN black",          # INVALID: player holds 0 black tokens
        "quit",
    ]
    monkeypatch.setattr(builtins, "input", _scripted_input(commands))

    play.run(seed=1, human_seat=0, checkpoint_path=None, render_enabled=False, resolution=(800, 600))
    out = capsys.readouterr().out

    assert "You only hold 0 black token(s)" in out
    assert state.phase == TurnPhase.TOKEN_RETURN  # still pending -- invalid return did not resolve it
    assert state.player(0).total_tokens == 11  # unchanged by the invalid attempt
    assert state.current_player == 0  # turn not consumed


def test_multi_token_return_when_excess_is_larger(monkeypatch, capsys, bundle, config):
    """Give the human exactly enough held tokens that a 3-distinct pick
    pushes them from 9 to 12 (excess=2), then return two tokens at once."""
    state = real_setup_game(bundle, config, seed=2, starting_player=0)
    player0 = state.player(0)
    player0.tokens[Color.WHITE] = 3
    player0.tokens[Color.BLUE] = 3
    player0.tokens[Color.BLACK] = 3
    monkeypatch.setattr(play, "setup_game", lambda *a, **k: state)

    commands = [
        "P green red gold",      # INVALID SYNTAX: gold is not a pickable color
        "P green red white",     # VALID: 3 distinct colors -> 9 + 3 = 12 tokens, excess=2
        "RETURN white blue",     # return 2 different tokens at once
        "quit",
    ]
    monkeypatch.setattr(builtins, "input", _scripted_input(commands))

    play.run(seed=2, human_seat=0, checkpoint_path=None, render_enabled=False, resolution=(800, 600))
    out = capsys.readouterr().out

    assert "INVALID COMMAND" in out and "gold cannot be picked" in out
    assert "You must return exactly 2 token" in out
    player0 = state.player(0)
    assert player0.total_tokens == 10
    assert player0.tokens[Color.WHITE] == 3  # 3 + 1 (picked) - 1 (returned)
    assert player0.tokens[Color.BLUE] == 2  # 3 - 1 (returned), never picked


def test_returned_tokens_go_back_to_the_bank_via_cli_parsed_action(bundle, config):
    """Isolated (no AI interference) check that a RETURN command parsed by
    the CLI, once applied through the shared engine, actually credits the
    bank -- not just debits the player."""
    state = real_setup_game(bundle, config, seed=3, starting_player=0)
    player0 = state.player(0)
    player0.tokens[Color.WHITE] = 4
    player0.tokens[Color.BLUE] = 4
    player0.tokens[Color.GREEN] = 3  # 11 tokens already -- one over the limit

    # Force the phase directly (equivalent to having just picked) so we can
    # isolate the RETURN step itself.
    state.phase = TurnPhase.MAIN_ACTION
    from splendor_ai.game.engine import _advance_after_main_action

    _advance_after_main_action(state, config)
    assert state.phase == TurnPhase.TOKEN_RETURN

    bank_white_before = state.board.bank[Color.WHITE]
    parsed = parse_command("RETURN white", state, config)
    assert parsed.kind == "action"
    apply_action(state, config, parsed.action)

    assert player0.total_tokens == 10
    assert state.board.bank[Color.WHITE] == bank_white_before + 1
    assert state.phase == TurnPhase.MAIN_ACTION  # resolved and turn finalized
    assert state.current_player == 1
