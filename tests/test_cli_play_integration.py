"""End-to-end smoke test: drive splendor_ai.cli.play.run() with scripted
terminal input (rendering disabled) to verify a human can actually play
a game against random AI opponents through the real command parser and
engine, including invalid input, meta commands, and quitting.
"""
import builtins

import pytest

from splendor_ai.cli import play


def _scripted_input(commands):
    it = iter(commands)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError()

    return fake_input


def test_human_can_play_several_turns_then_quit(monkeypatch, capsys):
    commands = [
        "help",
        "state",
        "legal",
        "bogus command",
        "P white blue green",
        "history",
        "quit",
    ]
    monkeypatch.setattr(builtins, "input", _scripted_input(commands))

    play.run(seed=1, human_seat=0, checkpoint_path=None, render_enabled=False, resolution=(800, 600))

    out = capsys.readouterr().out
    assert "You are Player 1" in out
    assert "INVALID COMMAND: unknown command" in out
    assert "Exiting." in out


def test_human_eof_quits_gracefully(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", _scripted_input([]))
    play.run(seed=2, human_seat=0, checkpoint_path=None, render_enabled=False, resolution=(800, 600))
    out = capsys.readouterr().out
    assert "Exiting." in out


def test_illegal_action_reprompts_without_crashing(monkeypatch, capsys):
    commands = [
        "P white white",  # illegal: bank has 7 white which IS >=4, so this is actually legal...
    ]
    # Use an action that's clearly illegal instead: buying a nonexistent reserved card.
    commands = ["B R1", "quit"]
    monkeypatch.setattr(builtins, "input", _scripted_input(commands))
    play.run(seed=3, human_seat=0, checkpoint_path=None, render_enabled=False, resolution=(800, 600))
    out = capsys.readouterr().out
    assert "ILLEGAL ACTION:" in out
    assert "Your turn has not been consumed." in out
    assert "Exiting." in out


def test_full_scripted_human_vs_three_ai_game_reaches_game_over(monkeypatch, capsys, bundle, config):
    """A complete human-vs-3-random-AI game, played to a real GAME OVER,
    with rendering disabled. The "human" is driven by an auto-input
    function that always submits a currently-legal command (generated via
    the exact same legal_actions()/format_action() the real `legal`
    terminal command uses), so this exercises the full parse -> explain
    -> engine -> render pipeline turn after turn until the game actually
    ends -- not just a handful of scripted moves."""
    import random

    from splendor_ai.cli.human_input import format_action
    from splendor_ai.game.engine import setup_game as real_setup_game
    from splendor_ai.game.rules import legal_actions

    holder = {}

    def fake_setup_game(*a, **k):
        state = real_setup_game(*a, **k)
        holder["state"] = state
        return state

    monkeypatch.setattr(play, "setup_game", fake_setup_game)

    rng = random.Random(0)

    def fake_input(prompt=""):
        state = holder["state"]
        actions = legal_actions(state, config)
        action = rng.choice(actions)
        return format_action(action, state)

    def fake_input_or_quit(prompt=""):
        state = holder.get("state")
        if state is not None and state.game_over:
            return "quit"
        return fake_input(prompt)

    monkeypatch.setattr(builtins, "input", fake_input_or_quit)

    play.run(seed=42, human_seat=0, checkpoint_path=None, render_enabled=False, resolution=(800, 600))

    state = holder["state"]
    out = capsys.readouterr().out
    assert state.game_over
    assert "GAME OVER" in out


def test_human_seat_defaults_to_random_but_is_reproducible_with_seed(monkeypatch, capsys):
    def _seat_line(seed):
        monkeypatch.setattr(builtins, "input", _scripted_input(["quit"]))
        play.run(seed=seed, human_seat=None, checkpoint_path=None, render_enabled=False, resolution=(800, 600))
        out = capsys.readouterr().out
        return next(line for line in out.splitlines() if line.startswith("You are Player"))

    line_a = _seat_line(99)
    line_b = _seat_line(99)
    assert line_a == line_b  # same seed -> same randomly-chosen seat, both times


def test_human_seat_random_choice_is_within_range(monkeypatch, capsys):
    seats_seen = set()
    for seed in range(20):
        monkeypatch.setattr(builtins, "input", _scripted_input(["quit"]))
        play.run(seed=seed, human_seat=None, checkpoint_path=None, render_enabled=False, resolution=(800, 600))
        out = capsys.readouterr().out
        line = next(l for l in out.splitlines() if l.startswith("You are Player"))
        seat_num = int(line.split()[3].rstrip("."))
        assert 1 <= seat_num <= 4
        seats_seen.add(seat_num)
    assert len(seats_seen) > 1  # confirms it actually varies, not stuck on one seat


def test_explicit_human_seat_overrides_random_default(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", _scripted_input(["quit"]))
    play.run(seed=1, human_seat=2, checkpoint_path=None, render_enabled=False, resolution=(800, 600))
    out = capsys.readouterr().out
    assert "You are Player 3" in out


def test_viewer_stays_open_after_game_over_until_quit(monkeypatch, capsys, bundle, config):
    """The window must not close the instant the game ends -- only when
    the human explicitly types 'quit' at the post-game prompt."""
    import random

    import splendor_ai.ui.pygame_view as pygame_view_module
    from splendor_ai.cli.human_input import format_action
    from splendor_ai.game.engine import setup_game as real_setup_game
    from splendor_ai.game.rules import legal_actions

    class FakeView:
        instances = []

        def __init__(self, *a, **k):
            self._closed = False
            self.stopped = False
            self.difficulty_label = k.get("difficulty_label")
            FakeView.instances.append(self)

        def start(self):
            pass

        def update(self, state):
            pass

        def stop(self):
            self.stopped = True

        @property
        def closed(self):
            return self._closed

    FakeView.instances.clear()
    monkeypatch.setattr(pygame_view_module, "SplendorView", FakeView)

    holder = {}

    def fake_setup_game(*a, **k):
        state = real_setup_game(*a, **k)
        holder["state"] = state
        return state

    monkeypatch.setattr(play, "setup_game", fake_setup_game)

    rng = random.Random(0)
    game_over_reached = {"value": False}

    def fake_input(prompt=""):
        state = holder["state"]
        if state.game_over:
            game_over_reached["value"] = True
            return "quit"
        actions = legal_actions(state, config)
        return format_action(rng.choice(actions), state)

    monkeypatch.setattr(builtins, "input", fake_input)

    play.run(seed=42, human_seat=0, checkpoint_path=None, render_enabled=True, resolution=(800, 600))

    assert game_over_reached["value"]
    assert len(FakeView.instances) == 1
    assert FakeView.instances[0].stopped is True
    out = capsys.readouterr().out
    assert "Game over. Type 'again' to play a new game (same difficulty), or 'quit' to close the viewer and exit." in out


def test_again_replays_a_new_game_with_a_fresh_board(monkeypatch, capsys, bundle, config):
    """'again' must start a genuinely new game (freshly shuffled board)
    rather than silently ending the process -- and must not keep reusing
    the original --seed, or every replay would be an identical board."""
    import random

    from splendor_ai.cli.human_input import format_action
    from splendor_ai.game.engine import setup_game as real_setup_game
    from splendor_ai.game.rules import legal_actions

    seeds_used = []
    states = []

    def fake_setup_game(bundle_, config_, seed=None, starting_player=0):
        seeds_used.append(seed)
        state = real_setup_game(bundle_, config_, seed=seed, starting_player=starting_player)
        states.append(state)
        return state

    monkeypatch.setattr(play, "setup_game", fake_setup_game)

    rng = random.Random(0)
    game_count = {"n": 0}

    def fake_input(prompt=""):
        state = states[-1]
        if state.game_over:
            game_count["n"] += 1
            return "again" if game_count["n"] == 1 else "quit"
        actions = legal_actions(state, config)
        return format_action(rng.choice(actions), state)

    monkeypatch.setattr(builtins, "input", fake_input)

    play.run(seed=7, human_seat=None, checkpoint_path=None, render_enabled=False, resolution=(800, 600))

    assert len(states) == 2  # played exactly two games, not one
    assert states[0].game_over and states[1].game_over
    assert seeds_used == [7, None]  # first game honors --seed; the replay always reshuffles

    out = capsys.readouterr().out
    assert out.count("GAME OVER") == 2
    assert "Exiting." in out


def test_again_keeps_an_explicit_human_seat_fixed_across_replays(monkeypatch, capsys, bundle, config):
    import random

    from splendor_ai.cli.human_input import format_action
    from splendor_ai.game.engine import setup_game as real_setup_game
    from splendor_ai.game.rules import legal_actions

    states = []

    def fake_setup_game(bundle_, config_, seed=None, starting_player=0):
        state = real_setup_game(bundle_, config_, seed=seed, starting_player=starting_player)
        states.append(state)
        return state

    monkeypatch.setattr(play, "setup_game", fake_setup_game)

    rng = random.Random(1)
    game_count = {"n": 0}

    def fake_input(prompt=""):
        state = states[-1]
        if state.game_over:
            game_count["n"] += 1
            return "again" if game_count["n"] == 1 else "quit"
        actions = legal_actions(state, config)
        return format_action(rng.choice(actions), state)

    monkeypatch.setattr(builtins, "input", fake_input)

    play.run(seed=3, human_seat=2, checkpoint_path=None, render_enabled=False, resolution=(800, 600))

    out = capsys.readouterr().out
    seat_lines = [l for l in out.splitlines() if l.startswith("You are Player")]
    assert len(seat_lines) == 2
    assert all(l.startswith("You are Player 3.") for l in seat_lines)  # explicit --human-seat sticks across replays


def test_post_game_prompt_reprompts_on_unrecognized_input(monkeypatch, capsys, bundle, config):
    import random

    from splendor_ai.cli.human_input import format_action
    from splendor_ai.game.engine import setup_game as real_setup_game
    from splendor_ai.game.rules import legal_actions

    holder = {}

    def fake_setup_game(*a, **k):
        state = real_setup_game(*a, **k)
        holder["state"] = state
        return state

    monkeypatch.setattr(play, "setup_game", fake_setup_game)

    rng = random.Random(0)
    responses = iter(["blah", "quit"])

    def fake_input(prompt=""):
        state = holder["state"]
        if state.game_over:
            return next(responses)
        actions = legal_actions(state, config)
        return format_action(rng.choice(actions), state)

    monkeypatch.setattr(builtins, "input", fake_input)

    play.run(seed=42, human_seat=0, checkpoint_path=None, render_enabled=False, resolution=(800, 600))

    out = capsys.readouterr().out
    assert "Please type 'again' or 'quit'." in out
    assert "Exiting." in out


def test_again_mid_game_abandons_current_game_and_starts_a_new_one(monkeypatch, capsys):
    """'again' must work DURING a game too, not just at game over --
    abandoning the in-progress game immediately rather than requiring the
    human to play it out first."""
    from splendor_ai.game.engine import setup_game as real_setup_game

    seeds_used = []

    def fake_setup_game(bundle_, config_, seed=None, starting_player=0):
        seeds_used.append(seed)
        return real_setup_game(bundle_, config_, seed=seed, starting_player=starting_player)

    monkeypatch.setattr(play, "setup_game", fake_setup_game)
    monkeypatch.setattr(builtins, "input", _scripted_input(["again", "quit"]))

    play.run(seed=5, human_seat=0, checkpoint_path=None, render_enabled=False, resolution=(800, 600))

    out = capsys.readouterr().out
    assert "Abandoning this game and starting a new one..." in out
    assert seeds_used == [5, None]  # first game honors --seed; the restarted game always reshuffles
    assert "Exiting." in out


def test_parse_command_recognizes_again_and_its_aliases(bundle, config):
    from splendor_ai.cli.human_input import parse_command
    from splendor_ai.game.engine import setup_game as real_setup_game

    state = real_setup_game(bundle, config, seed=1)
    for word in ("again", "AGAIN", "restart", "new"):
        assert parse_command(word, state, config).kind == "again"
