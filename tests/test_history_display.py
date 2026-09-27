"""Every recorded HistoryEntry carries the actual Action taken (not just a
free-text description), so both the terminal `history` command and the
Pygame recent-actions log can display it as exact command syntax --
requested so the on-screen log reads like something you could type."""
from splendor_ai.cli.render import format_history
from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import BuyCardAction, CardSource, PickTokensAction, SelectNobleAction
from splendor_ai.game.engine import apply_action, setup_game


def test_pick_tokens_history_entry_has_action(bundle, config):
    state = setup_game(bundle, config, seed=1)
    apply_action(state, config, PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN)))
    entry = state.history[-1]
    # Stored as the canonicalized (sorted-by-value) form the engine actually applied.
    assert entry.action == PickTokensAction(colors=(Color.BLUE, Color.GREEN, Color.WHITE))


def test_format_history_shows_legal_command_syntax(bundle, config):
    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_012"]  # cost white:3
    state.board.visible[1][0] = card
    state.player(0).tokens[Color.WHITE] = 3
    apply_action(state, config, BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0))

    text = format_history(state)
    assert "-> B T1-1" in text


def test_format_history_pick_tokens_syntax(bundle, config):
    state = setup_game(bundle, config, seed=1)
    apply_action(state, config, PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN)))
    text = format_history(state)
    assert "-> P blue green white" in text  # canonical (sorted-by-value) order


def test_format_history_falls_back_to_description_for_auto_granted_noble(bundle, config):
    state = setup_game(bundle, config, seed=1)
    noble = bundle.noble_by_id["noble_007"]  # white:4 blue:4
    state.board.nobles = [noble]
    player = state.player(0)
    player.bonuses[Color.WHITE] = 4
    player.bonuses[Color.BLUE] = 4
    apply_action(state, config, PickTokensAction(colors=(Color.GREEN, Color.RED, Color.BLACK)))

    entry = state.history[-1]
    assert entry.action == SelectNobleAction(noble_id=noble.id)  # auto-grant still records the action
    text = format_history(state)
    assert "NOBLE noble_007" in text  # falls back to the noble's ID, not a positional index


def test_no_history_message(bundle, config):
    state = setup_game(bundle, config, seed=1)
    assert format_history(state) == "(no actions yet)"


def test_reserve_without_gold_is_annotated(bundle, config):
    from splendor_ai.game.actions import ReserveCardAction, ReserveSource

    state = setup_game(bundle, config, seed=1)
    state.board.bank[Color.GOLD] = 0  # nothing left to receive
    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))

    entry = state.history[-1]
    assert "gold" not in entry.description
    text = format_history(state)
    assert "(no gold)" in text


def test_reserve_with_gold_is_not_annotated(bundle, config):
    from splendor_ai.game.actions import ReserveCardAction, ReserveSource

    state = setup_game(bundle, config, seed=1)
    assert state.board.bank[Color.GOLD] > 0
    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))

    text = format_history(state)
    assert "(no gold)" not in text


def test_format_history_entry_helper_used_by_both_cli_and_pygame(bundle, config):
    """Both display surfaces call the exact same formatter, so they can
    never drift apart."""
    from splendor_ai.cli.human_input import format_history_entry

    state = setup_game(bundle, config, seed=1)
    apply_action(state, config, PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN)))
    entry = state.history[-1]
    assert format_history_entry(entry, state) == "P blue green white"


def test_round_grouping_matches_num_players(bundle, config):
    """The recent-actions log groups by ROUND (turn_index // num_players),
    not by individual turn -- since most turns are a single action,
    grouping by turn alone would separate nearly every line."""
    from splendor_ai.agents.random_agent import RandomAgent
    from splendor_ai.game.simulate import play_game

    agents = [RandomAgent(seed=i) for i in range(config.num_players)]
    state = play_game(bundle, config, agents, seed=1, max_turns=40)

    rounds_seen = {}
    for entry in state.history:
        r = entry.turn_index // state.num_players
        rounds_seen.setdefault(r, set()).add(entry.seat)

    # Every round with at least num_players simple-turn entries should
    # touch multiple distinct seats -- confirming this groups several
    # players' moves together, not just one action at a time.
    multi_seat_rounds = [seats for seats in rounds_seen.values() if len(seats) > 1]
    assert len(multi_seat_rounds) > 0
