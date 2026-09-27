from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import PickTokensAction, SelectNobleAction
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.state import TurnPhase


def test_single_eligible_noble_is_auto_granted(bundle, config):
    state = setup_game(bundle, config, seed=1)
    noble = bundle.noble_by_id["noble_007"]  # white:4 blue:4
    state.board.nobles = [noble]
    player = state.player(0)
    player.bonuses[Color.WHITE] = 4
    player.bonuses[Color.BLUE] = 4

    apply_action(state, config, PickTokensAction(colors=(Color.GREEN, Color.RED, Color.BLACK)))

    assert noble in player.nobles
    assert noble not in state.board.nobles
    assert player.prestige == noble.points
    assert state.phase == TurnPhase.MAIN_ACTION
    assert state.current_player == 1


def test_multiple_eligible_nobles_requires_explicit_choice(bundle, config):
    state = setup_game(bundle, config, seed=1)
    noble_a = bundle.noble_by_id["noble_001"]  # white:3 blue:3 black:3
    noble_b = bundle.noble_by_id["noble_005"]  # white:3 blue:3 green:3
    state.board.nobles = [noble_a, noble_b]
    player = state.player(0)
    player.bonuses[Color.WHITE] = 3
    player.bonuses[Color.BLUE] = 3
    player.bonuses[Color.GREEN] = 3
    player.bonuses[Color.BLACK] = 3

    apply_action(state, config, PickTokensAction(colors=(Color.RED, Color.RED)))

    assert state.phase == TurnPhase.NOBLE_CHOICE
    assert state.current_player == 0  # turn not finalized yet
    assert set(state.pending_eligible_nobles) == {noble_a.id, noble_b.id}
    assert player.nobles == []


def test_selecting_a_noble_finalizes_turn_leaving_the_other_on_board(bundle, config):
    state = setup_game(bundle, config, seed=1)
    noble_a = bundle.noble_by_id["noble_001"]
    noble_b = bundle.noble_by_id["noble_005"]
    state.board.nobles = [noble_a, noble_b]
    player = state.player(0)
    player.bonuses[Color.WHITE] = 3
    player.bonuses[Color.BLUE] = 3
    player.bonuses[Color.GREEN] = 3
    player.bonuses[Color.BLACK] = 3

    apply_action(state, config, PickTokensAction(colors=(Color.RED, Color.RED)))
    apply_action(state, config, SelectNobleAction(noble_id=noble_a.id))

    assert player.nobles == [noble_a]
    assert noble_b in state.board.nobles
    assert state.phase == TurnPhase.MAIN_ACTION
    assert state.current_player == 1


def test_only_one_noble_granted_per_turn_even_if_still_eligible(bundle, config):
    state = setup_game(bundle, config, seed=1)
    noble_a = bundle.noble_by_id["noble_001"]
    noble_b = bundle.noble_by_id["noble_005"]
    state.board.nobles = [noble_a, noble_b]
    player = state.player(0)
    player.bonuses[Color.WHITE] = 3
    player.bonuses[Color.BLUE] = 3
    player.bonuses[Color.GREEN] = 3
    player.bonuses[Color.BLACK] = 3

    apply_action(state, config, PickTokensAction(colors=(Color.RED, Color.RED)))
    apply_action(state, config, SelectNobleAction(noble_id=noble_a.id))
    assert player.nobles == [noble_a]  # exactly one this turn

    # Fast-forward back to player 0's next turn; they remain eligible for
    # noble_b, and since only one noble is left it should now auto-grant.
    state.current_player = 0
    state.phase = TurnPhase.MAIN_ACTION
    apply_action(state, config, PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN)))
    assert noble_b in player.nobles
