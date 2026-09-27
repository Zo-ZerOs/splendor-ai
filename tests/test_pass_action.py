from splendor_ai.data.schemas import Color, NORMAL_COLORS
from splendor_ai.game.actions import PassAction
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.rules import legal_actions
from splendor_ai.game.state import TurnPhase


def test_pass_is_the_only_legal_action_when_bank_is_dry_and_nothing_affordable(bundle, config):
    state = setup_game(bundle, config, seed=1)
    for c in NORMAL_COLORS:
        state.board.bank[c] = 0
    card = bundle.card_by_id["t1_001"]
    state.player(0).reserved_cards = [card, card, card]

    actions = legal_actions(state, config)
    assert actions == [PassAction()]


def test_pass_is_not_offered_when_another_action_exists(bundle, config):
    state = setup_game(bundle, config, seed=1)
    actions = legal_actions(state, config)
    assert PassAction() not in actions


def test_applying_pass_advances_turn_without_changing_board(bundle, config):
    state = setup_game(bundle, config, seed=1)
    for c in NORMAL_COLORS:
        state.board.bank[c] = 0
    card = bundle.card_by_id["t1_001"]
    state.player(0).reserved_cards = [card, card, card]

    before_bank = dict(state.board.bank)
    apply_action(state, config, PassAction())

    assert state.board.bank == before_bank
    assert state.current_player == 1
    assert state.phase == TurnPhase.MAIN_ACTION


def test_full_round_of_passes_ends_the_game(bundle, config):
    """A structural deadlock (bank dry of normal colors, everyone maxed
    on reserved cards, nobody can afford anything) can never resolve
    itself -- once every player has passed in a row, the game must end
    rather than loop forever."""
    state = setup_game(bundle, config, seed=1)
    for c in NORMAL_COLORS:
        state.board.bank[c] = 0
    card = bundle.card_by_id["t1_001"]
    for p in state.players:
        p.reserved_cards = [card, card, card]

    for i in range(config.num_players - 1):
        assert not state.game_over
        apply_action(state, config, PassAction())
    apply_action(state, config, PassAction())  # the num_players-th consecutive pass

    assert state.game_over
    assert state.phase == TurnPhase.GAME_OVER
    assert state.winners is not None


def test_a_single_buy_action_resets_the_pass_counter(bundle, config):
    from splendor_ai.data.schemas import Color
    from splendor_ai.game.actions import BuyCardAction, CardSource

    state = setup_game(bundle, config, seed=1)
    for c in NORMAL_COLORS:
        state.board.bank[c] = 0
    card = bundle.card_by_id["t1_001"]  # cost: green:1, red:3, black:1
    for p in state.players:
        p.reserved_cards = [card, card, card]

    apply_action(state, config, PassAction())
    apply_action(state, config, PassAction())
    assert state.consecutive_passes == 2

    # Give the about-to-act player exactly enough tokens to buy their
    # reserved card instead of passing.
    mover = state.player(state.current_player)
    mover.tokens[Color.GREEN] = 1
    mover.tokens[Color.RED] = 3
    mover.tokens[Color.BLACK] = 1

    apply_action(state, config, BuyCardAction(source=CardSource.RESERVED, reserved_index=0))
    assert state.consecutive_passes == 0
    assert not state.game_over
