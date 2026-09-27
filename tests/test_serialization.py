from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import PickTokensAction, ReserveCardAction, ReserveSource
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.state import GameState


def test_round_trip_preserves_state(bundle, config):
    state = setup_game(bundle, config, seed=7)
    apply_action(state, config, PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN)))

    data = state.to_dict()
    restored = GameState.from_dict(data, bundle)

    assert restored.to_dict() == data


def test_restored_state_continues_identically(bundle, config):
    state = setup_game(bundle, config, seed=7)
    apply_action(state, config, ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0))

    data = state.to_dict()
    restored = GameState.from_dict(data, bundle)

    action = PickTokensAction(colors=(Color.RED, Color.BLACK, Color.WHITE))
    apply_action(state, config, action)
    apply_action(restored, config, action)

    assert state.to_dict() == restored.to_dict()


def test_clone_is_independent(bundle, config):
    state = setup_game(bundle, config, seed=7)
    clone = state.clone()
    apply_action(state, config, PickTokensAction(colors=(Color.WHITE, Color.BLUE, Color.GREEN)))

    assert state.player(0).total_tokens == 3
    assert clone.player(0).total_tokens == 0
