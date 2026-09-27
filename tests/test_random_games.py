"""Headless random-play smoke tests. These don't test individual rules
(see the other test_*.py files) -- they play many full random games and
assert core invariants never break, to catch unreachable/invalid states
that unit tests on isolated rules might miss.
"""
import random

import pytest

from splendor_ai.data.schemas import Color, NORMAL_COLORS
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.rules import legal_actions
from splendor_ai.game.state import TurnPhase

MAX_TURNS = 1000


def _assert_invariants(state, bundle, config):
    # Token conservation: bank + all players' tokens == total minted per color.
    for color in NORMAL_COLORS:
        total = state.board.bank[color] + sum(p.tokens[color] for p in state.players)
        assert total == config.tokens_per_normal_color, f"{color} token conservation broken"
    gold_total = state.board.bank[Color.GOLD] + sum(p.tokens[Color.GOLD] for p in state.players)
    assert gold_total == config.gold_tokens, "gold token conservation broken"

    # No player ever holds a negative token count.
    for p in state.players:
        for color, n in p.tokens.items():
            assert n >= 0, f"negative token count for player {p.seat} color {color}"
    for color, n in state.board.bank.items():
        assert n >= 0, f"negative bank count for {color}"

    # Card conservation per tier: deck + visible + reserved + purchased == original tier size.
    for tier in config.tiers:
        original = len(bundle.cards_by_tier[tier])
        deck_n = state.board.deck_remaining(tier)
        visible_n = sum(1 for c in state.board.visible[tier] if c is not None)
        reserved_n = sum(1 for p in state.players for c in p.reserved_cards if c.tier == tier)
        purchased_n = sum(1 for p in state.players for c in p.purchased_cards if c.tier == tier)
        assert deck_n + visible_n + reserved_n + purchased_n == original, f"tier {tier} card conservation broken"

    # Reserved-card limit and token limit are never violated at rest (MAIN_ACTION boundary).
    if state.phase == TurnPhase.MAIN_ACTION:
        for p in state.players:
            assert len(p.reserved_cards) <= config.max_reserved_cards
            assert p.total_tokens <= config.max_tokens_held

    # Nobles: no duplicate, none both on board and owned.
    board_noble_ids = {n.id for n in state.board.nobles}
    owned_noble_ids = [n.id for p in state.players for n in p.nobles]
    assert len(owned_noble_ids) == len(set(owned_noble_ids)), "a noble was granted more than once"
    assert board_noble_ids.isdisjoint(owned_noble_ids), "a noble is both on the board and owned"


@pytest.mark.parametrize("seed", range(15))
def test_random_full_game_maintains_invariants(bundle, config, seed):
    rng = random.Random(seed)
    state = setup_game(bundle, config, seed=seed)

    turns = 0
    while not state.game_over and turns < MAX_TURNS:
        actions = legal_actions(state, config)
        assert actions, f"no legal actions available in phase {state.phase} (seed={seed}, turn={turns})"
        action = rng.choice(actions)
        apply_action(state, config, action)
        _assert_invariants(state, bundle, config)
        turns += 1

    assert state.game_over, f"game did not terminate within {MAX_TURNS} turns (seed={seed})"
    assert state.winners is not None
    assert len(state.winners) >= 1
    # Every player played the same number of turns in the final round.
    assert state.turns_completed % state.num_players == 0
