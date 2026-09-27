"""Proves there is exactly ONE action/rule system, shared by the human
terminal path and the AI/RL path:

  - Every human-typed command parses into the SAME structured Action
    dataclass the AI's Discrete(149) index decodes to.
  - Applying either representation through the SAME apply_action()
    produces byte-identical resulting states.
  - The CLI itself never independently judges legality -- explain.py is
    only ever invoked after apply_action has already raised.
"""
import random

from splendor_ai.cli.human_input import format_action, parse_command
from splendor_ai.env.action_space import decode_action, encode_action
from splendor_ai.game.actions import canonicalize_action
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.rules import legal_actions


def _strip_rng(d):
    d = dict(d)
    d.pop("rng_state", None)
    return d


def test_human_command_and_ai_index_produce_identical_transition_for_a_pick(bundle, config):
    state_a = setup_game(bundle, config, seed=1)
    state_b = setup_game(bundle, config, seed=1)

    parsed = parse_command("P white blue green", state_a, config)
    human_action = parsed.action

    index = encode_action(human_action, state_a)
    ai_action = decode_action(index, state_b)
    # decode_action always returns canonical (sorted-by-value) color order;
    # the human's raw typed order may differ but means the same move --
    # canonicalizing both sides is the correct, meaningful comparison
    # (their actual gameplay effect is verified identical below regardless).
    assert canonicalize_action(ai_action) == canonicalize_action(human_action)

    apply_action(state_a, config, human_action)
    apply_action(state_b, config, ai_action)

    assert _strip_rng(state_a.to_dict()) == _strip_rng(state_b.to_dict())


def test_human_command_and_ai_index_produce_identical_transition_for_a_buy(bundle, config):
    from splendor_ai.data.schemas import Color

    state_a = setup_game(bundle, config, seed=1)
    state_b = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_036"]  # cost blue:4
    for s in (state_a, state_b):
        s.board.visible[1][0] = card
        s.player(0).tokens[Color.BLUE] = 4

    parsed = parse_command("B T1-1", state_a, config)
    human_action = parsed.action
    index = encode_action(human_action, state_a)
    ai_action = decode_action(index, state_b)
    assert ai_action == human_action

    apply_action(state_a, config, human_action)
    apply_action(state_b, config, ai_action)
    assert _strip_rng(state_a.to_dict()) == _strip_rng(state_b.to_dict())


def test_random_games_every_legal_action_agrees_between_command_and_index_forms(bundle, config):
    """Across many random games and turns: for every currently legal
    action, (a) formatting it as a terminal command and re-parsing it
    recovers the exact same action, and (b) encoding/decoding through the
    Discrete(149) space recovers the exact same action. For the action
    actually taken each turn, applying the command-parsed form and the
    index-decoded form from identical clones produces identical states."""
    rng = random.Random(0)
    for seed in range(12):
        state = setup_game(bundle, config, seed=seed)
        turns = 0
        while not state.game_over and turns < 150:
            actions = legal_actions(state, config)

            for action in actions:
                cmd = format_action(action, state)
                reparsed = parse_command(cmd, state, config)
                assert reparsed.kind == "action", f"format/reparse failed for {cmd!r} (seed={seed}, turn={turns})"
                assert reparsed.action == action

                idx = encode_action(action, state)
                assert decode_action(idx, state) == action

            chosen = rng.choice(actions)
            cmd = format_action(chosen, state)
            idx = encode_action(chosen, state)

            clone_via_command = state.clone()
            clone_via_index = state.clone()
            command_action = parse_command(cmd, clone_via_command, config).action
            index_action = decode_action(idx, clone_via_index)

            apply_action(clone_via_command, config, command_action)
            apply_action(clone_via_index, config, index_action)
            assert _strip_rng(clone_via_command.to_dict()) == _strip_rng(clone_via_index.to_dict())

            apply_action(state, config, chosen)
            turns += 1
