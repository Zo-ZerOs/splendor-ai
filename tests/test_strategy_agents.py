"""Tests for the hand-crafted comparison strategies: NoNobleAgent,
ReserveRushAgent, BonusEngineAgent, SynergyReserveAgent. Covers (a)
general legal-action compliance across many full random-opponent games
(they must never crash or pick an illegal action), and (b) behavior
specific to what makes each one distinct.
"""
import pytest

from splendor_ai.agents.blocker_agent import BlockerAgent
from splendor_ai.agents.bonus_engine_agent import BonusEngineAgent
from splendor_ai.agents.no_noble_agent import NoNobleAgent
from splendor_ai.agents.random_agent import RandomAgent
from splendor_ai.agents.reserve_rush_agent import ReserveRushAgent
from splendor_ai.agents.synergy_reserve_agent import SynergyReserveAgent
from splendor_ai.data.schemas import Card, Color
from splendor_ai.game.actions import BuyCardAction, ReserveCardAction, ReserveSource
from splendor_ai.game.engine import apply_action, setup_game
from splendor_ai.game.rules import legal_actions
from splendor_ai.game.simulate import play_game

STRATEGY_AGENTS = [NoNobleAgent, ReserveRushAgent, BonusEngineAgent, SynergyReserveAgent]


@pytest.mark.parametrize("agent_cls", STRATEGY_AGENTS)
def test_agent_never_picks_illegal_action_across_many_games(bundle, config, agent_cls):
    for seed in range(8):
        agents = [agent_cls() if seat == 0 else RandomAgent(seed=seed * 10 + seat) for seat in range(config.num_players)]
        state = play_game(bundle, config, agents, seed=seed, max_turns=3000)
        assert state.game_over, f"{agent_cls.__name__} game did not terminate (seed={seed})"


@pytest.mark.parametrize("agent_cls", STRATEGY_AGENTS)
def test_all_four_strategies_together_produce_valid_games(bundle, config, agent_cls):
    """Sanity check the actual 4-distinct-strategy tournament setup used
    for comparison runs terminates cleanly too."""
    agents = [NoNobleAgent(), ReserveRushAgent(), BonusEngineAgent(), RandomAgent(seed=1)]
    state = play_game(bundle, config, agents, seed=3, max_turns=3000)
    assert state.game_over


def test_no_noble_agent_card_value_ignores_noble_progress(bundle, config):
    """The whole point of NoNobleAgent: its card-value function must not
    change based on noble requirements, unlike HeuristicAgent's."""
    from splendor_ai.agents.heuristic_agent import HeuristicAgent

    state = setup_game(bundle, config, seed=1)
    card = bundle.card_by_id["t1_001"]  # bonus=black
    player = state.player(0)

    no_noble = NoNobleAgent()
    heuristic = HeuristicAgent()

    value_before = no_noble._card_value(card, player, state.board)

    # Add a noble that's almost satisfied by this card's bonus color --
    # HeuristicAgent's value should jump for this reason, NoNobleAgent's must not.
    noble = bundle.noble_by_id["noble_001"]  # requires black:3 among others
    state.board.nobles = [noble]
    player.bonuses[Color.BLACK] = 2  # one black bonus away from qualifying

    value_after = no_noble._card_value(card, player, state.board)
    heuristic_value_after = heuristic._card_value(card, player, state.board)

    assert value_before == value_after  # unaffected by noble proximity
    assert heuristic_value_after > heuristic._card_value(card, player, state.board) - 1.0  # sanity: heuristic's function runs
    # HeuristicAgent explicitly adds +1.0 for a noble this bonus is partially toward.
    heuristic_no_noble_board_value = card.points * 3.0 + 0.1 * sum(
        1 for t in state.board.visible for c in state.board.visible[t] if c is not None and c.cost_for(card.bonus) > 0
    )
    assert heuristic_value_after == pytest.approx(heuristic_no_noble_board_value + 1.0)


def test_reserve_rush_agent_reserves_its_target_early(bundle, config):
    """Force a scenario where nothing is affordable, so the agent's first
    real decision must be to reserve its chosen target (the best
    points/cost ratio card currently visible)."""
    state = setup_game(bundle, config, seed=1)
    agent = ReserveRushAgent()

    action = agent.act(state, config)
    assert action.kind() == "reserve_card"
    assert agent.target_card_id is not None

    # The reserved slot must actually correspond to the chosen target.
    from splendor_ai.game.actions import ReserveSource

    assert action.source == ReserveSource.VISIBLE
    reserved_card = state.board.visible[action.tier][action.slot]
    assert reserved_card.id == agent.target_card_id

    # And it must be the best worth = points/cost ratio among ALL visible cards.
    best_worth, best_card = -1.0, None
    for tier in (1, 2, 3):
        for card in state.board.visible[tier]:
            total_cost = sum(card.cost.values())
            worth = card.points / total_cost if total_cost > 0 else 0.0
            if worth > best_worth:
                best_worth, best_card = worth, card
    assert reserved_card.id == best_card.id


def test_reserve_rush_agent_retargets_after_buying(bundle, config):
    state = setup_game(bundle, config, seed=1)
    agent = ReserveRushAgent()
    target_card = bundle.card_by_id["t1_037"]  # cost white:4, points:1 -- decent worth ratio
    state.board.visible[1][0] = target_card
    agent.target_card_id = target_card.id

    player = state.player(0)
    player.tokens[Color.WHITE] = 4  # afford it outright

    from splendor_ai.game.actions import BuyCardAction, CardSource

    action = agent.act(state, config)
    assert isinstance(action, BuyCardAction)
    assert action.source == CardSource.VISIBLE
    apply_action(state, config, action)
    assert agent.target_card_id is None  # cleared after buying -- will re-target next call


def test_reserve_rush_agent_retargets_if_target_disappears(bundle, config):
    state = setup_game(bundle, config, seed=1)
    agent = ReserveRushAgent()
    agent.target_card_id = "does-not-exist-anymore"

    action = agent.act(state, config)  # must not crash -- should pick a fresh target
    assert action is not None
    assert agent.target_card_id is not None
    assert agent.target_card_id != "does-not-exist-anymore"


def test_reserve_rush_agent_buys_something_else_when_target_is_stalled(bundle, config):
    """Regression test: if the target is already reserved (so it can't be
    re-reserved) and currently unaffordable, the agent must still buy
    whatever ELSE is legally affordable rather than passing -- found via a
    heuristic-vs-heuristic stress test (4 non-random seats interacting
    produces token/reservation dynamics single-heuristic-vs-random games
    never exercised), where this previously fell through all the way to
    an illegal PassAction()."""
    state = setup_game(bundle, config, seed=1)
    agent = ReserveRushAgent()

    stalled_target = bundle.card_by_id["t2_021"]  # cost white:5, points:2 -- expensive
    other_card = bundle.card_by_id["t1_012"]  # cost white:3, points:0 -- affordable
    player = state.player(0)
    player.reserved_cards = [stalled_target]  # already reserved -- can't re-reserve
    agent.target_card_id = stalled_target.id
    # Isolate the board to just `other_card` so no incidentally-affordable
    # seeded card can make the assertion flaky.
    for tier in (1, 2, 3):
        state.board.visible[tier] = [None] * len(state.board.visible[tier])
    state.board.visible[1][0] = other_card
    player.tokens[Color.WHITE] = 3  # affords other_card, NOT stalled_target (needs 5)

    from splendor_ai.game.actions import BuyCardAction, CardSource

    action = agent.act(state, config)
    assert isinstance(action, BuyCardAction)
    assert action.source == CardSource.VISIBLE
    assert state.board.visible[action.tier][action.slot].id == other_card.id


def test_bonus_engine_agent_buys_cheap_tier1_during_bootstrap(bundle, config):
    """With zero bonuses (deep in bootstrap), if a tier-1 buy is
    affordable it must be preferred over a more expensive higher-tier buy,
    even if the higher-tier one is worth more points."""
    state = setup_game(bundle, config, seed=1)
    agent = BonusEngineAgent()
    player = state.player(0)

    cheap = bundle.card_by_id["t1_012"]  # cost white:3, points:0
    expensive = bundle.card_by_id["t2_021"]  # cost white:5, points:2
    state.board.visible[1][0] = cheap
    state.board.visible[2][0] = expensive
    player.tokens[Color.WHITE] = 5  # can afford both

    action = agent.act(state, config)
    from splendor_ai.game.actions import BuyCardAction

    assert isinstance(action, BuyCardAction)
    bought = state.board.visible[action.tier][action.slot]
    assert bought.id == cheap.id  # bootstrap mode picks the tier-1 card, not the bigger one


def test_bonus_engine_agent_switches_to_payoff_mode_once_bonuses_built(bundle, config):
    """Once bonus threshold is reached, it should buy the highest-points
    affordable card rather than continuing to prefer tier-1."""
    state = setup_game(bundle, config, seed=1)
    agent = BonusEngineAgent()
    player = state.player(0)
    for c in (Color.WHITE, Color.BLUE, Color.GREEN):
        player.bonuses[c] = 1  # 3 total bonuses -- clears BONUS_THRESHOLD

    cheap = bundle.card_by_id["t1_012"]  # cost white:3, points:0
    expensive = bundle.card_by_id["t2_021"]  # cost white:5, points:2
    state.board.visible[1][0] = cheap
    state.board.visible[2][0] = expensive
    player.tokens[Color.WHITE] = 5

    action = agent.act(state, config)
    from splendor_ai.game.actions import BuyCardAction

    assert isinstance(action, BuyCardAction)
    bought = state.board.visible[action.tier][action.slot]
    assert bought.id == expensive.id  # payoff mode: highest points wins


def test_blocker_agent_reserves_a_card_the_target_can_afford_right_now(bundle, config):
    """Denial should take priority over the blocker's own economy: if the
    target seat can afford a visible card right now, reserve it out from
    under them even if the blocker itself gets nothing from it."""
    state = setup_game(bundle, config, seed=1)
    agent = BlockerAgent(target_seat=1)

    card = bundle.card_by_id["t1_012"]  # cost white:3
    state.board.visible[1][0] = card
    state.player(1).tokens[Color.WHITE] = 3  # target can afford it right now
    # Blocker itself (seat 0) has nothing -- must still choose to reserve, not pass.

    from splendor_ai.game.actions import ReserveCardAction, ReserveSource

    action = agent.act(state, config)
    assert action == ReserveCardAction(source=ReserveSource.VISIBLE, tier=1, slot=0)


def test_blocker_agent_picks_target_needed_tokens_when_reservation_unavailable(bundle, config):
    """If the blocker already has 3 reserved cards (can't reserve the
    threat away), it should instead grab tokens the target needs."""
    state = setup_game(bundle, config, seed=1)
    agent = BlockerAgent(target_seat=1)

    card = bundle.card_by_id["t1_016"]  # cost white:1 blue:2 green:1 red:1
    state.board.visible[1][0] = card
    state.player(1).tokens[Color.WHITE] = 1  # target needs blue:2, green:1, red:1 (shortfall=4, above threshold)
    # Force a tighter threat instead: target already has most of it.
    state.player(1).tokens[Color.BLUE] = 2
    state.player(1).tokens[Color.GREEN] = 1
    # shortfall now just red:1 -- within BLOCK_SHORTFALL_THRESHOLD

    filler = bundle.card_by_id["t1_001"]
    state.player(0).reserved_cards = [filler, filler, filler]  # blocker is maxed out, can't reserve

    action = agent.act(state, config)
    assert action.kind() == "pick_tokens"
    assert Color.RED in action.colors  # the color the target is missing


def test_blocker_agent_plays_for_itself_when_no_threat_detected(bundle, config):
    """With no target threat (everything far from affordable), the
    blocker should still play productively rather than passing uselessly."""
    state = setup_game(bundle, config, seed=1)
    agent = BlockerAgent(target_seat=1)

    card = bundle.card_by_id["t1_037"]  # cost white:4, points:1
    state.board.visible[1][0] = card
    state.player(0).tokens[Color.WHITE] = 4  # the BLOCKER (not the target) can afford it

    from splendor_ai.game.actions import BuyCardAction

    action = agent.act(state, config)
    assert isinstance(action, BuyCardAction)
    bought = state.board.visible[action.tier][action.slot]
    assert bought.id == card.id


def test_blocker_agent_never_picks_illegal_action_across_many_games(bundle, config):
    for seed in range(8):
        agents = [
            BlockerAgent(target_seat=1) if seat == 0 else RandomAgent(seed=seed * 10 + seat)
            for seat in range(config.num_players)
        ]
        state = play_game(bundle, config, agents, seed=seed, max_turns=3000)
        assert state.game_over, f"blocker game did not terminate (seed={seed})"


def _make_target_and_bridge():
    """A tier-3 target right at the strategy's own worth floor (cost/point
    = 2.0, i.e. points/cost = 0.5) whose cost includes white and blue, plus
    a cheap tier-1 bridge card whose bonus is white -- discounting the
    target once bought."""
    target = Card(id="test_target", tier=3, bonus=Color.RED, points=5,
                  cost={Color.WHITE: 3, Color.BLUE: 3, Color.GREEN: 3, Color.BLACK: 1}, image_rel="")
    bridge = Card(id="test_bridge", tier=1, bonus=Color.WHITE, points=0,
                  cost={Color.BLUE: 3}, image_rel="")
    return target, bridge


def _isolated_board(state, target=None, bridge=None):
    """Wipe every visible tier-2/tier-3 slot and all-but-one tier-1 slot so
    the only possible synergy candidate is exactly the one under test --
    real dealt cards can otherwise coincidentally form their own
    (undesired) synergy and make these tests flaky."""
    for tier in (2, 3):
        state.board.visible[tier] = [None] * len(state.board.visible[tier])
    state.board.visible[1] = [None] * len(state.board.visible[1])
    if target is not None:
        state.board.visible[target.tier][0] = target
    if bridge is not None:
        state.board.visible[1][0] = bridge
    return state


def test_synergy_reserve_agent_reserves_a_reachable_target(bundle, config):
    state = setup_game(bundle, config, seed=1)
    target, bridge = _make_target_and_bridge()
    _isolated_board(state, target, bridge)
    agent = SynergyReserveAgent()

    action = agent.act(state, config)

    assert action == ReserveCardAction(source=ReserveSource.VISIBLE, tier=3, slot=0)
    assert agent.target_card_id == target.id
    assert agent.bridge_card_id == bridge.id


def test_synergy_reserve_agent_ignores_a_target_with_no_bridge(bundle, config):
    """Without any tier-1 card on board whose bonus matches one of the
    target's cost colors, this is just an expensive card -- no reason to
    reserve it over normal play."""
    state = setup_game(bundle, config, seed=1)
    target, _bridge = _make_target_and_bridge()
    _isolated_board(state, target, bridge=None)
    agent = SynergyReserveAgent()

    action = agent.act(state, config)

    assert not (isinstance(action, ReserveCardAction) and action.tier == 3 and action.slot == 0)
    assert agent.target_card_id is None


def test_synergy_reserve_agent_buys_the_bridge_card_once_affordable(bundle, config):
    """Simulates being mid-combo on a LATER turn for the same seat (the
    agent instance persists per-seat across turns in real play, just like
    ReserveRushAgent's own tests set target_card_id directly rather than
    replaying apply_action() across a turn boundary, which would hand
    control to the next seat instead)."""
    state = setup_game(bundle, config, seed=1)
    target, bridge = _make_target_and_bridge()
    _isolated_board(state, target, bridge)
    player = state.player(0)
    player.reserved_cards = [target]
    state.board.visible[3][0] = None  # reserving removes it from the visible board

    agent = SynergyReserveAgent()
    agent.target_card_id = target.id
    agent.bridge_card_id = bridge.id

    player.tokens[Color.BLUE] = 3  # exactly affords the bridge card now

    from splendor_ai.game.actions import CardSource

    action = agent.act(state, config)
    assert action == BuyCardAction(source=CardSource.VISIBLE, tier=1, slot=0)


def test_synergy_reserve_agent_picks_a_double_of_a_still_needed_color(bundle, config):
    """Once the target is reserved and the bridge isn't affordable yet,
    the agent should grab 2 of a single still-needed color (bank
    permitting) rather than spreading across colors."""
    state = setup_game(bundle, config, seed=1)
    target, bridge = _make_target_and_bridge()
    _isolated_board(state, target, bridge)
    player = state.player(0)
    player.reserved_cards = [target]
    state.board.visible[3][0] = None

    agent = SynergyReserveAgent()
    agent.target_card_id = target.id
    agent.bridge_card_id = bridge.id

    action = agent.act(state, config)
    from splendor_ai.game.actions import PickTokensAction

    assert isinstance(action, PickTokensAction)
    assert len(action.colors) == 2 and action.colors[0] == action.colors[1]
    assert action.colors[0] in (Color.WHITE, Color.BLUE, Color.GREEN, Color.BLACK)


def test_synergy_reserve_agent_buys_completed_target(bundle, config):
    state = setup_game(bundle, config, seed=1)
    target, bridge = _make_target_and_bridge()
    _isolated_board(state, target, bridge)
    player = state.player(0)
    player.reserved_cards = [target]
    state.board.visible[3][0] = None

    agent = SynergyReserveAgent()
    agent.target_card_id = target.id
    agent.bridge_card_id = bridge.id

    player.bonuses[Color.WHITE] = 3  # as if the bridge (and more) had been bought
    player.tokens[Color.BLUE] = 3
    player.tokens[Color.GREEN] = 3
    player.tokens[Color.BLACK] = 1

    from splendor_ai.game.actions import CardSource

    action = agent.act(state, config)
    assert action == BuyCardAction(source=CardSource.RESERVED, reserved_index=0)
    apply_action(state, config, action)
    assert agent.target_card_id is None
    assert agent.bridge_card_id is None
