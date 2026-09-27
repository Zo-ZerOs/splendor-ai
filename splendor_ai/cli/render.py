"""Plain-text rendering of game state for the terminal. Independent of
Pygame -- this is what the human actually reads to decide their move.
"""
from __future__ import annotations

from splendor_ai.config import GameConfig
from splendor_ai.game.state import GameState, TurnPhase


def _cost_str(cost: dict) -> str:
    parts = [f"{c.value}:{n}" for c, n in cost.items() if n > 0]
    return "{" + ", ".join(parts) + "}" if parts else "{free}"


def format_card_line(card, prefix: str) -> str:
    return f"  {prefix}: {card.id}  tier={card.tier} points={card.points} bonus={card.bonus.value} cost={_cost_str(card.cost)}"


def format_board(state: GameState) -> str:
    lines = ["--- Board ---"]
    lines.append("Nobles:")
    for i, n in enumerate(state.board.nobles):
        lines.append(f"  N{i + 1}: {n.id} points={n.points} requires={_cost_str(n.requirements)}")
    if not state.board.nobles:
        lines.append("  (none remaining)")

    for tier in (1, 2, 3):
        lines.append(f"Tier {tier} (deck: {state.board.deck_remaining(tier)} remaining):")
        for slot, card in enumerate(state.board.visible[tier]):
            if card is None:
                lines.append(f"  T{tier}-{slot + 1}: (empty)")
            else:
                lines.append(format_card_line(card, f"T{tier}-{slot + 1}"))

    bank = ", ".join(f"{c.value}:{n}" for c, n in state.board.bank.items())
    lines.append(f"Bank: {bank}")
    return "\n".join(lines)


def format_player(state: GameState, seat: int, reveal_reserved: bool) -> str:
    p = state.player(seat)
    marker = " <-- current" if state.current_player == seat and not state.game_over else ""
    lines = [f"Player {seat + 1}{marker}  (prestige={p.prestige}, cards={p.purchased_card_count})"]
    tokens = ", ".join(f"{c.value}:{n}" for c, n in p.tokens.items() if n > 0) or "none"
    lines.append(f"  tokens: {tokens}")
    bonuses = ", ".join(f"{c.value}:{n}" for c, n in p.bonuses.items() if n > 0) or "none"
    lines.append(f"  bonuses: {bonuses}")
    nobles = ", ".join(n.id for n in p.nobles) or "none"
    lines.append(f"  nobles: {nobles}")
    if reveal_reserved:
        if p.reserved_cards:
            for i, card in enumerate(p.reserved_cards):
                lines.append(format_card_line(card, f"  R{i + 1}"))
        else:
            lines.append("  reserved: none")
    else:
        lines.append(f"  reserved: {len(p.reserved_cards)} card(s) (hidden)")
    return "\n".join(lines)


def format_state_summary(state: GameState, config: GameConfig, human_seat: int) -> str:
    lines = [f"=== Round {state.round_number}{' (FINAL ROUND)' if state.final_round_started else ''} ==="]
    lines.append(format_board(state))
    lines.append("--- Players ---")
    for seat in range(state.num_players):
        lines.append(format_player(state, seat, reveal_reserved=(seat == human_seat)))
    return "\n".join(lines)


def format_history(state: GameState, n: int = 15) -> str:
    if not state.history:
        return "(no actions yet)"
    from splendor_ai.cli.human_input import format_history_entry

    lines = []
    for entry in state.history[-n:]:
        command = format_history_entry(entry, state)
        lines.append(f"turn {entry.turn_index}: Player {entry.seat + 1} -> {command}")
    return "\n".join(lines)


def format_winner(state: GameState) -> str:
    if not state.game_over:
        return "(game not over)"
    ranking = _ranking_lines(state)
    if len(state.winners) == 1:
        header = f"GAME OVER -- Winner: Player {state.winners[0] + 1}!"
    else:
        names = ", ".join(f"Player {s + 1}" for s in state.winners)
        header = f"GAME OVER -- Tie between: {names}"
    return header + "\n" + "\n".join(ranking)


def _ranking_lines(state: GameState) -> list:
    from splendor_ai.game.rules import compute_ranking

    lines = []
    for rank, group in enumerate(compute_ranking(state), start=1):
        names = ", ".join(f"Player {s + 1} (prestige={state.player(s).prestige}, cards={state.player(s).purchased_card_count})" for s in group)
        lines.append(f"  #{rank}: {names}")
    return lines
