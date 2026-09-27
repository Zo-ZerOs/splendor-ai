"""Terminal command parsing and the reverse mapping (Action -> command
string), used both to interpret human input and to print the list of
currently legal commands.

Syntax (case-insensitive):
    P white blue green      take 3 different-colored tokens
    P red red                take 2 tokens of the same color
    B T1-1                   buy the card in visible tier-1 slot 1
    B R2                     buy your 2nd reserved card
    R T2-3                   reserve the card in visible tier-2 slot 3
    R D3                     reserve the face-down top card of tier 3
    RETURN white white       (only during a token-return decision)
    NOBLE 2                  (only during a noble-choice decision, by
                              1-based index into the offered list)
    help / state / legal / history / again / quit
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from splendor_ai.config import GameConfig
from splendor_ai.data.schemas import Color
from splendor_ai.game.actions import (
    BuyCardAction,
    CardSource,
    PassAction,
    PickTokensAction,
    ReserveCardAction,
    ReserveSource,
    ReturnTokensAction,
    SelectNobleAction,
)
from splendor_ai.game.state import GameState

META_COMMANDS = {"HELP", "STATE", "LEGAL", "HISTORY", "QUIT", "EXIT", "AGAIN", "RESTART", "NEW"}

HELP_TEXT = """\
Commands:
  P <color> <color> <color>   take 3 different-colored tokens (white/blue/green/red/black)
  P <color> <color>           take 2 tokens of the same color (bank must have had >=4)
  P <color> [<color>]         take 1 of each remaining color (only legal when the bank
                               has fewer than 3 normal colors left)
  B T<tier>-<slot>             buy the visible card at that tier/slot, e.g. B T1-1
  B R<n>                       buy your n-th reserved card, e.g. B R2
  R T<tier>-<slot>             reserve the visible card at that tier/slot, e.g. R T2-3
  R D<tier>                    reserve the face-down top card of that tier, e.g. R D3
  RETURN <color> [<color>...]  return tokens down to the 10-token limit (only when prompted)
  NOBLE <n>  (or N <n>)         choose noble #n from the offered list (only when prompted)
  PASS                         pass (only ever offered when no other action is legal)
  help                         show this help
  state                        show the full board/player summary again
  legal                        list every currently legal command
  history                      show recent actions
  again                         abandon this game and start a new one (same difficulty)
  quit                         exit the program
"""

_COLOR_NAMES = {c.value: c for c in Color if c != Color.GOLD}


@dataclass
class ParsedCommand:
    kind: str  # "action" | "help" | "state" | "legal" | "history" | "again" | "quit" | "error"
    action: Optional[object] = None
    error: Optional[str] = None


def _parse_colors(tokens: List[str]) -> Optional[List[Color]]:
    colors = []
    for t in tokens:
        c = _COLOR_NAMES.get(t.lower())
        if c is None:
            return None
        colors.append(c)
    return colors


def parse_command(raw: str, state: GameState, config: GameConfig) -> ParsedCommand:
    raw = raw.strip()
    if not raw:
        return ParsedCommand(kind="error", error="Empty command. Type 'help' for a list of commands.")

    parts = raw.split()
    cmd = parts[0].upper()

    if cmd in META_COMMANDS:
        if cmd == "EXIT":
            return ParsedCommand(kind="quit")
        if cmd in ("RESTART", "NEW"):
            return ParsedCommand(kind="again")
        return ParsedCommand(kind=cmd.lower())

    if cmd == "PASS":
        return ParsedCommand(kind="action", action=PassAction())

    if cmd == "P":
        color_tokens = parts[1:]
        if not (1 <= len(color_tokens) <= 3):
            return ParsedCommand(
                kind="error",
                error=(
                    "INVALID COMMAND: 'P' takes 1 to 3 colors.\n"
                    "Examples:\n  P white blue green\n  P red red\n"
                    "  P white   (only legal when very few colors remain in the bank)"
                ),
            )
        colors = _parse_colors(color_tokens)
        if colors is None:
            return ParsedCommand(
                kind="error",
                error='INVALID COMMAND: unknown color. Use white, blue, green, red, or black (gold cannot be picked with "P").',
            )
        if len(colors) == 3 and len(set(colors)) != 3:
            return ParsedCommand(kind="error", error="INVALID COMMAND: taking 3 tokens requires 3 DIFFERENT colors, e.g. 'P white blue green'.")
        # A 2-different-color or 1-color pick is NOT always legal -- it only
        # is when the bank has fewer than 3 normal colors left. That's a
        # game-state question, not a syntax question, so we build the
        # action here and let the shared engine decide (see explain.py for
        # the friendly reason if apply_action rejects it).
        return ParsedCommand(kind="action", action=PickTokensAction(colors=tuple(colors)))

    if cmd == "B":
        if len(parts) != 2:
            return ParsedCommand(kind="error", error="INVALID COMMAND: 'B' needs exactly one position.\nExamples:\n  B T1-1\n  B T2-3\n  B R1")
        spec = parts[1].upper()
        m = re.fullmatch(r"T([123])-([1-4])", spec)
        if m:
            tier, slot = int(m.group(1)), int(m.group(2)) - 1
            return ParsedCommand(kind="action", action=BuyCardAction(source=CardSource.VISIBLE, tier=tier, slot=slot))
        m = re.fullmatch(r"R([1-3])", spec)
        if m:
            idx = int(m.group(1)) - 1
            return ParsedCommand(kind="action", action=BuyCardAction(source=CardSource.RESERVED, reserved_index=idx))
        return ParsedCommand(
            kind="error",
            error=f'INVALID COMMAND: "{parts[1]}" is not a valid visible-card or reserved-card position.\nExamples:\n  B T1-1\n  B T2-3\n  B R1',
        )

    if cmd == "R":
        if len(parts) != 2:
            return ParsedCommand(kind="error", error="INVALID COMMAND: 'R' needs exactly one position.\nExamples:\n  R T2-3\n  R D3")
        spec = parts[1].upper()
        m = re.fullmatch(r"T([123])-([1-4])", spec)
        if m:
            tier, slot = int(m.group(1)), int(m.group(2)) - 1
            return ParsedCommand(kind="action", action=ReserveCardAction(source=ReserveSource.VISIBLE, tier=tier, slot=slot))
        m = re.fullmatch(r"D([123])", spec)
        if m:
            tier = int(m.group(1))
            return ParsedCommand(kind="action", action=ReserveCardAction(source=ReserveSource.BLIND, tier=tier))
        return ParsedCommand(
            kind="error",
            error=f'INVALID COMMAND: "{parts[1]}" is not a valid visible-card or face-down-tier position.\nExamples:\n  R T2-3\n  R D3',
        )

    if cmd == "RETURN":
        colors = _parse_colors_incl_gold(parts[1:])
        if colors is None:
            return ParsedCommand(kind="error", error="INVALID COMMAND: unknown color in RETURN command.")
        if not colors:
            return ParsedCommand(
                kind="error",
                error="INVALID COMMAND: 'RETURN' needs at least one color.\nExamples:\n  RETURN blue\n  RETURN red red\n  RETURN white black gold",
            )
        return ParsedCommand(kind="action", action=ReturnTokensAction(colors=tuple(colors)))

    if cmd in ("NOBLE", "N"):
        if len(parts) != 2 or not parts[1].isdigit():
            return ParsedCommand(kind="error", error="INVALID COMMAND: usage is NOBLE <n> (n = index shown in the noble-choice prompt), e.g. 'NOBLE 1'.")
        idx = int(parts[1]) - 1
        if idx < 0 or idx >= len(state.pending_eligible_nobles):
            return ParsedCommand(
                kind="error",
                error=f"INVALID COMMAND: noble index out of range -- choose between 1 and {len(state.pending_eligible_nobles)}.",
            )
        return ParsedCommand(kind="action", action=SelectNobleAction(noble_id=state.pending_eligible_nobles[idx]))

    return ParsedCommand(kind="error", error=f"INVALID COMMAND: unknown command '{parts[0]}'. Type 'help' for a list of commands.")


def _parse_colors_incl_gold(tokens: List[str]) -> Optional[List[Color]]:
    names = {c.value: c for c in Color}
    colors = []
    for t in tokens:
        c = names.get(t.lower())
        if c is None:
            return None
        colors.append(c)
    return colors


def format_action(action, state: Optional[GameState] = None) -> str:
    """Render an Action as the exact terminal command that produces it.
    `state` is only needed to resolve a SelectNobleAction's POSITIONAL
    index (which noble slot in the currently-pending list) -- when
    formatting a HISTORICAL action, that pending list no longer reflects
    the moment the choice was made, so this falls back to the noble's
    actual ID instead of a (now meaningless) index."""
    if isinstance(action, PickTokensAction):
        return "P " + " ".join(c.value for c in action.colors)
    if isinstance(action, BuyCardAction):
        if action.source == CardSource.VISIBLE:
            return f"B T{action.tier}-{action.slot + 1}"
        return f"B R{action.reserved_index + 1}"
    if isinstance(action, ReserveCardAction):
        if action.source == ReserveSource.VISIBLE:
            return f"R T{action.tier}-{action.slot + 1}"
        return f"R D{action.tier}"
    if isinstance(action, ReturnTokensAction):
        return "RETURN " + " ".join(c.value for c in action.colors)
    if isinstance(action, SelectNobleAction):
        if state is not None and action.noble_id in state.pending_eligible_nobles:
            idx = state.pending_eligible_nobles.index(action.noble_id) + 1
            return f"NOBLE {idx}"
        return f"NOBLE {action.noble_id}"
    if isinstance(action, PassAction):
        return "PASS"
    return str(action)


def format_history_entry(entry, state: Optional[GameState] = None) -> str:
    """Command-syntax rendering of one HistoryEntry for display in the
    terminal `history` command and the Pygame recent-actions log -- both
    call this so they never drift apart. Falls back to the entry's
    free-text description when no structured action was recorded (e.g. an
    automatic single-noble grant). A reserve that didn't earn a gold token
    (the bank had none left) is annotated "(no gold)", since the plain
    command syntax alone ("R T2-3") can't tell you that."""
    if entry.action is None:
        return entry.description
    command = format_action(entry.action, state)
    if entry.action_kind == "reserve_card" and "gold" not in entry.description:
        command += " (no gold)"
    return command
