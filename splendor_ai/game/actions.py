"""Structured (non-RL) representation of Splendor actions.

These dataclasses are the engine's native action representation, used by
the terminal CLI, tests, and baseline agents. The RL flat discrete
action-index mapping (splendor_ai.env.action_space) is built ON TOP of
these -- it encodes/decodes to/from these same structures so the engine
itself never needs to know about action indices.

A turn may require more than one Action to fully resolve:
  1. exactly one "main action" (PickTokens, BuyCard, or ReserveCard)
  2. optionally a ReturnTokens action, if the main action left the player
     holding more than the token limit
  3. optionally a SelectNoble action, if more than one noble became
     eligible at the end of the turn

The engine's `phase` attribute tracks which of these it is currently
expecting; see splendor_ai.game.engine.TurnPhase.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from splendor_ai.data.schemas import Color


class CardSource(str, Enum):
    VISIBLE = "visible"
    RESERVED = "reserved"


class ReserveSource(str, Enum):
    VISIBLE = "visible"
    BLIND = "blind"


@dataclass(frozen=True)
class PickTokensAction:
    """Take tokens: 3 distinct normal colors (1 each), 2 of one color (bank
    must have had >=4 before the action), or -- only when fewer than 3
    colors remain in the bank -- 1 each of every remaining color."""

    colors: Tuple[Color, ...]

    def kind(self) -> str:
        return "pick_tokens"


@dataclass(frozen=True)
class BuyCardAction:
    source: CardSource
    tier: Optional[int] = None  # required if source == VISIBLE
    slot: Optional[int] = None  # required if source == VISIBLE, 0..3
    reserved_index: Optional[int] = None  # required if source == RESERVED

    def kind(self) -> str:
        return "buy_card"


@dataclass(frozen=True)
class ReserveCardAction:
    source: ReserveSource
    tier: int
    slot: Optional[int] = None  # required if source == VISIBLE, 0..3

    def kind(self) -> str:
        return "reserve_card"


@dataclass(frozen=True)
class ReturnTokensAction:
    """Tokens the player gives back to the bank to get back to the token
    limit. `colors` is a multiset expressed as a tuple (may include GOLD)."""

    colors: Tuple[Color, ...]

    def kind(self) -> str:
        return "return_tokens"


@dataclass(frozen=True)
class SelectNobleAction:
    noble_id: str

    def kind(self) -> str:
        return "select_noble"


@dataclass(frozen=True)
class PassAction:
    """Only ever legal when no Pick/Buy/Reserve action is available -- an
    exceedingly rare state (e.g. the bank has run dry of normal-color
    tokens and the player can afford nothing and has 3 reserved cards
    already). Matches the official rules' handling of a player with no
    available move."""

    def kind(self) -> str:
        return "pass"


Action = "PickTokensAction | BuyCardAction | ReserveCardAction | ReturnTokensAction | SelectNobleAction | PassAction"


def canonicalize_action(action):
    """Normalize the color-tuple ORDER on Pick/Return actions to the
    canonical (sorted-by-value) form used everywhere legality is checked
    or an action is encoded to a Discrete-space index. Every OTHER part of
    an action (which card, which tier/slot, which reserved index, which
    noble) already has a single unambiguous representation, so this only
    ever touches PickTokensAction/ReturnTokensAction.

    This exists because a human can type colors in any order ("P white
    blue green" vs "P green white blue" mean the same thing), but
    equality/dict-lookup on the underlying tuple is order-sensitive.
    Both the engine (legality/apply) and the RL action-space encoder call
    this so a human-parsed action and its Discrete-index form are always
    treated identically."""
    if isinstance(action, PickTokensAction):
        return PickTokensAction(colors=tuple(sorted(action.colors, key=lambda c: c.value)))
    if isinstance(action, ReturnTokensAction):
        return ReturnTokensAction(colors=tuple(sorted(action.colors, key=lambda c: c.value)))
    return action


def action_to_dict(action) -> dict:
    """JSON-serializable representation of an Action, used so the full
    game history (GameState.history) can store the actual action taken,
    not just a free-text description -- e.g. so a UI can re-render it as
    exact terminal-command syntax later, or a game can be replayed."""
    if isinstance(action, PickTokensAction):
        return {"kind": "pick_tokens", "colors": [c.value for c in action.colors]}
    if isinstance(action, BuyCardAction):
        return {"kind": "buy_card", "source": action.source.value, "tier": action.tier, "slot": action.slot, "reserved_index": action.reserved_index}
    if isinstance(action, ReserveCardAction):
        return {"kind": "reserve_card", "source": action.source.value, "tier": action.tier, "slot": action.slot}
    if isinstance(action, ReturnTokensAction):
        return {"kind": "return_tokens", "colors": [c.value for c in action.colors]}
    if isinstance(action, SelectNobleAction):
        return {"kind": "select_noble", "noble_id": action.noble_id}
    if isinstance(action, PassAction):
        return {"kind": "pass"}
    raise ValueError(f"cannot serialize action {action!r}")


def action_from_dict(d: dict):
    kind = d["kind"]
    if kind == "pick_tokens":
        return PickTokensAction(colors=tuple(Color(c) for c in d["colors"]))
    if kind == "buy_card":
        return BuyCardAction(source=CardSource(d["source"]), tier=d.get("tier"), slot=d.get("slot"), reserved_index=d.get("reserved_index"))
    if kind == "reserve_card":
        return ReserveCardAction(source=ReserveSource(d["source"]), tier=d["tier"], slot=d.get("slot"))
    if kind == "return_tokens":
        return ReturnTokensAction(colors=tuple(Color(c) for c in d["colors"]))
    if kind == "select_noble":
        return SelectNobleAction(noble_id=d["noble_id"])
    if kind == "pass":
        return PassAction()
    raise ValueError(f"unknown action kind {kind!r}")
