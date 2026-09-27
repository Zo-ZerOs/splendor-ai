"""Game state containers: Board, GameState, TurnPhase, and (de)serialization.

GameState never stores gameplay-data (Card/Noble) definitions itself --
it only ever references the shared, read-only DataBundle loaded once at
startup. Serialization therefore stores IDs, and `GameState.from_dict`
needs that same DataBundle to reconstruct object references.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from splendor_ai.data.loader import DataBundle
from splendor_ai.data.schemas import Card, Color, Noble
from splendor_ai.game.actions import action_from_dict, action_to_dict
from splendor_ai.game.player import PlayerState


class TurnPhase(str, Enum):
    MAIN_ACTION = "main_action"
    TOKEN_RETURN = "token_return"
    NOBLE_CHOICE = "noble_choice"
    GAME_OVER = "game_over"


@dataclass
class Board:
    bank: Dict[Color, int]
    decks: Dict[int, List[Card]]  # face-down remaining cards; end of list = top of deck
    visible: Dict[int, List[Optional[Card]]]
    nobles: List[Noble]

    def deck_remaining(self, tier: int) -> int:
        return len(self.decks[tier])

    def clone(self) -> "Board":
        return Board(
            bank=dict(self.bank),
            decks={t: list(cards) for t, cards in self.decks.items()},
            visible={t: list(slots) for t, slots in self.visible.items()},
            nobles=list(self.nobles),
        )


@dataclass
class HistoryEntry:
    turn_index: int
    seat: int
    action_kind: str
    description: str
    action: object = None  # the actual Action taken, if any (None for automatic events like an auto-granted single noble)
    # Colored-tokens-first payment breakdown (see PlayerState.payment_for),
    # recorded only for action_kind == "buy_card" -- the board slot/player's
    # own token counts have already moved on by the time this is displayed
    # later, so the payment has to be captured at the moment of the buy.
    payment: Optional[Dict[Color, int]] = None


@dataclass
class GameState:
    board: Board
    players: List[PlayerState]
    current_player: int
    phase: TurnPhase
    starting_player: int
    turns_completed: int = 0
    final_round_end_at: Optional[int] = None
    game_over: bool = False
    winners: Optional[List[int]] = None
    pending_eligible_nobles: List[str] = field(default_factory=list)
    history: List[HistoryEntry] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)
    consecutive_passes: int = 0

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def round_number(self) -> int:
        """1-indexed round currently in progress."""
        return self.turns_completed // self.num_players + 1

    @property
    def final_round_started(self) -> bool:
        return self.final_round_end_at is not None

    def player(self, seat: int) -> PlayerState:
        return self.players[seat]

    def clone(self) -> "GameState":
        new_rng = random.Random()
        new_rng.setstate(self.rng.getstate())
        return GameState(
            board=self.board.clone(),
            players=[p.clone() for p in self.players],
            current_player=self.current_player,
            phase=self.phase,
            starting_player=self.starting_player,
            turns_completed=self.turns_completed,
            final_round_end_at=self.final_round_end_at,
            game_over=self.game_over,
            winners=list(self.winners) if self.winners is not None else None,
            pending_eligible_nobles=list(self.pending_eligible_nobles),
            history=list(self.history),
            rng=new_rng,
            consecutive_passes=self.consecutive_passes,
        )

    # ---------------- serialization ----------------

    def to_dict(self) -> dict:
        return {
            "board": {
                "bank": {c.value: n for c, n in self.board.bank.items()},
                "decks": {str(t): [c.id for c in cards] for t, cards in self.board.decks.items()},
                "visible": {str(t): [c.id if c else None for c in slots] for t, slots in self.board.visible.items()},
                "nobles": [n.id for n in self.board.nobles],
            },
            "players": [
                {
                    "seat": p.seat,
                    "name": p.name,
                    "tokens": {c.value: n for c, n in p.tokens.items()},
                    "bonuses": {c.value: n for c, n in p.bonuses.items()},
                    "purchased_cards": [c.id for c in p.purchased_cards],
                    "reserved_cards": [c.id for c in p.reserved_cards],
                    "reserved_card_sources": list(p.reserved_card_sources),
                    "nobles": [n.id for n in p.nobles],
                }
                for p in self.players
            ],
            "current_player": self.current_player,
            "phase": self.phase.value,
            "starting_player": self.starting_player,
            "turns_completed": self.turns_completed,
            "final_round_end_at": self.final_round_end_at,
            "game_over": self.game_over,
            "winners": self.winners,
            "pending_eligible_nobles": self.pending_eligible_nobles,
            "history": [
                {
                    "turn_index": h.turn_index,
                    "seat": h.seat,
                    "action_kind": h.action_kind,
                    "description": h.description,
                    "action": action_to_dict(h.action) if h.action is not None else None,
                    "payment": {c.value: n for c, n in h.payment.items()} if h.payment is not None else None,
                }
                for h in self.history
            ],
            "rng_state": _rng_state_to_jsonable(self.rng.getstate()),
            "consecutive_passes": self.consecutive_passes,
        }

    @staticmethod
    def from_dict(data: dict, bundle: DataBundle) -> "GameState":
        board = Board(
            bank={Color(c): n for c, n in data["board"]["bank"].items()},
            decks={int(t): [bundle.card_by_id[cid] for cid in ids] for t, ids in data["board"]["decks"].items()},
            visible={int(t): [bundle.card_by_id[cid] if cid else None for cid in slots] for t, slots in data["board"]["visible"].items()},
            nobles=[bundle.noble_by_id[nid] for nid in data["board"]["nobles"]],
        )
        players = []
        for pd in data["players"]:
            p = PlayerState(
                seat=pd["seat"],
                name=pd["name"],
                tokens={Color(c): n for c, n in pd["tokens"].items()},
                bonuses={Color(c): n for c, n in pd["bonuses"].items()},
                purchased_cards=[bundle.card_by_id[cid] for cid in pd["purchased_cards"]],
                reserved_cards=[bundle.card_by_id[cid] for cid in pd["reserved_cards"]],
                reserved_card_sources=list(pd.get("reserved_card_sources", [])),
                nobles=[bundle.noble_by_id[nid] for nid in pd["nobles"]],
            )
            players.append(p)
        rng = random.Random()
        rng.setstate(_rng_state_from_jsonable(data["rng_state"]))
        return GameState(
            board=board,
            players=players,
            current_player=data["current_player"],
            phase=TurnPhase(data["phase"]),
            starting_player=data["starting_player"],
            turns_completed=data["turns_completed"],
            final_round_end_at=data["final_round_end_at"],
            game_over=data["game_over"],
            winners=data["winners"],
            pending_eligible_nobles=list(data["pending_eligible_nobles"]),
            history=[
                HistoryEntry(
                    turn_index=h["turn_index"],
                    seat=h["seat"],
                    action_kind=h["action_kind"],
                    description=h["description"],
                    action=action_from_dict(h["action"]) if h.get("action") else None,
                    payment={Color(c): n for c, n in h["payment"].items()} if h.get("payment") else None,
                )
                for h in data["history"]
            ],
            rng=rng,
            consecutive_passes=data.get("consecutive_passes", 0),
        )


def _rng_state_to_jsonable(state):
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _rng_state_from_jsonable(state):
    version, internal, gauss = state
    return (version, tuple(internal), gauss)
