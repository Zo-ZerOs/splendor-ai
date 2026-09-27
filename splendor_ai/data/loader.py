"""Strict, read-only loader for the user-prepared Splendor game data.

This module NEVER writes to cards/, coins/, or nobles/. It only reads
cards.json, coins.json, and nobles.json and resolves image paths relative
to each JSON file's own directory (e.g. "tier1/t1_001.png" referenced from
cards/cards.json resolves to cards/tier1/t1_001.png).

Loading is strict about gameplay-relevant schema problems (missing fields,
wrong types, negative numbers, invalid colors/tiers, duplicate IDs) and
raises DataValidationError listing every problem found. It deliberately
does NOT treat missing image files as fatal -- image availability is a
presentation concern handled by splendor_ai.ui.asset_loader (placeholders)
and reported (not fixed) by splendor_ai.data.validator.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

from splendor_ai.data.schemas import (
    Card,
    Coin,
    Color,
    Noble,
    ValidationIssue,
    validate_card_dict,
    validate_coin_dict,
    validate_noble_dict,
)

# Splendor/  (two levels up from splendor_ai/data/loader.py)
DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class DataValidationError(Exception):
    """Raised when the prepared game data fails strict schema validation."""

    def __init__(self, issues: List[ValidationIssue]):
        self.issues = issues
        message = "Invalid Splendor game data:\n" + "\n".join(f"  - {i}" for i in issues)
        super().__init__(message)


@dataclass(frozen=True)
class DataBundle:
    """All loaded, validated, read-only game data plus lookup indices."""

    cards: List[Card]
    nobles: List[Noble]
    coins: List[Coin]
    cards_by_tier: Dict[int, List[Card]]
    card_by_id: Dict[str, Card]
    noble_by_id: Dict[str, Noble]
    coin_by_color: Dict[Color, Coin]
    root: str


def _read_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_image(json_dir: str, rel_path: str) -> str:
    return os.path.normpath(os.path.join(json_dir, rel_path))


def load_cards(root: str) -> tuple[List[Card], List[ValidationIssue]]:
    cards_dir = os.path.join(root, "cards")
    json_path = os.path.join(cards_dir, "cards.json")
    raw_list = _read_json(json_path)

    issues: List[ValidationIssue] = []
    cards: List[Card] = []
    seen_ids: Dict[str, int] = {}

    if not isinstance(raw_list, list):
        issues.append(ValidationIssue("error", "cards.json", "top-level JSON must be a list"))
        return cards, issues

    for i, raw in enumerate(raw_list):
        entry_issues = validate_card_dict(i, raw)
        issues.extend(entry_issues)
        if any(iss.severity == "error" for iss in entry_issues):
            continue  # can't safely construct a Card from this entry

        card_id = raw["id"]
        if card_id in seen_ids:
            issues.append(ValidationIssue("error", f"card[{i}] (id={card_id})", f"duplicate id, first seen at card[{seen_ids[card_id]}]"))
            continue
        seen_ids[card_id] = i

        cost = {Color(k): v for k, v in raw["cost"].items()}
        image_abs = _resolve_image(cards_dir, raw["image"])
        cards.append(
            Card(
                id=card_id,
                tier=raw["tier"],
                bonus=Color(raw["bonus"]),
                points=raw["points"],
                cost=cost,
                image_rel=raw["image"],
                image_abs=image_abs,
            )
        )

    return cards, issues


def load_nobles(root: str) -> tuple[List[Noble], List[ValidationIssue]]:
    nobles_dir = os.path.join(root, "nobles")
    json_path = os.path.join(nobles_dir, "nobles.json")
    raw_list = _read_json(json_path)

    issues: List[ValidationIssue] = []
    nobles: List[Noble] = []
    seen_ids: Dict[str, int] = {}

    if not isinstance(raw_list, list):
        issues.append(ValidationIssue("error", "nobles.json", "top-level JSON must be a list"))
        return nobles, issues

    for i, raw in enumerate(raw_list):
        entry_issues = validate_noble_dict(i, raw)
        issues.extend(entry_issues)
        if any(iss.severity == "error" for iss in entry_issues):
            continue

        noble_id = raw["id"]
        if noble_id in seen_ids:
            issues.append(ValidationIssue("error", f"noble[{i}] (id={noble_id})", f"duplicate id, first seen at noble[{seen_ids[noble_id]}]"))
            continue
        seen_ids[noble_id] = i

        reqs = {Color(k): v for k, v in raw["requirements"].items()}
        image_abs = _resolve_image(nobles_dir, raw["image"])
        nobles.append(
            Noble(
                id=noble_id,
                points=raw["points"],
                requirements=reqs,
                image_rel=raw["image"],
                image_abs=image_abs,
            )
        )

    return nobles, issues


def load_coins(root: str) -> tuple[List[Coin], List[ValidationIssue]]:
    coins_dir = os.path.join(root, "coins")
    json_path = os.path.join(coins_dir, "coins.json")
    raw_list = _read_json(json_path)

    issues: List[ValidationIssue] = []
    coins: List[Coin] = []
    seen_colors: Dict[str, int] = {}

    if not isinstance(raw_list, list):
        issues.append(ValidationIssue("error", "coins.json", "top-level JSON must be a list"))
        return coins, issues

    for i, raw in enumerate(raw_list):
        entry_issues = validate_coin_dict(i, raw)
        issues.extend(entry_issues)
        if any(iss.severity == "error" for iss in entry_issues):
            continue

        color_name = raw["color"]
        if color_name in seen_colors:
            issues.append(ValidationIssue("error", f"coin[{i}] (color={color_name})", f"duplicate color, first seen at coin[{seen_colors[color_name]}]"))
            continue
        seen_colors[color_name] = i

        image_abs = _resolve_image(coins_dir, raw["image"])
        coins.append(
            Coin(
                color=Color(color_name),
                is_joker=raw["is_joker"],
                image_rel=raw["image"],
                image_abs=image_abs,
            )
        )

    return coins, issues


def load_all(root: str = DEFAULT_ROOT, strict: bool = True) -> DataBundle:
    """Load cards, coins, and nobles from disk.

    If strict=True (default), raises DataValidationError when any fatal
    ("error" severity) schema issue is found across all three files. This
    is the mode used by the game engine at runtime -- it should never run
    on broken gameplay data.
    """
    all_issues: List[ValidationIssue] = []

    cards, cards_issues = load_cards(root)
    nobles, nobles_issues = load_nobles(root)
    coins, coins_issues = load_coins(root)
    all_issues.extend(cards_issues)
    all_issues.extend(nobles_issues)
    all_issues.extend(coins_issues)

    if strict:
        fatal = [i for i in all_issues if i.severity == "error"]
        if fatal:
            raise DataValidationError(fatal)

    cards_by_tier: Dict[int, List[Card]] = {1: [], 2: [], 3: []}
    for c in cards:
        cards_by_tier.setdefault(c.tier, []).append(c)

    bundle = DataBundle(
        cards=cards,
        nobles=nobles,
        coins=coins,
        cards_by_tier=cards_by_tier,
        card_by_id={c.id: c for c in cards},
        noble_by_id={n.id: n for n in nobles},
        coin_by_color={c.color: c for c in coins},
        root=root,
    )
    return bundle
