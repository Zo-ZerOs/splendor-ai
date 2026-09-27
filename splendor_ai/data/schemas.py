"""Schema definitions and pure validation helpers for Splendor's prepared
game data (cards.json, coins.json, nobles.json).

Validation here only ever reads dicts parsed from JSON; nothing in this
module writes to the user's data files. Functions return lists of
human-readable issue strings rather than raising, so callers (the strict
loader and the reporting validator) can decide what to do with them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Color(str, Enum):
    WHITE = "white"
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    BLACK = "black"
    GOLD = "gold"

    def __str__(self) -> str:  # nicer printing in CLI/UI
        return self.value


NORMAL_COLORS: List[Color] = [Color.WHITE, Color.BLUE, Color.GREEN, Color.RED, Color.BLACK]
ALL_COLORS: List[Color] = NORMAL_COLORS + [Color.GOLD]
VALID_COLOR_NAMES = {c.value for c in NORMAL_COLORS}
VALID_TIERS = {1, 2, 3}

CARD_REQUIRED_FIELDS = {"id", "tier", "bonus", "points", "cost", "image"}
COST_REQUIRED_FIELDS = set(VALID_COLOR_NAMES)
COIN_REQUIRED_FIELDS = {"color", "is_joker", "image"}
NOBLE_REQUIRED_FIELDS = {"id", "points", "requirements", "image"}


@dataclass(frozen=True)
class Card:
    id: str
    tier: int
    bonus: Color
    points: int
    cost: Dict[Color, int]
    image_rel: str
    image_abs: Optional[str] = None

    def cost_for(self, color: Color) -> int:
        return self.cost.get(color, 0)


@dataclass(frozen=True)
class Noble:
    id: str
    points: int
    requirements: Dict[Color, int]
    image_rel: str
    image_abs: Optional[str] = None

    def requirement_for(self, color: Color) -> int:
        return self.requirements.get(color, 0)


@dataclass(frozen=True)
class Coin:
    color: Color
    is_joker: bool
    image_rel: str
    image_abs: Optional[str] = None


@dataclass
class ValidationIssue:
    severity: str  # "error" or "warning"
    entity: str  # e.g. "card[3] (id=t1_004)"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.entity}: {self.message}"


def _cost_like_issues(entity: str, field_name: str, values: object) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not isinstance(values, dict):
        issues.append(ValidationIssue("error", entity, f"'{field_name}' must be an object, got {type(values).__name__}"))
        return issues
    fields = set(values.keys())
    missing = COST_REQUIRED_FIELDS - fields
    extra = fields - COST_REQUIRED_FIELDS
    if missing:
        issues.append(ValidationIssue("error", entity, f"'{field_name}' missing colors: {sorted(missing)}"))
    if extra:
        issues.append(ValidationIssue("error", entity, f"'{field_name}' has unknown color keys: {sorted(extra)}"))
    for color, val in values.items():
        if not isinstance(val, int) or isinstance(val, bool):
            issues.append(ValidationIssue("error", entity, f"'{field_name}.{color}' must be an int, got {val!r}"))
        elif val < 0:
            issues.append(ValidationIssue("error", entity, f"'{field_name}.{color}' is negative ({val})"))
    return issues


def validate_card_dict(index: int, raw: dict) -> List[ValidationIssue]:
    entity = f"card[{index}] (id={raw.get('id', '?')})"
    issues: List[ValidationIssue] = []
    if not isinstance(raw, dict):
        return [ValidationIssue("error", entity, f"entry is not an object: {raw!r}")]

    fields = set(raw.keys())
    missing = CARD_REQUIRED_FIELDS - fields
    extra = fields - CARD_REQUIRED_FIELDS
    if missing:
        issues.append(ValidationIssue("error", entity, f"missing required fields: {sorted(missing)}"))
    if extra:
        issues.append(ValidationIssue("warning", entity, f"unknown fields present (ignored): {sorted(extra)}"))

    if "id" in raw and not isinstance(raw["id"], str):
        issues.append(ValidationIssue("error", entity, f"'id' must be a string, got {raw['id']!r}"))

    if "tier" in raw:
        if not isinstance(raw["tier"], int) or isinstance(raw["tier"], bool) or raw["tier"] not in VALID_TIERS:
            issues.append(ValidationIssue("error", entity, f"'tier' must be one of {sorted(VALID_TIERS)}, got {raw['tier']!r}"))

    if "bonus" in raw and raw["bonus"] not in VALID_COLOR_NAMES:
        issues.append(ValidationIssue("error", entity, f"'bonus' must be one of {sorted(VALID_COLOR_NAMES)}, got {raw['bonus']!r}"))

    if "points" in raw:
        if not isinstance(raw["points"], int) or isinstance(raw["points"], bool):
            issues.append(ValidationIssue("error", entity, f"'points' must be an int, got {raw['points']!r}"))
        elif raw["points"] < 0:
            issues.append(ValidationIssue("error", entity, f"'points' is negative ({raw['points']})"))

    if "cost" in raw:
        issues.extend(_cost_like_issues(entity, "cost", raw["cost"]))

    if "image" in raw and not isinstance(raw["image"], str):
        issues.append(ValidationIssue("error", entity, f"'image' must be a string path, got {raw['image']!r}"))

    return issues


def validate_noble_dict(index: int, raw: dict) -> List[ValidationIssue]:
    entity = f"noble[{index}] (id={raw.get('id', '?')})"
    issues: List[ValidationIssue] = []
    if not isinstance(raw, dict):
        return [ValidationIssue("error", entity, f"entry is not an object: {raw!r}")]

    fields = set(raw.keys())
    missing = NOBLE_REQUIRED_FIELDS - fields
    extra = fields - NOBLE_REQUIRED_FIELDS
    if missing:
        issues.append(ValidationIssue("error", entity, f"missing required fields: {sorted(missing)}"))
    if extra:
        issues.append(ValidationIssue("warning", entity, f"unknown fields present (ignored): {sorted(extra)}"))

    if "id" in raw and not isinstance(raw["id"], str):
        issues.append(ValidationIssue("error", entity, f"'id' must be a string, got {raw['id']!r}"))

    if "points" in raw:
        if not isinstance(raw["points"], int) or isinstance(raw["points"], bool):
            issues.append(ValidationIssue("error", entity, f"'points' must be an int, got {raw['points']!r}"))
        elif raw["points"] < 0:
            issues.append(ValidationIssue("error", entity, f"'points' is negative ({raw['points']})"))

    if "requirements" in raw:
        issues.extend(_cost_like_issues(entity, "requirements", raw["requirements"]))

    if "image" in raw and not isinstance(raw["image"], str):
        issues.append(ValidationIssue("error", entity, f"'image' must be a string path, got {raw['image']!r}"))

    return issues


def validate_coin_dict(index: int, raw: dict) -> List[ValidationIssue]:
    entity = f"coin[{index}] (color={raw.get('color', '?')})"
    issues: List[ValidationIssue] = []
    if not isinstance(raw, dict):
        return [ValidationIssue("error", entity, f"entry is not an object: {raw!r}")]

    fields = set(raw.keys())
    missing = COIN_REQUIRED_FIELDS - fields
    extra = fields - COIN_REQUIRED_FIELDS
    if missing:
        issues.append(ValidationIssue("error", entity, f"missing required fields: {sorted(missing)}"))
    if extra:
        issues.append(ValidationIssue("warning", entity, f"unknown fields present (ignored): {sorted(extra)}"))

    valid_names = VALID_COLOR_NAMES | {"gold"}
    if "color" in raw and raw["color"] not in valid_names:
        issues.append(ValidationIssue("error", entity, f"'color' must be one of {sorted(valid_names)}, got {raw['color']!r}"))

    if "is_joker" in raw and not isinstance(raw["is_joker"], bool):
        issues.append(ValidationIssue("error", entity, f"'is_joker' must be a bool, got {raw['is_joker']!r}"))

    if "image" in raw and not isinstance(raw["image"], str):
        issues.append(ValidationIssue("error", entity, f"'image' must be a string path, got {raw['image']!r}"))

    return issues
