"""Read-only validation report for the prepared Splendor game data.

Usage:
    python -m splendor_ai.data.validator [--root PATH]

This NEVER modifies cards/, coins/, or nobles/. It loads the data
non-strictly (so it can report every problem instead of stopping at the
first one) and prints a full report: per-tier card counts, noble/coin
counts, duplicate IDs, invalid tiers/colors, missing fields, negative
values, broken image paths, duplicate image references, and unknown
fields. Exits with a non-zero status if any fatal schema issue is found.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import List

from splendor_ai.data.loader import DEFAULT_ROOT, load_cards, load_coins, load_nobles
from splendor_ai.data.schemas import NORMAL_COLORS, ValidationIssue


def _check_images_exist(entity_prefix: str, items, root_subdir: str, root: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for item in items:
        if item.image_abs and not os.path.isfile(item.image_abs):
            issues.append(ValidationIssue("warning", f"{entity_prefix}(id={getattr(item, 'id', item.color)})", f"broken image path -> {item.image_abs}"))
    return issues


def _check_duplicate_images(entity_prefix: str, items) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    counts = Counter(i.image_rel for i in items)
    for image_rel, n in counts.items():
        if n > 1:
            dupes = [getattr(i, "id", str(i.color)) for i in items if i.image_rel == image_rel]
            issues.append(ValidationIssue("warning", entity_prefix, f"image '{image_rel}' referenced by {n} entries: {dupes}"))
    return issues


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _check_unreferenced_files(label: str, dir_path: str, referenced_rel_paths: List[str], root_dir_for_rel: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.isdir(dir_path):
        return issues
    disk_files = set()
    for dirpath, _dirnames, filenames in os.walk(dir_path):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            rel = os.path.normpath(os.path.relpath(os.path.join(dirpath, fn), root_dir_for_rel))
            disk_files.add(rel)
    referenced = {os.path.normpath(p) for p in referenced_rel_paths}
    unreferenced = disk_files - referenced
    if unreferenced:
        issues.append(ValidationIssue("warning", label, f"{len(unreferenced)} image file(s) on disk not referenced by any JSON entry: {sorted(unreferenced)}"))
    return issues


def run_validation(root: str = DEFAULT_ROOT) -> int:
    print(f"Validating Splendor data at: {root}\n")
    all_issues: List[ValidationIssue] = []

    try:
        cards, cards_issues = load_cards(root)
    except FileNotFoundError as e:
        print(f"FATAL: could not read cards.json: {e}")
        return 2
    try:
        nobles, nobles_issues = load_nobles(root)
    except FileNotFoundError as e:
        print(f"FATAL: could not read nobles.json: {e}")
        return 2
    try:
        coins, coins_issues = load_coins(root)
    except FileNotFoundError as e:
        print(f"FATAL: could not read coins.json: {e}")
        return 2

    all_issues.extend(cards_issues)
    all_issues.extend(nobles_issues)
    all_issues.extend(coins_issues)

    # --- Cards summary ---
    print("=== Cards ===")
    tier_counts = Counter(c.tier for c in cards)
    print(f"Total cards loaded: {len(cards)}")
    for t in sorted(tier_counts):
        print(f"  Tier {t}: {tier_counts[t]} cards")
    all_issues.extend(_check_duplicate_images("cards", cards))
    all_issues.extend(_check_images_exist("card", cards, "cards", root))
    all_issues.extend(
        _check_unreferenced_files("cards/", os.path.join(root, "cards"), [c.image_rel for c in cards], os.path.join(root, "cards"))
    )

    # --- Nobles summary ---
    print("\n=== Nobles ===")
    print(f"Total nobles loaded: {len(nobles)}")
    for n in nobles:
        nonzero = sum(1 for v in n.requirements.values() if v > 0)
        if nonzero not in (2, 3):
            all_issues.append(ValidationIssue("warning", f"noble(id={n.id})", f"unusual requirement shape: {nonzero} nonzero colors (standard nobles use 2 or 3)"))
    all_issues.extend(_check_duplicate_images("nobles", nobles))
    all_issues.extend(_check_images_exist("noble", nobles, "nobles", root))
    all_issues.extend(
        _check_unreferenced_files("nobles/", os.path.join(root, "nobles"), [n.image_rel for n in nobles], os.path.join(root, "nobles"))
    )

    # --- Coins summary ---
    print("\n=== Coins ===")
    coin_colors = [c.color.value for c in coins]
    print(f"Available coin colors: {coin_colors}")
    normal_present = {c.color.value for c in coins if not c.is_joker}
    missing_normal = {c.value for c in NORMAL_COLORS} - normal_present
    if missing_normal:
        all_issues.append(ValidationIssue("error", "coins.json", f"missing normal coin colors: {sorted(missing_normal)}"))
    if not any(c.is_joker for c in coins):
        all_issues.append(ValidationIssue("error", "coins.json", "no joker (gold) coin defined"))
    all_issues.extend(_check_images_exist("coin", coins, "coins", root))
    all_issues.extend(
        _check_unreferenced_files("coins/", os.path.join(root, "coins"), [c.image_rel for c in coins], os.path.join(root, "coins"))
    )

    # --- Summary ---
    print("\n=== Summary ===")
    errors = [i for i in all_issues if i.severity == "error"]
    warnings = [i for i in all_issues if i.severity == "warning"]
    if not all_issues:
        print("No validation errors or warnings found. Data is clean and ready to use.")
        return 0

    if errors:
        print(f"{len(errors)} FATAL error(s) -- the game engine cannot load this data as-is:")
        for i in errors:
            print(f"  - {i}")
    if warnings:
        print(f"{len(warnings)} warning(s) (non-fatal, e.g. missing images use placeholders):")
        for i in warnings:
            print(f"  - {i}")

    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prepared Splendor game data (read-only).")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Path to the Splendor project root (containing cards/, coins/, nobles/).")
    args = parser.parse_args()
    sys.exit(run_validation(args.root))


if __name__ == "__main__":
    main()
