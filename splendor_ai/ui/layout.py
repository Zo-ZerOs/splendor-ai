"""Screen-space geometry for the read-only Pygame viewer. Pure math, no
Pygame drawing calls, so it can be unit tested without a display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pygame

from splendor_ai.data.schemas import NORMAL_COLORS, Color

CARD_SIZE = (90, 120)
NOBLE_SIZE = (70, 70)
COIN_SIZE = (44, 44)
CARD_GAP = 10
TIER_ROW_GAP = 14
MARGIN = 16
# Reserved strip across the very top of the window for status banners
# (FINAL ROUND / pending token-return or noble-choice) so they never
# overlap the nobles row or the recent-action log.
TOP_BANNER_HEIGHT = 36


@dataclass
class Layout:
    resolution: Tuple[int, int]
    noble_rects: List[pygame.Rect] = field(default_factory=list)
    tier_deck_rects: Dict[int, pygame.Rect] = field(default_factory=dict)
    tier_card_rects: Dict[int, List[pygame.Rect]] = field(default_factory=dict)
    bank_coin_rects: Dict[Color, pygame.Rect] = field(default_factory=dict)
    player_panel_rects: List[pygame.Rect] = field(default_factory=list)
    log_rect: pygame.Rect = None
    help_rect: pygame.Rect = None
    banner_rect: pygame.Rect = None

    @staticmethod
    def compute(resolution: Tuple[int, int], num_nobles: int = 5, num_players: int = 4) -> "Layout":
        w, h = resolution
        layout = Layout(resolution=resolution)

        # --- Nobles row (top) ---
        nobles_y = MARGIN + TOP_BANNER_HEIGHT
        total_noble_w = num_nobles * NOBLE_SIZE[0] + (num_nobles - 1) * CARD_GAP
        start_x = (w - total_noble_w) // 2
        for i in range(num_nobles):
            x = start_x + i * (NOBLE_SIZE[0] + CARD_GAP)
            layout.noble_rects.append(pygame.Rect(x, nobles_y, *NOBLE_SIZE))

        # --- Tier rows (tier 3 on top, tier 1 closest to bank) ---
        tiers_top = nobles_y + NOBLE_SIZE[1] + 20
        deck_w = CARD_SIZE[0]
        row_w = deck_w + CARD_GAP + 4 * CARD_SIZE[0] + 3 * CARD_GAP
        row_x = (w - row_w) // 2
        for row_index, tier in enumerate((3, 2, 1)):
            y = tiers_top + row_index * (CARD_SIZE[1] + TIER_ROW_GAP)
            layout.tier_deck_rects[tier] = pygame.Rect(row_x, y, deck_w, CARD_SIZE[1])
            cards_x = row_x + deck_w + CARD_GAP
            slots = []
            for slot in range(4):
                x = cards_x + slot * (CARD_SIZE[0] + CARD_GAP)
                slots.append(pygame.Rect(x, y, *CARD_SIZE))
            layout.tier_card_rects[tier] = slots

        # --- Bank token row ---
        bank_y = tiers_top + 3 * (CARD_SIZE[1] + TIER_ROW_GAP) + 10
        colors = NORMAL_COLORS + [Color.GOLD]
        total_bank_w = len(colors) * COIN_SIZE[0] + (len(colors) - 1) * CARD_GAP
        bank_x = (w - total_bank_w) // 2
        for i, color in enumerate(colors):
            x = bank_x + i * (COIN_SIZE[0] + CARD_GAP)
            layout.bank_coin_rects[color] = pygame.Rect(x, bank_y, *COIN_SIZE)

        # --- Player panels (bottom row) ---
        panels_y = bank_y + COIN_SIZE[1] + 16
        panel_h = h - panels_y - MARGIN
        panel_w = (w - MARGIN * 2 - (num_players - 1) * CARD_GAP) // num_players
        for i in range(num_players):
            x = MARGIN + i * (panel_w + CARD_GAP)
            layout.player_panel_rects.append(pygame.Rect(x, panels_y, panel_w, max(120, panel_h)))

        # --- Recent-action log (right side) and "How to play" help panel
        # (left side) -- both run the full height alongside the board, from
        # the nobles row down to just above the player panels, since that
        # side space next to the centered board is otherwise unused. Width
        # is clamped to whatever fits beside the (centered) tier rows so
        # they never overlap the board at narrower resolutions.
        side_panel_height = max(0, panels_y - nobles_y - 10)
        side_panel_width = max(0, min(260, row_x - MARGIN - 10))
        layout.log_rect = pygame.Rect(w - side_panel_width - MARGIN, nobles_y, side_panel_width, side_panel_height)
        layout.help_rect = pygame.Rect(MARGIN, nobles_y, side_panel_width, side_panel_height)

        layout.banner_rect = pygame.Rect(0, 0, w, h)
        return layout
