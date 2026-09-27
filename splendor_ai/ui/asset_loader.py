"""Loads images ONLY from Splendor/cards, Splendor/coins, Splendor/nobles
(via the paths already resolved by splendor_ai.data.loader) and caches
them as Pygame surfaces. Never generates or writes replacement game data
-- a missing/corrupt image falls back to an in-memory placeholder surface
that shows the object's ID and a warning is logged, but the process never
crashes and no file on disk is touched.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import pygame

from splendor_ai.data.schemas import Card, Coin, Color, Noble

logger = logging.getLogger("splendor_ai.ui")

_COLOR_RGB = {
    Color.WHITE: (245, 245, 235),
    Color.BLUE: (60, 110, 220),
    Color.GREEN: (50, 160, 90),
    Color.RED: (200, 60, 50),
    Color.BLACK: (40, 40, 45),
    Color.GOLD: (215, 175, 40),
}


class AssetLoader:
    def __init__(self):
        self._cache: Dict[Tuple[str, int, int], pygame.Surface] = {}
        self._warned_paths: set = set()
        self._font: Optional[pygame.font.Font] = None

    def _font_for(self, size: int) -> pygame.font.Font:
        if not pygame.font.get_init():
            pygame.font.init()
        return pygame.font.SysFont("arial", size)

    def _load_or_placeholder(self, path: Optional[str], size: Tuple[int, int], label: str, accent: Tuple[int, int, int]) -> pygame.Surface:
        key = (path or f"__placeholder__{label}", *size)
        if key in self._cache:
            return self._cache[key]

        surface: Optional[pygame.Surface] = None
        if path:
            try:
                surface = pygame.image.load(path)
                if pygame.display.get_surface() is not None:
                    surface = surface.convert_alpha()
            except Exception as e:  # noqa: BLE001 - any load failure -> placeholder, never crash
                if path not in self._warned_paths:
                    logger.warning("Could not load image '%s' (%s); using placeholder.", path, e)
                    self._warned_paths.add(path)
                surface = None

        if surface is None:
            surface = self._make_placeholder(size, label, accent)
        else:
            surface = pygame.transform.smoothscale(surface, size)

        self._cache[key] = surface
        return surface

    def _make_placeholder(self, size: Tuple[int, int], label: str, accent: Tuple[int, int, int]) -> pygame.Surface:
        w, h = size
        surf = pygame.Surface(size, pygame.SRCALPHA)
        surf.fill((230, 230, 230, 255))
        pygame.draw.rect(surf, accent, surf.get_rect(), width=max(2, w // 20))
        pygame.draw.line(surf, (160, 160, 160), (0, 0), (w, h), 1)
        pygame.draw.line(surf, (160, 160, 160), (0, h), (w, 0), 1)
        font = self._font_for(max(10, h // 8))
        text_surf = font.render(label, True, (30, 30, 30))
        rect = text_surf.get_rect(center=(w // 2, h // 2))
        surf.blit(text_surf, rect)
        return surf

    def card_image(self, card: Card, size: Tuple[int, int]) -> pygame.Surface:
        return self._load_or_placeholder(card.image_abs, size, card.id, _COLOR_RGB.get(card.bonus, (120, 120, 120)))

    def noble_image(self, noble: Noble, size: Tuple[int, int]) -> pygame.Surface:
        return self._load_or_placeholder(noble.image_abs, size, noble.id, (150, 120, 60))

    def coin_image(self, coin: Coin, size: Tuple[int, int]) -> pygame.Surface:
        return self._load_or_placeholder(coin.image_abs, size, coin.color.value, _COLOR_RGB.get(coin.color, (120, 120, 120)))

    def card_back(self, size: Tuple[int, int]) -> pygame.Surface:
        return self._make_placeholder(size, "?", (90, 90, 90))
