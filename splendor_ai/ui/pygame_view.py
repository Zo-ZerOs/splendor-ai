"""Read-only Pygame viewer.

Runs its own render loop in a background thread so the terminal can block
on human input without freezing the window (Windows would otherwise mark
it "Not Responding"). The main thread never touches Pygame directly --
it only calls `update(state)` after every action, which hands the viewer
a private snapshot (`GameState.clone()`) to draw from. All human actions
happen through the terminal; this window draws only.

Images are loaded exclusively from Splendor/cards, Splendor/coins, and
Splendor/nobles via AssetLoader; anything missing renders as a labeled
placeholder and logs a warning instead of crashing.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import pygame

from splendor_ai.config import GameConfig
from splendor_ai.data.loader import DataBundle
from splendor_ai.data.schemas import ALL_COLORS, Color, NORMAL_COLORS
from splendor_ai.game.actions import CardSource, ReserveSource
from splendor_ai.game.state import GameState, TurnPhase
from splendor_ai.ui.asset_loader import AssetLoader
from splendor_ai.ui.layout import CARD_SIZE, COIN_SIZE, NOBLE_SIZE, Layout

logger = logging.getLogger("splendor_ai.ui")

# Representative RGB for each color's token circle / bonus rectangle fill.
_TOKEN_RGB = {
    Color.WHITE: (238, 238, 230),
    Color.BLUE: (55, 105, 215),
    Color.GREEN: (45, 155, 85),
    Color.RED: (195, 55, 45),
    Color.BLACK: (40, 40, 45),
    Color.GOLD: (210, 170, 35),
}

# Soft, non-neon tier identity: D1=green, D2=yellow, D3=blue. Used both for
# the face-down deck tiles (a light fill, dark text drawn on top) and, in a
# brighter/more legible-on-dark variant, for the purchased-card-count text
# in player panels (drawn directly on the dark panel background) -- same
# color per tier in both places so the two visually match.
_TIER_FILL_RGB = {
    1: (150, 195, 155),
    2: (215, 195, 120),
    3: (145, 180, 220),
}
_TIER_TEXT_RGB = {
    1: (120, 210, 140),
    2: (230, 195, 90),
    3: (110, 165, 230),
}


# Matches the "(<card id>)" that _apply_buy_card/_apply_reserve_card always
# embed in a buy's or a visible-sourced reserve's HistoryEntry.description,
# to recover which card was involved for display (the board slot it came
# from may already be refilled/gone). Not anchored to the end of the
# string -- a reserve's description can have " and received 1 gold"
# appended after the "(id)" group.
_CARD_ID_IN_DESCRIPTION_RE = re.compile(r"\(([^()]+)\)")


def _net_circles(picked: tuple, returned: tuple) -> tuple:
    """Cancels colors pairwise between what was picked/gained and what
    was then forced back (a token-return triggered by exceeding the
    hand limit), one-for-one per matching color. Returns (leftover_gained,
    leftover_returned) -- e.g. picked=(blue,blue,white), returned=
    (blue,green,red) -> one blue cancels, leaving ([blue,white],
    [green,red])."""
    remaining_returned = list(returned)
    leftover_gained = []
    for color in picked:
        if color in remaining_returned:
            remaining_returned.remove(color)
        else:
            leftover_gained.append(color)
    return leftover_gained, remaining_returned


def _count_player_actions(state: GameState, seat: int) -> dict:
    counts = {"pick_tokens": 0, "buy_card": 0, "reserve_card": 0}
    for entry in state.history:
        if entry.seat == seat and entry.action_kind in counts:
            counts[entry.action_kind] += 1
    return counts


# Dark theme: the main background is dark, so text drawn directly on it
# (the log/help panels, the round counter) uses TEXT_COLOR_ON_BG (light).
# Player panels are also dark now, so their own header text (name/score)
# uses TEXT_COLOR_ON_BG too; TEXT_COLOR (dark) remains for text drawn on
# genuinely light surfaces (the deck tiles, empty card slots, the white
# action-count mini-table cells).
BG_COLOR = (26, 28, 33)
PANEL_COLOR = (48, 51, 59)
PANEL_BORDER = (95, 92, 85)
HIGHLIGHT_BORDER = (235, 90, 70)
TEXT_COLOR = (30, 30, 30)
TEXT_COLOR_ON_BG = (222, 222, 222)
FINAL_ROUND_COLOR = (255, 100, 80)


class SplendorView:
    def __init__(self, bundle: DataBundle, config: GameConfig, human_seat: int = 0, resolution=(1280, 860), difficulty_label: Optional[str] = None):
        self.bundle = bundle
        self.config = config
        self.human_seat = human_seat
        self.resolution = resolution
        self.difficulty_label = difficulty_label
        self.layout = Layout.compute(resolution, num_nobles=config.nobles_count_in_play, num_players=config.num_players)
        self.assets = AssetLoader()

        self._lock = threading.Lock()
        self._state: Optional[GameState] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._closed_by_user = False
        # Fixed left-to-right noble slot assignment for the CURRENT game,
        # captured once (see _noble_slot_order) so a claimed noble's slot
        # goes blank in place rather than the remaining nobles shifting
        # left to fill the gap.
        self._noble_order: Optional[list] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._closed_by_user = False
        self._thread = threading.Thread(target=self._run, name="splendor-view", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def closed(self) -> bool:
        """True once the user has closed the window. The game itself keeps
        running in the terminal -- callers should check this once per turn
        and print a one-time notice (see splendor_ai.cli.play.run)."""
        return self._closed_by_user

    def update(self, state: GameState) -> None:
        snapshot = state.clone() if state is not None else None
        with self._lock:
            self._state = snapshot

    # ---------------- render thread ----------------

    def _run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode(self.resolution)
        pygame.display.set_caption("Splendor -- read-only viewer")
        clock = pygame.time.Clock()
        fonts = {
            "small": pygame.font.SysFont("arial", 14),
            "med": pygame.font.SysFont("arial", 17),
            "big": pygame.font.SysFont("arial", 30, bold=True),
            "deck": pygame.font.SysFont("arial", 22, bold=True),
            "coin": pygame.font.SysFont("arial", 19, bold=True),
        }

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Viewer window closed; rendering stopped (game continues in the terminal).")
                    self._running = False
                    self._closed_by_user = True

            with self._lock:
                state = self._state

            screen.fill(BG_COLOR)
            if state is not None:
                try:
                    self._draw(screen, state, fonts)
                except Exception:  # noqa: BLE001 -- never let a draw glitch kill the viewer/game
                    logger.exception("Error while rendering board state; skipping this frame.")
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    # ---------------- drawing ----------------

    def _text(self, screen, font, text, pos, color=TEXT_COLOR):
        screen.blit(font.render(text, True, color), pos)

    def _draw_outlined_text(self, screen, font, text, center, text_color=(0, 0, 0), outline_color=(255, 255, 255)):
        """Bold text with a thin outline so it stays legible over any coin
        artwork color (light or dark)."""
        outline = font.render(text, True, outline_color)
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
            screen.blit(outline, outline.get_rect(center=(center[0] + dx, center[1] + dy)))
        base = font.render(text, True, text_color)
        screen.blit(base, base.get_rect(center=center))

    def _draw_deck_tile(self, screen, rect, tier: int, remaining: int, fonts) -> None:
        """Face-down deck pile -- no source image exists for a card back,
        so this draws a plain tile with the tier label and remaining count
        centered inside it (e.g. "D1" / "36 left"). Filled with this
        tier's identity color (green/yellow/blue for D1/D2/D3, see
        _TIER_FILL_RGB) so it's visually distinct at a glance."""
        pygame.draw.rect(screen, _TIER_FILL_RGB[tier], rect)
        pygame.draw.rect(screen, PANEL_BORDER, rect, 2)
        label_surf = fonts["deck"].render(f"D{tier}", True, (50, 48, 45))
        screen.blit(label_surf, label_surf.get_rect(center=(rect.centerx, rect.centery - 14)))
        sub_surf = fonts["small"].render(f"{remaining} left", True, (50, 48, 45))
        screen.blit(sub_surf, sub_surf.get_rect(center=(rect.centerx, rect.centery + 16)))

    def _noble_slot_order(self, state: GameState) -> list:
        """Ids of every noble in this game, in a fixed left-to-right slot
        order that never reshuffles as nobles get claimed -- captured
        once (or re-captured if a genuinely new game's nobles show up,
        e.g. this view outlives one game) from whatever's still on the
        board plus whatever players have already claimed."""
        current_ids = {n.id for n in state.board.nobles}
        if self._noble_order is None or not current_ids.issubset(self._noble_order):
            self._noble_order = [n.id for n in state.board.nobles] + [n.id for p in state.players for n in p.nobles]
        return self._noble_order

    def _draw(self, screen, state: GameState, fonts) -> None:
        L = self.layout

        round_label = f"Round {state.round_number}"
        if self.difficulty_label:
            round_label += f" ({self.difficulty_label})"
        round_surf = fonts["deck"].render(round_label, True, TEXT_COLOR_ON_BG)
        screen.blit(round_surf, round_surf.get_rect(midtop=(self.resolution[0] // 2, 2)))

        noble_by_id = {n.id: n for n in state.board.nobles}
        for i, noble_id in enumerate(self._noble_slot_order(state)):
            if i >= len(L.noble_rects):
                break
            rect = L.noble_rects[i]
            noble = noble_by_id.get(noble_id)
            if noble is not None:
                screen.blit(self.assets.noble_image(noble, rect.size), rect)
            else:
                # claimed -- blank slot, no outline, same as an exhausted
                # deck's empty visible-card slot elsewhere in this draw.
                pygame.draw.rect(screen, BG_COLOR, rect)

        for tier in (1, 2, 3):
            deck_rect = L.tier_deck_rects[tier]
            self._draw_deck_tile(screen, deck_rect, tier, state.board.deck_remaining(tier), fonts)

            for slot, rect in enumerate(L.tier_card_rects[tier]):
                card = state.board.visible[tier][slot]
                if card is None:
                    # Deck exhausted, nothing to refill this slot with --
                    # blend it into the background instead of drawing a
                    # placeholder box (no fill, no border, no text).
                    pygame.draw.rect(screen, BG_COLOR, rect)
                else:
                    screen.blit(self.assets.card_image(card, rect.size), rect)

        for color, rect in L.bank_coin_rects.items():
            coin = self.bundle.coin_by_color.get(color)
            if coin is not None:
                screen.blit(self.assets.coin_image(coin, rect.size), rect)
            count = state.board.bank.get(color, 0)
            self._draw_outlined_text(screen, fonts["coin"], str(count), rect.center)

        for seat in range(state.num_players):
            self._draw_player_panel(screen, state, seat, L.player_panel_rects[seat], fonts)

        self._draw_log(screen, state, L.log_rect, fonts)
        self._draw_card_tracker(screen, state, L.help_rect, fonts)
        self._draw_banner(screen, state, fonts)

    def _draw_token_circle(self, screen, center, radius, color: Color, count: int, fonts) -> None:
        rgb = _TOKEN_RGB.get(color, (150, 150, 150))
        pygame.draw.circle(screen, rgb, center, radius)
        pygame.draw.circle(screen, (0, 0, 0), center, radius, 2)
        if count > 0:
            self._draw_outlined_text(screen, fonts["small"], str(count), center, text_color=(255, 255, 255), outline_color=(0, 0, 0))

    def _draw_bonus_rect(self, screen, rect, color: Color, count: int, fonts) -> None:
        rgb = _TOKEN_RGB.get(color, (150, 150, 150))
        pygame.draw.rect(screen, rgb, rect)
        pygame.draw.rect(screen, (0, 0, 0), rect, 2)
        if count > 0:
            self._draw_outlined_text(screen, fonts["small"], str(count), rect.center, text_color=(255, 255, 255), outline_color=(0, 0, 0))

    def _draw_player_panel(self, screen, state: GameState, seat: int, rect, fonts) -> None:
        player = state.players[seat]
        is_current = state.current_player == seat and not state.game_over
        border_color = HIGHLIGHT_BORDER if is_current else PANEL_BORDER
        pygame.draw.rect(screen, PANEL_COLOR, rect)
        pygame.draw.rect(screen, border_color, rect, 3 if is_current else 1)

        pad = 8
        x = rect.x + pad
        y = rect.y + 6

        # --- header: player number (left) + score (right) ---
        name_label = f"Player {seat + 1}" + (" (You)" if seat == self.human_seat else "")
        self._text(screen, fonts["med"], name_label, (x, y), color=TEXT_COLOR_ON_BG)
        score_surf = fonts["med"].render(f"Score: {player.prestige}", True, TEXT_COLOR_ON_BG)
        screen.blit(score_surf, (rect.right - pad - score_surf.get_width(), y))
        y += 26

        # --- nobles: small pictures, stacked vertically in a left column
        # right under the header. The column only takes up width when the
        # player actually has a noble, so panels don't waste space on it
        # for most of the game (nobles are rare). Everything else flows
        # to the right of this column. ---
        content_top = y
        noble_size = (28, 28)
        noble_col_width = (noble_size[0] + 10) if player.nobles else 0
        ny = content_top
        for noble in player.nobles:
            screen.blit(self.assets.noble_image(noble, noble_size), pygame.Rect(x, ny, *noble_size))
            ny += noble_size[1] + 6
        noble_col_bottom = ny

        right_x = x + noble_col_width
        ry = content_top

        # --- small action-count table: P / B / R ---
        counts = _count_player_actions(state, seat)
        cell_w, cell_h = 46, 20
        for i, (letter, key) in enumerate((("P", "pick_tokens"), ("B", "buy_card"), ("R", "reserve_card"))):
            cell_rect = pygame.Rect(right_x + i * (cell_w + 4), ry, cell_w, cell_h)
            pygame.draw.rect(screen, (255, 255, 255), cell_rect)
            pygame.draw.rect(screen, PANEL_BORDER, cell_rect, 1)
            self._text(screen, fonts["small"], f"{letter} {counts[key]}", (cell_rect.x + 5, cell_rect.y + 2))
        ry += cell_h + 10

        # --- tokens: colored circles for every color, always shown; "-"
        # in place of the count when a player holds none of that color ---
        # Followed by the running total across all colors (incl. gold),
        # in plain white text, right after the last circle.
        radius = 14
        cx, cy = right_x + radius, ry + radius
        for color in ALL_COLORS:
            n = player.tokens.get(color, 0)
            self._draw_token_circle(screen, (cx, cy), radius, color, n, fonts)
            cx += radius * 2 + 7
        total_tokens = sum(player.tokens.get(color, 0) for color in ALL_COLORS)
        total_surf = fonts["coin"].render(str(total_tokens), True, (255, 255, 255))
        screen.blit(total_surf, total_surf.get_rect(midleft=(cx + 4, cy)))
        ry += radius * 2 + 10

        # --- bonuses: colored rectangles for every normal color, always
        # shown; "-" in place of the count when a player has none yet ---
        bonus_w, bonus_h = 26, 22
        bx = right_x
        for color in NORMAL_COLORS:
            n = player.bonuses.get(color, 0)
            self._draw_bonus_rect(screen, pygame.Rect(bx, ry, bonus_w, bonus_h), color, n, fonts)
            bx += bonus_w + 6

        # --- purchased-card counts by tier, right after the bonus row --
        # color-coded to match the D1/D2/D3 deck tiles (green/yellow/blue,
        # see _TIER_TEXT_RGB) so it's clear which number is which tier. ---
        tier_counts = {1: 0, 2: 0, 3: 0}
        for c in player.purchased_cards:
            tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
        tx = bx + 8
        for tier in (1, 2, 3):
            surf = fonts["med"].render(str(tier_counts[tier]), True, _TIER_TEXT_RGB[tier])
            if tx + surf.get_width() > rect.right - pad:
                break
            screen.blit(surf, (tx, ry + (bonus_h - surf.get_height()) // 2))
            tx += surf.get_width() + 10

        ry += bonus_h + 10

        y = max(noble_col_bottom, ry)

        # --- reserved cards: visible-sourced shown face up (not secret in
        # physical Splendor -- everyone saw it when it was reserved), blind
        # (deck-top) shown face down for opponents; always face up for the
        # human's own reservations. ---
        card_w, card_h = 46, 62
        for i, card in enumerate(player.reserved_cards):
            small_rect = pygame.Rect(x + i * (card_w + 6), y, card_w, card_h)
            source = player.reserved_card_sources[i] if i < len(player.reserved_card_sources) else "blind"
            face_up = seat == self.human_seat or source == "visible"
            if face_up:
                screen.blit(self.assets.card_image(card, small_rect.size), small_rect)
            else:
                screen.blit(self.assets.card_back(small_rect.size), small_rect)

    def _card_from_description(self, description: str):
        """Looks up the Card a buy's or a visible-sourced reserve's
        HistoryEntry refers to, via the card id embedded in its
        description (see _CARD_ID_IN_DESCRIPTION_RE) -- the board slot it
        came from may already show a different, refilled card by the
        time this is drawn."""
        match = _CARD_ID_IN_DESCRIPTION_RE.search(description)
        if match is None:
            return None
        return self.bundle.card_by_id.get(match.group(1))

    def _card_tuple_items(self, card, starred: bool = False) -> list:
        """[T<tier>, <score>, <bonus color circle>] -- the log's compact
        stand-in for a fully-known (publicly visible) card. `starred`
        marks a card that is (or was, for a buy) held in reserve rather
        than bought straight off the board."""
        return [
            ("text", f"[T{card.tier}, {card.points}, "),
            ("circle", card.bonus),
            ("text", "]*" if starred else "]"),
        ]

    def _minus_suffix_items(self, colors) -> list:
        """" - (return <circle> ...)" -- the log's shared notation for
        tokens given back (a forced return after exceeding the hand
        limit)."""
        items = [("text", " - (return ")]
        items.extend(("circle", c) for c in colors)
        items.append(("text", ")"))
        return items

    def _draw_inline(self, screen, fonts, items, x, y) -> None:
        """Draws a left-to-right sequence of ("text", str) / ("circle",
        Color) items on one log line, starting at (x, y)."""
        font = fonts["small"]
        line_h = 16
        radius = 5
        cx = x
        for kind, value in items:
            if kind == "text":
                surf = font.render(value, True, TEXT_COLOR_ON_BG)
                screen.blit(surf, (cx, y))
                cx += surf.get_width()
            else:  # "circle"
                center = (cx + radius, y + line_h // 2)
                pygame.draw.circle(screen, _TOKEN_RGB.get(value, (150, 150, 150)), center, radius)
                pygame.draw.circle(screen, (0, 0, 0), center, radius, 1)
                cx += radius * 2 + 3

    def _merge_history_for_display(self, state: GameState) -> list:
        """Scans state.history in chronological order, merging a
        pick_tokens/reserve_card entry with an immediately-following
        return_tokens entry from the SAME seat and turn (a forced return
        after exceeding the 10-token hand limit -- turns_completed only
        increments once a turn fully resolves, so both entries always
        share turn_index) into one display unit. Returns a list of
        1- or 2-tuples of HistoryEntry, still in chronological order."""
        units = []
        hist = state.history
        i, n = 0, len(hist)
        while i < n:
            entry = hist[i]
            nxt = hist[i + 1] if i + 1 < n else None
            if (
                entry.action_kind in ("pick_tokens", "reserve_card")
                and nxt is not None
                and nxt.action_kind == "return_tokens"
                and nxt.seat == entry.seat
                and nxt.turn_index == entry.turn_index
            ):
                units.append((entry, nxt))
                i += 2
            else:
                units.append((entry,))
                i += 1
        return units

    def _log_line_items(self, unit: tuple, state: GameState) -> list:
        """Builds the ("text"|"circle", value) sequence for one recent-
        action log line -- SCREEN ONLY, entirely separate from
        format_history_entry (used by the terminal's `history` command
        and human-typed command syntax, which this never touches). Coins
        are colored circles rather than color names; a fully-known card
        is the [T<tier>, <score>, <bonus circle>] tuple, starred when
        it's a reserved card (currently held, or bought out of reserve)
        rather than a straight board buy. A blind (face-down)
        reservation's card is never revealed here, same as it isn't on
        the board.

        `unit` is 1 HistoryEntry, or 2 when _merge_history_for_display
        paired a pick/reserve with the forced return it triggered."""
        entry = unit[0]
        items = [("text", f"P{entry.seat + 1}: ")]

        if entry.action_kind == "buy_card" and entry.payment is not None:
            card = self._card_from_description(entry.description)
            starred = entry.action is not None and entry.action.source == CardSource.RESERVED
            items.append(("text", "B "))
            items.extend(self._card_tuple_items(card, starred) if card is not None else [("text", "?")])
            paid_colors = [c for c in ALL_COLORS for _ in range(entry.payment.get(c, 0))]
            if paid_colors:
                items.append(("text", " - (paid "))
                items.extend(("circle", c) for c in paid_colors)
                items.append(("text", ")"))
            return items

        if entry.action_kind == "pick_tokens" and entry.action is not None:
            returned_colors = unit[1].action.colors if len(unit) > 1 else ()
            gained, returned = _net_circles(entry.action.colors, returned_colors)
            items.append(("text", "P "))
            items.extend(("circle", c) for c in gained)
            if returned:
                items.extend(self._minus_suffix_items(returned))
            return items

        if entry.action_kind == "reserve_card" and entry.action is not None:
            items.append(("text", "R "))
            if entry.action.source == ReserveSource.VISIBLE:
                card = self._card_from_description(entry.description)
                items.extend(self._card_tuple_items(card, starred=True) if card is not None else [("text", "?")])
            else:
                items.append(("text", f"D{entry.action.tier}"))

            got_gold = "gold" in entry.description
            if len(unit) > 1:
                # exceeded the hand limit -- merged with the forced return
                # (always exactly 1 token back, since reserving can only
                # ever push the total 1 over the cap).
                returned_colors = unit[1].action.colors
                if Color.GOLD not in returned_colors:
                    # kept the gold -- show it, then what was actually given back
                    items.append(("text", " & "))
                    items.append(("circle", Color.GOLD))
                    items.extend(self._minus_suffix_items(returned_colors))
                # else: gave the just-granted gold right back -- net zero,
                # nothing more to show beyond the bare card.
            elif got_gold:
                items.append(("text", " & "))
                items.append(("circle", Color.GOLD))
            # else: no gold was available to grant -- bare card, nothing more.
            return items

        from splendor_ai.cli.human_input import format_history_entry

        items.append(("text", format_history_entry(entry, state)))
        return items

    def _draw_log(self, screen, state: GameState, rect, fonts) -> None:
        if rect.width <= 0:
            return

        self._text(screen, fonts["med"], "Recent actions:", (rect.x, rect.y), color=TEXT_COLOR_ON_BG)
        y = rect.y + 20
        prev_round = None
        for unit in self._merge_history_for_display(state)[::-1]:
            entry = unit[0]
            # Group by ROUND (one full cycle of all players), not by
            # individual turn: most turns are a single action, so
            # separating turn-by-turn would draw a line after nearly every
            # line -- indistinguishable from separating action-by-action.
            entry_round = entry.turn_index // state.num_players
            if prev_round is not None and entry_round != prev_round:
                pygame.draw.line(screen, (95, 95, 100), (rect.x, y + 2), (rect.right, y + 2), 1)
                y += 8
                if y > rect.bottom:
                    break
            self._draw_inline(screen, fonts, self._log_line_items(unit, state), rect.x, y)
            y += 16
            prev_round = entry_round
            if y > rect.bottom:
                break

    def _draw_card_tracker(self, screen, state: GameState, rect, fonts) -> None:
        """Full-deck overview, grouped by tier in a fixed 10-column grid --
        tier1: 10x4 (cards t1_040..t1_001), tier2: 10x3 (t2_030..t2_001),
        tier3: 10x2 (t3_020..t3_001) -- matching the physical Splendor
        deck counts, in REVERSE id order (row-major: t{tier}_040..031 is
        the first row, 030..021 the second, etc).

        Each cell shows that card's actual artwork (a tiny thumbnail via
        AssetLoader.card_image, same source as the main board's cards, no
        outline), skipped entirely (background shows through) once the
        card has been bought OR reserved (by ANY player -- a reserved
        card is off the shared board/deck the same as a bought one, even
        though it hasn't been paid for yet)."""
        if rect.width <= 0:
            return

        out_ids = {c.id for player in state.players for c in player.purchased_cards}
        out_ids.update(c.id for player in state.players for c in player.reserved_cards)

        cols = 10
        gap = 2
        cell_w = (rect.width - (cols - 1) * gap) / cols * 1.3  # 1.3x the base fit-to-panel size
        cell_h = cell_w * (CARD_SIZE[1] / CARD_SIZE[0])  # match real card aspect ratio

        x0 = rect.x
        y = rect.y
        self._text(screen, fonts["med"], "Card tracker:", (x0, y), color=TEXT_COLOR_ON_BG)
        y += 20

        for tier in (1, 2, 3):
            if y > rect.bottom:
                break
            cards = sorted(self.bundle.cards_by_tier[tier], key=lambda c: c.id, reverse=True)
            remaining = sum(1 for c in cards if c.id not in out_ids)
            label = f"Tier {tier} ({remaining}/{len(cards)} left)"
            self._text(screen, fonts["small"], label, (x0, y), color=_TIER_TEXT_RGB[tier])
            y += 14

            for i, card in enumerate(cards):
                if card.id in out_ids:
                    continue  # blank cell -- background already shows through
                row, col = divmod(i, cols)
                cell_rect = pygame.Rect(
                    round(x0 + col * (cell_w + gap)), round(y + row * (cell_h + gap)), round(cell_w), round(cell_h)
                )
                screen.blit(self.assets.card_image(card, cell_rect.size), cell_rect)

            rows = -(-len(cards) // cols)  # ceil division
            y += rows * (cell_h + gap) + 10

    def _draw_banner(self, screen, state: GameState, fonts) -> None:
        w, h = self.resolution

        if state.game_over:
            self._draw_game_over_overlay(screen, state, fonts)
        else:
            if state.final_round_started:
                self._text(screen, fonts["med"], "FINAL ROUND", (12, 4), color=FINAL_ROUND_COLOR)
            pending_text = None
            if state.phase == TurnPhase.TOKEN_RETURN:
                mover = state.player(state.current_player)
                excess = mover.total_tokens - self.config.max_tokens_held
                pending_text = f"PENDING: Player {state.current_player + 1} must return {excess} token(s)"
            elif state.phase == TurnPhase.NOBLE_CHOICE:
                pending_text = f"PENDING: Player {state.current_player + 1} must choose a noble"
            if pending_text:
                text_surf = fonts["med"].render(pending_text, True, (255, 255, 255))
                bg_rect = text_surf.get_rect()
                bg_rect.topright = (w - 12, 4)
                bg = pygame.Surface((bg_rect.width + 12, bg_rect.height + 8), pygame.SRCALPHA)
                bg.fill((150, 90, 20, 220))
                screen.blit(bg, (bg_rect.x - 6, bg_rect.y - 4))
                screen.blit(text_surf, bg_rect)

    def _draw_game_over_overlay(self, screen, state: GameState, fonts) -> None:
        from splendor_ai.game.rules import compute_ranking

        w, h = self.resolution
        ranking = compute_ranking(state)
        overlay_h = 110 + 22 * len(ranking)
        overlay = pygame.Surface((w, overlay_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 190))
        top = h // 2 - overlay_h // 2
        screen.blit(overlay, (0, top))

        if len(state.winners) == 1:
            header = f"GAME OVER -- Winner: Player {state.winners[0] + 1}"
        else:
            names = ", ".join(f"Player {s + 1}" for s in state.winners)
            header = f"GAME OVER -- Tie between: {names}"
        surf = fonts["big"].render(header, True, (255, 255, 255))
        screen.blit(surf, surf.get_rect(center=(w // 2, top + 30)))

        y = top + 65
        for rank, group in enumerate(ranking, start=1):
            names = ", ".join(f"Player {s + 1} (prestige={state.player(s).prestige}, cards={state.player(s).purchased_card_count})" for s in group)
            line = f"#{rank}: {names}"
            line_surf = fonts["small"].render(line, True, (240, 240, 240))
            screen.blit(line_surf, line_surf.get_rect(center=(w // 2, y)))
            y += 22
