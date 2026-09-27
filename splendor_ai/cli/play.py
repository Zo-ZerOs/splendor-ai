"""Entry point: one human (via terminal) plays against three AI seats,
with an optional read-only Pygame window mirroring the board.

    python -m splendor_ai.cli.play
    python -m splendor_ai.cli.play --difficulty easy
    python -m splendor_ai.cli.play --difficulty medium
    python -m splendor_ai.cli.play --difficulty expert
    python -m splendor_ai.cli.play --difficulty medium --mixed 2=expert   # mixed difficulty: seat 3 (index 2) is Expert
    python -m splendor_ai.cli.play --human-seat 2 --seed 123
    python -m splendor_ai.cli.play --no-render
    python -m splendor_ai.cli.play --checkpoint checkpoints/best.pt       # explicit override, bypasses --difficulty

Training never imports this module, and this module never imports any
training/RL code at module scope -- checkpoint loading (Phase 6) is
deferred to inside main() so headless/test environments that never pass
--checkpoint/--difficulty don't need torch installed for the CLI to work.

Type 'again' (or 'restart'/'new') at any time -- during a game or at game
over -- to abandon/finish the current game and start a new one at the
same difficulty (a fresh shuffled board, and a fresh random seat unless
--human-seat was given explicitly). Type 'quit' to exit.
"""
from __future__ import annotations

import argparse
import logging
import os
import random

from splendor_ai.agents.random_agent import RandomAgent
from splendor_ai.cli import render
from splendor_ai.cli.explain import explain_illegal_reason
from splendor_ai.cli.human_input import HELP_TEXT, format_action, parse_command
from splendor_ai.config import DEFAULT_ROOT, GameConfig
from splendor_ai.data.loader import load_all
from splendor_ai.game.engine import IllegalActionError, apply_action, setup_game
from splendor_ai.game.rules import legal_actions
from splendor_ai.game.state import TurnPhase

logger = logging.getLogger("splendor_ai.cli")

# Ordered weakest to strongest. "expert" is always the current best
# checkpoint from the most recent training; as training improves, older
# tiers get demoted down this list (e.g. today's expert becomes tomorrow's
# medium) rather than overwritten -- see splendor_ai/training/select_difficulty.py.
DIFFICULTIES = ("easy", "medium", "expert")


class QuitRequested(Exception):
    pass


class RestartRequested(Exception):
    """Raised when the human types 'again'/'restart'/'new' mid-game --
    abandons the current game immediately in favor of a fresh one at the
    same difficulty, without waiting for it to reach game over."""


def difficulty_checkpoint_path(difficulty: str, root: str = DEFAULT_ROOT) -> str:
    return os.path.join(root, "checkpoints", difficulty, "best.pt")


def parse_mixed_overrides(mixed_args, num_players: int) -> dict:
    overrides = {}
    for item in mixed_args:
        if "=" not in item:
            raise SystemExit(f"--mixed expects SEAT=DIFFICULTY (e.g. --mixed 2=expert), got '{item}'")
        seat_str, difficulty = item.split("=", 1)
        try:
            seat = int(seat_str)
        except ValueError:
            raise SystemExit(f"--mixed seat must be an integer, got '{seat_str}'")
        if not (0 <= seat < num_players):
            raise SystemExit(f"--mixed seat {seat} out of range [0, {num_players - 1}]")
        if difficulty not in DIFFICULTIES:
            raise SystemExit(f"--mixed difficulty must be one of {DIFFICULTIES}, got '{difficulty}'")
        overrides[seat] = difficulty
    return overrides


def build_agents(config: GameConfig, human_seat: int, seed, checkpoint_path, difficulty=None, mixed_overrides=None):
    """Resolution order per AI seat: --mixed override > --difficulty >
    --checkpoint (applied to every AI seat) > RandomAgent fallback."""
    mixed_overrides = mixed_overrides or {}
    agents = [None] * config.num_players

    for seat in range(config.num_players):
        if seat == human_seat:
            continue

        seat_difficulty = mixed_overrides.get(seat, difficulty)
        path = None
        if seat_difficulty:
            path = difficulty_checkpoint_path(seat_difficulty)
            if not os.path.isfile(path):
                raise SystemExit(
                    f"No '{seat_difficulty}' checkpoint found at {path}. "
                    "Run training then `python -m splendor_ai.training.select_difficulty` first, "
                    "or pass --checkpoint to use a specific checkpoint file directly."
                )
        elif checkpoint_path:
            path = checkpoint_path

        if path:
            from splendor_ai.agents.rl_agent import RLAgent  # deferred: needs torch

            label = f"AI-{seat + 1} ({seat_difficulty or 'checkpoint'})"
            agents[seat] = RLAgent.load(path, seat_name=label)
        else:
            agents[seat] = RandomAgent(seed=None if seed is None else seed + seat)
    return agents


def _format_requirements_prose(reqs) -> str:
    parts = [f"{n} {c.value}" for c, n in reqs.items() if n > 0]
    if not parts:
        return "nothing"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _print_pending_prompt(state, config, human_seat: int) -> None:
    phase = state.phase
    if phase == TurnPhase.MAIN_ACTION:
        print()
        print(render.format_state_summary(state, config, human_seat))
        return

    if phase == TurnPhase.TOKEN_RETURN:
        player = state.player(human_seat)
        excess = player.total_tokens - config.max_tokens_held
        plural = "" if excess == 1 else "s"
        print(f"\nYou now have {player.total_tokens} tokens. The maximum is {config.max_tokens_held}.")
        print(f"You must return exactly {excess} token{plural}.")
        print("\nYour tokens:")
        for color, n in player.tokens.items():
            print(f"  {color.value}: {n}")
        print("\nEnter:")
        print("  RETURN <color> [<color> ...]")
        print("\nExample:")
        example_color = next((c.value for c, n in player.tokens.items() if n > 0), "white")
        print(f"  RETURN {example_color}")

    elif phase == TurnPhase.NOBLE_CHOICE:
        print("\nYou qualify for multiple nobles. Choose one:")
        for i, nid in enumerate(state.pending_eligible_nobles):
            noble = next(n for n in state.board.nobles if n.id == nid)
            print(f"  N{i + 1}: {noble.points} points -- requires {_format_requirements_prose(noble.requirements)}")
        print("\nEnter:")
        print("  NOBLE <n>")
        print("\nExample:")
        print("  NOBLE 1")


def _human_turn(state, config, human_seat: int) -> None:
    _print_pending_prompt(state, config, human_seat)

    while True:
        try:
            raw = input(f"Player {human_seat + 1}> ")
        except EOFError:
            raise QuitRequested()

        parsed = parse_command(raw, state, config)
        if parsed.kind == "quit":
            raise QuitRequested()
        if parsed.kind == "again":
            raise RestartRequested()
        if parsed.kind == "help":
            print(HELP_TEXT)
            continue
        if parsed.kind == "state":
            print(render.format_state_summary(state, config, human_seat))
            continue
        if parsed.kind == "legal":
            _print_legal(state, config)
            continue
        if parsed.kind == "history":
            print(render.format_history(state))
            continue
        if parsed.kind == "error":
            print(f"\n{parsed.error}")
            print("\nEnter another action:")
            continue

        try:
            apply_action(state, config, parsed.action)
        except IllegalActionError:
            reason = explain_illegal_reason(parsed.action, state, config)
            print(f"\nILLEGAL ACTION: {reason}")
            print("\nYour turn has not been consumed.")
            print('Enter another action or type "legal":')
            continue
        return


def _print_legal(state, config) -> None:
    actions = legal_actions(state, config)
    print("Legal commands:")
    for a in actions:
        print("  " + format_action(a, state))


def _ai_turn(state, config, agent, seat: int) -> None:
    action = agent.act(state, config)
    command_str = format_action(action, state)
    explanation = agent.explain(state, config, action)
    apply_action(state, config, action)
    print(f"Player {seat + 1} (AI): {command_str}   -- {explanation}")


def _difficulty_label(difficulty, mixed_overrides, checkpoint_path) -> str:
    """Short label shown in the viewer's top banner describing what the 3
    AI seats are actually playing at, since that isn't otherwise visible
    from the board itself."""
    if checkpoint_path:
        return "custom checkpoint"
    if mixed_overrides:
        return "mixed"
    if difficulty:
        return difficulty
    return "random"


def run(seed, human_seat, checkpoint_path, render_enabled: bool, resolution, difficulty=None, mixed=None) -> None:
    bundle = load_all()
    config = GameConfig.load()

    # human_seat given explicitly (--human-seat) stays fixed across replays;
    # otherwise every game -- including each "again" replay -- gets its own
    # fresh random seat, not just a fresh board.
    fixed_human_seat = human_seat
    mixed_overrides = parse_mixed_overrides(mixed or [], config.num_players)
    current_seed = seed

    view = None
    if render_enabled:
        from splendor_ai.ui.pygame_view import SplendorView

        label = _difficulty_label(difficulty, mixed_overrides, checkpoint_path)
        view = SplendorView(bundle, config, human_seat=fixed_human_seat or 0, resolution=resolution, difficulty_label=label)
        view.start()

    while True:
        this_human_seat = fixed_human_seat
        if this_human_seat is None:
            # Derived from `current_seed` when given, so --seed reproduces
            # both the board layout AND which seat the human sits in for the
            # first game; replays pass current_seed=None here so this is
            # freshly randomized every time, same as the board itself.
            this_human_seat = random.Random(current_seed).randrange(config.num_players)
        if not (0 <= this_human_seat < config.num_players):
            raise SystemExit(f"--human-seat must be in [0, {config.num_players - 1}]")

        agents = build_agents(config, this_human_seat, current_seed, checkpoint_path, difficulty=difficulty, mixed_overrides=mixed_overrides)
        state = setup_game(bundle, config, seed=current_seed, starting_player=0)

        if view is not None:
            view.human_seat = this_human_seat
            view.update(state)

        print(f"You are Player {this_human_seat + 1}. Type 'help' for commands.")
        for seat, agent in enumerate(agents):
            if agent is not None:
                print(f"  Player {seat + 1}: {agent.name}")
        print(render.format_state_summary(state, config, this_human_seat))

        view_close_notified = False
        quit_requested = False
        restart_requested = False
        try:
            while not state.game_over:
                seat = state.current_player
                if seat == this_human_seat:
                    _human_turn(state, config, this_human_seat)
                else:
                    _ai_turn(state, config, agents[seat], seat)
                if view is not None:
                    view.update(state)
                    if view.closed and not view_close_notified:
                        print("\n[Viewer window closed. The game continues here in the terminal only.]")
                        view_close_notified = True
        except QuitRequested:
            quit_requested = True
        except RestartRequested:
            restart_requested = True

        if view is not None:
            view.update(state)

        if quit_requested:
            break

        if restart_requested:
            print("\nAbandoning this game and starting a new one...")
            current_seed = None
            continue

        print()
        print(render.format_winner(state))

        # The window (if any) stays open showing the final board/ranking
        # until the human explicitly responds here -- it must not vanish
        # the instant the game ends.
        if view is not None and view.closed:
            break  # nothing left to prompt on -- the window is already gone

        if view is not None:
            print("\nGame over. Type 'again' to play a new game (same difficulty), or 'quit' to close the viewer and exit.")
        else:
            print("\nGame over. Type 'again' to play a new game (same difficulty), or 'quit' to exit.")

        play_again = False
        while view is None or not view.closed:
            try:
                raw = input("> ").strip().lower()
            except EOFError:
                break
            if raw in ("quit", "exit", "q"):
                break
            if raw in ("again", "restart", "new"):
                play_again = True
                break
            print("Please type 'again' or 'quit'.")

        if not play_again:
            break

        # A fresh board and (if not explicitly fixed) a fresh seat every
        # replay -- never silently repeat the same game.
        current_seed = None

    print("Exiting.")
    if view is not None:
        view.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Splendor: one human (terminal) vs three AI seats.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deck/noble shuffling.")
    parser.add_argument("--human-seat", type=int, default=None, help="0-indexed human seat (default from configs/game.yaml).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a specific trained RL checkpoint for all 3 AI seats (bypasses --difficulty). Falls back to random agents if nothing is given.")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default=None, help="Difficulty for all 3 AI seats by default -- resolves to checkpoints/<difficulty>/best.pt.")
    parser.add_argument("--mixed", action="append", default=[], metavar="SEAT=DIFFICULTY", help="Override one AI seat's difficulty (0-indexed), e.g. --mixed 2=expert. Repeatable.")
    parser.add_argument("--render", dest="render", action="store_true", default=True, help="Show the read-only Pygame window (default on).")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Disable the graphical window.")
    parser.add_argument("--resolution", type=str, default="1280x860", help="Viewer resolution as WxH, e.g. 1280x860.")
    args = parser.parse_args()

    try:
        w, h = (int(x) for x in args.resolution.lower().split("x"))
    except ValueError:
        raise SystemExit("--resolution must look like WIDTHxHEIGHT, e.g. 1280x860")

    run(args.seed, args.human_seat, args.checkpoint, args.render, (w, h), difficulty=args.difficulty, mixed=args.mixed)


if __name__ == "__main__":
    main()
