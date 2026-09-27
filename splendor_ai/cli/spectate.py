"""Entry point: watch all 4 seats played by AI, stepping one action at a
time, with an optional read-only Pygame window mirroring the board.

    python -m splendor_ai.cli.spectate
    python -m splendor_ai.cli.spectate --seed 123
    python -m splendor_ai.cli.spectate --difficulty expert
    python -m splendor_ai.cli.spectate --mixed 0=expert --mixed 2=easy
    python -m splendor_ai.cli.spectate --checkpoint checkpoints/best.pt
    python -m splendor_ai.cli.spectate --no-render

Type 'next' (or just press Enter) to execute the upcoming seat's action.
'state'/'history' inspect the board without advancing. 'again' abandons
the current game for a fresh one (a new random board, same as
splendor_ai.cli.play -- --seed only pins the FIRST game). 'quit' exits.
"""
from __future__ import annotations

import argparse

from splendor_ai.cli import render
from splendor_ai.cli.play import (
    DIFFICULTIES,
    QuitRequested,
    RestartRequested,
    _ai_turn,
    _difficulty_label,
    build_agents,
    parse_mixed_overrides,
)
from splendor_ai.config import GameConfig
from splendor_ai.data.loader import load_all
from splendor_ai.game.engine import setup_game
from splendor_ai.game.state import TurnPhase

HELP_TEXT = """Commands:
  next (or Enter)  -- execute the upcoming seat's action
  state            -- show the current board without advancing
  history          -- show the recent action log
  again            -- abandon this game, start a fresh one
  quit             -- exit
"""


def _phase_note(state) -> str:
    if state.phase == TurnPhase.TOKEN_RETURN:
        return " (returning excess tokens)"
    if state.phase == TurnPhase.NOBLE_CHOICE:
        return " (choosing a noble)"
    return ""


def run(seed, checkpoint_path, render_enabled: bool, resolution, difficulty=None, mixed=None) -> None:
    bundle = load_all()
    config = GameConfig.load()
    mixed_overrides = parse_mixed_overrides(mixed or [], config.num_players)
    current_seed = seed

    view = None
    if render_enabled:
        from splendor_ai.ui.pygame_view import SplendorView

        label = _difficulty_label(difficulty, mixed_overrides, checkpoint_path)
        view = SplendorView(bundle, config, human_seat=0, resolution=resolution, difficulty_label=label)
        view.start()

    while True:
        # human_seat=None -> build_agents gives every seat an AI (see its
        # docstring: it only skips `seat == human_seat`, and no real seat
        # index ever equals None).
        agents = build_agents(config, None, current_seed, checkpoint_path, difficulty=difficulty, mixed_overrides=mixed_overrides)
        state = setup_game(bundle, config, seed=current_seed, starting_player=0)

        if view is not None:
            view.update(state)

        print("Spectating -- all 4 seats are AI. Type 'help' for commands.")
        for seat, agent in enumerate(agents):
            print(f"  Player {seat + 1}: {agent.name}")
        print(render.format_state_summary(state, config, 0))

        view_close_notified = False
        quit_requested = False
        restart_requested = False
        try:
            while not state.game_over:
                seat = state.current_player
                print(f"\nUp next: Player {seat + 1} ({agents[seat].name}){_phase_note(state)}")
                try:
                    raw = input("> ").strip().lower()
                except EOFError:
                    raise QuitRequested()

                if raw in ("quit", "exit", "q"):
                    raise QuitRequested()
                if raw in ("again", "restart", "new"):
                    raise RestartRequested()
                if raw in ("help", "?"):
                    print(HELP_TEXT)
                    continue
                if raw in ("state", "board"):
                    print(render.format_state_summary(state, config, seat))
                    continue
                if raw in ("history", "log"):
                    print(render.format_history(state))
                    continue
                if raw not in ("next", "n", ""):
                    print("Type 'next' to advance (or 'help' for other commands).")
                    continue

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
        # until explicitly responded to here -- it must not vanish the
        # instant the game ends.
        if view is not None and view.closed:
            break

        if view is not None:
            print("\nGame over. Type 'again' to watch a new game, or 'quit' to close the viewer and exit.")
        else:
            print("\nGame over. Type 'again' to watch a new game, or 'quit' to exit.")

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

        # A fresh board every replay -- never silently repeat the same game.
        current_seed = None

    print("Exiting.")
    if view is not None:
        view.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch 4 AI seats play Splendor against each other, one action at a time.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deck/noble shuffling (pins the first game's board).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a specific trained RL checkpoint for all 4 seats (bypasses --difficulty). Falls back to random agents if nothing is given.")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default=None, help="Difficulty for all 4 seats by default -- resolves to checkpoints/<difficulty>/best.pt.")
    parser.add_argument("--mixed", action="append", default=[], metavar="SEAT=DIFFICULTY", help="Override one seat's difficulty (0-indexed), e.g. --mixed 2=expert. Repeatable.")
    parser.add_argument("--render", dest="render", action="store_true", default=True, help="Show the read-only Pygame window (default on).")
    parser.add_argument("--no-render", dest="render", action="store_false", help="Disable the graphical window.")
    parser.add_argument("--resolution", type=str, default="1280x860", help="Viewer resolution as WxH, e.g. 1280x860.")
    args = parser.parse_args()

    try:
        w, h = (int(x) for x in args.resolution.lower().split("x"))
    except ValueError:
        raise SystemExit("--resolution must look like WIDTHxHEIGHT, e.g. 1280x860")

    run(args.seed, args.checkpoint, args.render, (w, h), difficulty=args.difficulty, mixed=args.mixed)


if __name__ == "__main__":
    main()
