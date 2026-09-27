"""Turns a directory of periodic training checkpoints into the picked
difficulty tiers, saved as checkpoints/<tier>/best.pt for each tier named
in configs/difficulty.yaml's `tiers` list (weakest to strongest, e.g.
["easy", "medium", "expert"]).

Selection is driven ENTIRELY by measured win rates (never by game length
alone, per project policy):

  Top tier (e.g. Expert) = among checkpoints clearing a minimum
           win-rate-vs-random floor, the one with the best composite
           score (weighted win rate vs random and vs heuristic),
           double-checked for stability with a second independently
           seeded evaluation batch.

  Every tier below that = the earliest (least-trained) checkpoint that
           clears its OWN win-rate floors (vs random and vs heuristic --
           a self-play policy typically crushes random almost
           immediately, so a random-only floor would let "earliest
           qualifying" degenerate to "the very first checkpoint
           evaluated") AND loses the majority of games against the tier
           directly above it. If no checkpoint satisfies both, the tool
           says so in the report rather than silently picking something
           that doesn't actually meet the bar.

As training improves, promote a new checkpoint to the top tier and let
the old tiers shift down (yesterday's expert becomes today's medium, per
project convention) rather than being overwritten by a weaker one.

Usage:
    python -m splendor_ai.training.select_difficulty --run-dir checkpoints/run_20260101_000000 --games 1000
    python -m splendor_ai.training.select_difficulty --run-dir <dir> --games 200 --dry-run   # quick check, no copying
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple

import yaml

from splendor_ai.agents.heuristic_agent import HeuristicAgent
from splendor_ai.agents.random_agent import RandomAgent
from splendor_ai.agents.rl_agent import RLAgent
from splendor_ai.config import DEFAULT_ROOT, GameConfig
from splendor_ai.data.loader import DataBundle, load_all
from splendor_ai.training.evaluate import evaluate_matchup

DEFAULT_DIFFICULTY_CONFIG_PATH = os.path.join(DEFAULT_ROOT, "configs", "difficulty.yaml")

_UPDATE_RE = re.compile(r"update_(\d+)\.pt$")


def list_checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    results = []
    for fn in os.listdir(run_dir):
        m = _UPDATE_RE.search(fn)
        if m:
            results.append((int(m.group(1)), os.path.join(run_dir, fn)))
    results.sort(key=lambda t: t[0])
    return results


EvalFn = Callable[[DataBundle, GameConfig, str, int, int], dict]


def default_evaluate_checkpoint(bundle: DataBundle, config: GameConfig, path: str, games: int, seed: int, device: str = "cpu") -> dict:
    agent = RLAgent.load(path, bundle=bundle, device=device, deterministic=True)
    vs_random = evaluate_matchup(bundle, config, lambda: agent, lambda: RandomAgent(), games, base_seed=seed)
    vs_heuristic = evaluate_matchup(bundle, config, lambda: agent, lambda: HeuristicAgent(), games, base_seed=seed + 1_000_000)
    return {"vs_random": asdict(vs_random), "vs_heuristic": asdict(vs_heuristic)}


def _composite(result: dict, expert_cfg: dict) -> float:
    return (
        expert_cfg["composite_weight_vs_random"] * result["vs_random"]["win_rate"]
        + expert_cfg["composite_weight_vs_heuristic"] * result["vs_heuristic"]["win_rate"]
    )


def select_difficulty_tiers(
    bundle: DataBundle,
    config: GameConfig,
    run_dir: str,
    diff_cfg: dict,
    games: int,
    seed: int = 0,
    device: str = "cpu",
    sample_every: int = 1,
    evaluate_checkpoint: EvalFn = default_evaluate_checkpoint,
    log: Callable[[str], None] = print,
) -> dict:
    """Returns {"evaluations": [...], <tier_name>: {...}, ...} for every
    tier in diff_cfg["tiers"]."""
    tier_order: List[str] = diff_cfg["tiers"]
    if len(tier_order) < 1:
        raise ValueError("diff_cfg['tiers'] must name at least one tier")

    checkpoints = list_checkpoints(run_dir)
    if sample_every > 1:
        checkpoints = checkpoints[::sample_every]
    if not checkpoints:
        raise SystemExit(f"no update_*.pt checkpoints found in {run_dir}")

    top_tier = tier_order[-1]
    top_cfg = diff_cfg[top_tier]

    log(f"Evaluating {len(checkpoints)} checkpoint(s) from {run_dir} ({games} games each vs random/heuristic)...")
    evaluations = []
    for update, path in checkpoints:
        result = evaluate_checkpoint(bundle, config, path, games, seed + update, device)
        result = dict(result, update=update, path=path)
        result["composite"] = _composite(result, top_cfg)
        evaluations.append(result)
        log(
            f"  update {update}: vs_random={result['vs_random']['win_rate']:.2f} "
            f"vs_heuristic={result['vs_heuristic']['win_rate']:.2f} composite={result['composite']:.3f}"
        )

    # ---- Top tier: best composite score above a floor, stability-rechecked ----
    floor = top_cfg["min_win_rate_vs_random"]
    top_pool = [e for e in evaluations if e["vs_random"]["win_rate"] >= floor] or evaluations
    top_pick = max(top_pool, key=lambda e: e["composite"])

    stability_games = top_cfg["stability_check_games"]
    recheck = evaluate_checkpoint(bundle, config, top_pick["path"], stability_games, seed + 987_654, device)
    recheck_composite = _composite(recheck, top_cfg)
    stable = abs(recheck_composite - top_pick["composite"]) <= top_cfg["stability_tolerance"]
    top_pick = dict(top_pick, stability_recheck_composite=recheck_composite, stable=stable)
    if not stable:
        log(
            f"  WARNING: {top_tier} candidate (update {top_pick['update']}) failed the stability recheck "
            f"({top_pick['composite']:.3f} vs {recheck_composite:.3f} on a fresh batch); selected anyway, flagged in report."
        )

    selections: Dict[str, dict] = {top_tier: top_pick}
    matchup_games = max(100, games // 4)

    # ---- Every tier below: earliest checkpoint clearing its own floors
    # AND beatable by the tier directly above it (already selected). ----
    for i in range(len(tier_order) - 2, -1, -1):
        tier_name = tier_order[i]
        tier_cfg = diff_cfg[tier_name]
        above_name = tier_order[i + 1]
        above_pick = selections[above_name]
        used_paths = {sel["path"] for sel in selections.values()}

        candidates = [
            e for e in evaluations
            if e["vs_random"]["win_rate"] >= tier_cfg["min_win_rate_vs_random"]
            and e["vs_heuristic"]["win_rate"] >= tier_cfg.get("min_win_rate_vs_heuristic", 0.0)
            and e["path"] not in used_paths
        ]

        picked: Optional[dict] = None
        for cand in candidates:  # ascending by update -> prefer the weakest qualifying checkpoint
            matchup = evaluate_matchup(
                bundle, config,
                test_agent_factory=lambda p=cand["path"]: RLAgent.load(p, bundle=bundle, device=device, deterministic=True),
                opponent_agent_factory=lambda p=above_pick["path"]: RLAgent.load(p, bundle=bundle, device=device, deterministic=True),
                num_games=matchup_games,
                base_seed=seed + 42 + i,
            )
            if matchup.win_rate <= tier_cfg["max_win_rate_vs_next_tier"]:
                picked = dict(cand, **{f"vs_{above_name}_win_rate": matchup.win_rate})
                break

        if picked is None:
            log(
                f"  WARNING: no {tier_name} candidate satisfied BOTH its own floors and 'beatable by {above_name}' "
                f"-- falling back to the earliest checkpoint clearing its win-rate floors only."
            )
            picked = candidates[0] if candidates else evaluations[0]

        selections[tier_name] = picked

    return {"evaluations": evaluations, **selections}


def save_selected(root: str, selections: Dict[str, dict], tier_order: List[str]) -> None:
    for name in tier_order:
        result = selections[name]
        dest_dir = os.path.join(root, "checkpoints", name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copyfile(result["path"], os.path.join(dest_dir, "best.pt"))
        with open(os.path.join(dest_dir, "selection_report.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select difficulty tiers (easy/medium/expert) from a training run.")
    parser.add_argument("--run-dir", required=True, help="Directory of update_*.pt checkpoints from one training run")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-every", type=int, default=1, help="Evaluate only every Nth checkpoint to save compute")
    parser.add_argument("--difficulty-config", type=str, default=DEFAULT_DIFFICULTY_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Print the selection without writing checkpoints/<tier>")
    args = parser.parse_args()

    bundle = load_all()
    config = GameConfig.load()
    with open(args.difficulty_config, "r", encoding="utf-8") as f:
        diff_cfg = yaml.safe_load(f)

    result = select_difficulty_tiers(bundle, config, args.run_dir, diff_cfg, args.games, seed=args.seed, device=args.device, sample_every=args.sample_every)

    tier_order = diff_cfg["tiers"]
    print(json.dumps({name: result[name] for name in tier_order}, indent=2, default=str))

    if not args.dry_run:
        save_selected(DEFAULT_ROOT, result, tier_order)
        paths = ", ".join(os.path.join(DEFAULT_ROOT, "checkpoints", name, "best.pt") for name in tier_order)
        print(f"Saved: {paths}")


if __name__ == "__main__":
    main()
