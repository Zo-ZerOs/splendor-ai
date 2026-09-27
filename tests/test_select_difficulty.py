"""Tests the Easy/Medium/Expert SELECTION LOGIC using synthetic win-rate
data (mocked evaluation), so this doesn't require real trained checkpoints
or long-running evaluation -- it only checks that select_difficulty_tiers()
picks the right checkpoint given known numbers.
"""
import os

import pytest

from splendor_ai.training import select_difficulty as sd

DIFF_CFG = {
    "tiers": ["easy", "medium", "expert"],
    "easy": {"min_win_rate_vs_random": 0.40, "min_win_rate_vs_heuristic": 0.10, "max_win_rate_vs_next_tier": 0.45},
    "medium": {"min_win_rate_vs_random": 0.60, "min_win_rate_vs_heuristic": 0.25, "max_win_rate_vs_next_tier": 0.45},
    "expert": {
        "min_win_rate_vs_random": 0.80,
        "composite_weight_vs_random": 0.3,
        "composite_weight_vs_heuristic": 0.7,
        "stability_check_games": 50,
        "stability_tolerance": 0.10,
    },
}

FAKE_RESULTS = {
    5: {"vs_random": {"win_rate": 0.45}, "vs_heuristic": {"win_rate": 0.15}},  # qualifies for easy only
    10: {"vs_random": {"win_rate": 0.55}, "vs_heuristic": {"win_rate": 0.20}},  # below medium floor, above easy floor
    20: {"vs_random": {"win_rate": 0.65}, "vs_heuristic": {"win_rate": 0.30}},  # qualifies for medium
    30: {"vs_random": {"win_rate": 0.85}, "vs_heuristic": {"win_rate": 0.55}},  # best composite -> expert
    40: {"vs_random": {"win_rate": 0.82}, "vs_heuristic": {"win_rate": 0.50}},  # clears floor, lower composite
}


def _make_run_dir(tmp_path, results=None):
    results = results or FAKE_RESULTS
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for update in results:
        (run_dir / f"update_{update}.pt").write_bytes(b"")
    return str(run_dir)


def _fake_evaluate_checkpoint(bundle, config, path, games, seed, device="cpu"):
    update = int(os.path.basename(path).split("_")[1].split(".")[0])
    return dict(FAKE_RESULTS[update])


def test_expert_is_highest_composite_above_floor(tmp_path, bundle, config, monkeypatch):
    run_dir = _make_run_dir(tmp_path)
    monkeypatch.setattr(sd, "evaluate_matchup", lambda *a, **k: _FakeMatchup(0.30))

    result = sd.select_difficulty_tiers(
        bundle, config, run_dir, DIFF_CFG, games=10, evaluate_checkpoint=_fake_evaluate_checkpoint, log=lambda *_: None,
    )

    assert result["expert"]["update"] == 30
    assert result["expert"]["stable"] is True


def test_medium_is_earliest_qualifying_and_beatable_by_expert(tmp_path, bundle, config, monkeypatch):
    run_dir = _make_run_dir(tmp_path)
    monkeypatch.setattr(sd, "evaluate_matchup", lambda *a, **k: _FakeMatchup(0.30))  # loses to the tier above 70% of the time

    result = sd.select_difficulty_tiers(
        bundle, config, run_dir, DIFF_CFG, games=10, evaluate_checkpoint=_fake_evaluate_checkpoint, log=lambda *_: None,
    )

    assert result["medium"]["update"] == 20  # earliest checkpoint clearing medium's own floors (excluding expert=30)
    assert result["medium"]["vs_expert_win_rate"] == 0.30


def test_easy_is_earliest_qualifying_and_beatable_by_medium(tmp_path, bundle, config, monkeypatch):
    run_dir = _make_run_dir(tmp_path)
    monkeypatch.setattr(sd, "evaluate_matchup", lambda *a, **k: _FakeMatchup(0.30))

    result = sd.select_difficulty_tiers(
        bundle, config, run_dir, DIFF_CFG, games=10, evaluate_checkpoint=_fake_evaluate_checkpoint, log=lambda *_: None,
    )

    assert result["easy"]["update"] == 5  # earliest checkpoint clearing easy's own floors
    assert result["easy"]["vs_medium_win_rate"] == 0.30


def test_medium_falls_back_with_warning_when_nothing_is_beatable(tmp_path, bundle, config, monkeypatch):
    run_dir = _make_run_dir(tmp_path)
    # Every medium candidate beats the expert most of the time -- no clean choice exists.
    monkeypatch.setattr(sd, "evaluate_matchup", lambda *a, **k: _FakeMatchup(0.90))

    logs = []
    result = sd.select_difficulty_tiers(
        bundle, config, run_dir, DIFF_CFG, games=10, evaluate_checkpoint=_fake_evaluate_checkpoint, log=logs.append,
    )

    assert result["medium"]["update"] == 20  # fallback: earliest qualifying by its own floors alone
    assert any("WARNING" in line for line in logs)


def test_medium_heuristic_floor_skips_an_early_checkpoint_that_only_beats_random(tmp_path, bundle, config, monkeypatch):
    """Regression test: a self-play policy typically crushes random almost
    immediately, so a checkpoint can clear the random floor while still
    being barely-trained (very low win rate vs heuristic). Without a
    second floor on vs_heuristic, 'earliest qualifying' would pick that
    undertrained checkpoint instead of a genuinely intermediate one."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = {
        5: {"vs_random": {"win_rate": 0.90}, "vs_heuristic": {"win_rate": 0.15}},  # clears random floor, NOT heuristic floor
        20: {"vs_random": {"win_rate": 0.65}, "vs_heuristic": {"win_rate": 0.35}},  # clears both
        30: {"vs_random": {"win_rate": 0.85}, "vs_heuristic": {"win_rate": 0.55}},  # best composite -> expert
    }
    for update in results:
        (run_dir / f"update_{update}.pt").write_bytes(b"")

    def fake_eval(bundle, config, path, games, seed, device="cpu"):
        update = int(os.path.basename(path).split("_")[1].split(".")[0])
        return dict(results[update])

    cfg_with_heuristic_floor = {
        "tiers": ["easy", "medium", "expert"],
        "easy": {"min_win_rate_vs_random": 0.0, "min_win_rate_vs_heuristic": 0.0, "max_win_rate_vs_next_tier": 1.0},
        "medium": {"min_win_rate_vs_random": 0.60, "min_win_rate_vs_heuristic": 0.30, "max_win_rate_vs_next_tier": 0.45},
        "expert": DIFF_CFG["expert"],
    }
    monkeypatch.setattr(sd, "evaluate_matchup", lambda *a, **k: _FakeMatchup(0.20))

    result = sd.select_difficulty_tiers(
        bundle, config, str(run_dir), cfg_with_heuristic_floor, games=10, evaluate_checkpoint=fake_eval, log=lambda *_: None,
    )

    assert result["medium"]["update"] == 20  # NOT update 5, despite it also clearing the random-only floor


class _FakeMatchup:
    def __init__(self, win_rate):
        self.win_rate = win_rate
