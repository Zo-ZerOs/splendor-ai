import os

import pytest

from splendor_ai.agents.random_agent import RandomAgent
from splendor_ai.cli import play


def test_parse_mixed_overrides_basic():
    overrides = play.parse_mixed_overrides(["1=expert", "3=medium"], num_players=4)
    assert overrides == {1: "expert", 3: "medium"}


def test_parse_mixed_overrides_invalid_format():
    with pytest.raises(SystemExit):
        play.parse_mixed_overrides(["not-valid"], num_players=4)


def test_parse_mixed_overrides_bad_seat_range():
    with pytest.raises(SystemExit):
        play.parse_mixed_overrides(["9=expert"], num_players=4)


def test_parse_mixed_overrides_bad_difficulty_name():
    with pytest.raises(SystemExit):
        play.parse_mixed_overrides(["1=super-hard"], num_players=4)


def test_build_agents_defaults_to_random_when_no_difficulty_or_checkpoint(bundle, config):
    agents = play.build_agents(config, human_seat=0, seed=1, checkpoint_path=None)
    for seat, agent in enumerate(agents):
        if seat == 0:
            assert agent is None
        else:
            assert isinstance(agent, RandomAgent)


def test_build_agents_missing_difficulty_checkpoint_raises_helpful_error(bundle, config, tmp_path, monkeypatch):
    # Point the difficulty->path resolver at an empty tmp dir so nothing exists.
    monkeypatch.setattr(play, "difficulty_checkpoint_path", lambda difficulty, root=str(tmp_path): str(tmp_path / difficulty / "best.pt"))
    with pytest.raises(SystemExit, match="No 'expert' checkpoint found"):
        play.build_agents(config, human_seat=0, seed=1, checkpoint_path=None, difficulty="expert")


def test_difficulty_checkpoint_path_resolution(tmp_path):
    path = play.difficulty_checkpoint_path("medium", root=str(tmp_path))
    assert os.path.join(str(tmp_path), "checkpoints", "medium", "best.pt") == path


def test_difficulty_label_defaults_to_random():
    assert play._difficulty_label(None, {}, None) == "random"


def test_difficulty_label_shows_uniform_difficulty():
    assert play._difficulty_label("expert", {}, None) == "expert"


def test_difficulty_label_shows_mixed():
    assert play._difficulty_label("medium", {2: "expert"}, None) == "mixed"


def test_difficulty_label_shows_custom_checkpoint():
    assert play._difficulty_label(None, {}, "checkpoints/some/run.pt") == "custom checkpoint"
