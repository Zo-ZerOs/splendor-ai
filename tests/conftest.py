import pytest

from splendor_ai.config import GameConfig
from splendor_ai.data.loader import load_all


@pytest.fixture(scope="session")
def bundle():
    return load_all()


@pytest.fixture(scope="session")
def config():
    return GameConfig.load()
