import hashlib
import json
import os

import pytest

from splendor_ai.data.loader import DEFAULT_ROOT, DataValidationError, load_all
from splendor_ai.data.schemas import Color


def _hash_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_load_all_does_not_modify_source_files():
    paths = [
        os.path.join(DEFAULT_ROOT, "cards", "cards.json"),
        os.path.join(DEFAULT_ROOT, "coins", "coins.json"),
        os.path.join(DEFAULT_ROOT, "nobles", "nobles.json"),
    ]
    before = {p: _hash_file(p) for p in paths}
    load_all()
    after = {p: _hash_file(p) for p in paths}
    assert before == after


def test_load_all_counts(bundle):
    assert len(bundle.cards) == 90
    assert len(bundle.cards_by_tier[1]) == 40
    assert len(bundle.cards_by_tier[2]) == 30
    assert len(bundle.cards_by_tier[3]) == 20
    assert len(bundle.nobles) == 10
    assert len(bundle.coins) == 6


def test_coin_colors(bundle):
    colors = {c.color for c in bundle.coins}
    assert colors == set(Color)
    gold = bundle.coin_by_color[Color.GOLD]
    assert gold.is_joker is True
    for c in bundle.coins:
        if c.color != Color.GOLD:
            assert c.is_joker is False


def test_image_paths_resolve_to_real_files(bundle):
    for card in bundle.cards:
        assert os.path.isfile(card.image_abs), f"missing image for {card.id}: {card.image_abs}"
    for noble in bundle.nobles:
        assert os.path.isfile(noble.image_abs), f"missing image for {noble.id}: {noble.image_abs}"
    for coin in bundle.coins:
        assert os.path.isfile(coin.image_abs), f"missing image for {coin.color}: {coin.image_abs}"


def test_no_duplicate_ids(bundle):
    ids = [c.id for c in bundle.cards]
    assert len(ids) == len(set(ids))
    nids = [n.id for n in bundle.nobles]
    assert len(nids) == len(set(nids))


def test_strict_loader_rejects_missing_field(tmp_path):
    root = tmp_path
    _write_fixture(root, cards=[{"tier": 1, "bonus": "white", "points": 0, "cost": _zero_cost(), "image": "tier1/x.png"}])
    with pytest.raises(DataValidationError):
        load_all(str(root))


def test_strict_loader_rejects_invalid_tier(tmp_path):
    root = tmp_path
    _write_fixture(root, cards=[_card(tier=4)])
    with pytest.raises(DataValidationError):
        load_all(str(root))


def test_strict_loader_rejects_invalid_color(tmp_path):
    root = tmp_path
    _write_fixture(root, cards=[_card(bonus="purple")])
    with pytest.raises(DataValidationError):
        load_all(str(root))


def test_strict_loader_rejects_negative_cost(tmp_path):
    root = tmp_path
    card = _card()
    card["cost"]["white"] = -1
    _write_fixture(root, cards=[card])
    with pytest.raises(DataValidationError):
        load_all(str(root))


def test_strict_loader_rejects_duplicate_ids(tmp_path):
    root = tmp_path
    _write_fixture(root, cards=[_card(id="dup"), _card(id="dup")])
    with pytest.raises(DataValidationError):
        load_all(str(root))


def test_strict_loader_allows_missing_image_file(tmp_path):
    """Missing images are a presentation concern, not a fatal data error."""
    root = tmp_path
    _write_fixture(root, cards=[_card(image="tier1/does_not_exist.png")])
    bundle = load_all(str(root))
    assert len(bundle.cards) == 1
    assert not os.path.isfile(bundle.cards[0].image_abs)


def _zero_cost():
    return {"white": 0, "blue": 0, "green": 0, "red": 0, "black": 0}


def _card(id="t1_001", tier=1, bonus="white", points=0, image="tier1/t1_001.png"):
    return {"id": id, "tier": tier, "bonus": bonus, "points": points, "cost": _zero_cost(), "image": image}


def _write_fixture(root, cards=None, nobles=None, coins=None):
    os.makedirs(os.path.join(root, "cards", "tier1"), exist_ok=True)
    os.makedirs(os.path.join(root, "nobles", "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "coins", "images"), exist_ok=True)

    with open(os.path.join(root, "cards", "cards.json"), "w") as f:
        json.dump(cards if cards is not None else [], f)
    with open(os.path.join(root, "nobles", "nobles.json"), "w") as f:
        json.dump(nobles if nobles is not None else [], f)
    with open(os.path.join(root, "coins", "coins.json"), "w") as f:
        default_coins = [
            {"color": "white", "is_joker": False, "image": "images/white.png"},
            {"color": "blue", "is_joker": False, "image": "images/blue.png"},
            {"color": "green", "is_joker": False, "image": "images/green.png"},
            {"color": "red", "is_joker": False, "image": "images/red.png"},
            {"color": "black", "is_joker": False, "image": "images/black.png"},
            {"color": "gold", "is_joker": True, "image": "images/gold.png"},
        ]
        json.dump(coins if coins is not None else default_coins, f)
