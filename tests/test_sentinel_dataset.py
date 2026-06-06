"""Tests for learned_ai/sentinel/dataset.py (loads from data/games)."""

from __future__ import annotations

import os

import numpy as np

from learned_ai.sentinel.dataset import (
    SentinelDataset,
    examples_from_game,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.sentinel.feature_builder import FEATURE_DIM

_GAME_DIR = "data/games"


def _have_games():
    return os.path.isdir(_GAME_DIR) and any(
        f.endswith(".jsonl") for f in os.listdir(_GAME_DIR)
    )


def test_load_from_games_no_crash():
    assert _have_games(), "expected game logs in data/games"
    ds = SentinelDataset.load_from_games(_GAME_DIR, db=ExternalSolvedDB(""), limit=20)
    assert len(ds) > 0


def test_item_shape_and_targets():
    ds = SentinelDataset.load_from_games(_GAME_DIR, db=ExternalSolvedDB(""), limit=10)
    feat, label = ds[0]
    assert tuple(feat.shape) == (FEATURE_DIM,)
    assert set(label.keys()) == {
        "mistake_risk", "opportunity_score",
        "trajectory_value_delta", "turning_point_confidence", "weight",
    }


def test_dataset_length_positive_per_game():
    # A handful of games should each yield at least one example.
    ds = SentinelDataset.load_from_games(_GAME_DIR, db=ExternalSolvedDB(""), limit=5)
    assert len(ds) >= 5


def test_save_load_roundtrip(tmp_path):
    ds = SentinelDataset.load_from_games(_GAME_DIR, db=ExternalSolvedDB(""), limit=10)
    path = str(tmp_path / "ds.npz")
    ds.save_to_disk(path)
    ds2 = SentinelDataset.load_from_disk(path)
    assert len(ds2) == len(ds)
    f1, l1 = ds[0]
    f2, l2 = ds2[0]
    assert np.allclose(np.asarray(f1), np.asarray(f2), atol=1e-6)
    for k in l1:
        assert abs(float(l1[k]) - float(l2[k])) < 1e-5


def test_class_distribution_has_multiple_types():
    # With single-candidate game logs and no external DB, the labeller produces
    # proxy-driven categories. We require at least three distinct label types so
    # the dataset is not degenerate. (missed_opportunity needs multi-candidate
    # enriched logs and is exercised in test_sentinel_labels.py.)
    ds = SentinelDataset.load_from_games(_GAME_DIR, db=ExternalSolvedDB(""), limit=60)
    dist = ds.class_distribution()
    assert len(dist) >= 3
    assert sum(dist.values()) == len(ds)


def test_examples_from_game_handles_empty():
    assert examples_from_game({"moves": []}) == []
    assert examples_from_game({}) == []
