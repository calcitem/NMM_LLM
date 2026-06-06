"""Tests for learned_ai/sentinel/feature_builder.py."""

from __future__ import annotations

import numpy as np

from game.board import BoardState
from learned_ai.models.state_encoder import encode_state
from learned_ai.sentinel.feature_builder import (
    BASE_DIM,
    CONTEXT_DIM,
    FEATURE_DIM,
    build_features,
)


def _board():
    return BoardState.from_fen_string("BBW....B.W.W............|W|3|3")


def test_output_shape():
    feats = build_features(_board(), {})
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (FEATURE_DIM,)
    assert feats.dtype == np.float32
    assert FEATURE_DIM == 120
    assert BASE_DIM == 84
    assert CONTEXT_DIM == 36


def test_base_features_match_state_encoder():
    board = _board()
    feats = build_features(board, {})
    base = encode_state(board).detach().cpu().numpy()
    assert np.allclose(feats[:BASE_DIM], base, atol=1e-6)


def test_context_padding_zero_candidates():
    feats = build_features(_board(), {"candidates": []})
    # No candidates -> top-5 score slots and one-hots all zero.
    assert np.allclose(feats[BASE_DIM:BASE_DIM + 25], 0.0)
    # n_candidates_norm = 0
    assert feats[BASE_DIM + 34] == 0.0


def test_context_padding_one_candidate():
    ctx = {
        "candidates": [{"move": {"from": None, "to": "a4"}, "score": 1.0}],
        "chosen_rank": 0,
    }
    feats = build_features(_board(), ctx)
    # single candidate => rank feature is 0
    assert feats[BASE_DIM + 25] == 0.0
    # first candidate score slot is populated, the rest padded
    assert feats[BASE_DIM + 0] > 0.0
    assert np.allclose(feats[BASE_DIM + 1:BASE_DIM + 5], 0.0)


def test_context_values_normalised():
    ctx = {
        "candidates": [
            {"move": {"from": None, "to": "a4"}, "score": 12.0, "type": "place"},
            {"move": {"from": "a4", "to": "a1", "capture": "d1"}, "score": -5.0},
            {"move": {"from": "b2", "to": "b4"}, "score": 0.0, "type": "move"},
        ],
        "chosen_rank": 1,
        "closes_mill": True,
        "opens_mill_threat": False,
        "reduces_own_mobility": True,
        "trajectory_scores": [0.1, -0.2, 0.3, 0.4],
        "game_source": "human_vs_ai",
    }
    feats = build_features(_board(), ctx)
    ctxv = feats[BASE_DIM:]
    assert ctxv.shape == (CONTEXT_DIM,)
    # all context features bounded
    assert np.all(ctxv >= 0.0)
    assert np.all(ctxv <= 1.0)
    # flags reflect input
    assert ctxv[26] == 1.0   # closes_mill
    assert ctxv[27] == 0.0   # opens_mill_threat
    assert ctxv[28] == 1.0   # reduces_own_mobility
    assert ctxv[33] == 1.0   # human source
    # capture candidate (index 1) sets the capture one-hot
    assert ctxv[5 + 1 * 4 + 3] == 1.0


def test_missing_context_does_not_crash():
    # None context, and a context with junk values, both must work.
    build_features(_board(), None)
    build_features(_board(), {"candidates": [{"score": "nan-ish"}], "chosen_rank": "x"})


def test_no_nans():
    ctx = {"candidates": [{"score": float("inf")}, {"score": float("nan")}]}
    feats = build_features(_board(), ctx)
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))
