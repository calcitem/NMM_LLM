"""Tests for learned_ai/sentinel/infer.py (SentinelAdvisor)."""

from __future__ import annotations

import time

import torch

from game.board import BoardState
from learned_ai.sentinel.config import SentinelConfig
from learned_ai.sentinel.infer import SentinelAdvice, SentinelAdvisor
from learned_ai.sentinel.model import SentinelNet


def _board():
    return BoardState.from_fen_string("BBW....B.W.W............|W|3|3")


def _trained_advisor(tmp_path):
    cfg = SentinelConfig(hidden_dims=[64, 32], dropout=0.0)
    net = SentinelNet(input_dim=cfg.input_dim, hidden_dims=cfg.hidden_dims, dropout=0.0)
    ckpt = tmp_path / "sentinel.pt"
    torch.save({"state_dict": net.state_dict(), "config": cfg.to_dict()}, ckpt)
    return SentinelAdvisor(str(ckpt), config=cfg, device="cpu")


def test_advise_returns_sentinel_advice(tmp_path):
    advisor = _trained_advisor(tmp_path)
    assert advisor.is_loaded()
    advice = advisor.advise(_board(), {"candidates": [{"score": 1.0}]})
    assert isinstance(advice, SentinelAdvice)
    assert 0.0 <= advice.mistake_risk <= 1.0
    assert 0.0 <= advice.opportunity_score <= 1.0
    assert -1.0 <= advice.trajectory_value_delta <= 1.0
    assert 0.0 <= advice.turning_point_confidence <= 1.0
    assert isinstance(advice.is_turning_point, bool)
    assert advice.advisory_message in (
        "safe", "possible_mistake", "missed_opportunity", "critical"
    )


def test_advise_fast(tmp_path):
    advisor = _trained_advisor(tmp_path)
    board = _board()
    ctx = {"candidates": [{"score": 1.0}]}
    advisor.advise(board, ctx)  # warmup
    t0 = time.perf_counter()
    advisor.advise(board, ctx)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"advise took {elapsed_ms:.2f}ms"


def test_advise_no_crash_empty_context(tmp_path):
    advisor = _trained_advisor(tmp_path)
    # None context and an empty dict must both work.
    advisor.advise(_board(), None)
    advisor.advise(_board(), {})


def test_unloaded_advisor_returns_neutral():
    advisor = SentinelAdvisor()  # no checkpoint
    assert not advisor.is_loaded()
    advice = advisor.advise(_board(), {})
    assert advice == SentinelAdvice.neutral()
