"""learned_ai/models/scaffolded_encoder.py — per-position feature encoding for
the scaffolded meta-policy.

Each legal move is encoded as a 62-float vector that combines the existing
58-float sentinel feature vector with 4 additional expert-context floats:
  [58]  sentinel_score : SentinelAdvisor quality score in [0, 1]
  [59]  blended_abs    : 0.5 * h_abs_norm + 0.5 * vn_abs_norm → [0,1]
                         (heuristic + value-net absolute eval, each mapped from [-1,1])
  [60]  is_engine_top1 : 1.0 if this is the heuristic's best move
  [61]  blended_delta  : tanh(0.5 * h_delta + 0.5 * vn_delta)
                         (blended signed improvement from heuristic and value-net)

When value_net=None (inference without VN), features 59/61 fall back to pure
heuristic values, preserving exact backward-compatibility with existing checkpoints.

The value head takes a fixed 23-float board-level vector:
  [0:20)  board_context_features (phase, piece counts, mobility, mills)
  [20]    h_eval_abs: absolute heuristic evaluation, raw from evaluate() in [-1,1]
  [21]    max_sentinel_score across legal moves
  [22]    mean_sentinel_score across legal moves

NOTE — Future Option A: when the next checkpoint is trained from scratch, extend
MOVE_FEAT_DIM to 64 by adding vn_score_abs and vn_delta_tanh as independent
features [62] and [63] instead of blending them into [59] and [61].  This gives
the model clean, separable signals from the two evaluators.  See Learned_ai.md.

Public API
----------
  MOVE_FEAT_DIM   = 62
  VALUE_INPUT_DIM = 23
  VN_BLEND        = 0.5   (weight of value-net signal in features 59 and 61)

  encode_position(board, player, sentinel_advisor, db, value_net) -> EncodedPosition
    Returns feat_matrix (k, 62), value_input (23,), legal_moves list, and
    a dict of raw scores for use in reward computation.

  build_enriched_row(...) -> np.ndarray (62,)  [for unit tests / custom use]
  build_value_input(...) -> np.ndarray (23,)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from learned_ai.sentinel.feature_builder import (
    FEATURE_DIM as _SENT_FEAT_DIM,
    board_context_features,
    build_move_features,
)

MOVE_FEAT_DIM: int = _SENT_FEAT_DIM + 4   # 62
VALUE_INPUT_DIM: int = 23

# Blend weight: fraction of value-net signal mixed into features 59 and 61.
# 0.0 = pure heuristic (old behaviour); 1.0 = pure value-net; 0.5 = equal blend.
VN_BLEND: float = 0.5


# ── heuristic evaluate import (avoids ai/__init__ heavy imports) ───────────────

def _get_evaluate():
    """Return ai.heuristics.evaluate.  Safe after HeuristicAgent has been imported
    (which registers the ai namespace package); also works standalone."""
    import importlib
    import importlib.util
    import os
    import sys
    import types

    if "ai" not in sys.modules:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        ai_dir = os.path.join(repo_root, "ai")
        ai_pkg = types.ModuleType("ai")
        ai_pkg.__path__ = [ai_dir]
        sys.modules["ai"] = ai_pkg

    ai_dir = sys.modules["ai"].__path__[0]
    for name in ("heuristics",):
        full = f"ai.{name}"
        if full in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(ai_dir, f"{name}.py")
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot find ai/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)

    return sys.modules["ai.heuristics"].evaluate


_evaluate_fn = None


def _heuristic_eval(board, player: str) -> float:
    """evaluate(board, player, strength_mode=True) → float in [-1, 1]."""
    global _evaluate_fn
    if _evaluate_fn is None:
        _evaluate_fn = _get_evaluate()
    try:
        return float(_evaluate_fn(board, player, strength_mode=True))
    except Exception:
        return 0.0


# ── per-move enriched feature row ─────────────────────────────────────────────

def build_enriched_row(
    board,
    move: Dict[str, Any],
    player: str,
    sentinel_score: float,
    h_abs_norm: float,
    is_top1: bool,
    h_delta: float,
    vn_abs_norm: float = 0.0,
    vn_delta: float = 0.0,
    move_ctx: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Build one 62-float row for ``move``.

    Parameters
    ----------
    board          : BoardState before the move
    move           : apply-move dict {from, to, capture}
    player         : mover ("W" or "B")
    sentinel_score : [0,1] quality from SentinelAdvisor (0.5 if unavailable)
    h_abs_norm     : heuristic eval after move, mapped to [0,1]: (h+1)/2
    is_top1        : True if this is the heuristic's best-ranked move
    h_delta        : h_after - h_before (raw; used in blend, not tanh'd here)
    vn_abs_norm    : value-net eval after move, mapped to [0,1]: (vn+1)/2
                     (0.0 when value_net not available)
    vn_delta       : vn_after - vn_before (0.0 when value_net not available)
    move_ctx       : dict for build_move_features() counterfactual block
    """
    base = build_move_features(board, move, player, move_ctx)  # (58,)

    # Feature 59: blend of heuristic and value-net absolute eval
    blended_abs = (1.0 - VN_BLEND) * h_abs_norm + VN_BLEND * vn_abs_norm

    # Feature 61: blend of heuristic and value-net delta, then tanh
    blended_delta = (1.0 - VN_BLEND) * h_delta + VN_BLEND * vn_delta

    extra = np.array(
        [
            float(np.clip(sentinel_score, 0.0, 1.0)),
            float(np.clip(blended_abs, 0.0, 1.0)),
            1.0 if is_top1 else 0.0,
            float(math.tanh(blended_delta)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extra]).astype(np.float32)


def build_value_input(
    board,
    player: str,
    h_eval_abs: float,
    sentinel_scores: List[float],
) -> np.ndarray:
    """Build the 23-float board-level input for the value head."""
    ctx = board_context_features(board, player)          # (20,)
    if sentinel_scores:
        max_s = float(max(sentinel_scores))
        mean_s = float(sum(sentinel_scores) / len(sentinel_scores))
    else:
        max_s = mean_s = 0.5
    extra = np.array([h_eval_abs, max_s, mean_s], dtype=np.float32)
    return np.concatenate([ctx, extra]).astype(np.float32)


# ── encoded position ───────────────────────────────────────────────────────────

@dataclass
class EncodedPosition:
    """All data produced by encode_position() for one board state."""

    feat_matrix: np.ndarray          # (k, MOVE_FEAT_DIM) — one row per legal move
    value_input: np.ndarray          # (VALUE_INPUT_DIM,) — for the value head
    legal_moves: List[Dict[str, Any]]   # same order as feat_matrix rows
    # For reward computation:
    sentinel_scores: List[float]     # raw sentinel quality per move
    h_scores_abs: List[float]        # evaluate(board_after, player, sm=True) per move
    h_before: float                  # evaluate(board_before, player, sm=True)
    h_top1_idx: int                  # index of heuristic's best move
    db_moves: List[Dict[str, Any]]   # query_all_moves() output (may be empty)
    # Value-net scores (0.0 when value_net=None):
    vn_scores_abs: List[float] = field(default_factory=list)
    vn_before: float = 0.0


def encode_position(
    board,
    player: str,
    sentinel_advisor=None,
    db=None,
    value_net=None,
) -> Optional[EncodedPosition]:
    """Encode all legal moves at ``board`` into the scaffolded feature format.

    Parameters
    ----------
    board           : BoardState (before any move)
    player          : mover ("W" or "B")
    sentinel_advisor: SentinelAdvisor or None (scores default to 0.5)
    db              : ExternalSolvedDB or None (DB features default to 0)
    value_net       : ValueNet or None — when provided, blends VN signal into
                      features 59 (abs eval) and 61 (delta) via VN_BLEND weight.
                      Also populates EncodedPosition.vn_scores_abs and .vn_before
                      for use in per-move reward shaping during training.

    Returns None when there are no legal moves (terminal position).
    """
    from game.rules import get_all_legal_moves

    legal_moves = get_all_legal_moves(board)
    if not legal_moves:
        return None

    k = len(legal_moves)

    # ── Sentinel scores ────────────────────────────────────────────────────────
    sentinel_scores: List[float]
    if sentinel_advisor is not None and sentinel_advisor.is_loaded():
        try:
            advice = sentinel_advisor.advise(board, legal_moves, player)
            sentinel_scores = advice.move_scores if advice else [0.5] * k
        except Exception:
            sentinel_scores = [0.5] * k
    else:
        sentinel_scores = [0.5] * k

    # ── Heuristic + Value-net 1-ply evaluations (single loop) ─────────────────
    h_before = _heuristic_eval(board, player)
    vn_before = float(value_net.predict(board, player)) if value_net is not None else 0.0

    h_scores_abs: List[float] = []
    vn_scores_abs: List[float] = []

    for mv in legal_moves:
        try:
            board_after = board.apply_move(mv)
            h_scores_abs.append(_heuristic_eval(board_after, player))
            if value_net is not None:
                vn_scores_abs.append(float(value_net.predict(board_after, player)))
            else:
                vn_scores_abs.append(0.0)
        except Exception:
            h_scores_abs.append(h_before)
            vn_scores_abs.append(vn_before)

    # Rank moves by heuristic score (higher = better for player)
    sorted_by_h = sorted(range(k), key=lambda i: -h_scores_abs[i])
    h_ranks = [0] * k
    for rank, idx in enumerate(sorted_by_h):
        h_ranks[idx] = rank
    h_top1_idx = sorted_by_h[0]

    h_min = min(h_scores_abs)
    h_max = max(h_scores_abs)
    h_range = h_max - h_min + 1e-8
    h_norms = [(s - h_min) / h_range for s in h_scores_abs]

    # ── DB annotations ─────────────────────────────────────────────────────────
    db_moves: List[Dict[str, Any]] = []
    if db is not None:
        try:
            db_moves = db.query_all_moves(board, player) or []
        except Exception:
            db_moves = []

    # ── Build feature matrix ───────────────────────────────────────────────────
    # DB data is kept in EncodedPosition.db_moves for reward computation but
    # must NOT flow into feature slots — inference has no DB access.
    rows: List[np.ndarray] = []
    for i, mv in enumerate(legal_moves):
        ctx = {
            "all_moves": [],
            "heuristic_rank": h_ranks[i],
            "n_legal": k,
            "heuristic_score_norm": h_norms[i],
        }
        h_abs_norm = (h_scores_abs[i] + 1.0) / 2.0   # map [-1,1] to [0,1]
        h_delta    = h_scores_abs[i] - h_before
        vn_abs_norm = (vn_scores_abs[i] + 1.0) / 2.0  # map [-1,1] to [0,1]
        vn_delta    = vn_scores_abs[i] - vn_before
        row = build_enriched_row(
            board,
            mv,
            player,
            sentinel_score=sentinel_scores[i],
            h_abs_norm=h_abs_norm,
            is_top1=(i == h_top1_idx),
            h_delta=h_delta,
            vn_abs_norm=vn_abs_norm,
            vn_delta=vn_delta,
            move_ctx=ctx,
        )
        rows.append(row)

    feat_matrix = np.stack(rows).astype(np.float32)     # (k, 62)
    value_input = build_value_input(board, player, h_before, sentinel_scores)

    return EncodedPosition(
        feat_matrix=feat_matrix,
        value_input=value_input,
        legal_moves=legal_moves,
        sentinel_scores=sentinel_scores,
        h_scores_abs=h_scores_abs,
        h_before=h_before,
        h_top1_idx=h_top1_idx,
        db_moves=db_moves,
        vn_scores_abs=vn_scores_abs,
        vn_before=vn_before,
    )
