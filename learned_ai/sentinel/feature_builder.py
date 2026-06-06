"""learned_ai/sentinel/feature_builder.py — extended sentinel feature vector.

The sentinel input is a 120-float vector:

  [0:84)   base board-state encoding from learned_ai.models.state_encoder.encode_state
           (REUSED — never duplicated here).
  [84:120) 36 context features describing the *decision* at this ply.

Context layout (36 floats):
  [0:5)   top-5 heuristic scores (normalised to [0,1], 0-padded)
  [5:25)  top-5 move-type one-hots (4-way place/move/fly/capture each), 0-padded
  [25]    chosen_move_rank  (rank / max(n-1,1); 0 if a single candidate)
  [26]    closes_mill        (bool)
  [27]    opens_mill_threat  (bool)
  [28]    reduces_own_mobility (bool)
  [29:33) trajectory score trend (last 4 heuristic scores, normalised, 0-padded)
  [33]    game_source_is_human (1.0 if human-vs-ai else 0.0)
  [34]    n_candidates_norm  (n_candidates / 30.0, clipped to 1.0)
  [35]    reserved padding   (0.0) — keeps the block exactly 36 wide.

Public API:
  build_features(board_state, move_context: dict) -> np.ndarray (120,)
  CONTEXT_DIM, BASE_DIM, FEATURE_DIM constants.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from learned_ai.models.state_encoder import encode_state

BASE_DIM = 84
CONTEXT_DIM = 36
FEATURE_DIM = BASE_DIM + CONTEXT_DIM  # 120

_MOVE_TYPES = ("place", "move", "fly", "capture")
_MOVE_TYPE_IDX = {t: i for i, t in enumerate(_MOVE_TYPES)}
_TOP_K = 5
_MAX_CANDIDATES = 30.0


def _squash(x: float) -> float:
    """Map an unbounded heuristic score into (0,1) with a smooth logistic.

    Heuristic scores in this engine are roughly centred near 0 and can be
    moderately large; a logistic keeps the feature bounded and well-scaled
    without needing dataset-wide statistics.
    """
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(xf):
        return 0.0
    if xf >= 0:
        z = math.exp(-xf)
        return 1.0 / (1.0 + z)
    z = math.exp(xf)
    return z / (1.0 + z)


def _candidate_move_type(cand: Dict[str, Any]) -> Optional[str]:
    """Infer the 4-way move type of a candidate from its dict.

    A candidate may carry an explicit ``type`` (from a game log) or just a raw
    move dict {from,to,capture}. Capture takes precedence (it is the most
    strategically salient channel), then explicit type, then from/to shape.
    """
    move = cand.get("move", cand)
    if not isinstance(move, dict):
        move = cand
    if move.get("capture"):
        return "capture"
    t = cand.get("type") or move.get("type")
    if t in _MOVE_TYPE_IDX:
        return t
    if move.get("from") is None and move.get("to") is not None:
        return "place"
    if move.get("from") is not None:
        return "move"
    return None


def _build_context(ctx: Dict[str, Any]) -> np.ndarray:
    """Build the 36-float context block from a move_context dict.

    Recognised keys (all optional — missing keys are treated as empty/zero):
      candidates:           list of {move|score|type} sorted desc by score
      chosen_rank:          int
      closes_mill:          bool
      opens_mill_threat:    bool
      reduces_own_mobility: bool
      trajectory_scores:    list of up to 4 recent heuristic scores (chrono order)
      game_source:          "human_vs_ai" | "ai_vs_ai"
    """
    out = np.zeros(CONTEXT_DIM, dtype=np.float32)
    ctx = ctx or {}

    candidates: List[Dict[str, Any]] = list(ctx.get("candidates") or [])
    n_cand = len(candidates)

    # [0:5) top-5 scores, [5:25) top-5 move-type one-hots
    for i in range(_TOP_K):
        if i >= n_cand:
            break
        cand = candidates[i] if isinstance(candidates[i], dict) else {}
        score = cand.get("score", cand.get("game_ai_score", 0.0))
        out[i] = _squash(score)
        mt = _candidate_move_type(cand)
        if mt is not None:
            out[5 + i * 4 + _MOVE_TYPE_IDX[mt]] = 1.0

    # [25] chosen move rank, normalised so 0 = best, 1 = worst.
    chosen_rank = ctx.get("chosen_rank", 0)
    try:
        chosen_rank = int(chosen_rank)
    except (TypeError, ValueError):
        chosen_rank = 0
    if n_cand > 1:
        out[25] = max(0.0, min(1.0, chosen_rank / float(n_cand - 1)))
    else:
        out[25] = 0.0

    # [26:29) boolean flags
    out[26] = 1.0 if ctx.get("closes_mill") else 0.0
    out[27] = 1.0 if ctx.get("opens_mill_threat") else 0.0
    out[28] = 1.0 if ctx.get("reduces_own_mobility") else 0.0

    # [29:33) trajectory trend — last 4 scores, right-aligned (most recent last).
    traj = list(ctx.get("trajectory_scores") or [])[-4:]
    for j, s in enumerate(traj):
        out[29 + (4 - len(traj)) + j] = _squash(s)

    # [33] game source, [34] candidate-count norm, [35] reserved.
    out[33] = 1.0 if ctx.get("game_source") == "human_vs_ai" else 0.0
    out[34] = min(1.0, n_cand / _MAX_CANDIDATES)
    out[35] = 0.0
    return out


def build_features(board_state, move_context: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Return the 120-float sentinel feature vector for a board + decision context.

    The first 84 values come from the shared state encoder; the last 36 encode
    the move-decision context. Robust to a missing/empty context dict.
    """
    base_t = encode_state(board_state)            # torch.Tensor (84,)
    base = np.asarray(base_t.detach().cpu().numpy(), dtype=np.float32)
    ctx = _build_context(move_context or {})
    return np.concatenate([base, ctx]).astype(np.float32)
