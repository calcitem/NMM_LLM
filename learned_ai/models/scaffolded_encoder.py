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
VN_BLEND: float = 0.0

# ── Lookahead extension ────────────────────────────────────────────────────────
# 12 half-plies × 5 signals (h_norm, learner_sent, opp_sent, vn_norm, gap_norm) = 60 floats
LOOKAHEAD_PLIES:              int = 12
LOOKAHEAD_SIGNALS:            int = 5
LOOKAHEAD_FEAT_DIM:           int = LOOKAHEAD_PLIES * LOOKAHEAD_SIGNALS   # 60
MOVE_FEAT_DIM_WITH_LOOKAHEAD: int = MOVE_FEAT_DIM + LOOKAHEAD_FEAT_DIM  # 122

# ── Top-K search-informed extension (v3, 2026-07-16) ──────────────────────────
# Per-candidate row on top of the 122-float base+lookahead:
#   [0] ab_score_norm : alpha-beta root score, min-max scaled to [0, 1] across the K candidates
#   [1] ab_rank_norm  : (K - rank) / (K - 1)  →  1.0 for #1, 0.0 for #K
#   [2] human_freq    : probability human plays this move (from HumanDB/TrajectoryDB/N-gram, or 0)
#   [3] human_rank    : normalised rank among human choices — 1.0 for most-played, 0.0 for absent
TOPK_EXTRA_DIM: int = 4
MOVE_FEAT_DIM_WITH_TOPK: int = MOVE_FEAT_DIM_WITH_LOOKAHEAD + TOPK_EXTRA_DIM  # 126


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


def encode_position_with_lookahead(
    board,
    player: str,
    sentinel_advisor=None,
    db=None,
    value_net=None,
    lookahead_advisor=None,
    lookahead_dim: Optional[int] = None,
) -> Optional[EncodedPosition]:
    """Encode legal moves with a lookahead block appended.

    When lookahead_advisor is provided, calls score_moves_matrix() and appends
    its (k, N) result to the base (k, 62) feat_matrix.  N = advisor.feat_dim
    (e.g. 15 for 5-ply specialists, 36 for 12-ply Overseer).

    When lookahead_advisor is None, a zero block is appended whose width is:
      - lookahead_dim  if explicitly provided (e.g. OVERSEER_LOOKAHEAD_DIM=36)
      - LOOKAHEAD_FEAT_DIM (15)  otherwise — backward-compatible default

    All other fields of EncodedPosition are identical to encode_position().
    """
    enc = encode_position(board, player, sentinel_advisor, db, value_net)
    if enc is None:
        return None

    k = len(enc.legal_moves)
    if lookahead_advisor is not None:
        try:
            la_block = lookahead_advisor.score_moves_matrix(board, enc, player)
        except Exception:
            _dim = getattr(lookahead_advisor, "feat_dim", LOOKAHEAD_FEAT_DIM)
            la_block = np.zeros((k, _dim), dtype=np.float32)
    else:
        _dim = lookahead_dim if lookahead_dim is not None else LOOKAHEAD_FEAT_DIM
        la_block = np.zeros((k, _dim), dtype=np.float32)

    enc.feat_matrix = np.concatenate([enc.feat_matrix, la_block], axis=1).astype(np.float32)
    return enc


# ── Top-K encoder (v3) ────────────────────────────────────────────────────────

def _move_notation(mv: Dict[str, Any]) -> str:
    """Return the notation string used by TrajectoryDB / HumanDB / NGram.

    Matches ai/game_ai.py's move-notation convention:
      placement           → "d3"
      movement            → "d3-a1"
      movement + capture  → "d3-a1xd7"
      placement + capture → "d3xd7"
    """
    frm = mv.get("from")
    to  = mv.get("to") or ""
    cap = mv.get("capture")
    s = f"{frm}-{to}" if frm else to
    if cap:
        s += f"x{cap}"
    return s


def _human_prior_freqs(
    board,
    color: str,
    human_db=None,               # ai.human_db.HumanDB   (preferred, ELO-stratified)
    trajectory_db=None,          # ai.trajectory_db.TrajectoryDB (fallback)
    ngram_model=None,            # ai.ngram_opponent_model.NGramOpponentModel (last resort)
    game_notations: Optional[List[str]] = None,   # required for ngram fallback
) -> Dict[str, float]:
    """Return {notation: probability} for the next move by `color`.

    Precedence: HumanDB > TrajectoryDB > N-gram.  Returns {} when nothing hits.
    """
    # 1. HumanDB — richest source (ELO/win-rate stratified)
    if human_db is not None:
        try:
            freqs = human_db.query_all_frequencies(board)
            if freqs:
                return freqs
        except Exception:
            pass

    # 2. TrajectoryDB — position-based frequency
    if trajectory_db is not None:
        try:
            freqs = trajectory_db.query_all_frequencies(board)
            if freqs:
                return freqs
        except Exception:
            pass

    # 3. N-gram fallback — sequence-based; needs game_notations
    if ngram_model is not None and game_notations is not None:
        try:
            freqs = ngram_model.predict(color, game_notations)
            if freqs:
                return freqs
        except Exception:
            pass

    return {}


def encode_top_k_candidates(
    board,
    player: str,
    gameai,                                    # ai.game_ai.GameAI — provides alpha-beta search
    top_k: int = 5,
    ab_depth: Optional[int] = None,            # None → gameai.max_search_depth
    ab_time_budget: Optional[float] = None,    # None → gameai's own per-diff cap
    sentinel_advisor=None,
    db=None,
    value_net=None,
    lookahead_advisor=None,
    lookahead_dim: Optional[int] = None,
    human_db=None,
    trajectory_db=None,
    ngram_model=None,
    game_notations: Optional[List[str]] = None,
    ab_preserve_tt: bool = False,
) -> Optional[Any]:
    """Encode the ``top_k`` alpha-beta-best candidates as a (K, MOVE_FEAT_DIM_WITH_TOPK) matrix.

    The specialist is only ever asked to re-rank the classical engine's top K moves,
    not to score every legal move from scratch.  This mirrors how strong players work:
    narrow to a promising short-list, then evaluate deeply.

    Per-candidate feature row (126 floats):
      * 62 base features         (same as encode_position)
      * 60 lookahead features    (same as encode_position_with_lookahead)
      * 4  top-K extras          (ab_score_norm, ab_rank_norm, human_freq, human_rank)

    Steps:
      1. Call ``gameai.score_root_moves(board, depth, time_budget)`` to get
         alpha-beta-scored candidates.  Take the top-K.
      2. Encode the full position with lookahead as normal, then filter to the K rows.
      3. Compute the alpha-beta score/rank normalisations.
      4. Look up human-prior frequencies (HumanDB > TrajectoryDB > N-gram).
      5. Concatenate the 4 extra floats onto each row.

    Returns an EncodedPosition-like object with:
        feat_matrix: (K, MOVE_FEAT_DIM_WITH_TOPK)
        value_input: (23,)
        legal_moves: the K candidate move dicts, in alpha-beta rank order

    Returns None if there are no legal moves, or if the search fails, or if none
    of the top-K survived the encoder's own legal-move filter.
    """
    if gameai is None:
        # No alpha-beta search available — fall back to old-style encoding.
        return encode_position_with_lookahead(board, player,
                                              sentinel_advisor=sentinel_advisor,
                                              db=db,
                                              value_net=value_net,
                                              lookahead_advisor=lookahead_advisor,
                                              lookahead_dim=lookahead_dim)

    # 1. Alpha-beta score all legal moves via the classical engine.
    # NB: GameAI.score_root_moves does NOT respect its time_budget — it runs
    # to full depth regardless.  So the depth default matters a lot.
    # Depth 1 = ~30 ms/call and gives real scores for all moves.
    # Depth 2 = ~500 ms/call, deeper structure but 500× slower.
    # Depth 3+ = starts aborting inside negamax → returns mostly-zero scores anyway.
    # For training we cap at depth 2; at inference the router passes coordinator's
    # last_depth_reached (which reuses the warm TT for near-instant probes).
    _depth  = int(ab_depth) if ab_depth is not None else 1
    _budget = float(ab_time_budget) if ab_time_budget is not None else float(getattr(gameai, "_override_time_budget", None) or 2.0)
    try:
        scored = gameai.score_root_moves(board, depth=_depth, time_budget=_budget,
                                         preserve_tt=ab_preserve_tt)
    except Exception:
        scored = []
    if not scored:
        return None
    scored_top_k = list(scored[:max(1, int(top_k))])

    # 1b. If the sentinel's top pick is NOT in the α-β top-K, force-include it as
    # an outsider candidate.  Rationale: sentinel scores moves differently from
    # α-β (it's an ML advisor over blunder patterns) and its argmax may be a
    # legitimately strong non-obvious move the specialist should have the chance
    # to prefer.  Same for the highest-freq human move — if the human favourite
    # is not covered by α-β top-K, expose it too.
    def _key(m): return (m.get("from"), m.get("to"), m.get("capture"))
    top_k_keys = {_key(mv) for mv, _ in scored_top_k}
    scored_all_map = {_key(mv): float(s) for mv, s in scored}

    def _add_outsider(mv: Dict[str, Any]) -> None:
        k = _key(mv)
        if k in top_k_keys:
            return
        top_k_keys.add(k)
        scored_top_k.append((mv, scored_all_map.get(k, 0.0)))

    if sentinel_advisor is not None:
        try:
            from game.rules import get_all_legal_moves
            all_legal = get_all_legal_moves(board)
            sent_advice = sentinel_advisor.advise(board, all_legal, player, 0)
            if sent_advice and getattr(sent_advice, "move_scores", None):
                _idx = int(np.argmax(np.asarray(sent_advice.move_scores)))
                if 0 <= _idx < len(all_legal):
                    _add_outsider(all_legal[_idx])
        except Exception:
            pass

    # 2. Base encoding only — 62-float rows for all legal moves.  We defer
    #    the expensive 15-ply lookahead until AFTER we've filtered to top-K
    #    (~3-4× speedup vs the previous "encode all then discard" pattern).
    base_enc = encode_position(
        board, player,
        sentinel_advisor=sentinel_advisor,
        db=db,
        value_net=value_net,
    )
    if base_enc is None or not base_enc.legal_moves:
        return None

    enc_idx = {_key(m): i for i, m in enumerate(base_enc.legal_moves)}

    kept_rows: List[np.ndarray] = []
    kept_moves: List[Dict[str, Any]] = []
    kept_scores: List[float] = []
    kept_orig_idx: List[int] = []
    for mv, score_norm in scored_top_k:
        idx = enc_idx.get(_key(mv))
        if idx is None:
            continue
        kept_rows.append(base_enc.feat_matrix[idx])
        kept_moves.append(mv)
        kept_scores.append(float(score_norm))
        kept_orig_idx.append(idx)

    if not kept_rows:
        return None

    K = len(kept_rows)
    feat_62 = np.stack(kept_rows).astype(np.float32)   # (K, MOVE_FEAT_DIM=62)

    # 2b. Compute lookahead ONLY for the K candidates (biggest single speedup).
    if lookahead_advisor is not None:
        try:
            la_block = lookahead_advisor.score_moves_matrix(
                board, base_enc, player, moves_subset=kept_moves,
            )
        except Exception:
            _dim = getattr(lookahead_advisor, "feat_dim", LOOKAHEAD_FEAT_DIM)
            la_block = np.zeros((K, _dim), dtype=np.float32)
    else:
        _dim = lookahead_dim if lookahead_dim is not None else LOOKAHEAD_FEAT_DIM
        la_block = np.zeros((K, _dim), dtype=np.float32)

    feat_base = np.concatenate([feat_62, la_block], axis=1).astype(np.float32)   # (K, 122)

    # 3. ab_score_norm (already normalised by GameAI.score_root_moves)
    #    ab_rank_norm: 1.0 for #1, 0.0 for #K (evenly spaced)
    ab_scores_arr = np.asarray(kept_scores, dtype=np.float32)
    if K > 1:
        ab_ranks_arr = np.linspace(1.0, 0.0, num=K, dtype=np.float32)
    else:
        ab_ranks_arr = np.ones(1, dtype=np.float32)

    # 4. Human-prior probability per candidate.
    freqs = _human_prior_freqs(
        board, player,
        human_db=human_db,
        trajectory_db=trajectory_db,
        ngram_model=ngram_model,
        game_notations=game_notations,
    )
    if freqs:
        sorted_ntns = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)
        ntn_rank = {ntn: r + 1 for r, (ntn, _) in enumerate(sorted_ntns)}
    else:
        ntn_rank = {}

    human_freq_arr = np.zeros(K, dtype=np.float32)
    human_rank_arr = np.zeros(K, dtype=np.float32)
    for i, mv in enumerate(kept_moves):
        ntn = _move_notation(mv)
        human_freq_arr[i] = float(freqs.get(ntn, 0.0))
        r = ntn_rank.get(ntn)
        # Normalise rank: rank 1 → 1.0, rank 2 → 0.8, ..., rank 5 → 0.2, unseen → 0.0
        if r is not None and r <= 5:
            human_rank_arr[i] = 1.0 - (r - 1) / 5.0
        # else: leave at 0.0

    # 5. Concatenate the 4 extras.
    extras = np.column_stack([
        ab_scores_arr,
        ab_ranks_arr,
        human_freq_arr,
        human_rank_arr,
    ]).astype(np.float32)                                  # (K, 4)
    feat_final = np.concatenate([feat_base, extras], axis=1).astype(np.float32)  # (K, 126)

    # 6. Return an EncodedPosition-like object.  Reuse the base's other fields but
    #    filter the per-move lists to the top-K.
    def _pick(lst):
        if lst is None:
            return lst
        try:
            return [lst[i] for i in kept_orig_idx]
        except Exception:
            return lst

    base_enc.feat_matrix = feat_final
    base_enc.legal_moves = kept_moves
    if hasattr(base_enc, "sentinel_scores"):
        base_enc.sentinel_scores = _pick(base_enc.sentinel_scores)
    if hasattr(base_enc, "h_scores_abs"):
        base_enc.h_scores_abs = _pick(base_enc.h_scores_abs)
    if hasattr(base_enc, "vn_scores_abs"):
        base_enc.vn_scores_abs = _pick(base_enc.vn_scores_abs)
    if hasattr(base_enc, "db_moves"):
        base_enc.db_moves = _pick(base_enc.db_moves)
    # h_top1_idx is over the K subset now; keep 0 if the AB #1 also matches heuristic top1.
    if hasattr(base_enc, "h_top1_idx") and base_enc.h_scores_abs:
        try:
            base_enc.h_top1_idx = int(np.argmax(np.asarray(base_enc.h_scores_abs)))
        except Exception:
            base_enc.h_top1_idx = 0
    return base_enc
