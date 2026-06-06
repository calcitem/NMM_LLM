"""learned_ai/sentinel/feature_builder.py — per-move sentinel feature vector.

The sentinel is a *move-level* scorer: every example describes ONE candidate
move in ONE position, encoded from the mover's perspective. The model predicts
a single move-quality score in [0, 1].

Feature layout (FEATURE_DIM = 58 floats)
----------------------------------------
Board context (mover-normalised, 20 floats) — same for all moves in a position:
  [0:4)   phase one-hot [placement, midgame, endgame, flying]
  [4]     mover piece count / 9
  [5]     opponent piece count / 9
  [6]     mover mills count / 3
  [7]     opponent mills count / 3
  [8]     mover mobility / 24
  [9]     opponent mobility / 24
  [10]    mover pieces in double mills / 9
  [11]    opponent pieces in double mills / 9
  [12]    mover placed / 9
  [13]    opponent placed / 9
  [14]    mover potential mills (2-of-3 with empty) / 8
  [15]    opponent potential mills / 8
  [16]    side-to-move is black (0.0 white, 1.0 black)
  [17:20) reserved padding (0.0)

Move-specific (20 floats):
  [20]    from-square index / 24 (0 for placements)
  [21]    to-square index / 24
  [22]    is_placement (bool)
  [23]    is_mill_closing (bool)
  [24]    is_capture (bool)
  [25]    captured piece index / 24 (0 if no capture)
  [26]    would_create_double_mill (bool)
  [27]    would_block_opponent_mill (bool)
  [28]    resulting mover piece count / 9
  [29]    resulting opponent piece count / 9
  [30]    resulting mover mobility / 24
  [31]    resulting opponent mobility / 24
  [32]    resulting mover mills / 3
  [33]    resulting opponent mills / 3
  [34]    destination is a "strong" (junction, deg>=3) square (bool)
  [35]    destination is a "corner" (deg==2) square (bool)
  [36]    move reduces own mobility (bool)
  [37]    move opens a new mill threat (bool)
  [38:40) reserved padding (0.0)

Counterfactual context (18 floats) — same for all moves in a position:
  [40]    n_legal_moves / 24
  [41]    frac_winning_moves   (DB)
  [42]    frac_losing_moves    (DB)
  [43]    frac_draw_moves      (DB)
  [44]    best_available_wdl   (1.0 win / 0.5 draw / 0.0 loss)
  [45]    worst_available_wdl
  [46]    heuristic_rank / n_legal (0 = top)
  [47]    heuristic_score_normalised (across candidates)
  [48]    winning_move_available (bool, DB)
  [49]    losing_move_available  (bool, DB)
  [50]    this move is DB win (bool)
  [51]    this move is DB loss (bool)
  [52]    this move is DB draw (bool)
  [53]    this move WDL known (bool)
  [54]    db_available (bool)
  [55]    a better DB move existed than this one (bool)
  [56:58) reserved padding (0.0)

At inference time the DB is not queried, so the DB-derived slots are populated
from whatever the caller can compute cheaply (frac/best/worst left at 0 when the
DB is unavailable). Training enriches them from query_all_moves().

Public API:
  build_move_features(board, move, player, move_ctx) -> np.ndarray (FEATURE_DIM,)
  board_context_features(board, player) -> np.ndarray (BOARD_CTX_DIM,)
  FEATURE_DIM, BOARD_CTX_DIM, MOVE_DIM, COUNTERFACTUAL_DIM constants.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from game.board import ADJACENCY, MILLS, POSITIONS

BOARD_CTX_DIM = 20
MOVE_DIM = 20
COUNTERFACTUAL_DIM = 18
FEATURE_DIM = BOARD_CTX_DIM + MOVE_DIM + COUNTERFACTUAL_DIM  # 58

_POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}
_N_POS = float(len(POSITIONS))  # 24

# A square is "strong" (junction) when it has >= 3 neighbours; "corner" when 2.
_DEGREE = {p: len(ADJACENCY.get(p, [])) for p in POSITIONS}

_WDL_SCALAR = {"win": 1.0, "draw": 0.5, "loss": 0.0}


# ── small helpers ──────────────────────────────────────────────────────────────

def _opponent(player: str) -> str:
    return "B" if player == "W" else "W"


def _piece_count(board, color: str) -> int:
    return sum(1 for v in board.positions.values() if v == color)


def _mills_count(board, color: str) -> int:
    return sum(1 for ml in MILLS if all(board.positions[p] == color for p in ml))


def _potential_mills(board, color: str) -> int:
    """Count mill lines with exactly 2 of ``color`` and 1 empty square."""
    n = 0
    for ml in MILLS:
        vals = [board.positions[p] for p in ml]
        if vals.count(color) == 2 and vals.count("") == 1:
            n += 1
    return n


def _double_mill_pieces(board, color: str) -> int:
    """Count pieces that participate in two or more completed mills."""
    membership: Dict[str, int] = {}
    for ml in MILLS:
        if all(board.positions[p] == color for p in ml):
            for p in ml:
                membership[p] = membership.get(p, 0) + 1
    return sum(1 for c in membership.values() if c >= 2)


def _mobility(board, color: str) -> int:
    """Total legal moves available to ``color`` from this board (place or move)."""
    try:
        if board.phase == "place":
            return len(board.legal_placements(color))
        return len(board.legal_moves(color))
    except Exception:
        return 0


def _phase_onehot(board, player: str) -> List[float]:
    """[placement, midgame, endgame, flying] one-hot from the mover's view."""
    oh = [0.0, 0.0, 0.0, 0.0]
    try:
        if board.phase == "place":
            oh[0] = 1.0
            return oh
        on = board.pieces_on_board.get(player, _piece_count(board, player))
        if on <= 3:
            oh[3] = 1.0
        elif on <= 4:
            oh[2] = 1.0
        else:
            oh[1] = 1.0
    except Exception:
        oh[1] = 1.0
    return oh


# ── board context ───────────────────────────────────────────────────────────────

def board_context_features(board, player: str) -> np.ndarray:
    """20-float board context, normalised so ``player`` is the mover."""
    opp = _opponent(player)
    out = np.zeros(BOARD_CTX_DIM, dtype=np.float32)
    out[0:4] = _phase_onehot(board, player)
    out[4] = min(1.0, _piece_count(board, player) / 9.0)
    out[5] = min(1.0, _piece_count(board, opp) / 9.0)
    out[6] = min(1.0, _mills_count(board, player) / 3.0)
    out[7] = min(1.0, _mills_count(board, opp) / 3.0)
    out[8] = min(1.0, _mobility(board, player) / _N_POS)
    out[9] = min(1.0, _mobility(board, opp) / _N_POS)
    out[10] = min(1.0, _double_mill_pieces(board, player) / 9.0)
    out[11] = min(1.0, _double_mill_pieces(board, opp) / 9.0)
    out[12] = min(1.0, board.pieces_placed.get(player, 0) / 9.0)
    out[13] = min(1.0, board.pieces_placed.get(opp, 0) / 9.0)
    out[14] = min(1.0, _potential_mills(board, player) / 8.0)
    out[15] = min(1.0, _potential_mills(board, opp) / 8.0)
    out[16] = 0.0 if player == "W" else 1.0
    # [17:20) reserved
    return out


# ── move-specific ─────────────────────────────────────────────────────────────

def _would_block_opponent_mill(board, move: Dict[str, Any], player: str) -> bool:
    """True if the destination square fills an opponent 2-of-3 mill line."""
    opp = _opponent(player)
    to = move.get("to")
    if to is None:
        return False
    for ml in MILLS:
        if to not in ml:
            continue
        vals = [board.positions[p] for p in ml]
        if vals.count(opp) == 2 and vals.count("") == 1:
            return True
    return False


def _opens_new_mill_threat(after, player: str) -> bool:
    for ml in MILLS:
        vals = [after.positions[p] for p in ml]
        if vals.count(player) == 2 and vals.count("") == 1:
            return True
    return False


def move_features(board, move: Dict[str, Any], player: str) -> np.ndarray:
    """20-float move-specific block for ``move`` applied by ``player``."""
    opp = _opponent(player)
    out = np.zeros(MOVE_DIM, dtype=np.float32)

    frm = move.get("from")
    to = move.get("to")
    cap = move.get("capture")

    out[0] = (_POS_INDEX.get(frm, 0) / _N_POS) if frm is not None else 0.0
    out[1] = (_POS_INDEX.get(to, 0) / _N_POS) if to is not None else 0.0
    out[2] = 1.0 if frm is None else 0.0          # is_placement
    out[4] = 1.0 if cap else 0.0                  # is_capture
    out[5] = (_POS_INDEX.get(cap, 0) / _N_POS) if cap else 0.0

    apply_dict = {"from": frm, "to": to, "capture": cap}
    after = None
    try:
        after = board.apply_move(apply_dict)
    except Exception:
        after = None

    is_mill_closing = False
    if after is not None and to is not None:
        try:
            is_mill_closing = after.is_mill(to, player)
        except Exception:
            is_mill_closing = False
    out[3] = 1.0 if is_mill_closing else 0.0

    # would_create_double_mill: closing produces a piece in >=2 mills.
    would_double = False
    if after is not None and is_mill_closing:
        would_double = _double_mill_pieces(after, player) > _double_mill_pieces(board, player)
    out[6] = 1.0 if would_double else 0.0

    out[7] = 1.0 if _would_block_opponent_mill(board, move, player) else 0.0

    if after is not None:
        out[8] = min(1.0, _piece_count(after, player) / 9.0)
        out[9] = min(1.0, _piece_count(after, opp) / 9.0)
        out[10] = min(1.0, _mobility(after, player) / _N_POS)
        out[11] = min(1.0, _mobility(after, opp) / _N_POS)
        out[12] = min(1.0, _mills_count(after, player) / 3.0)
        out[13] = min(1.0, _mills_count(after, opp) / 3.0)
    else:
        out[8] = min(1.0, _piece_count(board, player) / 9.0)
        out[9] = min(1.0, _piece_count(board, opp) / 9.0)

    deg = _DEGREE.get(to, 0) if to is not None else 0
    out[14] = 1.0 if deg >= 3 else 0.0            # strong / junction
    out[15] = 1.0 if deg == 2 else 0.0            # corner

    reduces_mobility = False
    if after is not None:
        reduces_mobility = _mobility(after, player) < _mobility(board, player)
    out[16] = 1.0 if reduces_mobility else 0.0

    out[17] = 1.0 if (after is not None and _opens_new_mill_threat(after, player)) else 0.0
    # [18:20) reserved
    return out


# ── counterfactual context ──────────────────────────────────────────────────────

def _move_key(mv: Dict[str, Any]):
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def counterfactual_features(
    move: Dict[str, Any],
    all_moves: Optional[List[Dict[str, Any]]],
    heuristic_rank: int = 0,
    n_legal: int = 0,
    heuristic_score_norm: float = 0.5,
) -> np.ndarray:
    """18-float counterfactual block for ``move`` within its candidate set.

    ``all_moves`` is the list from ``ExternalSolvedDB.query_all_moves`` (dicts
    with ``move`` and ``wdl``). When None/empty the DB-derived slots stay zero.
    """
    out = np.zeros(COUNTERFACTUAL_DIM, dtype=np.float32)
    moves = list(all_moves or [])
    n_db = len(moves)
    n = n_legal if n_legal > 0 else n_db

    out[0] = min(1.0, n / _N_POS) if n > 0 else 0.0

    if n_db > 0:
        n_win = sum(1 for m in moves if m.get("wdl") == "win")
        n_loss = sum(1 for m in moves if m.get("wdl") == "loss")
        n_draw = sum(1 for m in moves if m.get("wdl") == "draw")
        out[1] = n_win / n_db
        out[2] = n_loss / n_db
        out[3] = n_draw / n_db
        # best/worst available WDL
        if n_win > 0:
            out[4] = 1.0
        elif n_draw > 0:
            out[4] = 0.5
        else:
            out[4] = 0.0
        if n_loss > 0:
            out[5] = 0.0
        elif n_draw > 0:
            out[5] = 0.5
        else:
            out[5] = 1.0
        out[8] = 1.0 if n_win > 0 else 0.0
        out[9] = 1.0 if n_loss > 0 else 0.0

        played_wdl = next(
            (m.get("wdl") for m in moves if _move_key(m.get("move", {})) == _move_key(move)),
            "unknown",
        )
        out[10] = 1.0 if played_wdl == "win" else 0.0
        out[11] = 1.0 if played_wdl == "loss" else 0.0
        out[12] = 1.0 if played_wdl == "draw" else 0.0
        out[13] = 1.0 if played_wdl in ("win", "draw", "loss") else 0.0
        out[14] = 1.0  # db_available
        played_v = _WDL_SCALAR.get(played_wdl, 0.0)
        out[15] = 1.0 if out[4] > played_v else 0.0  # a strictly better DB move existed

    if n > 0:
        out[6] = min(1.0, heuristic_rank / float(max(1, n)))
    out[7] = float(min(1.0, max(0.0, heuristic_score_norm)))
    # [16:18) reserved
    return out


# ── public assembly ─────────────────────────────────────────────────────────────

def build_move_features(
    board,
    move: Dict[str, Any],
    player: str,
    move_ctx: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Return the FEATURE_DIM-float per-move feature vector.

    ``move`` is an apply-move dict {from,to,capture}. ``player`` is the mover.
    ``move_ctx`` may carry (all optional):
      all_moves:            query_all_moves() output for counterfactual block
      heuristic_rank:       int rank of this move (0 = top)
      n_legal:              int number of legal moves
      heuristic_score_norm: float [0,1] heuristic score across candidates
    """
    ctx = move_ctx or {}
    board_block = board_context_features(board, player)
    mv_block = move_features(board, move, player)
    cf_block = counterfactual_features(
        move,
        ctx.get("all_moves"),
        heuristic_rank=int(ctx.get("heuristic_rank", 0) or 0),
        n_legal=int(ctx.get("n_legal", 0) or 0),
        heuristic_score_norm=float(ctx.get("heuristic_score_norm", 0.5)),
    )
    return np.concatenate([board_block, mv_block, cf_block]).astype(np.float32)
