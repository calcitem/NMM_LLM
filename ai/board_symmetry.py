"""ai/board_symmetry.py — D4 board symmetry helpers.

Provides canonical-form utilities shared by TrajectoryDB and EndgameDB.
Pooling symmetric positions multiplies effective sample size by up to 8×.

The D4 group has 8 elements: 4 rotations (0°, 90°, 180°, 270°) and 4
reflections (x-axis, y-axis, main diagonal, anti-diagonal).  The centre of
the NMM board is d4 in centred coordinates.

All public functions are side-effect free and safe to call from any thread.
"""

from __future__ import annotations
from typing import Optional

# ── Position tables ───────────────────────────────────────────────────────────

# Exact order used by BoardState.to_fen_string() / game/board.py POSITIONS.
_POSITIONS: list[str] = [
    "a7", "d7", "g7", "g4", "g1", "d1", "a1", "a4",  # outer ring
    "b6", "d6", "f6", "f4", "f2", "d2", "b2", "b4",  # middle ring
    "c5", "d5", "e5", "e4", "e3", "d3", "c3", "c4",  # inner ring
]
_POS_IDX: dict[str, int] = {p: i for i, p in enumerate(_POSITIONS)}

# Centred coordinates: file a=−3, d=0, g=3; rank 1→−3, 4→0, 7→+3
_POSITION_COORDS: dict[str, tuple[int, int]] = {
    "a7": (-3,  3), "d7": (0,  3), "g7": (3,  3),
    "g4": ( 3,  0), "g1": (3, -3), "d1": (0, -3), "a1": (-3, -3), "a4": (-3,  0),
    "b6": (-2,  2), "d6": (0,  2), "f6": (2,  2),
    "f4": ( 2,  0), "f2": (2, -2), "d2": (0, -2), "b2": (-2, -2), "b4": (-2,  0),
    "c5": (-1,  1), "d5": (0,  1), "e5": (1,  1),
    "e4": ( 1,  0), "e3": (1, -1), "d3": (0, -1), "c3": (-1, -1), "c4": (-1,  0),
}
_COORDS_POSITION: dict[tuple[int, int], str] = {v: k for k, v in _POSITION_COORDS.items()}

# ── D4 group ──────────────────────────────────────────────────────────────────
# Matrix (a,b,c,d): (x,y) → (ax+by, cx+dy)

_SYMMETRIES: list[tuple[int, int, int, int]] = [
    ( 1,  0,  0,  1),  # 0: identity
    ( 0, -1,  1,  0),  # 1: 90° CCW       inverse → 3
    (-1,  0,  0, -1),  # 2: 180°          inverse → 2
    ( 0,  1, -1,  0),  # 3: 270° CCW      inverse → 1
    (-1,  0,  0,  1),  # 4: flip x-axis   inverse → 4
    ( 1,  0,  0, -1),  # 5: flip y-axis   inverse → 5
    ( 0,  1,  1,  0),  # 6: main diagonal inverse → 6
    ( 0, -1, -1,  0),  # 7: anti-diagonal inverse → 7
]
SYM_INVERSE: list[int] = [0, 3, 2, 1, 4, 5, 6, 7]

# ── Pre-computed board permutations ───────────────────────────────────────────
# _BOARD_PERM[sym_idx] maps POSITIONS index → new POSITIONS index.
# Built once at module load; used by _apply_board_sym for O(24) transforms.

_BOARD_PERM: list[Optional[list[int]]] = []

def _build_board_perms() -> None:
    for a, b, c, d in _SYMMETRIES:
        perm: list[int] = []
        valid = True
        for pos in _POSITIONS:
            x, y = _POSITION_COORDS[pos]
            new_pos = _COORDS_POSITION.get((a * x + b * y, c * x + d * y))
            if new_pos is None:
                valid = False
                break
            perm.append(_POS_IDX[new_pos])
        _BOARD_PERM.append(perm if valid else None)

_build_board_perms()


# ── Single-position transform ─────────────────────────────────────────────────

def transform_pos(pos: str, sym_idx: int) -> Optional[str]:
    """Transform one position label under symmetry sym_idx; None if unmapped."""
    if sym_idx == 0:
        return pos
    coords = _POSITION_COORDS.get(pos)
    if coords is None:
        return None
    x, y = coords
    a, b, c, d = _SYMMETRIES[sym_idx]
    return _COORDS_POSITION.get((a * x + b * y, c * x + d * y))


# ── Notation transform ────────────────────────────────────────────────────────

def transform_notation(notation: str, sym_idx: int) -> Optional[str]:
    """
    Transform a move notation string by symmetry sym_idx.

    Handles all formats used by GameEngine:
        "d2"         placement
        "a1-b1"      movement
        "a1-b1xa4"   movement + capture
        "d2xa4"      placement + capture (rare)
    Returns None if any position is unmapped by this symmetry.
    """
    if sym_idx == 0:
        return notation

    cap_suffix = ""
    base = notation
    if "x" in notation:
        xi = notation.index("x")
        base = notation[:xi]
        t_cap = transform_pos(notation[xi + 1:], sym_idx)
        if t_cap is None:
            return None
        cap_suffix = f"x{t_cap}"

    if "-" in base:
        from_pos, to_pos = base.split("-", 1)
        t_from = transform_pos(from_pos, sym_idx)
        t_to   = transform_pos(to_pos,   sym_idx)
        if t_from is None or t_to is None:
            return None
        return f"{t_from}-{t_to}{cap_suffix}"

    t_pos = transform_pos(base, sym_idx)
    if t_pos is None:
        return None
    return f"{t_pos}{cap_suffix}"


# ── Board-string (FEN) transforms ────────────────────────────────────────────

def _apply_board_sym(board_24: str, sym_idx: int) -> Optional[str]:
    """Apply symmetry sym_idx to the 24-char POSITIONS-ordered board string."""
    if sym_idx == 0:
        return board_24
    perm = _BOARD_PERM[sym_idx]
    if perm is None:
        return None
    result = ['?'] * 24
    for old_idx, new_idx in enumerate(perm):
        result[new_idx] = board_24[old_idx]
    return "".join(result)


def canonical_board_str(board_24: str) -> tuple[str, int]:
    """
    Return ``(canonical_str, sym_idx)`` where canonical_str is the
    lexicographically smallest D4 transform of board_24.
    sym_idx is the lowest-index transform that achieves the minimum (tiebreaker
    for symmetric boards, ensuring a single deterministic inverse transform).
    """
    best = board_24
    best_idx = 0
    for sym_idx in range(1, 8):
        t = _apply_board_sym(board_24, sym_idx)
        if t is not None and t < best:
            best = t
            best_idx = sym_idx
    return best, best_idx


def board_query_canonicals(board_24: str) -> list[tuple[str, int]]:
    """
    Return all ``(unique_canonical, lowest_sym_idx)`` pairs for a query board.

    Typically returns 8 entries (one per D4 element).  For a board that is
    invariant under some non-identity transform (true board symmetry), multiple
    elements produce the same string — only the lowest-index one is kept.
    Each returned ``(canonical, sym_idx)`` pair can be looked up in the index;
    move notations found there must be inverse-transformed by
    ``SYM_INVERSE[sym_idx]`` to recover actual board moves.
    """
    seen: dict[str, int] = {}  # canonical → lowest sym_idx that produced it
    for sym_idx in range(8):
        t = _apply_board_sym(board_24, sym_idx)
        if t is not None and t not in seen:
            seen[t] = sym_idx
    return list(seen.items())


# ── Sequence (notation-list) transforms ──────────────────────────────────────

def _transform_sequence(notations: list[str], sym_idx: int) -> Optional[list[str]]:
    if sym_idx == 0:
        return notations
    result: list[str] = []
    for n in notations:
        t = transform_notation(n, sym_idx)
        if t is None:
            return None
        result.append(t)
    return result


def canonical_sequence(notations: list[str]) -> tuple[list[str], int]:
    """
    Return ``(canonical_sequence, sym_idx)`` for a move sequence.
    The canonical form is the D4 transform that yields the lexicographically
    smallest pipe-joined sequence.  Lowest-index sym is the tiebreaker.
    """
    best_joined = "|".join(notations)
    best_seq = notations
    best_idx = 0
    for sym_idx in range(1, 8):
        transformed = _transform_sequence(notations, sym_idx)
        if transformed is None:
            continue
        j = "|".join(transformed)
        if j < best_joined:
            best_joined = j
            best_seq = transformed
            best_idx = sym_idx
    return best_seq, best_idx


def prefix_query_canonicals(notations: list[str], depth: int) -> list[tuple[str, int]]:
    """
    Return all ``(unique_canonical_prefix, lowest_sym_idx)`` pairs for a
    query prefix of length depth.  Analogous to board_query_canonicals but
    for notation sequences.
    """
    prefix = notations[:depth]
    seen: dict[str, int] = {}
    for sym_idx in range(8):
        transformed = _transform_sequence(prefix, sym_idx)
        if transformed is None:
            continue
        key = "|".join(transformed)
        if key not in seen:
            seen[key] = sym_idx
    return list(seen.items())
