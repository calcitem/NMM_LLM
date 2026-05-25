"""ai/endgame_solved_db.py — Retrograde endgame database for ≤6-piece positions.

Stores WDL (Win/Draw/Loss) results for positions where each side has exactly 3
pieces (fly-phase positions).  The database uses a direct O(1) combinatorial
index — no binary search needed.

Position ID formula (v1: nW = nB = 3)
---------------------------------------
white_rank  = combo_rank(sorted(W_indices), 24)     # in [0, C(24,3)) = [0, 2024)
remaining   = sorted([0..23] \\ W_indices)           # 21 squares not occupied by W
B_remapped  = [remaining.index(b) for b in sorted(B_indices)]
black_rank  = combo_rank(B_remapped, 21)             # in [0, C(21,3)) = [0, 1330)
pos_id      = white_rank * 1330 * 2 + black_rank * 2 + turn_bit

Total slots: C(24,3) × C(21,3) × 2 = 2024 × 1330 × 2 = 5,383,840

WDL encoding (2 bits per slot, packed 4 per byte)
-------------------------------------------------
  0 = unknown / unsolved
  1 = W (side to move wins)
  2 = L (side to move loses)
  3 = D (draw)
Stored in a flat byte array of length ceil(5,383,840 / 4) = 1,345,960 bytes.

Public surface
--------------
    EndgameSolvedDB(db_dir)
        .is_available()
        .query(board: BoardState) -> "W" | "L" | "D" | None
        .close()
"""

from __future__ import annotations

import logging
from math import comb
from pathlib import Path
from typing import Optional

from game.board import BoardState, POSITIONS

logger = logging.getLogger(__name__)

# ── Board constants ────────────────────────────────────────────────────────────

_POS_TO_IDX: dict[str, int] = {pos: i for i, pos in enumerate(POSITIONS)}
_N_POS = 24

# ── Combinatorial number system ────────────────────────────────────────────────

def combo_rank(sorted_indices: list[int]) -> int:
    """Rank a k-subset of {0..n-1} given as a sorted ascending list.

    Combinatorial number system: rank = Σ_i C(c_i, i+1).
    Returns 0 for an empty list.
    """
    return sum(comb(c, i + 1) for i, c in enumerate(sorted_indices))


def combo_unrank(rank: int, k: int, n: int) -> list[int]:
    """Recover the sorted k-subset of {0..n-1} with the given combinatorial rank.

    Greedy from the highest index downward.
    """
    result: list[int] = []
    remaining = rank
    upper = n - 1
    for i in range(k, 0, -1):
        c = upper
        while c >= i - 1 and comb(c, i) > remaining:
            c -= 1
        remaining -= comb(c, i)
        result.append(c)
        upper = c - 1
    result.reverse()
    return result


# ── Position ID encoding ───────────────────────────────────────────────────────

def encode_position_id(
    white_pieces: list[str], black_pieces: list[str], turn: str
) -> int:
    """Encode a fly-phase board position to a direct integer index.

    white_pieces / black_pieces: lists of position labels (e.g. ["a7", "d5", "c3"])
    turn: "W" or "B"

    Returns an int in [0, TABLE_SIZE_3_3) for (3,3) positions.
    Raises KeyError if any label is not a valid board position.
    """
    nW = len(white_pieces)
    nB = len(black_pieces)
    W = sorted(_POS_TO_IDX[p] for p in white_pieces)
    B_raw = sorted(_POS_TO_IDX[p] for p in black_pieces)
    W_set = set(W)
    remaining = [i for i in range(_N_POS) if i not in W_set]
    B_remapped = [remaining.index(b) for b in B_raw]
    white_rank = combo_rank(W)
    black_rank = combo_rank(B_remapped)
    black_combinations = comb(_N_POS - nW, nB)
    turn_bit = 0 if turn == "W" else 1
    return white_rank * black_combinations * 2 + black_rank * 2 + turn_bit


def decode_position_id(
    pos_id: int, nW: int, nB: int
) -> tuple[list[str], list[str], str]:
    """Inverse of encode_position_id.

    Returns (white_pieces, black_pieces, turn) as lists of position labels.
    """
    n_remaining = _N_POS - nW
    black_combinations = comb(n_remaining, nB)
    turn_bit = pos_id & 1
    remainder = pos_id >> 1
    black_rank = remainder % black_combinations
    white_rank = remainder // black_combinations
    W = combo_unrank(white_rank, nW, _N_POS)
    W_set = set(W)
    remaining = [i for i in range(_N_POS) if i not in W_set]
    B_remapped = combo_unrank(black_rank, nB, n_remaining)
    B = [remaining[j] for j in B_remapped]
    white_pieces = [POSITIONS[i] for i in W]
    black_pieces = [POSITIONS[i] for i in B]
    turn = "W" if turn_bit == 0 else "B"
    return white_pieces, black_pieces, turn


# ── Table size constants ───────────────────────────────────────────────────────

TABLE_SIZE_3_3: int = comb(24, 3) * comb(21, 3) * 2  # = 5,383,840
_WDL_BYTES_3_3: int = (TABLE_SIZE_3_3 + 3) >> 2       # = 1,345,960

# ── WDL value constants ────────────────────────────────────────────────────────

WDL_UNKNOWN = 0
WDL_WIN = 1   # side to move wins
WDL_LOSS = 2  # side to move loses
WDL_DRAW = 3

_WDL_CHAR: dict[int, Optional[str]] = {
    WDL_WIN: "W",
    WDL_LOSS: "L",
    WDL_DRAW: "D",
    WDL_UNKNOWN: None,
}

# ── WDL bit-packing helpers ────────────────────────────────────────────────────

def get_wdl(table: bytes | bytearray, pos_id: int) -> int:
    """Read the 2-bit WDL value for pos_id from a packed byte table."""
    byte_idx = pos_id >> 2
    shift = (pos_id & 3) << 1
    return (table[byte_idx] >> shift) & 3


def set_wdl(table: bytearray, pos_id: int, value: int) -> None:
    """Write the 2-bit WDL value for pos_id into a mutable packed byte table."""
    byte_idx = pos_id >> 2
    shift = (pos_id & 3) << 1
    table[byte_idx] = (table[byte_idx] & ~(3 << shift)) | ((value & 3) << shift)


# ── EndgameSolvedDB ────────────────────────────────────────────────────────────

class EndgameSolvedDB:
    """Read-only query interface for the retrograde endgame WDL database.

    The DB is optional: ``is_available()`` returns False when no WDL file is
    found in db_dir.  When available, ``query()`` returns "W", "L", "D", or
    None.  None means the position is outside scope or has not been solved.
    """

    _WDL_FILENAME = "endgame_3_3.wdl"

    def __init__(self, db_dir: str | Path | None) -> None:
        self._table_3_3: Optional[bytes] = None
        if db_dir is None:
            return
        wdl_path = Path(db_dir) / self._WDL_FILENAME
        if not wdl_path.exists():
            return
        try:
            data = wdl_path.read_bytes()
            if len(data) != _WDL_BYTES_3_3:
                logger.warning(
                    "EndgameSolvedDB: %s has wrong size %d (expected %d)",
                    wdl_path, len(data), _WDL_BYTES_3_3,
                )
                return
            self._table_3_3 = data
            logger.info("EndgameSolvedDB: loaded %s (%d positions)", wdl_path, TABLE_SIZE_3_3)
        except OSError as exc:
            logger.warning("EndgameSolvedDB: could not read %s: %s", wdl_path, exc)

    def is_available(self) -> bool:
        return self._table_3_3 is not None

    def query(self, board: BoardState) -> Optional[str]:
        """Return "W" (side to move wins), "L" (loses), "D" (draw), or None.

        Returns None when:
        - the DB is not loaded
        - either side does not have exactly 3 pieces on the board
        - neither side has fully placed all 9 pieces (not yet fly phase)
        - the position has not been solved (WDL_UNKNOWN)
        """
        if self._table_3_3 is None:
            return None
        w_pieces = [p for p, owner in board.positions.items() if owner == "W"]
        b_pieces = [p for p, owner in board.positions.items() if owner == "B"]
        if len(w_pieces) != 3 or len(b_pieces) != 3:
            return None
        if board.pieces_placed.get("W", 0) < 9 or board.pieces_placed.get("B", 0) < 9:
            return None
        try:
            pos_id = encode_position_id(w_pieces, b_pieces, board.turn)
        except (KeyError, ValueError, IndexError):
            return None
        val = get_wdl(self._table_3_3, pos_id)
        return _WDL_CHAR[val]

    def close(self) -> None:
        self._table_3_3 = None
