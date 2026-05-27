"""ai/endgame_solved_db.py — Retrograde endgame database for post-placement positions.

Stores WDL (Win/Draw/Loss) results for positions where all pieces have been
placed and both sides have between 3 and 7 pieces on the board.  The database
uses a direct O(1) combinatorial index — no binary search needed.

Position ID formula (general: any nW, nB ≥ 3)
-----------------------------------------------
white_rank  = combo_rank(sorted(W_indices), 24)           # C(24,nW) possible values
remaining   = sorted([0..23] \\ W_indices)                 # 24-nW squares
B_remapped  = [remaining.index(b) for b in sorted(B_indices)]
black_rank  = combo_rank(B_remapped, 24-nW)                # C(24-nW,nB) possible values
pos_id      = white_rank * C(24-nW,nB) * 2 + black_rank * 2 + turn_bit

File names: endgame_{nW}_{nB}.wdl in the configured database directory.
The 3v3 table (5,383,840 positions, ~1.3 MB) is the base case.

WDL encoding (2 bits per slot, packed 4 per byte)
-------------------------------------------------
  0 = unknown / unsolved
  1 = W (side to move wins)
  2 = L (side to move loses)
  3 = D (draw)

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

def _table_size(nW: int, nB: int) -> int:
    return comb(_N_POS, nW) * comb(_N_POS - nW, nB) * 2


def _expected_bytes(nW: int, nB: int) -> int:
    return (_table_size(nW, nB) + 3) >> 2


class EndgameSolvedDB:
    """Read-only query interface for the retrograde endgame WDL database.

    Loads every endgame_{nW}_{nB}.wdl file found in db_dir on construction.
    ``is_available()`` returns True when at least one table is loaded.
    ``query()`` dispatches to the correct table by piece count, returning
    "W", "L", "D", or None.
    """

    def __init__(self, db_dir: str | Path | None) -> None:
        self._tables: dict[tuple[int, int], bytes] = {}
        if db_dir is None:
            return
        db_path = Path(db_dir)
        if not db_path.is_dir():
            return
        for wdl_path in sorted(db_path.glob("endgame_*.wdl")):
            stem = wdl_path.stem  # e.g. "endgame_3_3"
            parts = stem.split("_")
            if len(parts) != 3:
                continue
            try:
                nW, nB = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if nW < 3 or nB < 3:
                continue
            expected = _expected_bytes(nW, nB)
            try:
                data = wdl_path.read_bytes()
            except OSError as exc:
                logger.warning("EndgameSolvedDB: could not read %s: %s", wdl_path, exc)
                continue
            if len(data) != expected:
                logger.warning(
                    "EndgameSolvedDB: %s has wrong size %d (expected %d) — skipped.",
                    wdl_path, len(data), expected,
                )
                continue
            self._tables[(nW, nB)] = data
            logger.info(
                "EndgameSolvedDB: loaded %s (%d positions)", wdl_path, _table_size(nW, nB)
            )

    def is_available(self) -> bool:
        return bool(self._tables)

    def query(self, board: BoardState) -> Optional[str]:
        """Return "W" (side to move wins), "L" (loses), "D" (draw), or None.

        Returns None when:
        - no table for the current (nW, nB) piece count is loaded
        - either side has not yet placed all 9 pieces (pre-placement phase)
        - the position ID resolves to WDL_UNKNOWN (should not occur in solved tables)
        """
        if not self._tables:
            return None
        if board.pieces_placed.get("W", 0) < 9 or board.pieces_placed.get("B", 0) < 9:
            return None
        w_pieces = [p for p, owner in board.positions.items() if owner == "W"]
        b_pieces = [p for p, owner in board.positions.items() if owner == "B"]
        nW, nB = len(w_pieces), len(b_pieces)
        table = self._tables.get((nW, nB))
        if table is None:
            return None
        try:
            pos_id = encode_position_id(w_pieces, b_pieces, board.turn)
        except (KeyError, ValueError, IndexError):
            return None
        val = get_wdl(table, pos_id)
        return _WDL_CHAR[val]

    def close(self) -> None:
        self._tables.clear()
