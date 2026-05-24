"""ai/fullgame_db.py — Read-only query interface for the full-game position DB.

Companion to ``tools/build_fullgame_db.py``.  The database is built offline
(potentially over many hours) and consulted at query time by the GameAI.

Design contract
---------------
The DB is *optional*.  When ``FullGameDB.is_available()`` returns False the
GameAI falls back to its normal negamax search.  When the DB IS available but
the queried position is not present, ``query()`` returns ``None`` (also a
fallback signal).  Only when an exact canonical hit is found does the DB
override the search — and even then GameAI may blend the result with the
heuristic via ``score_delta()``.

The schema (see build_fullgame_db.py) stores:
    positions(key BLOB PK, outcome INT, depth INT, best_move TEXT,
              trajectories TEXT, samples INT)

`key` is a packed 9-byte canonical position id.  We rely on the existing
``ai.board_symmetry`` D4 helpers for canonicalization, so the DB only stores
one representative per equivalence class.

Public surface
--------------
    FullGameDB(path)
        .is_available()
        .query(board: BoardState) -> FullGameResult | None
        .score_delta(board, current_color) -> dict[str, float]
        .best_move(board) -> str | None        # actual-board notation
        .close()
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from game.board import BoardState
from .board_symmetry import (
    SYM_INVERSE,
    canonical_board_str,
    transform_notation,
)

logger = logging.getLogger(__name__)


# Same packing as the builder.  Kept duplicated here so this module has no
# tools/ dependency — the AI must never need to import the builder.
_PIECE_BITS = {".": 0b00, "W": 0b01, "B": 0b10}


def _encode_canonical(board24: str, turn: str, placed_w: int, placed_b: int) -> bytes:
    val = 0
    for i, ch in enumerate(board24):
        val |= _PIECE_BITS[ch] << (i * 2)
    return val.to_bytes(6, "little") + bytes(
        (0 if turn == "W" else 1, placed_w & 0xFF, placed_b & 0xFF)
    )


def _unpack_trajectories(blob: str) -> list[tuple[str, bytes, str]]:
    if not blob:
        return []
    out = []
    for part in blob.split("|"):
        try:
            n, ck, f = part.rsplit(":", 2)
        except ValueError:
            continue
        try:
            out.append((n, bytes.fromhex(ck), f))
        except ValueError:
            continue
    return out


@dataclass
class FullGameResult:
    """One row from the position table, with the symmetry index used to
    transform stored canonical move notations back into the actual board
    orientation the caller asked about."""

    outcome: Optional[int]          # 1=W win, -1=B win, 0=draw, None=unknown
    depth: Optional[int]            # plies to result, or None
    best_move_canonical: Optional[str]
    sym_idx: int                    # transform that maps actual board → canonical
    trajectories: list[tuple[str, bytes, str]]  # (canonical_notation, child_key, flag)


class FullGameDB:
    """Read-only wrapper around the position SQLite database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None
        if self.path.exists():
            try:
                self._conn = sqlite3.connect(
                    f"file:{self.path}?mode=ro", uri=True, check_same_thread=False,
                )
                # Sanity-check schema
                self._conn.execute("SELECT key FROM positions LIMIT 1").fetchone()
                logger.info("FullGameDB: opened %s (read-only).", self.path)
            except sqlite3.Error as exc:
                logger.warning("FullGameDB: could not open %s — %s", self.path, exc)
                self._conn = None

    def is_available(self) -> bool:
        return self._conn is not None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, board: BoardState) -> Optional[FullGameResult]:
        """Look up the canonical position for `board`.  Returns None on miss."""
        if self._conn is None:
            return None

        fen = board.to_fen_string()
        board24, turn, pw, pb = fen.split("|")
        canon, sym = canonical_board_str(board24)
        key = _encode_canonical(canon, turn, int(pw), int(pb))

        row = self._conn.execute(
            "SELECT outcome, depth, best_move, trajectories FROM positions WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        outcome, depth, best_move, traj_blob = row
        return FullGameResult(
            outcome=outcome,
            depth=depth,
            best_move_canonical=best_move,
            sym_idx=sym,
            trajectories=_unpack_trajectories(traj_blob or ""),
        )

    # ── Convenience helpers used by GameAI ──────────────────────────────────

    def best_move(self, board: BoardState) -> Optional[str]:
        """Return the best move notation in the actual board's orientation."""
        result = self.query(board)
        if result is None or not result.best_move_canonical:
            return None
        inv = SYM_INVERSE[result.sym_idx]
        return transform_notation(result.best_move_canonical, inv)

    def score_delta(self, board: BoardState, current_color: str) -> dict[str, float]:
        """Return a per-move score delta in [-0.5, +0.5] compatible with the
        TrajectoryDB / EndgameDB hint interface used by GameAI.

        Mapping:
            move flagged 'W' (winning for side-to-move) → +0.5
            move flagged 'L' (losing)                   → -0.5
            move flagged 'N' (neutral / unknown)        →  0.0

        Returns {} on miss so GameAI can fall back cleanly.
        """
        result = self.query(board)
        if result is None or not result.trajectories:
            return {}
        inv = SYM_INVERSE[result.sym_idx]
        # current_color is the colour whose turn it is.  Stored flags are
        # already from the side-to-move perspective (the builder writes 'W'
        # iff the move wins for side-to-move), so no extra sign flip needed.
        out: dict[str, float] = {}
        for canon_notation, _child_key, flag in result.trajectories:
            actual = transform_notation(canon_notation, inv)
            if actual is None:
                continue
            if flag == "W":
                out[actual] = 0.5
            elif flag == "L":
                out[actual] = -0.5
            else:
                out[actual] = 0.0
        return out

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        if self._conn is None:
            return {"available": 0}
        total = self._conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        resolved = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        return {"available": 1, "positions": total, "resolved": resolved}
