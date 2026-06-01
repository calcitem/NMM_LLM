"""ai/endgame_db.py — Position-based endgame learning database.

Indexes every endgame position encountered across all saved game files so the
AI can ask: "from this exact board position, which moves have historically led
to wins?"

Unlike TrajectoryDB (which indexes by move-notation prefix), EndgameDB indexes
by the exact board state reached after placement is complete and total pieces
fall to endgame threshold (≤11).  D4 board symmetry is used so all 8 rotations
and reflections of the same position share statistics.

After each game, call add_game() to keep the index current.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ai.board_symmetry import (
    canonical_board_str,
    board_query_canonicals,
    transform_notation,
    SYM_INVERSE,
)

logger = logging.getLogger(__name__)

# Index positions that occur at or below this total-piece count (post-placement).
_ENDGAME_PIECE_THRESHOLD = 11


def _total_pieces(board_fen: str) -> int:
    board_part = board_fen.split("|")[0]
    return board_part.count("W") + board_part.count("B")


def _placement_done(board_fen: str) -> bool:
    parts = board_fen.split("|")
    if len(parts) < 4:
        return False
    try:
        return int(parts[2]) >= 9 and int(parts[3]) >= 9
    except ValueError:
        return False


class EndgameDB:
    """
    In-memory index of historical endgame positions.

    The index maps  canon_key → {canon_notation → outcome_counts}
    where canon_key is "<canonical-24-char board>|<turn>" (D4 lex-min form)
    and outcome_counts is {"W": int, "B": int, "D": int, "total": int}.

    All 8 D4 equivalents of a position share the same bucket, multiplying
    effective sample size by up to 8×.

    query() returns a per-move score delta (positive = historically good for
    the side to move, negative = bad) so the engine can favour historically
    winning continuations.
    """

    def __init__(self, games_dir: Path | str) -> None:
        self._games_dir = Path(games_dir)
        self._index: dict[str, dict[str, dict]] = {}
        self._game_count = 0

    # ── Build / update ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Index every *.jsonl file in the games directory from scratch."""
        self._index.clear()
        self._game_count = 0
        if not self._games_dir.exists():
            logger.warning("EndgameDB: games directory not found: %s", self._games_dir)
            return
        for path in sorted(self._games_dir.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._index_game(json.loads(line))
                except Exception as exc:
                    logger.debug("EndgameDB: skipping line in %s — %s", path.name, exc)
        logger.info(
            "EndgameDB: indexed %d games → %d position entries.",
            self._game_count, len(self._index),
        )

    def _index_game(self, record: dict) -> None:
        # Skip adaptive-softened games — blunder-inflated play pollutes the library.
        if record.get("adaptive_softened"):
            return
        winner = record.get("winner")
        moves = record.get("moves", [])
        if not moves:
            return

        self._game_count += 1

        for move in moves:
            fen_before = move.get("board_fen_before", "")
            notation = move.get("notation", "")
            if not fen_before or not notation:
                continue
            if not _placement_done(fen_before):
                continue
            if _total_pieces(fen_before) > _ENDGAME_PIECE_THRESHOLD:
                continue

            parts = fen_before.split("|")
            board_24 = parts[0]
            turn = parts[1] if len(parts) >= 2 else "W"

            canon_board, sym_idx = canonical_board_str(board_24)
            canon_notation = transform_notation(notation, sym_idx)
            if canon_notation is None:
                continue

            key = f"{canon_board}|{turn}"
            bucket = self._index.setdefault(key, {})
            entry = bucket.setdefault(canon_notation, {"W": 0, "B": 0, "D": 0, "total": 0})
            entry["total"] += 1
            if winner in ("W", "B"):
                entry[winner] += 1
            else:
                entry["D"] += 1

    def add_game(self, record: dict) -> None:
        """Incrementally add one completed game without a full reload."""
        self._index_game(record)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        board,              # BoardState — provides to_fen_string()
        current_color: str,
        min_samples: int = 1,
    ) -> dict[str, float]:
        """
        Return a score-delta dict for candidate next-move notations.

        Positive delta → this move historically correlates with current_color
                         winning (max +0.5 at 100 % win rate).
        Negative delta → correlates with a loss (min −0.5).
        Returns {}     when no endgame data exist for this position.

        All 8 D4 symmetric equivalents of the board are queried and their
        statistics are merged; results are inverse-transformed back to the
        actual board orientation.
        """
        fen = board.to_fen_string()
        parts = fen.split("|")
        board_24 = parts[0]
        turn = parts[1] if len(parts) >= 2 else "W"

        merged: dict[str, dict] = {}

        for canon_board, sym_idx in board_query_canonicals(board_24):
            key = f"{canon_board}|{turn}"
            candidates = self._index.get(key)
            if not candidates:
                continue
            inv = SYM_INVERSE[sym_idx]
            for canon_notation, stats in candidates.items():
                actual_notation = transform_notation(canon_notation, inv)
                if actual_notation is None:
                    continue
                if actual_notation not in merged:
                    merged[actual_notation] = {"W": 0, "B": 0, "D": 0, "total": 0}
                entry = merged[actual_notation]
                entry["total"] += stats["total"]
                entry["W"]     += stats["W"]
                entry["B"]     += stats["B"]
                entry["D"]     += stats["D"]

        if not merged:
            return {}

        total_samples = sum(c["total"] for c in merged.values())
        if total_samples < min_samples:
            return {}

        result: dict[str, float] = {}
        for notation, stats in merged.items():
            total = stats["total"]
            if total == 0:
                continue
            wins  = stats.get(current_color, 0)
            draws = stats.get("D", 0)
            score = (wins + 0.4 * draws) / total
            result[notation] = score - 0.5
        return result

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def game_count(self) -> int:
        return self._game_count

    @property
    def position_count(self) -> int:
        return len(self._index)
