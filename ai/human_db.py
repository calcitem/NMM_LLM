"""ai/human_db.py — Runtime read/write adapter for data/human_db.sqlite.

Drop-in replacement for TrajectoryDB + MalomDB (for positions in the human
corpus).  Opens a single SQLite file at startup instead of scanning thousands
of JSONL game files.

Duck-typing compatibility with TrajectoryDB:
  query()                 → same signature & semantics
  query_all_frequencies() → same signature & semantics
  query_line()            → same signature & semantics
  add_game()              → incremental SQLite upsert (persistent across restarts)
  game_count              → property
  entry_count             → property

Additional methods for the 3D explorer (Phase 2):
  query_position()
  query_moves()
  get_malom_wdl()
  canonical_winning_line()
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ai.trajectory_db import make_board_state_key, _norm
from ai.board_symmetry import transform_notation, SYM_INVERSE

if TYPE_CHECKING:
    from game.board import BoardState

log = logging.getLogger(__name__)


@dataclass
class MoveStats:
    notation: str
    wins: int
    losses: int
    draws: int
    total: int
    win_pct: float                # wins / total
    avg_moves_to_end: float       # human-game average plies remaining after this move
    malom_wdl_after: str | None   # 'W'|'L'|'D' for successor position (next mover's view)
    malom_dtw_after: int | None   # Malom DTW for successor (positive=win, negative=loss)


@dataclass
class PositionStats:
    total_games: int
    wins: int
    losses: int
    draws: int
    malom_wdl: str | None          # 'W'|'L'|'D' for current mover
    malom_dtw: int | None
    canonical_winning_move: str | None


class HumanDB:
    """Read/write adapter for the pre-built human-game SQLite database."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._available = False
        self._game_count: int = 0
        self._entry_count: int = 0

        if not self._path.exists():
            log.info("HumanDB: file not found at %s — will remain unavailable.", self._path)
            return

        try:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
            self._available = True
            self._refresh_counts()
            log.info(
                "HumanDB loaded: %d positions, %d moves, %d games — %s",
                self._entry_count, self._move_count(), self._game_count, self._path,
            )
        except Exception as exc:
            log.warning("HumanDB: failed to open %s — %s", self._path, exc)
            self._conn = None

    def is_available(self) -> bool:
        return self._available and self._conn is not None

    def _refresh_counts(self) -> None:
        if not self._conn:
            return
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'total_games'"
        ).fetchone()
        self._game_count = int(row[0]) if row else 0
        self._entry_count = self._conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]

    def _move_count(self) -> int:
        if not self._conn:
            return 0
        return self._conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0]

    @property
    def game_count(self) -> int:
        return self._game_count

    @property
    def entry_count(self) -> int:
        return self._entry_count

    # ── Visualization queries ─────────────────────────────────────────────────

    def query_position(self, board: "BoardState") -> Optional[PositionStats]:
        """Return aggregate stats for this board position, or None if no data."""
        if not self.is_available():
            return None
        state_key, _ = make_board_state_key(board)
        row = self._conn.execute(
            "SELECT total_games, wins, losses, draws, malom_wdl, malom_dtw, "
            "canonical_winning_move FROM positions WHERE state_key = ?",
            (state_key,),
        ).fetchone()
        if not row:
            return None
        return PositionStats(
            total_games=row[0], wins=row[1], losses=row[2], draws=row[3],
            malom_wdl=row[4], malom_dtw=row[5], canonical_winning_move=row[6],
        )

    def query_moves(self, board: "BoardState") -> list[MoveStats]:
        """Return per-move stats for all next moves seen from this position."""
        if not self.is_available():
            return []
        state_key, sym_idx = make_board_state_key(board)
        rows = self._conn.execute(
            "SELECT notation, wins, losses, draws, total, moves_to_end_sum, "
            "malom_wdl_after, malom_dtw_after FROM moves WHERE state_key = ?",
            (state_key,),
        ).fetchall()
        if not rows:
            return []
        inv = SYM_INVERSE[sym_idx]
        result = []
        for r in rows:
            actual = transform_notation(r[0], inv)
            if actual is None:
                continue
            total = r[4]
            result.append(MoveStats(
                notation=actual,
                wins=r[1], losses=r[2], draws=r[3],
                total=total,
                win_pct=r[1] / total if total else 0.0,
                avg_moves_to_end=r[5] / total if total else 0.0,
                malom_wdl_after=r[6],
                malom_dtw_after=r[7],
            ))
        result.sort(key=lambda m: m.win_pct, reverse=True)
        return result

    def get_malom_wdl(self, board: "BoardState") -> Optional[dict]:
        """Return {"outcome": "W"|"L"|"D", "dtw": int|None} for this position, or None."""
        if not self.is_available():
            return None
        state_key, _ = make_board_state_key(board)
        row = self._conn.execute(
            "SELECT malom_wdl, malom_dtw FROM positions WHERE state_key = ?",
            (state_key,),
        ).fetchone()
        if not row or row[0] is None:
            return None
        return {"outcome": row[0], "dtw": row[1]}

    def canonical_winning_line(
        self, board: "BoardState", depth: int = 10
    ) -> list[str]:
        """Follow canonical_winning_move chain; returns list of notations (actual-game space)."""
        if not self.is_available():
            return []
        from game.board import BoardState as BS
        visited: set[str] = set()
        line: list[str] = []
        current = board
        for _ in range(depth):
            state_key, sym_idx = make_board_state_key(current)
            if state_key in visited:
                break
            visited.add(state_key)
            row = self._conn.execute(
                "SELECT canonical_winning_move FROM positions WHERE state_key = ?",
                (state_key,),
            ).fetchone()
            if not row or not row[0]:
                break
            inv = SYM_INVERSE[sym_idx]
            actual = transform_notation(row[0], inv)
            if actual is None:
                break
            line.append(actual)
            # Advance board — we need to apply the move.
            # Look up the next board via moves_to_end_sum / total ordering
            # (cheapest: use the fen stored in the move record if available).
            # Since we don't store FENs, advance by looking at query_moves for the
            # resulting state. This requires applying the move to the current board.
            try:
                current = _apply_notation(current, actual)
            except Exception:
                break
            if current is None:
                break
        return line

    # ── TrajectoryDB duck-type interface ──────────────────────────────────────

    def query(
        self,
        board: "BoardState",
        current_color: str,
        min_samples: int = 3,
        prefer_ai: bool = False,
    ) -> dict[str, float]:
        """Per-notation score delta [-0.5, +0.5], confidence-weighted.

        Positive = move historically correlates with current_color winning.
        Mirrors TrajectoryDB.query() exactly so game_ai.py callers need no changes.
        """
        if not self.is_available():
            return {}
        state_key, sym_idx = make_board_state_key(board)
        rows = self._conn.execute(
            "SELECT notation, wins, losses, draws, total FROM moves WHERE state_key = ?",
            (state_key,),
        ).fetchall()
        if not rows:
            return {}

        inv = SYM_INVERSE[sym_idx]
        result: dict[str, float] = {}

        for r in rows:
            total = r[4]
            if total < min_samples:
                continue
            actual = transform_notation(r[0], inv)
            if actual is None:
                continue
            wins  = r[1]
            draws = r[3]
            win_rate = (wins + 0.4 * draws) / max(1, total)
            raw = win_rate - 0.5
            confidence = min(1.0, math.log(total + 1) / math.log(20))
            result[actual] = max(-0.5, min(0.5, raw * confidence))

        return result

    def query_all_frequencies(
        self,
        board: "BoardState",
        min_samples: int = 5,
    ) -> dict[str, float]:
        """Per-notation relative frequency [0, 1] for use in SE-11 opponent extension.

        Cached on (state_key, min_samples) — the same canonical position is often
        re-queried many times during a rollout (learner-decide, enc_after, retry,
        confirm, branch rollouts) and per encoder call.  Cache holds up to 100k
        entries; SYM inverse transform applied per call since sym_idx varies.
        """
        if not self.is_available():
            return {}
        state_key, sym_idx = make_board_state_key(board)
        canon = self._query_all_frequencies_canonical(state_key, min_samples)
        if not canon:
            return {}
        inv = SYM_INVERSE[sym_idx]
        result: dict[str, float] = {}
        for canon_ntn, freq in canon:
            actual = transform_notation(canon_ntn, inv)
            if actual:
                result[actual] = freq
        return result

    @lru_cache(maxsize=100_000)
    def _query_all_frequencies_canonical(
        self, state_key: str, min_samples: int,
    ) -> tuple[tuple[str, float], ...]:
        """SQL lookup + canonical-form frequency computation (no SYM transform).

        Returned as a tuple so lru_cache can hold it; empty tuple when no data.
        """
        rows = self._conn.execute(
            "SELECT notation, total FROM moves WHERE state_key = ?",
            (state_key,),
        ).fetchall()
        if not rows:
            return ()
        total_all = sum(r[1] for r in rows)
        if total_all < min_samples:
            return ()
        return tuple((r[0], r[1] / total_all) for r in rows if r[1] > 0)

    def query_opponent_loss(
        self,
        board: "BoardState",
        opponent_color: str,
        min_samples: int = 3,
    ) -> dict[str, float]:
        """Compatibility shim — delegates to query() (same signal, different caller convention)."""
        return self.query(board, board.turn, min_samples=min_samples)

    def query_line(
        self,
        board: "BoardState",
        k: int = 4,
        min_samples: int = 3,
    ) -> list[tuple[str, float]]:
        """Top-k moves by score delta, descending. Mirrors TrajectoryDB.query_line()."""
        scores = self.query(board, board.turn, min_samples=min_samples)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # ── Incremental update ────────────────────────────────────────────────────

    def add_game(self, record: dict) -> None:
        """Incrementally add one completed game to the SQLite DB.

        Called from web/app.py on game completion. No Malom annotation is
        performed at runtime (too slow); malom_wdl/dtw columns stay NULL for
        newly added positions until the next --update build run.
        """
        if not self.is_available():
            return
        try:
            self._add_game_inner(record)
        except Exception as exc:
            log.warning("HumanDB.add_game failed: %s", exc)

    def _add_game_inner(self, record: dict) -> None:
        from ai.trajectory_db import _norm
        winner = record.get("winner")
        moves = record.get("moves", [])
        if not moves:
            return

        # Skip AI-vs-AI.
        source_type = record.get("source_type", "")
        if source_type == "ai_vs_ai":
            return
        if (record.get("self_play") or
                (record.get("white_difficulty") and record.get("black_difficulty")
                 and not record.get("human_color"))):
            return
        if record.get("adaptive_softened"):
            return

        total_plies = len(moves)
        pos_delta: dict = {}   # state_key → {wins, losses, draws, total}
        move_delta: dict = {}  # (state_key, canon_notation) → {...}

        for i, move in enumerate(moves):
            notation = _norm(move.get("notation", ""))
            fen = move.get("board_fen_before", "")
            if not notation or not fen:
                continue
            try:
                from game.board import BoardState
                board = BoardState.from_fen_string(fen)
            except Exception:
                continue

            state_key, sym_idx = make_board_state_key(board)
            canon_notation = transform_notation(notation, sym_idx)
            if canon_notation is None:
                continue

            color = move.get("color", "W")
            plies_remaining = total_plies - i

            # positions delta
            if state_key not in pos_delta:
                pos_delta[state_key] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
            pd = pos_delta[state_key]
            pd["total"] += 1
            if winner == color:
                pd["wins"] += 1
            elif winner is not None and winner != color:
                pd["losses"] += 1
            else:
                pd["draws"] += 1

            # moves delta
            key = (state_key, canon_notation)
            if key not in move_delta:
                move_delta[key] = {"wins": 0, "losses": 0, "draws": 0, "total": 0, "mte": 0.0}
            md = move_delta[key]
            md["total"] += 1
            md["mte"] += plies_remaining
            if winner == color:
                md["wins"] += 1
            elif winner is not None and winner != color:
                md["losses"] += 1
            else:
                md["draws"] += 1

        if not pos_delta:
            return

        with self._conn:
            self._conn.executemany("""
                INSERT INTO positions(state_key, total_games, wins, losses, draws)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    total_games = total_games + excluded.total_games,
                    wins        = wins        + excluded.wins,
                    losses      = losses      + excluded.losses,
                    draws       = draws       + excluded.draws
            """, [
                (sk, s["total"], s["wins"], s["losses"], s["draws"])
                for sk, s in pos_delta.items()
            ])

            self._conn.executemany("""
                INSERT INTO moves(state_key, notation, wins, losses, draws, total, moves_to_end_sum)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(state_key, notation) DO UPDATE SET
                    wins             = wins             + excluded.wins,
                    losses           = losses           + excluded.losses,
                    draws            = draws            + excluded.draws,
                    total            = total            + excluded.total,
                    moves_to_end_sum = moves_to_end_sum + excluded.moves_to_end_sum
            """, [
                (sk, cn, s["wins"], s["losses"], s["draws"], s["total"], s["mte"])
                for (sk, cn), s in move_delta.items()
            ])

            # Recompute canonical_winning_move for touched positions.
            for state_key in pos_delta:
                self._conn.execute("""
                    UPDATE positions
                    SET canonical_winning_move = (
                        SELECT notation FROM moves
                        WHERE moves.state_key = ?
                        ORDER BY wins DESC, total DESC
                        LIMIT 1
                    )
                    WHERE state_key = ?
                """, (state_key, state_key))

        # Update in-memory counts.
        self._game_count += 1
        self._entry_count = self._conn.execute(
            "SELECT COUNT(*) FROM positions"
        ).fetchone()[0]
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('total_games', ?)",
            (str(self._game_count),),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        self.close()


# ── Board application helper ──────────────────────────────────────────────────

def _apply_notation(board: "BoardState", notation: str) -> Optional["BoardState"]:
    """Apply a move notation string to a board and return the resulting board.

    Handles place (e.g. 'd6'), move (e.g. 'd6-d7'), move+capture ('d6-d7xb4'),
    and fly-phase moves (same as move).  Returns None on parse failure.
    """
    from game.board import BoardState
    notation = _norm(notation)

    capture: Optional[str] = None
    if "x" in notation:
        main, capture = notation.split("x", 1)
    else:
        main = notation

    if "-" in main:
        from_sq, to_sq = main.split("-", 1)
        return board.apply_move({"type": "move", "from": from_sq, "to": to_sq,
                                  "capture": capture})
    else:
        return board.apply_move({"type": "place", "from": None, "to": main,
                                  "capture": capture})
