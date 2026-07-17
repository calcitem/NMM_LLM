"""learned_ai/data/specialist_db.py — Self-built experience database for specialist AIs.

Each specialist maintains a persistent SQLite database that accumulates per-position
WDL statistics from self-play games, Malom-validated labels for key positions (training
only), tagged winning move sequences, and promoted preferred plays.

Position keys use D4 (dihedral-8) symmetry so that all 8 rotationally and reflectionally
equivalent board positions share the same database entry, giving up to 8× data efficiency
— identical to the approach used by the endgame databases.

At inference, the DB populates counterfactual feature slots with WDL fractions from
self-play history, substituting for Malom (not required at inference time).
Ships pre-seeded from training; grows further from every game the user plays.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Tuple

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS positions (
    pos_hash     TEXT    PRIMARY KEY,
    wins         INTEGER NOT NULL DEFAULT 0,
    draws        INTEGER NOT NULL DEFAULT 0,
    losses       INTEGER NOT NULL DEFAULT 0,
    malom_label  TEXT    DEFAULT NULL,
    last_seen    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS winning_lines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    move_seq     TEXT    NOT NULL,
    phase        TEXT    NOT NULL,
    result       TEXT    NOT NULL,
    wins         INTEGER NOT NULL DEFAULT 1,
    times_played INTEGER NOT NULL DEFAULT 1,
    win_rate     REAL    NOT NULL DEFAULT 1.0,
    last_seen    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wl_phase ON winning_lines(phase);
CREATE INDEX IF NOT EXISTS idx_wl_wr   ON winning_lines(win_rate);

CREATE TABLE IF NOT EXISTS preferred_plays (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tag          TEXT    NOT NULL,
    pos_sequence TEXT    NOT NULL,
    win_rate     REAL    NOT NULL,
    times_played INTEGER NOT NULL DEFAULT 0,
    promoted     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_pp_promoted ON preferred_plays(promoted);
"""

_PROMOTE_MIN_PLAYED = 5
_PROMOTE_WIN_RATE   = 0.65
_DEMOTE_WIN_RATE    = 0.45
_DEMOTE_MIN_RECENT  = 20
_MIN_SAMPLES_QUERY  = 10


def _board_hash(board) -> str:
    """D4-canonical hash — all 8 symmetric equivalents map to the same key."""
    from ai.board_symmetry import canonical_board_str
    fen   = board.to_fen_string()
    parts = fen.split("|")          # [board_24, turn, W_placed, B_placed]
    canon, _ = canonical_board_str(parts[0])
    key = f"{canon}|{parts[1]}|{parts[2]}|{parts[3]}"
    return hashlib.sha1(key.encode()).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


class SpecialistDB:
    """Persistent self-play experience database for one specialist.

    Not thread-safe. Each training process opens its own instance.
    Grows indefinitely — even 100 000 games stay under 1 GB.
    """

    def __init__(self, db_path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ── Position statistics ───────────────────────────────────────────────────

    def record_game(
        self,
        boards: List,
        result: str,
        move_seq: List[str],
        phase: str,
    ) -> None:
        """Record all positions from a completed game and update winning lines.

        Parameters
        ----------
        boards   : list of BoardState objects at each of the learner's turns
        result   : 'W', 'D', or 'L' from the learner's perspective
        move_seq : list of move notation strings (learner's moves only)
        phase    : 'open' | 'mid' | 'end'
        """
        now = _now()
        with self._conn:
            for board in boards:
                h = _board_hash(board)
                self._conn.execute("""
                    INSERT INTO positions (pos_hash, wins, draws, losses, last_seen)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(pos_hash) DO UPDATE SET
                        wins      = wins   + excluded.wins,
                        draws     = draws  + excluded.draws,
                        losses    = losses + excluded.losses,
                        last_seen = excluded.last_seen
                """, (h,
                      1 if result == "W" else 0,
                      1 if result == "D" else 0,
                      1 if result == "L" else 0,
                      now))

            if result in ("W", "D") and move_seq:
                seq_json = json.dumps(move_seq)
                row = self._conn.execute(
                    "SELECT id, times_played, wins FROM winning_lines WHERE move_seq=? AND phase=?",
                    (seq_json, phase)
                ).fetchone()
                if row:
                    wid, played, wins = row
                    new_played = played + 1
                    new_wins   = wins + (1 if result == "W" else 0)
                    self._conn.execute(
                        "UPDATE winning_lines SET times_played=?, wins=?, win_rate=?, last_seen=? WHERE id=?",
                        (new_played, new_wins, new_wins / new_played, now, wid)
                    )
                else:
                    self._conn.execute("""
                        INSERT INTO winning_lines (move_seq, phase, result, wins, times_played, win_rate, last_seen)
                        VALUES (?, ?, ?, ?, 1, 1.0, ?)
                    """, (seq_json, phase, result, 1 if result == "W" else 0, now))

            self._promote_lines(phase)

    def _promote_lines(self, phase: str) -> None:
        rows = self._conn.execute("""
            SELECT id, move_seq, times_played, win_rate
            FROM winning_lines
            WHERE phase=? AND times_played >= ? AND win_rate >= ?
        """, (phase, _PROMOTE_MIN_PLAYED, _PROMOTE_WIN_RATE)).fetchall()

        for wid, seq_json, played, wr in rows:
            existing = self._conn.execute(
                "SELECT id FROM preferred_plays WHERE pos_sequence=?", (seq_json,)
            ).fetchone()
            if not existing:
                self._conn.execute("""
                    INSERT INTO preferred_plays (tag, pos_sequence, win_rate, times_played, promoted)
                    VALUES (?, ?, ?, ?, 1)
                """, (f"{phase}_line_{wid}", seq_json, wr, played))
            else:
                self._conn.execute(
                    "UPDATE preferred_plays SET win_rate=?, times_played=?, promoted=1 WHERE id=?",
                    (wr, played, existing[0])
                )

        demote = self._conn.execute(
            "SELECT id, pos_sequence FROM preferred_plays WHERE promoted=1"
        ).fetchall()
        for pid, seq_json in demote:
            row = self._conn.execute(
                "SELECT times_played, win_rate FROM winning_lines WHERE move_seq=?", (seq_json,)
            ).fetchone()
            if row and row[0] >= _DEMOTE_MIN_RECENT and row[1] < _DEMOTE_WIN_RATE:
                self._conn.execute("UPDATE preferred_plays SET promoted=0 WHERE id=?", (pid,))

    # ── Malom validation (training-time only) ─────────────────────────────────

    def label_position_malom(self, board, wdl: str) -> None:
        """Store a Malom WDL label ('W'/'D'/'L') for a position (training time only)."""
        h = _board_hash(board)
        self._conn.execute("""
            INSERT INTO positions (pos_hash, wins, draws, losses, malom_label, last_seen)
            VALUES (?, 0, 0, 0, ?, ?)
            ON CONFLICT(pos_hash) DO UPDATE SET malom_label = excluded.malom_label
        """, (h, wdl, _now()))
        self._conn.commit()

    # ── Inference query ───────────────────────────────────────────────────────

    def query_wdl(self, board, min_samples: int = _MIN_SAMPLES_QUERY) -> Optional[Tuple[float, float, float]]:
        """Return (win_frac, draw_frac, loss_frac) or None if insufficient data.

        When a Malom label exists and self-play count is low, the Malom label
        provides a strong prior.  At inference without Malom, self-play statistics
        substitute once enough games have been played.
        """
        h = _board_hash(board)
        row = self._conn.execute(
            "SELECT wins, draws, losses, malom_label FROM positions WHERE pos_hash=?", (h,)
        ).fetchone()
        if row is None:
            return None
        wins, draws, losses, malom_label = row
        n = wins + draws + losses
        if malom_label and n < min_samples:
            if malom_label == "W":
                return (0.90, 0.05, 0.05)
            if malom_label == "D":
                return (0.05, 0.90, 0.05)
            if malom_label == "L":
                return (0.05, 0.05, 0.90)
        if n < min_samples:
            return None
        return (wins / n, draws / n, losses / n)

    def query_win_prob(self, board, min_samples: int = _MIN_SAMPLES_QUERY) -> float:
        """Return P(win) + 0.5*P(draw) or 0.5 if position is unknown."""
        wdl = self.query_wdl(board, min_samples)
        if wdl is None:
            return 0.5
        w, d, _ = wdl
        return w + 0.5 * d

    # ── Preferred plays ───────────────────────────────────────────────────────

    def get_promoted_plays(self, phase: str = "") -> List[Tuple[str, List[str], float]]:
        """Return [(tag, move_list, win_rate)] for all promoted plays."""
        rows = self._conn.execute(
            "SELECT tag, pos_sequence, win_rate FROM preferred_plays WHERE promoted=1 ORDER BY win_rate DESC"
        ).fetchall()
        result = []
        for tag, seq_json, wr in rows:
            try:
                result.append((tag, json.loads(seq_json), float(wr)))
            except Exception:
                pass
        return result

    # ── Stats / maintenance ───────────────────────────────────────────────────

    def stats(self) -> dict:
        pos  = self._conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        well = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE wins+draws+losses >= ?", (_MIN_SAMPLES_QUERY,)
        ).fetchone()[0]
        malom = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL"
        ).fetchone()[0]
        lines = self._conn.execute("SELECT COUNT(*) FROM winning_lines").fetchone()[0]
        prefs = self._conn.execute(
            "SELECT COUNT(*) FROM preferred_plays WHERE promoted=1"
        ).fetchone()[0]
        return {
            "positions": pos,
            "well_sampled": well,
            "malom_labeled": malom,
            "winning_lines": lines,
            "preferred_plays": prefs,
        }

    def close(self) -> None:
        self._conn.close()
