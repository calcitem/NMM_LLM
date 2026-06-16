"""tools/build_human_db.py — Build or incrementally update data/human_db.sqlite.

Reads human-vs-human game JSONL files, aggregates per-position move statistics,
annotates each resulting position with Malom WDL + DTW, and writes a SQLite
database that HumanDB (ai/human_db.py) reads at server startup in milliseconds
instead of scanning tens of thousands of files.

Usage
-----
    # Full build (first time, or after --rebuild):
    .venv/bin/python tools/build_human_db.py \\
        --games-dir data/human_games \\
        --output data/human_db.sqlite \\
        --malom-db /mnt/windows/NMM_DB/.../Std_DD_89adjusted

    # Incremental update (only processes new/changed files):
    .venv/bin/python tools/build_human_db.py --update

    # Skip Malom annotation (faster when DB is not mounted):
    .venv/bin/python tools/build_human_db.py --update --no-malom

    # Force full rebuild from scratch:
    .venv/bin/python tools/build_human_db.py --rebuild
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.trajectory_db import make_board_state_key, _norm
from ai.board_symmetry import transform_notation, SYM_INVERSE
from game.board import BoardState, POSITIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS positions (
    state_key              TEXT PRIMARY KEY,
    total_games            INTEGER NOT NULL DEFAULT 0,
    wins                   INTEGER NOT NULL DEFAULT 0,
    losses                 INTEGER NOT NULL DEFAULT 0,
    draws                  INTEGER NOT NULL DEFAULT 0,
    malom_wdl              TEXT,
    malom_dtw              INTEGER,
    canonical_winning_move TEXT
);

CREATE TABLE IF NOT EXISTS moves (
    state_key        TEXT    NOT NULL,
    notation         TEXT    NOT NULL,
    wins             INTEGER NOT NULL DEFAULT 0,
    losses           INTEGER NOT NULL DEFAULT 0,
    draws            INTEGER NOT NULL DEFAULT 0,
    total            INTEGER NOT NULL DEFAULT 0,
    moves_to_end_sum REAL    NOT NULL DEFAULT 0.0,
    malom_wdl_after  TEXT,
    malom_dtw_after  INTEGER,
    PRIMARY KEY (state_key, notation)
);

CREATE INDEX IF NOT EXISTS idx_moves_state ON moves(state_key);

CREATE TABLE IF NOT EXISTS processed_files (
    file_path   TEXT PRIMARY KEY,
    mtime       REAL    NOT NULL,
    games_found INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()


def _clear_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DELETE FROM positions;
        DELETE FROM moves;
        DELETE FROM processed_files;
        DELETE FROM meta WHERE key != 'schema_version';
    """)
    conn.commit()
    log.info("Cleared existing data (--rebuild).")


# ── Game parsing ──────────────────────────────────────────────────────────────

def _parse_game(record: dict) -> Optional[list[dict]]:
    """Return a list of ply dicts or None if the game should be skipped."""
    if record.get("adaptive_softened"):
        return None
    source_type = record.get("source_type", "")
    if source_type not in ("human_vs_human", "human_involved", ""):
        return None
    # Skip AI-vs-AI games (both sides have difficulty set, no human_color).
    if (
        record.get("self_play")
        or (record.get("white_difficulty") and record.get("black_difficulty")
            and not record.get("human_color"))
    ):
        return None
    moves = record.get("moves", [])
    if not moves:
        return None
    return moves


# ── Core aggregation ──────────────────────────────────────────────────────────

def _process_file(
    path: Path,
    pos_stats: dict,   # state_key → {wins, losses, draws, total}
    move_stats: dict,  # (state_key, canon_notation) → {wins, losses, draws, total, mte_sum}
    pos_boards: dict,  # state_key → BoardState (representative; for Malom annotation)
    move_boards: dict, # (state_key, canon_notation) → next BoardState (for Malom after-move)
) -> int:
    """Process one JSONL file; returns number of games indexed."""
    games_found = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        moves = _parse_game(record)
        if moves is None:
            continue

        winner = record.get("winner")
        games_found += 1
        total_plies = len(moves)

        parsed_plies: list[tuple] = []  # (state_key, sym_idx, canon_notation, board, next_board)

        for i, move in enumerate(moves):
            notation = _norm(move.get("notation", ""))
            fen = move.get("board_fen_before", "")
            if not notation or not fen:
                continue
            try:
                board = BoardState.from_fen_string(fen)
            except Exception:
                continue

            state_key, sym_idx = make_board_state_key(board)
            canon_notation = transform_notation(notation, sym_idx)
            if canon_notation is None:
                continue

            # Determine the board AFTER this move (= fen_before of next ply).
            next_board: Optional[BoardState] = None
            if i + 1 < len(moves):
                next_fen = moves[i + 1].get("board_fen_before", "")
                if next_fen:
                    try:
                        next_board = BoardState.from_fen_string(next_fen)
                    except Exception:
                        pass

            color = move.get("color", "W")
            parsed_plies.append((state_key, sym_idx, canon_notation, board, next_board, color))

        for i, (state_key, sym_idx, canon_notation, board, next_board, color) in enumerate(parsed_plies):
            plies_remaining = total_plies - i

            # — positions table —
            if state_key not in pos_stats:
                pos_stats[state_key] = {"wins": 0, "losses": 0, "draws": 0, "total": 0}
            ps = pos_stats[state_key]
            ps["total"] += 1
            if winner == color:
                ps["wins"] += 1
            elif winner is not None and winner != color:
                ps["losses"] += 1
            else:
                ps["draws"] += 1

            if state_key not in pos_boards:
                pos_boards[state_key] = board

            # — moves table —
            key = (state_key, canon_notation)
            if key not in move_stats:
                move_stats[key] = {"wins": 0, "losses": 0, "draws": 0,
                                   "total": 0, "mte_sum": 0.0}
            ms = move_stats[key]
            ms["total"] += 1
            ms["mte_sum"] += plies_remaining
            if winner == color:
                ms["wins"] += 1
            elif winner is not None and winner != color:
                ms["losses"] += 1
            else:
                ms["draws"] += 1

            if next_board is not None and key not in move_boards:
                move_boards[key] = next_board

    return games_found


# ── Malom annotation ──────────────────────────────────────────────────────────

def _annotate_malom(
    pos_boards: dict,
    move_boards: dict,
    malom_path: str,
) -> tuple[dict, dict]:
    """Query Malom for each unique board and return annotation dicts.

    Returns:
        pos_malom  : state_key  → {"wdl": "W"|"L"|"D", "dtw": int}
        move_malom : (sk, cn)   → {"wdl": "W"|"L"|"D", "dtw": int}
    """
    try:
        from ai.malom_db import MalomDB
        malom = MalomDB(malom_path)
    except Exception as exc:
        log.warning("Could not load MalomDB: %s — skipping annotation.", exc)
        return {}, {}

    if not malom.is_available():
        log.warning("Malom DB not available at %s — skipping annotation.", malom_path)
        return {}, {}

    log.info("Malom DB ready. Annotating %d positions + %d next-positions …",
             len(pos_boards), len(move_boards))

    pos_malom: dict = {}
    move_malom: dict = {}

    def _query(board) -> Optional[dict]:
        try:
            return malom.query(board)
        except Exception:
            return None

    total = len(pos_boards) + len(move_boards)
    done = 0
    log_every = max(1, total // 20)

    for state_key, board in pos_boards.items():
        res = _query(board)
        if res:
            pos_malom[state_key] = {"wdl": res["outcome"], "dtw": res.get("dtw")}
        done += 1
        if done % log_every == 0:
            log.info("  Malom annotation: %d / %d (%.0f%%)", done, total, 100 * done / total)

    for key, board in move_boards.items():
        res = _query(board)
        if res:
            move_malom[key] = {"wdl": res["outcome"], "dtw": res.get("dtw")}
        done += 1
        if done % log_every == 0:
            log.info("  Malom annotation: %d / %d (%.0f%%)", done, total, 100 * done / total)

    log.info("Malom annotation complete: %d position hits, %d move hits.",
             len(pos_malom), len(move_malom))
    return pos_malom, move_malom


# ── SQLite writes ─────────────────────────────────────────────────────────────

def _upsert_positions(
    conn: sqlite3.Connection,
    pos_stats: dict,
    pos_malom: dict,
) -> None:
    rows = []
    for state_key, s in pos_stats.items():
        pm = pos_malom.get(state_key, {})
        rows.append((
            state_key,
            s["total"], s["wins"], s["losses"], s["draws"],
            pm.get("wdl"), pm.get("dtw"),
        ))
    conn.executemany("""
        INSERT INTO positions(state_key, total_games, wins, losses, draws, malom_wdl, malom_dtw)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(state_key) DO UPDATE SET
            total_games = total_games + excluded.total_games,
            wins        = wins        + excluded.wins,
            losses      = losses      + excluded.losses,
            draws       = draws       + excluded.draws,
            malom_wdl   = COALESCE(malom_wdl, excluded.malom_wdl),
            malom_dtw   = COALESCE(malom_dtw, excluded.malom_dtw)
    """, rows)


def _upsert_moves(
    conn: sqlite3.Connection,
    move_stats: dict,
    move_malom: dict,
) -> None:
    rows = []
    for (state_key, canon_notation), s in move_stats.items():
        mm = move_malom.get((state_key, canon_notation), {})
        rows.append((
            state_key, canon_notation,
            s["wins"], s["losses"], s["draws"], s["total"], s["mte_sum"],
            mm.get("wdl"), mm.get("dtw"),
        ))
    conn.executemany("""
        INSERT INTO moves(state_key, notation, wins, losses, draws, total, moves_to_end_sum,
                          malom_wdl_after, malom_dtw_after)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(state_key, notation) DO UPDATE SET
            wins             = wins             + excluded.wins,
            losses           = losses           + excluded.losses,
            draws            = draws            + excluded.draws,
            total            = total            + excluded.total,
            moves_to_end_sum = moves_to_end_sum + excluded.moves_to_end_sum,
            malom_wdl_after  = COALESCE(malom_wdl_after,  excluded.malom_wdl_after),
            malom_dtw_after  = COALESCE(malom_dtw_after,  excluded.malom_dtw_after)
    """, rows)


def _recompute_canonical_winning_moves(conn: sqlite3.Connection) -> None:
    """Set canonical_winning_move to the most-played winning notation per state."""
    conn.execute("""
        UPDATE positions
        SET canonical_winning_move = (
            SELECT notation FROM moves
            WHERE moves.state_key = positions.state_key
            ORDER BY wins DESC, total DESC
            LIMIT 1
        )
    """)


def _update_meta(conn: sqlite3.Connection, game_count: int, file_count: int) -> None:
    from datetime import datetime
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('build_date', ?)",
                 (datetime.now().isoformat(timespec="seconds"),))
    existing_games = conn.execute(
        "SELECT value FROM meta WHERE key = 'total_games'"
    ).fetchone()
    prev = int(existing_games[0]) if existing_games and existing_games[0] else 0
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('total_games', ?)",
                 (str(prev + game_count),))
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('total_files', ?)",
                 (str(file_count),))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Build or update data/human_db.sqlite")
    ap.add_argument("--games-dir", default="data/human_games",
                    help="Directory containing human-vs-human *.jsonl files")
    ap.add_argument("--extra-dirs", nargs="*", default=[],
                    help="Additional game directories to include")
    ap.add_argument("--output", default="data/human_db.sqlite",
                    help="Output SQLite path")
    ap.add_argument("--malom-db", default="",
                    help="Path to Malom DB directory (e.g. .../Std_DD_89adjusted)")
    ap.add_argument("--no-malom", action="store_true",
                    help="Skip Malom annotation (malom_wdl/dtw columns stay NULL)")
    ap.add_argument("--update", action="store_true",
                    help="Only process files not yet in processed_files table")
    ap.add_argument("--rebuild", action="store_true",
                    help="Clear DB and reprocess all files from scratch")
    args = ap.parse_args()

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_db(conn)

    if args.rebuild:
        _clear_db(conn)

    # Discover files.
    all_dirs = [ROOT / args.games_dir] + [ROOT / d for d in args.extra_dirs]
    all_files: list[Path] = []
    for d in all_dirs:
        if d.exists():
            all_files.extend(sorted(d.rglob("*.jsonl")))
        else:
            log.warning("Directory not found: %s", d)

    if args.update and not args.rebuild:
        already = {
            row[0]: row[1]
            for row in conn.execute("SELECT file_path, mtime FROM processed_files")
        }
        new_files = []
        for p in all_files:
            mtime = p.stat().st_mtime
            if str(p) not in already or already[str(p)] != mtime:
                new_files.append(p)
        log.info("--update: %d / %d files need processing.", len(new_files), len(all_files))
        all_files = new_files

    if not all_files:
        log.info("No files to process. DB is up to date.")
        conn.close()
        return

    log.info("Processing %d files from %s…", len(all_files), args.games_dir)

    # Accumulate stats across all files.
    pos_stats: dict = {}   # state_key → {wins, losses, draws, total}
    move_stats: dict = {}  # (state_key, canon_notation) → {wins, losses, draws, total, mte_sum}
    pos_boards: dict = {}  # state_key → representative BoardState
    move_boards: dict = {} # (state_key, canon_notation) → next BoardState

    total_games = 0
    file_mtime_map: dict[str, tuple[float, int]] = {}  # file_path → (mtime, games_found)

    t0 = time.time()
    for i, path in enumerate(all_files):
        mtime = path.stat().st_mtime
        try:
            n = _process_file(path, pos_stats, move_stats, pos_boards, move_boards)
        except Exception as exc:
            log.warning("Skipping %s — %s", path.name, exc)
            n = 0
        total_games += n
        file_mtime_map[str(path)] = (mtime, n)
        if (i + 1) % 500 == 0 or (i + 1) == len(all_files):
            elapsed = time.time() - t0
            log.info("  Parsed %d / %d files, %d games, %.1f s",
                     i + 1, len(all_files), total_games, elapsed)

    log.info("Parsed %d games → %d unique positions, %d unique (position, move) pairs.",
             total_games, len(pos_stats), len(move_stats))

    # Malom annotation.
    pos_malom: dict = {}
    move_malom: dict = {}
    if not args.no_malom:
        malom_path = args.malom_db
        if not malom_path:
            try:
                from learned_ai.sentinel.config import load_config as _lc
                malom_path = getattr(_lc(), "external_db_path", "") or ""
            except Exception:
                pass
        if malom_path:
            pos_malom, move_malom = _annotate_malom(pos_boards, move_boards, malom_path)
        else:
            log.info("No --malom-db specified and no config path found; skipping annotation.")

    # Write to SQLite.
    log.info("Writing to %s …", output_path)
    with conn:
        _upsert_positions(conn, pos_stats, pos_malom)
        _upsert_moves(conn, move_stats, move_malom)
        _recompute_canonical_winning_moves(conn)
        _update_meta(conn, total_games, len(file_mtime_map))
        conn.executemany(
            "INSERT OR REPLACE INTO processed_files(file_path, mtime, games_found) VALUES (?, ?, ?)",
            [(fp, mt, gf) for fp, (mt, gf) in file_mtime_map.items()],
        )

    elapsed = time.time() - t0
    pos_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    move_count = conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0]
    conn.close()

    log.info(
        "Done in %.1f s. DB: %d positions, %d moves, %d games. → %s",
        elapsed, pos_count, move_count, total_games, output_path,
    )


if __name__ == "__main__":
    main()
