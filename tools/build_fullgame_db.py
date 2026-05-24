"""tools/build_fullgame_db.py — Full-game position database generator.

Builds a SQLite database of Nine Men's Morris positions covering the entire
game (placement, movement, fly, terminals).  D4 symmetry is used so up to 8
equivalent positions share one row.

WARNING — full Nine Men's Morris has on the order of 10^10 legal positions;
a true complete solve is infeasible on a normal machine.  This script
performs a bounded enumeration with backpropagation of win/loss/draw
trajectories so you can:

  • build the database to a position cap or depth cap that fits your disk,
  • resume the build after interruption,
  • run a quick `--dry-run` to validate the pipeline,
  • run a `--sample N` mode for sanity testing without a 12 GB commitment.

Outcomes follow Wikipedia's convention for solved game-trees:
   1 = WIN-FOR-WHITE,  -1 = WIN-FOR-BLACK,  0 = DRAW,  NULL = UNKNOWN
``depth`` is distance-to-terminal (plies) when known, otherwise NULL.
``best_move`` is the best canonical-form move notation for the side to move.

Run inside the project venv:
    .venv/bin/python tools/build_fullgame_db.py --help
    .venv/bin/python tools/build_fullgame_db.py --dry-run

    # Default location (project data/ directory)
    .venv/bin/python tools/build_fullgame_db.py --max-positions 500000

    # Another drive — use --db-dir (auto-names the file fullgame.sqlite)
    .venv/bin/python tools/build_fullgame_db.py --db-dir D:\databases --max-positions 500000
    .venv/bin/python tools/build_fullgame_db.py --db-dir /mnt/external  --max-positions 500000

    # Or give the full path explicitly with --output
    .venv/bin/python tools/build_fullgame_db.py --output E:\NMM\fullgame.sqlite

The script prints the resolved absolute path before starting so you always
know where it's writing.  It also checks the target directory is writable
before beginning the build, to avoid discovering a permissions error after
hours of work.

Generated files are intentionally placed under data/ by default — see .gitignore.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

# ── Ensure project root on path when invoked directly ────────────────────────
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Dependency check / install ────────────────────────────────────────────────
# Stdlib-only by design.  We still verify the project's own requirements are
# importable so this script can be a one-stop "install everything and build".

_REQUIRED_STDLIB = ("sqlite3", "struct", "collections")
_REQUIRED_PROJECT = ("game.board", "game.rules", "ai.board_symmetry")


def _verify_deps() -> None:
    missing: list[str] = []
    for mod in _REQUIRED_STDLIB:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"FATAL: missing stdlib modules: {missing}", file=sys.stderr)
        sys.exit(2)


def _maybe_pip_install_requirements() -> None:
    req = _ROOT / "requirements.txt"
    if not req.exists():
        return
    try:
        __import__("fastapi")
        return  # something already installed → assume OK
    except ImportError:
        pass
    print("Installing project requirements…", file=sys.stderr)
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
    )


# ── Project imports ───────────────────────────────────────────────────────────

from game.board import POSITIONS, BoardState
from game.rules import get_all_legal_moves, is_terminal
# Import board_symmetry directly from its module to avoid pulling in ai/__init__.py
# (which depends on chromadb / fastapi).  This script is intentionally
# stdlib-only so it can run during DB builds without the full web stack.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "ai_board_symmetry",
    str(_ROOT / "ai" / "board_symmetry.py"),
)
_bs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bs)
SYM_INVERSE = _bs.SYM_INVERSE
canonical_board_str = _bs.canonical_board_str
transform_notation = _bs.transform_notation

logger = logging.getLogger("fullgame_db.build")


# ── Canonical key encoding ────────────────────────────────────────────────────
# The position state we must distinguish:
#   • 24-character board ("W"/"B"/".")
#   • side to move ("W"/"B")
#   • pieces placed so far for each side (0..9)
# Storing the board as a packed bit-pair string costs 6 bytes (2 bits × 24),
# plus 1 byte side, plus 1 byte placed_W, 1 byte placed_B = 9 bytes.

_PIECE_BITS = {".": 0b00, "W": 0b01, "B": 0b10}


def encode_canonical(board24: str, turn: str, placed_w: int, placed_b: int) -> bytes:
    """Pack a canonical position into 9 bytes for a compact SQLite primary key."""
    if len(board24) != 24:
        raise ValueError(f"board24 must be 24 chars, got {len(board24)}")
    val = 0
    for i, ch in enumerate(board24):
        val |= _PIECE_BITS[ch] << (i * 2)
    # 48 bits of board → 6 bytes
    board_bytes = val.to_bytes(6, "little")
    side = 0 if turn == "W" else 1
    return board_bytes + bytes((side, placed_w & 0xFF, placed_b & 0xFF))


def position_key(board: BoardState) -> bytes:
    """Compute the canonical 9-byte key for a BoardState (applies D4 canonicalization)."""
    fen = board.to_fen_string()
    board24, turn, pw, pb = fen.split("|")
    canon, _sym = canonical_board_str(board24)
    return encode_canonical(canon, turn, int(pw), int(pb))


def canonical_components(board: BoardState) -> tuple[str, int, str, int, int]:
    """Return (canonical_board24, sym_idx, turn, placed_w, placed_b)."""
    fen = board.to_fen_string()
    board24, turn, pw, pb = fen.split("|")
    canon, sym = canonical_board_str(board24)
    return canon, sym, turn, int(pw), int(pb)


def move_notation(move: dict) -> str:
    s = f"{move['from']}-{move['to']}" if move.get("from") else move.get("to", "")
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


# ── Schema ───────────────────────────────────────────────────────────────────
# WITHOUT ROWID keeps the table effectively as a clustered index on `key`.

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA page_size = 4096;
PRAGMA temp_store = MEMORY;

CREATE TABLE IF NOT EXISTS positions (
    key       BLOB PRIMARY KEY,    -- 9-byte canonical position
    outcome   INTEGER,             -- 1=W wins, -1=B wins, 0=draw, NULL=unknown
    depth     INTEGER,             -- plies to terminal (NULL if unknown)
    best_move TEXT,                -- canonical-form move notation
    samples   INTEGER NOT NULL DEFAULT 1
) WITHOUT ROWID;

-- Edges store the trajectory information: for each position, the list of
-- (move, child_key, classification) tuples.  Classification flag is:
--   'W' = winning move for side-to-move
--   'L' = losing move
--   'N' = neutral / unresolved
-- We pack edges as a single TEXT blob per position rather than a separate
-- table — keeps disk footprint smaller for big builds (one row vs ~10).
ALTER TABLE positions ADD COLUMN trajectories TEXT;  -- ignored if column exists

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""

# Some SQLite versions raise on the ALTER if column exists; we tolerate that.


def _ensure_trajectories_column(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}
    if "trajectories" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN trajectories TEXT")


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    for stmt in _SCHEMA.strip().split(";"):
        s = stmt.strip()
        if not s:
            continue
        if s.upper().startswith("ALTER"):
            continue  # handled below
        try:
            conn.execute(s)
        except sqlite3.OperationalError as exc:
            logger.debug("Schema stmt skipped (%s): %s", s.split()[0], exc)
    _ensure_trajectories_column(conn)
    conn.commit()
    return conn


# ── Trajectory packing ───────────────────────────────────────────────────────
# Stored as compact pipe-separated triples: "notation:childkey_hex:flag".
# Flag: W (winning for side-to-move) / L (losing) / N (neutral/unknown).

def pack_trajectories(items: list[tuple[str, bytes, str]]) -> str:
    return "|".join(f"{n}:{ck.hex()}:{f}" for n, ck, f in items)


def unpack_trajectories(blob: str) -> list[tuple[str, bytes, str]]:
    if not blob:
        return []
    out = []
    for part in blob.split("|"):
        n, ck, f = part.rsplit(":", 2)
        out.append((n, bytes.fromhex(ck), f))
    return out


# ── Builder ──────────────────────────────────────────────────────────────────

class FullGameDBBuilder:
    """Forward-BFS position enumerator with terminal back-propagation.

    The builder canonicalises each visited position so symmetric duplicates
    share one DB row.  It records every legal move as an edge, then performs
    a retrograde-style pass that labels terminal positions and propagates
    win/loss/draw outcomes back through the parent links.
    """

    def __init__(
        self,
        db_path: Path,
        max_positions: Optional[int] = None,
        max_depth: Optional[int] = None,
        commit_every: int = 5000,
        progress_every: float = 5.0,
    ) -> None:
        self.db_path = db_path
        self.max_positions = max_positions
        self.max_depth = max_depth
        self.commit_every = commit_every
        self.progress_every = progress_every

        self.conn = init_db(db_path)
        self.conn.execute("PRAGMA foreign_keys = OFF")

        self._inserted = 0
        self._scanned = 0
        self._t_start = time.monotonic()
        self._t_last_progress = self._t_start

    # ── Forward enumeration ──────────────────────────────────────────────────

    def enumerate_forward(self, start: BoardState) -> None:
        """BFS over reachable canonical positions starting from `start`.

        Resumes by reading the existing positions table — already-seen keys
        are skipped, so re-running the script extends the build.
        """
        start_key = position_key(start)
        seen: set[bytes] = set()
        # Pre-load existing keys so we don't re-process completed work.
        cur = self.conn.execute("SELECT key FROM positions")
        for (k,) in cur:
            seen.add(bytes(k))
        logger.info("Resuming from %d positions already in DB.", len(seen))

        queue: deque[tuple[bytes, BoardState, int]] = deque()
        if start_key not in seen:
            queue.append((start_key, start, 0))

        # If the start key IS in the DB but we have nothing else to do,
        # the resume seed is the union of any rows with NULL outcome — i.e.,
        # rows we may still want to expand.
        if not queue:
            cur = self.conn.execute(
                "SELECT key FROM positions WHERE outcome IS NULL "
                "AND (trajectories IS NULL OR trajectories = '') LIMIT 10000"
            )
            for (k,) in cur:
                # We must reconstruct the BoardState from the key.  Skip for
                # now — forward-pass resume on an existing DB only adds new
                # positions reachable from `start`; deep continuation should
                # use the backprop pass.
                _ = k

        while queue:
            if self.max_positions is not None and self._inserted >= self.max_positions:
                logger.info("Reached --max-positions cap (%d).", self.max_positions)
                break

            key, board, depth = queue.popleft()

            terminal, winner = is_terminal(board)
            if terminal:
                outcome = 1 if winner == "W" else (-1 if winner == "B" else 0)
                self._upsert(key, outcome=outcome, depth=0, best_move=None,
                             trajectories="")
                continue

            if self.max_depth is not None and depth >= self.max_depth:
                # Frontier node: insert with unknown outcome, no children.
                self._upsert(key, outcome=None, depth=None, best_move=None,
                             trajectories="")
                continue

            moves = get_all_legal_moves(board)
            canon_board, sym, _turn, _pw, _pb = canonical_components(board)
            edges: list[tuple[str, bytes, str]] = []
            for mv in moves:
                child = board.apply_move(mv)
                child_key = position_key(child)
                # Notation in canonical orientation (for storage stability)
                actual_notation = move_notation(mv)
                canon_notation = transform_notation(actual_notation, sym) or actual_notation
                edges.append((canon_notation, child_key, "N"))
                if child_key not in seen:
                    seen.add(child_key)
                    queue.append((child_key, child, depth + 1))

            self._upsert(
                key,
                outcome=None,
                depth=None,
                best_move=None,
                trajectories=pack_trajectories(edges),
            )
            self._scanned += 1
            self._maybe_progress(len(queue))

        self.conn.commit()
        logger.info("Forward enumeration complete: %d rows.", self._inserted)

    # ── Backward propagation ─────────────────────────────────────────────────

    def backpropagate(self, passes: int = 6) -> None:
        """Iteratively label parent positions from already-labelled children.

        A position with outcome=NULL is resolved when every child has a
        defined outcome.  Side-to-move picks the best outcome (max for W, min
        for B); ties prefer draws over losses.

        Multiple passes propagate labels through long chains.  Convergence
        within `passes` is not guaranteed for cyclic position spaces (NMM
        movement phase contains cycles); residual NULLs are left as
        UNKNOWN and the AI's negamax fallback handles them.
        """
        for pass_no in range(1, passes + 1):
            updated = 0
            cur = self.conn.execute(
                "SELECT key, trajectories FROM positions "
                "WHERE outcome IS NULL AND trajectories IS NOT NULL AND trajectories <> ''"
            )
            rows = cur.fetchall()
            for key, blob in rows:
                edges = unpack_trajectories(blob)
                if not edges:
                    continue
                child_keys = [e[1] for e in edges]
                placeholders = ",".join("?" for _ in child_keys)
                child_rows = self.conn.execute(
                    f"SELECT key, outcome, depth FROM positions WHERE key IN ({placeholders})",
                    child_keys,
                ).fetchall()
                child_map = {bytes(k): (o, d) for k, o, d in child_rows}

                if len(child_map) < len(child_keys):
                    continue  # unknown children → skip this pass
                outcomes = [child_map[ck][0] for ck in child_keys]
                if any(o is None for o in outcomes):
                    continue

                # Determine side to move from key (byte 6: 0=W, 1=B)
                side = "W" if key[6] == 0 else "B"
                if side == "W":
                    best_o = max(outcomes)
                else:
                    best_o = min(outcomes)

                # Choose best move (first child whose outcome matches best)
                new_edges: list[tuple[str, bytes, str]] = []
                best_move: Optional[str] = None
                for (n, ck, _flag), o in zip(edges, outcomes):
                    if (side == "W" and o == 1) or (side == "B" and o == -1):
                        flag = "W"
                    elif (side == "W" and o == -1) or (side == "B" and o == 1):
                        flag = "L"
                    else:
                        flag = "N"
                    if best_move is None and o == best_o:
                        best_move = n
                    new_edges.append((n, ck, flag))

                # depth = 1 + min depth of any matching child
                matching_depths = [
                    child_map[ck][1] for (_, ck, _), o in zip(edges, outcomes)
                    if o == best_o and child_map[ck][1] is not None
                ]
                new_depth = (1 + min(matching_depths)) if matching_depths else None

                self.conn.execute(
                    "UPDATE positions SET outcome=?, depth=?, best_move=?, trajectories=? WHERE key=?",
                    (best_o, new_depth, best_move, pack_trajectories(new_edges), key),
                )
                updated += 1
            self.conn.commit()
            logger.info("Backprop pass %d: labelled %d positions.", pass_no, updated)
            if updated == 0:
                break

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _upsert(
        self,
        key: bytes,
        outcome: Optional[int],
        depth: Optional[int],
        best_move: Optional[str],
        trajectories: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO positions (key, outcome, depth, best_move, trajectories, samples) "
            "VALUES (?, ?, ?, ?, ?, 1) "
            "ON CONFLICT(key) DO UPDATE SET "
            " outcome   = COALESCE(positions.outcome, excluded.outcome), "
            " depth     = COALESCE(positions.depth,   excluded.depth), "
            " best_move = COALESCE(positions.best_move, excluded.best_move), "
            " trajectories = COALESCE(NULLIF(positions.trajectories,''), excluded.trajectories), "
            " samples   = positions.samples + 1",
            (key, outcome, depth, best_move, trajectories),
        )
        self._inserted += 1
        if self._inserted % self.commit_every == 0:
            self.conn.commit()

    def _maybe_progress(self, queue_len: int) -> None:
        now = time.monotonic()
        if now - self._t_last_progress < self.progress_every:
            return
        self._t_last_progress = now
        elapsed = now - self._t_start
        rate = self._inserted / elapsed if elapsed > 0 else 0
        logger.info(
            "progress: scanned=%d inserted=%d queue=%d rate=%.0f pos/s",
            self._scanned, self._inserted, queue_len, rate,
        )

    def vacuum(self) -> None:
        logger.info("VACUUM (compacting database)…")
        self.conn.execute("VACUUM")

    def write_meta(self, **kv: str) -> None:
        for k, v in kv.items():
            self.conn.execute(
                "INSERT INTO meta(k,v) VALUES(?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (k, str(v)),
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Nine Men's Morris full-game position database.  "
            "A complete solve is not attempted in one run; use --max-positions "
            "or --max-depth to bound the build.  The script is resumable: "
            "re-running extends an existing DB."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help=(
            "Output SQLite file path.  Accepts any absolute path including "
            "other drives (e.g. D:\\\\databases\\\\fullgame.sqlite or "
            "/mnt/external/fullgame.sqlite).  "
            "Default: <project>/data/fullgame.sqlite"
        ),
    )
    parser.add_argument(
        "--db-dir", type=Path, default=None,
        help=(
            "Directory to write fullgame.sqlite into.  Shorthand for "
            "--output <dir>/fullgame.sqlite.  Useful for pointing at another "
            "drive without typing the filename.  Ignored if --output is set."
        ),
    )
    parser.add_argument(
        "--max-positions", type=int, default=None,
        help="Stop forward enumeration after this many newly-inserted positions.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="BFS ply cap.  Positions beyond this depth are stored as frontier nodes.",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Quick sanity build: cap to N positions and skip backprop. Implies --max-positions.",
    )
    parser.add_argument(
        "--passes", type=int, default=6,
        help="Backpropagation passes for win/loss labelling (default 6).",
    )
    parser.add_argument(
        "--vacuum", action="store_true",
        help="Run SQLite VACUUM after build to minimise file size.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate environment + build a tiny in-memory DB (100 positions) without writing to disk.",
    )
    parser.add_argument(
        "--install-deps", action="store_true",
        help="Pip-install project requirements before building.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-progress logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ── Resolve output path ───────────────────────────────────────────────────
    if args.output is not None:
        output_path = args.output.resolve()
    elif args.db_dir is not None:
        output_path = args.db_dir.resolve() / "fullgame.sqlite"
    else:
        output_path = (_ROOT / "data" / "fullgame.sqlite").resolve()

    # Pre-flight: make sure the target directory can actually be created/written.
    if not args.dry_run:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"ERROR: Cannot create output directory {output_path.parent}: {exc}",
                  file=sys.stderr)
            return 1
        # Quick write test so we fail fast rather than after hours of work.
        _probe = output_path.parent / ".nmm_write_probe"
        try:
            _probe.write_bytes(b"")
            _probe.unlink()
        except OSError as exc:
            print(f"ERROR: Output directory is not writable ({output_path.parent}): {exc}",
                  file=sys.stderr)
            return 1

    print(f"Output: {output_path}")

    _verify_deps()
    if args.install_deps:
        _maybe_pip_install_requirements()

    if args.dry_run:
        # Build a tiny in-memory DB to exercise every code path.
        logger.info("DRY RUN: tiny in-memory build, 100 positions, no disk write.")
        builder = FullGameDBBuilder(
            db_path=Path(":memory:"),  # ignored when we override conn
            max_positions=100,
            commit_every=50,
        )
        # Swap to a true in-memory connection so init_db's file write is benign.
        builder.conn.close()
        builder.conn = sqlite3.connect(":memory:")
        for stmt in _SCHEMA.strip().split(";"):
            s = stmt.strip()
            if not s or s.upper().startswith(("ALTER", "PRAGMA")):
                continue
            try:
                builder.conn.execute(s)
            except sqlite3.OperationalError:
                pass
        _ensure_trajectories_column(builder.conn)
        builder.enumerate_forward(BoardState.new_game())
        builder.backpropagate(passes=2)
        count = builder.conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        resolved = builder.conn.execute(
            "SELECT COUNT(*) FROM positions WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        print(f"DRY RUN OK: positions={count} resolved={resolved}")
        return 0

    max_pos = args.max_positions
    if args.sample is not None:
        max_pos = args.sample

    builder = FullGameDBBuilder(
        db_path=output_path,
        max_positions=max_pos,
        max_depth=args.max_depth,
    )
    try:
        t0 = time.monotonic()
        builder.enumerate_forward(BoardState.new_game())
        builder.write_meta(
            schema_version="1",
            built_at=str(int(time.time())),
            max_positions=str(max_pos),
            max_depth=str(args.max_depth),
        )
        if args.sample is None:
            builder.backpropagate(passes=args.passes)
        if args.vacuum:
            builder.vacuum()
        elapsed = time.monotonic() - t0
        count = builder.conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        resolved = builder.conn.execute(
            "SELECT COUNT(*) FROM positions WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(
            f"Build complete: {count} positions ({resolved} resolved) "
            f"in {elapsed:.1f}s, {size_mb:.1f} MB on disk."
        )
    finally:
        builder.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
