"""tools/build_fullgame_db.py — Full-game position database generator.

Builds a binary position database by:

  1. Scanning human-played JSONL game records to identify positions with
     frequency counts (D4 canonicalisation — symmetric duplicates share one entry).
  2. BFS-expanding from high-frequency seed positions to cover opponent
     responses not present in the corpus, up to ``--expand-depth`` plies out.
  3. Backpropagating win/loss/draw outcomes through the expanded tree.

Positions are stored in a temporary SQLite database during the build so that
large expansions never require holding all data in RAM simultaneously.  Only
a small ``seen`` key-set (~9 bytes per position) and per-pass outcome caches
are kept in memory.  The temp DB is deleted automatically after the binary
output file is written.

Run inside the project venv:
    .venv/bin/python tools/build_fullgame_db.py --help
    .venv/bin/python tools/build_fullgame_db.py --dry-run

    # Default: scan data/games, write data/fullgame.bin
    .venv/bin/python tools/build_fullgame_db.py

    # Explicit games directory and output path
    .venv/bin/python tools/build_fullgame_db.py \\
        --expand-from-games data/games \\
        --min-seed-frequency 3 \\
        --expand-depth 6 \\
        --output /mnt/external/fullgame.bin

    # Store temp DB on a separate large drive:
    .venv/bin/python tools/build_fullgame_db.py \\
        --temp-db /mnt/bigdrive/nmm_build.tmp.db \\
        --max-db-gb 100

    python tools/build_fullgame_db.py --expand-from-games data/games --min-seed-frequency 4 --early-expand-depth 4 --expand-depth 6 --output /mnt/windows/fullgame.bin
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
from typing import Optional

# ── Ensure project root on path when invoked directly ────────────────────────
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Dependency check ──────────────────────────────────────────────────────────

_REQUIRED_PROJECT = ("game.board", "game.rules", "ai.board_symmetry")


def _verify_deps() -> None:
    missing: list[str] = []
    for mod in _REQUIRED_PROJECT:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"FATAL: missing project modules: {missing}", file=sys.stderr)
        sys.exit(2)


def _maybe_pip_install_requirements() -> None:
    req = _ROOT / "requirements.txt"
    if not req.exists():
        return
    try:
        __import__("fastapi")
        return
    except ImportError:
        pass
    print("Installing project requirements…", file=sys.stderr)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


# ── Project imports ───────────────────────────────────────────────────────────

from game.board import POSITIONS, BoardState
from game.rules import get_all_legal_moves, is_terminal
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "ai_board_symmetry", str(_ROOT / "ai" / "board_symmetry.py")
)
_bs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bs)
canonical_board_str = _bs.canonical_board_str
transform_notation = _bs.transform_notation

logger = logging.getLogger("fullgame_db.build")

# ── Binary output format (v2) ─────────────────────────────────────────────────

_HEADER_MAGIC = b"NMM_FGDB"
_FORMAT_VERSION_2 = 2
_HEADER_SIZE = 32
_RECORD_SIZE = 36
_HEADER_FMT = "<8sHI18x"
_RECORD_FMT = "<9sBHIIIIII"  # key(9)+outcome(1)+depth(2)+best_move(4)+4×child(4)+freq(4)

assert struct.calcsize(_HEADER_FMT) == _HEADER_SIZE
assert struct.calcsize(_RECORD_FMT) == _RECORD_SIZE

_POS_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(POSITIONS)}
_NO_POS_BIN = 31
_EMPTY_MOVE_BIN = 0xFFFFFFFF
_OUTCOME_ENCODE = {None: 0, 1: 1, -1: 2, 0: 3}
_OUTCOME_DECODE = {0: None, 1: 1, 2: -1, 3: 0}
_FLAG_ENCODE = {"N": 0, "W": 1, "L": 2}
_FLAG_DECODE = {0: "N", 1: "W", 2: "L"}
_BITS_TO_PIECE = {0b00: "", 0b01: "W", 0b10: "B"}

# ── Canonical key encoding ────────────────────────────────────────────────────

_PIECE_BITS = {".": 0b00, "W": 0b01, "B": 0b10}


def encode_canonical(board24: str, turn: str, placed_w: int, placed_b: int) -> bytes:
    """Pack a canonical position into a 9-byte key."""
    if len(board24) != 24:
        raise ValueError(f"board24 must be 24 chars, got {len(board24)}")
    val = 0
    for i, ch in enumerate(board24):
        val |= _PIECE_BITS[ch] << (i * 2)
    return val.to_bytes(6, "little") + bytes(
        (0 if turn == "W" else 1, placed_w & 0xFF, placed_b & 0xFF)
    )


def position_key(board: BoardState) -> bytes:
    """Compute the canonical 9-byte key for a BoardState."""
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


# ── Binary helpers ────────────────────────────────────────────────────────────

def _pack_move_bin(notation: Optional[str], flag: str = "N") -> int:
    if notation is None:
        return _EMPTY_MOVE_BIN
    if "x" in notation:
        move_part, cap_str = notation.split("x", 1)
        cap_idx = _POS_TO_IDX[cap_str]
    else:
        move_part = notation
        cap_idx = _NO_POS_BIN
    if "-" in move_part:
        from_str, to_str = move_part.split("-", 1)
        from_idx = _POS_TO_IDX[from_str]
    else:
        to_str = move_part
        from_idx = _NO_POS_BIN
    return (
        from_idx
        | (_POS_TO_IDX[to_str] << 5)
        | (cap_idx << 10)
        | (_FLAG_ENCODE.get(flag, 0) << 15)
    )


def _decode_key_to_board(key: bytes) -> BoardState:
    """Reconstruct a canonical BoardState from a 9-byte position key."""
    val = int.from_bytes(key[:6], "little")
    positions_dict: dict[str, str] = {}
    for i, pos in enumerate(POSITIONS):
        positions_dict[pos] = _BITS_TO_PIECE[(val >> (i * 2)) & 0x3]
    turn = "W" if key[6] == 0 else "B"
    placed_w, placed_b = key[7], key[8]
    on_w = sum(1 for v in positions_dict.values() if v == "W")
    on_b = sum(1 for v in positions_dict.values() if v == "B")
    return BoardState(
        positions=positions_dict,
        turn=turn,
        pieces_on_board={"W": on_w, "B": on_b},
        pieces_placed={"W": placed_w, "B": placed_b},
        pieces_captured={"W": placed_w - on_w, "B": placed_b - on_b},
    )


def _current_rss_gb() -> float:
    """Return current resident set size in GB (Linux /proc; 0.0 on other platforms)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


# ── Edge packing helpers ──────────────────────────────────────────────────────
# Edges are stored as compact blobs in SQLite.
# Per edge: flag(1) + notation_len(1) + child_key(9) + notation(N bytes).

def _pack_edges(edges: list) -> Optional[bytes]:
    """Pack [[notation, child_key_bytes, flag], ...] to a compact BLOB."""
    if not edges:
        return None
    parts = []
    for notation, child_key, flag in edges:
        n_enc = notation.encode()
        parts.append(bytes([_FLAG_ENCODE.get(flag, 0), len(n_enc)]) + child_key + n_enc)
    return b"".join(parts)


def _unpack_edges(data: Optional[bytes]) -> list:
    """Unpack a BLOB back to [[notation, child_key_bytes, flag], ...]."""
    if not data:
        return []
    edges = []
    i = 0
    while i < len(data):
        flag = _FLAG_DECODE.get(data[i], "N")
        n_len = data[i + 1]
        child_key = bytes(data[i + 2:i + 11])
        notation = data[i + 11:i + 11 + n_len].decode()
        edges.append([notation, child_key, flag])
        i += 11 + n_len
    return edges


# ── Game file helpers ─────────────────────────────────────────────────────────

def _is_human_game(game: dict) -> bool:
    if game.get("human_color"):
        return True
    moves = game.get("moves") or []
    return all(m.get("game_ai_score") is None for m in moves)


def _game_notation_to_move(move: dict) -> Optional[dict]:
    to = move.get("to")
    if not to:
        return None
    return {"from": move.get("from"), "to": to, "capture": move.get("capture")}


# ── Builder ───────────────────────────────────────────────────────────────────

class ExpandFromGamesBuilder:
    """Frequency-seeded BFS builder backed by a temporary SQLite database.

    Phase 1 — scan human JSONL game records; count how often each canonical
               position is visited.
    Phase 2 — BFS-expand from positions whose frequency meets
               ``min_seed_frequency``.
    Phase 3 — backpropagate win/loss/draw outcomes through the expanded tree.

    During the build, all position data is stored in a SQLite file on disk.
    Only a set of seen keys (~9 bytes each) and per-pass outcome caches are
    kept in RAM.  This allows arbitrarily large builds without OOM.

    Call ``build(games_dir, output_path)`` then ``write_binary(output_path)``.
    The temp DB is deleted automatically after ``write_binary`` completes.

Example;
python tools/build_fullgame_db.py --expand-from-games data/games --min-seed-frequency 3 --early-expand-depth 4 --expand-depth 6 --output /mnt/windows/NMM_DB/fullgame.bin --temp-db /mnt/windows/NMM_DB/ --max-db-gb 40

for 269 games → 5168 unique canonical positions; temp size = 4.7 GB, final DB size = 545 MB


    """

    _COMMIT_INTERVAL = 500  # commit every N BFS steps

    def __init__(
        self,
        min_seed_frequency: int = 2,
        expand_depth: int = 4,
        early_expand_depth: Optional[int] = None,
        max_expand_positions: Optional[int] = None,
        backprop_passes: int = 6,
        progress_every: float = 5.0,
        max_memory_gb: float = 10.0,
        temp_db_path: Optional[Path] = None,
        max_db_gb: float = 10.0,
    ) -> None:
        self.min_seed_frequency = min_seed_frequency
        self.expand_depth = expand_depth
        self.early_expand_depth = early_expand_depth if early_expand_depth is not None else expand_depth * 2
        self.max_expand_positions = max_expand_positions
        self.backprop_passes = backprop_passes
        self.progress_every = progress_every
        self.max_memory_gb = max_memory_gb
        self.temp_db_path = temp_db_path
        self.max_db_gb = max_db_gb
        self._db_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._seen: set[bytes] = set()  # BFS dedup; 9 bytes per key
        self._games_processed = 0
        self._expanded = 0
        self._t_start = time.monotonic()

    def _depth_for_seed(self, key: bytes) -> int:
        """Return the BFS depth budget for a seed based on pieces placed."""
        placed_total = key[7] + key[8]
        t = min(placed_total / 18.0, 1.0)
        depth = self.early_expand_depth * (1 - t) + self.expand_depth * t
        return max(self.expand_depth, round(depth))

    # ── DB lifecycle ─────────────────────────────────────────────────────────

    def _open_db(self, output_path: Optional[Path]) -> None:
        if self.temp_db_path is not None:
            self._db_path = Path(self.temp_db_path)
        elif output_path is not None:
            self._db_path = output_path.with_suffix(".tmp.db")
        else:
            import tempfile
            self._db_path = Path(tempfile.mktemp(suffix=".tmp.db", prefix="nmm_fgdb_"))
        # If user passed a directory, append a default filename.
        if self._db_path.is_dir():
            self._db_path = self._db_path / "fullgame.tmp.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Temp DB: %s (limit %.1f GB)", self._db_path, self.max_db_gb)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -65536;
            CREATE TABLE IF NOT EXISTS positions (
                key       BLOB    PRIMARY KEY,
                outcome   INTEGER NOT NULL DEFAULT 0,
                depth     INTEGER,
                best_move TEXT,
                frequency INTEGER NOT NULL DEFAULT 0,
                edges     BLOB
            ) WITHOUT ROWID;
        """)
        self._conn.commit()

    def _close_db(self, delete: bool = True) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if delete and self._db_path is not None and self._db_path.exists():
            try:
                self._db_path.unlink()
                logger.info("Deleted temp DB: %s", self._db_path)
            except OSError as exc:
                logger.warning("Could not delete temp DB %s: %s", self._db_path, exc)

    def _db_size_gb(self) -> float:
        if self._db_path is None:
            return 0.0
        try:
            return os.path.getsize(self._db_path) / (1024 ** 3)
        except OSError:
            return 0.0

    # ── Public API ───────────────────────────────────────────────────────────

    def build(self, games_dir: Path, output_path: Optional[Path] = None) -> None:
        self._open_db(output_path)
        try:
            self._scan_games(games_dir)
            self._bfs_expand()
            self._backpropagate()
        except Exception:
            self._close_db(delete=False)
            raise

    def write_binary(self, output_path: Path) -> int:
        """Read from the temp DB (sorted by key) and write the binary v2 file."""
        if self._conn is None:
            raise RuntimeError("write_binary called before build()")

        sorted_keys = [row[0] for row in self._conn.execute(
            "SELECT key FROM positions ORDER BY key"
        )]

        records: list[bytes] = []
        for key in sorted_keys:
            row = self._conn.execute(
                "SELECT outcome, depth, best_move, frequency, edges FROM positions WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                continue
            outcome_byte, depth_raw, best_move_str, freq, edges_blob = row
            depth_val = 0xFFFF if depth_raw is None else min(depth_raw, 0xFFFE)
            bm_packed = _pack_move_bin(best_move_str)
            edges = _unpack_edges(edges_blob)
            informative = [(n, f) for n, _, f in edges if f in ("W", "L")]
            neutral = [(n, f) for n, _, f in edges if f == "N"]
            top4 = (informative + neutral)[:4]
            children = [_pack_move_bin(n, f) for n, f in top4]
            while len(children) < 4:
                children.append(_EMPTY_MOVE_BIN)
            records.append(struct.pack(
                _RECORD_FMT,
                key, outcome_byte, depth_val, bm_packed,
                children[0], children[1], children[2], children[3],
                max(0, freq),
            ))

        record_count = len(records)
        header = struct.pack(_HEADER_FMT, _HEADER_MAGIC, _FORMAT_VERSION_2, record_count)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(header)
            for rec in records:
                fh.write(rec)
        logger.info("write_binary: %d records → %s", record_count, output_path)

        self._close_db(delete=True)
        return record_count

    def stats(self) -> tuple[int, int]:
        """Return (total_positions, resolved_positions)."""
        if self._conn is None:
            return 0, 0
        total = self._conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        resolved = self._conn.execute(
            "SELECT COUNT(*) FROM positions WHERE outcome != 0"
        ).fetchone()[0]
        return total, resolved

    # ── Phase 1: game scan ───────────────────────────────────────────────────

    def _scan_games(self, games_dir: Path) -> None:
        import glob
        files = sorted(glob.glob(str(games_dir / "*.jsonl")))
        logger.info("Scanning %d game files in %s", len(files), games_dir)
        for fpath in files:
            try:
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            import json
                            game = json.loads(line)
                        except Exception:
                            continue
                        if not _is_human_game(game):
                            continue
                        self._process_game(game)
            except OSError as exc:
                logger.warning("Cannot read %s — %s", fpath, exc)
            self._conn.commit()  # commit after each file

        total, _ = self.stats()
        logger.info(
            "Scanned %d games → %d unique canonical positions.",
            self._games_processed, total,
        )

    def _process_game(self, game: dict) -> None:
        import json as _json
        moves = game.get("moves") or []
        board = BoardState.new_game()
        for move_record in moves:
            mv = _game_notation_to_move(move_record)
            if mv is None:
                break
            try:
                key = position_key(board)
                self._conn.execute(
                    "INSERT INTO positions(key, frequency) VALUES(?, 1) "
                    "ON CONFLICT(key) DO UPDATE SET frequency = frequency + 1",
                    (key,),
                )
                board = board.apply_move(mv)
            except Exception:
                break
        try:
            key = position_key(board)
            winner = game.get("winner")
            if winner in ("W", "B"):
                outcome_val = 1 if winner == "W" else 2
                self._conn.execute(
                    "INSERT INTO positions(key, frequency, outcome, depth) VALUES(?, 1, ?, 0) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "  frequency = frequency + 1, "
                    "  outcome = CASE WHEN outcome = 0 THEN excluded.outcome ELSE outcome END, "
                    "  depth    = CASE WHEN outcome = 0 THEN 0 ELSE depth END",
                    (key, outcome_val),
                )
            else:
                self._conn.execute(
                    "INSERT INTO positions(key, frequency) VALUES(?, 1) "
                    "ON CONFLICT(key) DO UPDATE SET frequency = frequency + 1",
                    (key,),
                )
        except Exception:
            pass
        self._games_processed += 1

    # ── Phase 2: BFS expansion ───────────────────────────────────────────────

    def _bfs_expand(self) -> None:
        seeds = [row[0] for row in self._conn.execute(
            "SELECT key FROM positions WHERE frequency >= ?",
            (self.min_seed_frequency,),
        )]
        logger.info(
            "%d seed positions (freq >= %d); early depth %d → late depth %d.",
            len(seeds), self.min_seed_frequency,
            self.early_expand_depth, self.expand_depth,
        )
        if not seeds:
            logger.warning("No seeds — try lowering --min-seed-frequency.")
            return
        if self.early_expand_depth > 8:
            logger.warning(
                "--early-expand-depth %d is very large — this can consume large amounts of "
                "disk space. The build will stop at --max-db-gb %.1f GB.",
                self.early_expand_depth, self.max_db_gb,
            )

        # Populate seen set from all positions already in the DB.
        self._seen = set(row[0] for row in self._conn.execute("SELECT key FROM positions"))

        seed_entries = sorted(
            ((k, self._depth_for_seed(k)) for k in seeds),
            key=lambda x: -x[1],
        )
        queue: deque[tuple[bytes, int]] = deque(seed_entries)

        t_last = time.monotonic()
        _db_check_counter = 0
        _db_limit_hit = False
        _ops_since_commit = 0

        while queue:
            if (
                self.max_expand_positions is not None
                and self._expanded >= self.max_expand_positions
            ):
                logger.info("--max-expand-positions cap reached.")
                break

            key, remaining = queue.popleft()
            board = _decode_key_to_board(key)

            _db_check_counter += 1
            if _db_check_counter >= 1000:
                _db_check_counter = 0
                db_gb = self._db_size_gb()
                if db_gb >= self.max_db_gb:
                    logger.warning(
                        "Temp DB limit %.1f GB reached (%.2f GB) after %d expanded — "
                        "stopping BFS early and writing partial results.",
                        self.max_db_gb, db_gb, self._expanded,
                    )
                    _db_limit_hit = True
                    break

            terminal, winner = is_terminal(board)
            if terminal:
                outcome = 1 if winner == "W" else (-1 if winner == "B" else 0)
                outcome_val = _OUTCOME_ENCODE[outcome]
                self._conn.execute(
                    "INSERT INTO positions(key, outcome, depth) VALUES(?, ?, 0) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "  outcome = CASE WHEN outcome = 0 THEN excluded.outcome ELSE outcome END, "
                    "  depth   = CASE WHEN outcome = 0 THEN 0 ELSE depth END",
                    (key, outcome_val),
                )
                _ops_since_commit += 1
                self._expanded += 1
            else:
                _, sym, _turn, _pw, _pb = canonical_components(board)
                edges: list[list] = []
                for mv in get_all_legal_moves(board):
                    child = board.apply_move(mv)
                    child_key = position_key(child)
                    canon_n = transform_notation(move_notation(mv), sym) or move_notation(mv)
                    edges.append([canon_n, child_key, "N"])
                    if child_key not in self._seen and remaining > 1:
                        self._seen.add(child_key)
                        self._conn.execute(
                            "INSERT OR IGNORE INTO positions(key) VALUES(?)", (child_key,)
                        )
                        queue.append((child_key, remaining - 1))

                packed = _pack_edges(edges)
                self._conn.execute(
                    "INSERT INTO positions(key, edges) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "  edges = CASE WHEN edges IS NULL THEN excluded.edges ELSE edges END",
                    (key, packed),
                )
                _ops_since_commit += 1
                self._expanded += 1

            if _ops_since_commit >= self._COMMIT_INTERVAL:
                self._conn.commit()
                _ops_since_commit = 0

            now = time.monotonic()
            if now - t_last >= self.progress_every:
                t_last = now
                db_gb = self._db_size_gb()
                total, _ = self.stats()
                logger.info(
                    "BFS expand: %d expanded  %d queued  %d total  %.3f GB DB",
                    self._expanded, len(queue), total, db_gb,
                )
                if db_gb >= self.max_db_gb:
                    logger.warning(
                        "Temp DB limit %.1f GB reached (%.2f GB) after %d expanded — "
                        "stopping BFS early and writing partial results.",
                        self.max_db_gb, db_gb, self._expanded,
                    )
                    _db_limit_hit = True
                    break

        self._conn.commit()
        total, _ = self.stats()
        if _db_limit_hit:
            logger.info(
                "BFS stopped by DB size cap — %d expanded, %d total positions (partial).",
                self._expanded, total,
            )
        else:
            logger.info(
                "BFS complete — %d expanded, %d total positions.",
                self._expanded, total,
            )

    # ── Phase 3: backpropagation ─────────────────────────────────────────────

    def _backpropagate(self) -> None:
        for pass_no in range(1, self.backprop_passes + 1):
            # Load outcome + depth for every position into RAM once per pass.
            # ~17 bytes × N positions — tiny compared to full _PosData objects.
            outcome_cache: dict[bytes, tuple[Optional[int], Optional[int]]] = {}
            for row in self._conn.execute("SELECT key, outcome, depth FROM positions"):
                k, oc, d = row
                outcome_cache[bytes(k)] = (_OUTCOME_DECODE[oc], d)

            # Load all unresolved positions that have edges.
            pending = self._conn.execute(
                "SELECT key, edges FROM positions WHERE outcome = 0 AND edges IS NOT NULL"
            ).fetchall()

            updates: list[tuple] = []
            for key, edges_blob in pending:
                key = bytes(key)
                side = "W" if key[6] == 0 else "B"
                win_outcome = 1 if side == "W" else -1
                loss_outcome = -win_outcome

                edges = _unpack_edges(edges_blob)
                child_infos: list[tuple] = []
                for n, ck, _ in edges:
                    ck = bytes(ck)
                    entry = outcome_cache.get(ck)
                    child_outcome = entry[0] if entry is not None else None
                    child_depth = entry[1] if entry is not None else None
                    child_infos.append((n, ck, child_outcome, child_depth))

                outcomes = [o for _, _, o, _ in child_infos]

                if win_outcome in outcomes:
                    best_o = win_outcome
                elif any(o is None for o in outcomes):
                    continue
                else:
                    best_o = max(outcomes) if side == "W" else min(outcomes)

                new_edges: list[list] = []
                best_move: Optional[str] = None
                matching_depths: list[int] = []
                for n, ck, o, child_d in child_infos:
                    if o == win_outcome:
                        flag = "W"
                    elif o == loss_outcome:
                        flag = "L"
                    else:
                        flag = "N"
                    if best_move is None and o == best_o:
                        best_move = n
                    if o == best_o and child_d is not None:
                        matching_depths.append(child_d)
                    new_edges.append([n, ck, flag])

                new_depth = (1 + min(matching_depths)) if matching_depths else None
                new_outcome_byte = _OUTCOME_ENCODE[best_o]
                new_edges_blob = _pack_edges(new_edges)
                updates.append((new_outcome_byte, new_depth, best_move, new_edges_blob, key))

            if updates:
                self._conn.executemany(
                    "UPDATE positions SET outcome=?, depth=?, best_move=?, edges=? WHERE key=?",
                    updates,
                )
                self._conn.commit()

            logger.info("Backprop pass %d: labelled %d positions.", pass_no, len(updates))
            if not updates:
                break


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Build the Nine Men's Morris full-game position database. "
            "Scans human game records, BFS-expands around common positions, "
            "and writes a sorted binary .bin file. "
            "Uses a temporary SQLite file during the build so large expansions "
            "never require holding all data in RAM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--expand-from-games", type=Path, default=None, metavar="DIR",
        help=(
            "Directory of human-played JSONL game records to scan. "
            "Defaults to <project>/data/games."
        ),
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output binary file path.  Default: <project>/data/fullgame.bin",
    )
    parser.add_argument(
        "--db-dir", type=Path, default=None,
        help=(
            "Directory to write fullgame.bin into.  "
            "Shorthand for --output <dir>/fullgame.bin.  Ignored if --output is set."
        ),
    )
    parser.add_argument(
        "--temp-db", type=Path, default=None, metavar="PATH",
        help=(
            "Path for the temporary SQLite build database "
            "(default: <output>.tmp.db alongside the output file). "
            "Point this at a large drive for very large builds — e.g. /mnt/bigdrive/build.tmp.db"
        ),
    )
    parser.add_argument(
        "--max-db-gb", type=float, default=10.0, metavar="GB",
        help=(
            "Maximum size of the temporary SQLite database in GB (default: 10.0). "
            "BFS stops and writes partial results when the temp DB exceeds this. "
            "Raise to e.g. 100 or 1000 for builds on large drives."
        ),
    )
    parser.add_argument(
        "--min-seed-frequency", type=int, default=2, metavar="N",
        help="Minimum human-game visits for a position to seed the BFS expansion (default 2).",
    )
    parser.add_argument(
        "--expand-depth", type=int, default=4, metavar="D",
        help="BFS depth for late-game / end-of-placement seed positions (default 4).",
    )
    parser.add_argument(
        "--early-expand-depth", type=int, default=None, metavar="D",
        help=(
            "BFS depth for early-game seed positions (0 pieces placed). "
            "Tapers linearly to --expand-depth by the end of placement (18 pieces). "
            "Default: 2 × --expand-depth."
        ),
    )
    parser.add_argument(
        "--max-expand-positions", type=int, default=None, metavar="N",
        help="Hard cap on BFS-expanded positions (default: unlimited).",
    )
    parser.add_argument(
        "--max-gb", type=float, default=6.0, metavar="GB",
        help=(
            "Abort BFS and write partial results when process RSS exceeds this value in GB "
            "(default: 6.0).  Secondary guard alongside --max-db-gb."
        ),
    )
    parser.add_argument(
        "--passes", type=int, default=6,
        help="Backpropagation passes for win/loss labelling (default 6).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build from a tiny synthetic game set (no disk write).",
    )
    parser.add_argument(
        "--install-deps", action="store_true",
        help="Pip-install project requirements before building.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _verify_deps()
    if args.install_deps:
        _maybe_pip_install_requirements()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    games_dir = (args.expand_from_games or (_ROOT / "data" / "games")).resolve()

    if args.output is not None:
        output_path = args.output.resolve()
    elif args.db_dir is not None:
        output_path = args.db_dir.resolve() / "fullgame.bin"
    else:
        output_path = (_ROOT / "data" / "fullgame.bin").resolve()

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        import tempfile
        synthetic = [
            {"moves": [{"to": p} for p in ["a7","d6","a4","d3","a1","d1","b6","b4","g7","g4"]], "human_color": "W"},
            {"moves": [{"to": p} for p in ["a7","g4","a4","g1","g7","d6","b6","d3","b4","f4"]], "human_color": "W"},
            {"moves": [{"to": p} for p in ["a7","d6","g7","d3","b6","f6","a4","g4","a1","g1"]], "human_color": "W"},
        ]
        with tempfile.TemporaryDirectory() as td:
            gdir = Path(td)
            with open(gdir / "dry_run.jsonl", "w") as f:
                for g in synthetic:
                    f.write(json.dumps(g) + "\n")
            with tempfile.TemporaryDirectory() as db_td:
                builder = ExpandFromGamesBuilder(
                    min_seed_frequency=1, expand_depth=2, max_expand_positions=500,
                    temp_db_path=Path(db_td) / "dry_run.tmp.db",
                )
                builder.build(gdir, output_path=None)
        total, resolved = builder.stats()
        print(f"DRY RUN OK: positions={total} resolved={resolved}")
        return 0

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not games_dir.is_dir():
        print(f"ERROR: games directory not found: {games_dir}", file=sys.stderr)
        return 1
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: cannot create output directory {output_path.parent}: {exc}", file=sys.stderr)
        return 1
    _probe = output_path.parent / ".nmm_write_probe"
    try:
        _probe.write_bytes(b"")
        _probe.unlink()
    except OSError as exc:
        print(f"ERROR: output directory not writable ({output_path.parent}): {exc}", file=sys.stderr)
        return 1

    temp_db = args.temp_db.resolve() if args.temp_db else None
    if temp_db is not None and temp_db.is_dir():
        temp_db = temp_db / "fullgame.tmp.db"
    print(f"Games dir:  {games_dir}")
    print(f"Output:     {output_path}")
    print(f"Temp DB:    {temp_db or str(output_path.with_suffix('.tmp.db'))} (max {args.max_db_gb:.1f} GB)")

    # ── Build ─────────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    builder = ExpandFromGamesBuilder(
        min_seed_frequency=args.min_seed_frequency,
        expand_depth=args.expand_depth,
        early_expand_depth=args.early_expand_depth,
        max_expand_positions=args.max_expand_positions,
        backprop_passes=args.passes,
        max_memory_gb=args.max_gb,
        temp_db_path=temp_db,
        max_db_gb=args.max_db_gb,
    )
    builder.build(games_dir, output_path)
    total, resolved = builder.stats()
    elapsed = time.monotonic() - t0
    print(f"Build: {total} positions ({resolved} resolved) in {elapsed:.1f}s.")

    n = builder.write_binary(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Binary: {n} records, {size_mb:.2f} MB → {output_path}")

    _update_settings(output_path)
    return 0


def _update_settings(db_path: Path) -> None:
    import json
    settings_path = _ROOT / "data" / "settings.json"
    try:
        settings: dict = {}
        if settings_path.exists():
            with open(settings_path) as f:
                settings = json.load(f)
        settings["fullgame_db_path"] = str(db_path)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        print(f"Settings updated: fullgame_db_path = {db_path}")
    except OSError as exc:
        print(f"WARNING: could not update settings.json: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
