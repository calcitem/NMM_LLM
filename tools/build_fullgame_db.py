"""tools/build_fullgame_db.py — Full-game position database generator.

Builds a binary position database by:

  1. Scanning human-played JSONL game records to identify positions with
     frequency counts (D4 canonicalisation — symmetric duplicates share one entry).
  2. BFS-expanding from high-frequency seed positions to cover opponent
     responses not present in the corpus, up to ``--expand-depth`` plies out.
  3. Backpropagating win/loss/draw outcomes through the expanded tree.

The entire build runs in RAM.  Output is a sorted binary ``.bin`` file written
directly — no SQLite or other intermediate file is created.

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

	python tools/build_fullgame_db.py --expand-from-games data/games --min-seed-frequency 4 --early-expand-depth 4 --expand-depth 6 --output /mnt/windows/fullgame.bin
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass, field
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
_FLAG_ENCODE = {"N": 0, "W": 1, "L": 2}
_BITS_TO_PIECE = {0b00: "", 0b01: "W", 0b10: "B"}   # "" matches BoardState.positions empty sentinel

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


# ── In-memory position record ─────────────────────────────────────────────────

@dataclass
class _PosData:
    outcome: Optional[int] = None       # 1=W wins, -1=B wins, 0=draw, None=unknown
    depth: Optional[int] = None         # plies to terminal, or None
    best_move: Optional[str] = None     # canonical-form best move notation
    frequency: int = 0                  # human-game visit count
    edges: list = field(default_factory=list)  # [[notation, child_key_bytes, flag], ...]


# ── Builder ───────────────────────────────────────────────────────────────────

class ExpandFromGamesBuilder:
    """Frequency-seeded BFS builder.  Entirely in-memory — no intermediate files.

    Phase 1 — scan human JSONL game records; count how often each canonical
               position is visited.
    Phase 2 — BFS-expand from positions whose frequency meets
               ``min_seed_frequency``.  Each seed's depth budget scales with
               how early in the game it is: ``early_expand_depth`` at 0 pieces
               placed, tapering linearly to ``expand_depth`` at 18 pieces
               placed (end of placement phase).  Seeds are processed
               deepest-first so early-game positions claim the BFS frontier
               before late-game ones.
    Phase 3 — backpropagate win/loss/draw outcomes through the expanded tree.

    Call ``write_binary(output_path)`` after ``build()`` to flush to disk.
    """

    def __init__(
        self,
        min_seed_frequency: int = 2,
        expand_depth: int = 4,
        early_expand_depth: Optional[int] = None,
        max_expand_positions: Optional[int] = None,
        backprop_passes: int = 6,
        progress_every: float = 5.0,
        max_memory_gb: float = 10.0,
    ) -> None:
        self.min_seed_frequency = min_seed_frequency
        self.expand_depth = expand_depth
        self.early_expand_depth = early_expand_depth if early_expand_depth is not None else expand_depth * 2
        self.max_expand_positions = max_expand_positions
        self.backprop_passes = backprop_passes
        self.progress_every = progress_every
        self.max_memory_gb = max_memory_gb
        self._store: dict[bytes, _PosData] = {}
        self._games_processed = 0
        self._expanded = 0
        self._t_start = time.monotonic()

    def _depth_for_seed(self, key: bytes) -> int:
        """Return the BFS depth budget for a seed based on pieces placed."""
        placed_total = key[7] + key[8]          # 0 (start) … 18 (all placed)
        t = min(placed_total / 18.0, 1.0)       # 0.0 = early game, 1.0 = end of placement
        depth = self.early_expand_depth * (1 - t) + self.expand_depth * t
        return max(self.expand_depth, round(depth))

    # ── Public API ───────────────────────────────────────────────────────────

    def build(self, games_dir: Path) -> None:
        self._scan_games(games_dir)
        self._bfs_expand()
        self._backpropagate()

    def write_binary(self, output_path: Path) -> int:
        """Sort the in-memory store by key and write a binary v2 file."""
        sorted_keys = sorted(self._store.keys())
        records: list[bytes] = []
        for key in sorted_keys:
            data = self._store[key]
            outcome_byte = _OUTCOME_ENCODE.get(data.outcome, 0)
            depth_val = 0xFFFF if data.depth is None else min(data.depth, 0xFFFE)
            bm_packed = _pack_move_bin(data.best_move)
            freq_val = max(0, data.frequency)
            informative = [(n, f) for n, _, f in data.edges if f in ("W", "L")]
            neutral = [(n, f) for n, _, f in data.edges if f == "N"]
            top4 = (informative + neutral)[:4]
            children = [_pack_move_bin(n, f) for n, f in top4]
            while len(children) < 4:
                children.append(_EMPTY_MOVE_BIN)
            records.append(struct.pack(
                _RECORD_FMT,
                key, outcome_byte, depth_val, bm_packed,
                children[0], children[1], children[2], children[3],
                freq_val,
            ))
        record_count = len(records)
        header = struct.pack(_HEADER_FMT, _HEADER_MAGIC, _FORMAT_VERSION_2, record_count)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(header)
            for rec in records:
                fh.write(rec)
        logger.info("write_binary: %d records → %s", record_count, output_path)
        return record_count

    def stats(self) -> tuple[int, int]:
        """Return (total_positions, resolved_positions)."""
        total = len(self._store)
        resolved = sum(1 for d in self._store.values() if d.outcome is not None)
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
                            game = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not _is_human_game(game):
                            continue
                        self._process_game(game)
            except OSError as exc:
                logger.warning("Cannot read %s — %s", fpath, exc)
        logger.info(
            "Scanned %d games → %d unique canonical positions.",
            self._games_processed, len(self._store),
        )

    def _process_game(self, game: dict) -> None:
        moves = game.get("moves") or []
        board = BoardState.new_game()
        for move_record in moves:
            mv = _game_notation_to_move(move_record)
            if mv is None:
                break
            try:
                key = position_key(board)
                if key in self._store:
                    self._store[key].frequency += 1
                else:
                    self._store[key] = _PosData(frequency=1)
                board = board.apply_move(mv)
            except Exception:
                break
        try:
            key = position_key(board)
            data = self._store.setdefault(key, _PosData())
            data.frequency += 1
            # Seed outcome from game result if not already resolved.
            winner = game.get("winner")
            if data.outcome is None and winner in ("W", "B"):
                data.outcome = 1 if winner == "W" else -1
                data.depth = 0
        except Exception:
            pass
        self._games_processed += 1

    # ── Phase 2: BFS expansion ───────────────────────────────────────────────

    def _bfs_expand(self) -> None:
        seeds = [k for k, v in self._store.items() if v.frequency >= self.min_seed_frequency]
        logger.info(
            "%d seed positions (freq >= %d); early depth %d → late depth %d.",
            len(seeds), self.min_seed_frequency, self.early_expand_depth, self.expand_depth,
        )
        if not seeds:
            logger.warning("No seeds — try lowering --min-seed-frequency.")
            return
        if self.early_expand_depth > 8:
            logger.warning(
                "--early-expand-depth %d is very large — this can consume tens of GB of RAM. "
                "The build will stop at --max-gb %.1f GB (RSS).",
                self.early_expand_depth, self.max_memory_gb,
            )

        seen: set[bytes] = set(self._store.keys())
        # Sort seeds deepest-budget-first so early-game positions claim the
        # BFS frontier before late-game ones.  Store only (key, remaining) —
        # boards are reconstructed on demand to keep queue memory small.
        seed_entries = sorted(
            ((k, self._depth_for_seed(k)) for k in seeds),
            key=lambda x: -x[1],
        )
        queue: deque[tuple[bytes, int]] = deque(seed_entries)

        t_last = time.monotonic()
        _mem_check_counter = 0
        _mem_limit_hit = False
        while queue:
            if (
                self.max_expand_positions is not None
                and self._expanded >= self.max_expand_positions
            ):
                logger.info("--max-expand-positions cap reached.")
                break

            key, remaining = queue.popleft()
            board = _decode_key_to_board(key)

            # Memory guard — check every 1000 pops.
            _mem_check_counter += 1
            if _mem_check_counter >= 1000:
                _mem_check_counter = 0
                rss = _current_rss_gb()
                if rss >= self.max_memory_gb:
                    logger.warning(
                        "Memory limit %.1f GB reached (RSS %.2f GB) after %d expanded — "
                        "stopping BFS early and writing partial results.",
                        self.max_memory_gb, rss, self._expanded,
                    )
                    _mem_limit_hit = True
                    break

            terminal, winner = is_terminal(board)
            if terminal:
                outcome = 1 if winner == "W" else (-1 if winner == "B" else 0)
                data = self._store.setdefault(key, _PosData())
                if data.outcome is None:
                    data.outcome = outcome
                    data.depth = 0
                self._expanded += 1
                continue

            _, sym, _turn, _pw, _pb = canonical_components(board)
            edges: list[list] = []
            for mv in get_all_legal_moves(board):
                child = board.apply_move(mv)
                child_key = position_key(child)
                canon_n = transform_notation(move_notation(mv), sym) or move_notation(mv)
                edges.append([canon_n, child_key, "N"])
                if child_key not in seen and remaining > 1:
                    seen.add(child_key)
                    queue.append((child_key, remaining - 1))

            data = self._store.setdefault(key, _PosData())
            if not data.edges:
                data.edges = edges
            self._expanded += 1

            now = time.monotonic()
            if now - t_last >= self.progress_every:
                t_last = now
                rss = _current_rss_gb()
                logger.info(
                    "BFS expand: %d expanded  %d queued  %d total  %.2f GB RSS",
                    self._expanded, len(queue), len(self._store), rss,
                )
                if rss >= self.max_memory_gb:
                    logger.warning(
                        "Memory limit %.1f GB reached (RSS %.2f GB) after %d expanded — "
                        "stopping BFS early and writing partial results.",
                        self.max_memory_gb, rss, self._expanded,
                    )
                    _mem_limit_hit = True
                    break

        if _mem_limit_hit:
            logger.info(
                "BFS stopped by memory cap — %d expanded, %d total positions (partial).",
                self._expanded, len(self._store),
            )
        else:
            logger.info(
                "BFS complete — %d expanded, %d total positions.",
                self._expanded, len(self._store),
            )

    # ── Phase 3: backpropagation ─────────────────────────────────────────────

    def _backpropagate(self) -> None:
        for pass_no in range(1, self.backprop_passes + 1):
            updated = 0
            for key, data in self._store.items():
                if data.outcome is not None or not data.edges:
                    continue

                side = "W" if key[6] == 0 else "B"
                win_outcome = 1 if side == "W" else -1

                # Collect child outcomes (None if child absent or unresolved).
                child_infos: list[tuple[str, bytes, Optional[int], Optional[int]]] = []
                for n, ck, _ in data.edges:
                    ch = self._store.get(ck)
                    child_infos.append((n, ck, ch.outcome if ch else None, ch.depth if ch else None))

                outcomes = [o for _, _, o, _ in child_infos]

                # WIN: any child is a forced win for STM — safe with partial coverage.
                if win_outcome in outcomes:
                    best_o = win_outcome
                # LOSS/DRAW: require all children in store and resolved.
                elif any(o is None for o in outcomes):
                    continue
                else:
                    best_o = max(outcomes) if side == "W" else min(outcomes)

                loss_outcome = -win_outcome
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

                data.outcome = best_o
                data.depth = (1 + min(matching_depths)) if matching_depths else None
                data.best_move = best_move
                data.edges = new_edges
                updated += 1

            logger.info("Backprop pass %d: labelled %d positions.", pass_no, updated)
            if updated == 0:
                break


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Nine Men's Morris full-game position database. "
            "Scans human game records, BFS-expands around common positions, "
            "and writes a sorted binary .bin file directly — no intermediate files."
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
            "(default: 6.0).  Prevents OOM when using large --early-expand-depth values."
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
            builder = ExpandFromGamesBuilder(
                min_seed_frequency=1, expand_depth=2, max_expand_positions=500,
            )
            builder.build(gdir)
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

    print(f"Games dir: {games_dir}")
    print(f"Output:    {output_path}")

    # ── Build ─────────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    builder = ExpandFromGamesBuilder(
        min_seed_frequency=args.min_seed_frequency,
        expand_depth=args.expand_depth,
        early_expand_depth=args.early_expand_depth,
        max_expand_positions=args.max_expand_positions,
        backprop_passes=args.passes,
        max_memory_gb=args.max_gb,
    )
    builder.build(games_dir)
    total, resolved = builder.stats()
    elapsed = time.monotonic() - t0
    print(f"Build: {total} positions ({resolved} resolved) in {elapsed:.1f}s.")

    n = builder.write_binary(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Binary: {n} records, {size_mb:.2f} MB → {output_path}")

    _update_settings(output_path)
    return 0


def _update_settings(db_path: Path) -> None:
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
