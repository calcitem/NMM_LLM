#!/usr/bin/env python3
"""tools/build_endgame_db.py — Generalized retrograde endgame solver for NMM.

Usage
-----
    # Build a specific table (loads its sub-tables from disk automatically):
    python tools/build_endgame_db.py --nW 4 --nB 3

    # Build all tables up to nW+nB ≤ <max-sum> in dependency order:
    python tools/build_endgame_db.py --build-all [--max-sum 11]

    # Skip tables that already exist on disk:
    python tools/build_endgame_db.py --build-all --skip-existing

Output: data/endgame/endgame_{nW}_{nB}.wdl

Algorithm
---------
All C(24,nW)×C(24-nW,nB)×2 positions are enumerated via the combinatorial index.

Pass 0: mark every position whose outcome is immediately determinable (blocked
move-phase mover → LOSS; any mill-closing move with a WIN capture → WIN).

Iterative forward passes propagate WIN and LOSS until fixed-point.
Remaining UNKNOWN positions → DRAW.

Tables are built in ascending (nW+nB) order so sub-tables are always fully
solved before they are consulted.  The 3v3 base case requires no sub-tables
because any mill capture there immediately reduces the opponent below 3 pieces.

Performance notes
-----------------
* _CT: precomputed Pascal's triangle avoids math.comb() function-call overhead.
* _RI: reusable 24-int buffer for the B-remapping step in _encode.
* Inner move loop uses bitmask occupancy checks and incremental mask updates
  rather than set/list allocations.
* new_mover list is only built when needed (mill-closing or non-capture encode).
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import logging
import mmap
import sys
import time
import types
from math import comb
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Load ai.endgame_solved_db without triggering ai/__init__.py heavy deps.
_ai_pkg = types.ModuleType("ai")
_ai_pkg.__path__ = [str(_ROOT / "ai")]
sys.modules.setdefault("ai", _ai_pkg)
_spec = _ilu.spec_from_file_location(
    "ai.endgame_solved_db", str(_ROOT / "ai" / "endgame_solved_db.py")
)
_esdb_mod = _ilu.module_from_spec(_spec)
sys.modules["ai.endgame_solved_db"] = _esdb_mod
_spec.loader.exec_module(_esdb_mod)

# Load ai.board_symmetry for D4 canonicalization helpers.
_bs_spec = _ilu.spec_from_file_location(
    "ai.board_symmetry", str(_ROOT / "ai" / "board_symmetry.py")
)
_bs_mod = _ilu.module_from_spec(_bs_spec)
sys.modules["ai.board_symmetry"] = _bs_mod
_bs_spec.loader.exec_module(_bs_mod)
_BPERM = _bs_mod._BOARD_PERM  # _BPERM[sym_idx][old_idx] = new_idx

from game.board import ADJACENCY, MILLS, POSITIONS

get_wdl = _esdb_mod.get_wdl
set_wdl = _esdb_mod.set_wdl
WDL_UNKNOWN = _esdb_mod.WDL_UNKNOWN
WDL_WIN = _esdb_mod.WDL_WIN
WDL_LOSS = _esdb_mod.WDL_LOSS
WDL_DRAW = _esdb_mod.WDL_DRAW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Fast combinatorial helpers ─────────────────────────────────────────────────
# Precomputed Pascal's triangle: _CT[n][k] = C(n, k) for 0 ≤ n,k ≤ 24.
# Avoids math.comb() function-call overhead in tight inner loops.
_CT: list[list[int]] = [[0] * 25 for _ in range(25)]
for _n in range(25):
    _CT[_n][0] = 1
    for _k in range(1, 25):
        if _k <= _n:
            _CT[_n][_k] = _CT[_n - 1][_k - 1] + _CT[_n - 1][_k]

# Reusable buffer: _RI[sq] = rank of square sq in the "remaining" list after
# removing white pieces.  Single-threaded use only.
_RI: list[int] = [0] * 24

# ── Board constants ────────────────────────────────────────────────────────────

_POS_TO_IDX: dict[str, int] = {pos: i for i, pos in enumerate(POSITIONS)}
_N = 24

# ── Adjacency as index lists ───────────────────────────────────────────────────

_ADJACENCY_IDX: list[list[int]] = [[] for _ in range(_N)]
for _pos_name, _neighbors in ADJACENCY.items():
    _ADJACENCY_IDX[_POS_TO_IDX[_pos_name]] = [_POS_TO_IDX[nb] for nb in _neighbors]

# ── Mill bitmasks per square ────────────────────────────────────────────────────

_MILL_MASKS_FOR: list[list[int]] = [[] for _ in range(_N)]
for _mill in MILLS:
    _mask = 0
    for _p in _mill:
        _mask |= 1 << _POS_TO_IDX[_p]
    for _p in _mill:
        _MILL_MASKS_FOR[_POS_TO_IDX[_p]].append(_mask)


# ── D4 canonicalization ────────────────────────────────────────────────────────
# Precomputed bitmask permutation pairs for the 7 non-identity D4 transforms.
# Each entry is a list of (old_bit, new_bit) pairs covering all 24 squares.
# Using bitmasks avoids list allocations in the inner canonicalization loop.
_BPERM_MASKS: list[list[tuple[int, int]]] = []
for _sym_idx in range(1, 8):
    _perm = _BPERM[_sym_idx]
    if _perm is None:
        continue
    _BPERM_MASKS.append([(1 << _old, 1 << _perm[_old]) for _old in range(_N)])


def _canonical_indices(w: list[int], b: list[int]) -> tuple[list[int], list[int]]:
    """Return the D4-canonical (w, b) index lists (bitmask-minimum over 8 transforms)."""
    w_mask = 0
    for i in w:
        w_mask |= 1 << i
    b_mask = 0
    for i in b:
        b_mask |= 1 << i
    best_w, best_b = w_mask, b_mask
    for pairs in _BPERM_MASKS:
        tw = tb = 0
        for old_bit, new_bit in pairs:
            if w_mask & old_bit:
                tw |= new_bit
            if b_mask & old_bit:
                tb |= new_bit
        if (tw, tb) < (best_w, best_b):
            best_w, best_b = tw, tb
    w_can = [i for i in range(_N) if (best_w >> i) & 1]
    b_can = [i for i in range(_N) if (best_b >> i) & 1]
    return w_can, b_can


def _is_canonical(w: list[int], b: list[int]) -> bool:
    w_can, b_can = _canonical_indices(w, b)
    return w_can == w and b_can == b


def _closes_mill(piece_mask: int, to_idx: int) -> bool:
    for mm in _MILL_MASKS_FOR[to_idx]:
        if (piece_mask & mm) == mm:
            return True
    return False


def _in_mill(piece_idx: int, piece_mask: int) -> bool:
    for mm in _MILL_MASKS_FOR[piece_idx]:
        if (piece_mask & mm) == mm:
            return True
    return False


# ── General encode / decode ────────────────────────────────────────────────────

def _table_size(nW: int, nB: int) -> int:
    return _CT[_N][nW] * _CT[_N - nW][nB] * 2


def _encode(
    w_sorted: list[int], b_sorted: list[int], turn_bit: int, nC_b: int
) -> int:
    """Pack (W_indices, B_indices, turn_bit) into a table position-ID.

    nC_b = C(_N - nW, nB) must be pre-computed by the caller.
    Uses precomputed _CT and _RI buffer — no per-call heap allocations.
    W_indices are always passed first, regardless of who is to move.
    """
    nW = len(w_sorted)
    nB = len(b_sorted)
    # White rank: Σ C(w[i], i+1)
    wr = 0
    for i in range(nW):
        wr += _CT[w_sorted[i]][i + 1]
    # Fill _RI: for each non-white square, _RI[sq] = its rank in the remaining list.
    k = 0
    wp = 0
    for sq in range(_N):
        if wp < nW and w_sorted[wp] == sq:
            wp += 1
        else:
            _RI[sq] = k
            k += 1
    # Black rank: Σ C(_RI[b[i]], i+1)
    br = 0
    for i in range(nB):
        br += _CT[_RI[b_sorted[i]]][i + 1]
    return wr * nC_b * 2 + br * 2 + turn_bit


def _decode(
    pos_id: int, nW: int, nB: int, nC_b: int
) -> tuple[list[int], list[int], int]:
    """Unpack a position-ID into (w_sorted, b_sorted, turn_bit).

    Inlines combo_unrank using the _CT table for fast comb lookups.
    """
    turn_bit = pos_id & 1
    rem = pos_id >> 1
    br = rem % nC_b
    wr = rem // nC_b
    # Unrank white (inline combo_unrank(wr, nW, 24))
    w = []
    rr = wr
    up = _N - 1
    for i in range(nW, 0, -1):
        c = up
        while c >= i - 1 and _CT[c][i] > rr:
            c -= 1
        rr -= _CT[c][i]
        w.append(c)
        up = c - 1
    w.reverse()
    # Unrank black remapping (inline combo_unrank(br, nB, _N-nW))
    b_rem = []
    rr = br
    up = _N - nW - 1
    for i in range(nB, 0, -1):
        c = up
        while c >= i - 1 and _CT[c][i] > rr:
            c -= 1
        rr -= _CT[c][i]
        b_rem.append(c)
        up = c - 1
    b_rem.reverse()
    # Map remapped B indices back to actual squares
    k = 0
    wp = 0
    remaining = []
    for sq in range(_N):
        if wp < nW and w[wp] == sq:
            wp += 1
        else:
            remaining.append(sq)
    b = sorted(remaining[j] for j in b_rem)
    return w, b, turn_bit


# ── Capture helpers ────────────────────────────────────────────────────────────

def _valid_captures(other_list: list[int]) -> list[int]:
    """Non-mill opponent pieces; fall back to all if every piece is in a mill."""
    other_mask = 0
    for i in other_list:
        other_mask |= 1 << i
    non_mill = [i for i in other_list if not _in_mill(i, other_mask)]
    return non_mill if non_mill else list(other_list)


def _best_capture_wdl_for_mover(
    new_mover: list[int],
    other_list: list[int],
    turn_bit: int,
    nW: int,
    nB: int,
    sub_tables: dict[tuple[int, int], bytes],
) -> int:
    """WDL for the current mover after closing a mill, choosing the best capture.

    Returns WDL_WIN, WDL_DRAW, WDL_LOSS, or WDL_UNKNOWN.
    Cross-table convention:
      turn_bit == 0 (W moves): W captures B piece → sub-table (nW, nB-1), B next.
      turn_bit == 1 (B moves): B captures W piece → sub-table (nW-1, nB), W next.
    W_indices are always the first argument to _encode.
    """
    captures = _valid_captures(other_list)
    best = WDL_LOSS
    has_unknown = False

    for cap_idx in captures:
        new_other = sorted(i for i in other_list if i != cap_idx)
        n_new_other = len(new_other)

        if n_new_other < 3:
            return WDL_WIN  # opponent below 3 → immediate loss for them

        if turn_bit == 0:
            sub_key_nw, sub_key_nb = nW, n_new_other
            sub_nC_b = _CT[_N - sub_key_nw][sub_key_nb]
            w_c, b_c = _canonical_indices(new_mover, new_other)
            sub_key = _encode(w_c, b_c, 1, sub_nC_b)
        else:
            sub_key_nw, sub_key_nb = n_new_other, nB
            sub_nC_b = _CT[_N - sub_key_nw][sub_key_nb]
            w_c, b_c = _canonical_indices(new_other, new_mover)
            sub_key = _encode(w_c, b_c, 0, sub_nC_b)

        sub_tbl = sub_tables.get((sub_key_nw, sub_key_nb))
        if sub_tbl is None:
            has_unknown = True
            continue

        sub_val = get_wdl(sub_tbl, sub_key)
        if sub_val == WDL_LOSS:
            return WDL_WIN
        elif sub_val == WDL_WIN:
            pass
        elif sub_val == WDL_DRAW:
            if best != WDL_WIN:
                best = WDL_DRAW

    if has_unknown and best == WDL_LOSS:
        return WDL_UNKNOWN
    return best


# ── Core solver ────────────────────────────────────────────────────────────────

def _process_pos(
    w: list[int], b: list[int], turn_bit: int,
    table: bytearray | mmap.mmap,
    nW: int, nB: int, nC_b: int,
    sub_tables: dict,
) -> int:
    """Evaluate one position; return WDL_WIN/LOSS/UNKNOWN.

    Bitmask occupancy checks and incremental mask updates avoid Python object
    allocations in the hot path.  new_mover list is only built for mill-closing
    moves (rare) and non-capture moves that need encoding.
    """
    mover = w if turn_bit == 0 else b
    other = b if turn_bit == 0 else w
    n_mover = len(mover)
    fly_mover = n_mover <= 3
    next_bit = 1 - turn_bit

    mover_mask = 0
    for i in mover:
        mover_mask |= 1 << i
    occ_mask = mover_mask
    for i in other:
        occ_mask |= 1 << i

    all_opponent_win = True
    has_any_move = False

    for fi in range(n_mover):
        from_idx = mover[fi]
        from_bit = 1 << from_idx
        mover_no_fi = mover_mask ^ from_bit

        targets = range(_N) if fly_mover else _ADJACENCY_IDX[from_idx]

        for to_idx in targets:
            to_bit = 1 << to_idx
            if occ_mask & to_bit:
                continue
            has_any_move = True

            new_mover_mask = mover_no_fi | to_bit

            if _closes_mill(new_mover_mask, to_idx):
                new_mover = []
                for j in range(n_mover):
                    if j != fi:
                        new_mover.append(mover[j])
                new_mover.append(to_idx)
                new_mover.sort()

                outcome = _best_capture_wdl_for_mover(
                    new_mover, other, turn_bit, nW, nB, sub_tables
                )
                if outcome == WDL_WIN:
                    return WDL_WIN
                elif outcome in (WDL_DRAW, WDL_UNKNOWN):
                    all_opponent_win = False
            else:
                new_mover = []
                for j in range(n_mover):
                    if j != fi:
                        new_mover.append(mover[j])
                new_mover.append(to_idx)
                new_mover.sort()

                if turn_bit == 0:
                    w_c, b_c = _canonical_indices(new_mover, other)
                else:
                    w_c, b_c = _canonical_indices(other, new_mover)
                succ_id = _encode(w_c, b_c, next_bit, nC_b)
                sv = get_wdl(table, succ_id)
                if sv == WDL_LOSS:
                    return WDL_WIN
                elif sv != WDL_WIN:
                    all_opponent_win = False

    if not has_any_move:
        return WDL_LOSS
    if all_opponent_win:
        return WDL_LOSS
    return WDL_UNKNOWN


def solve_table(
    nW: int,
    nB: int,
    sub_tables: dict[tuple[int, int], bytes],
    out_path: Path,
    verbose: bool = True,
) -> None:
    """Solve all (nW, nB) positions and write the WDL file to *out_path*.

    sub_tables must contain fully-solved (nW, nB-1) and (nW-1, nB) tables.
    The 3v3 base case passes sub_tables={} because mill captures there
    always reduce the opponent to 2 pieces (immediate WIN).

    The file is pre-allocated as a sparse file (OS manages paging) so large
    tables (5v4, 6v5, …) never require the full bytes to be in RAM at once.
    """
    ts = _table_size(nW, nB)
    n_bytes = (ts + 3) >> 2
    nC_b = _CT[_N - nW][nB]
    t0 = time.time()

    # Pre-allocate sparse file (all zeros = WDL_UNKNOWN = valid start state).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as _pre:
        _pre.seek(max(n_bytes, 1) - 1)
        _pre.write(b"\x00")
    _fh = open(out_path, "r+b")
    table = mmap.mmap(_fh.fileno(), n_bytes)

    # ── Precompute canonical position IDs (~ts/8) ─────────────────────────────
    canonical_ids: list[int] = []
    for pos_id in range(ts):
        w, b, _tb = _decode(pos_id, nW, nB, nC_b)
        if _is_canonical(w, b):
            canonical_ids.append(pos_id)

    if verbose:
        logger.info(
            "(%d,%d) Canonical positions: %d / %d (%.1f%%)",
            nW, nB, len(canonical_ids), ts, 100.0 * len(canonical_ids) / ts,
        )

    # ── Pass 0: mark terminals (canonical positions only) ─────────────────────
    n_pass0 = 0
    for pos_id in canonical_ids:
        w, b, turn_bit = _decode(pos_id, nW, nB, nC_b)
        v = _process_pos(w, b, turn_bit, table, nW, nB, nC_b, sub_tables)
        if v != WDL_UNKNOWN:
            set_wdl(table, pos_id, v)
            n_pass0 += 1

    if verbose:
        logger.info(
            "(%d,%d) Pass 0: %d resolved (%.1fs)", nW, nB, n_pass0, time.time() - t0
        )

    # ── Iterative forward passes (canonical positions only) ───────────────────
    for pass_num in range(1, 60):
        changed = 0
        tp = time.time()
        for pos_id in canonical_ids:
            if get_wdl(table, pos_id) != WDL_UNKNOWN:
                continue
            w, b, turn_bit = _decode(pos_id, nW, nB, nC_b)
            v = _process_pos(w, b, turn_bit, table, nW, nB, nC_b, sub_tables)
            if v != WDL_UNKNOWN:
                set_wdl(table, pos_id, v)
                changed += 1

        if verbose:
            logger.info(
                "(%d,%d) Pass %d: %d newly resolved (%.1fs)",
                nW, nB, pass_num, changed, time.time() - tp,
            )
        if changed == 0:
            break

    # ── Mark remaining canonical UNKNOWN as DRAW ──────────────────────────────
    n_draw = 0
    for pos_id in canonical_ids:
        if get_wdl(table, pos_id) == WDL_UNKNOWN:
            set_wdl(table, pos_id, WDL_DRAW)
            n_draw += 1

    # ── Fill non-canonical positions from their canonical equivalents ─────────
    try:
        for pos_id in range(ts):
            w, b, turn_bit = _decode(pos_id, nW, nB, nC_b)
            w_can, b_can = _canonical_indices(w, b)
            if w_can == w and b_can == b:
                continue  # canonical: already solved
            can_id = _encode(w_can, b_can, turn_bit, nC_b)
            set_wdl(table, pos_id, get_wdl(table, can_id))

        if verbose:
            n_win = sum(1 for i in range(ts) if get_wdl(table, i) == WDL_WIN)
            n_loss = sum(1 for i in range(ts) if get_wdl(table, i) == WDL_LOSS)
            logger.info(
                "(%d,%d) Solved: %d WIN  %d LOSS  %d DRAW  (total %d, %.1fs)",
                nW, nB, n_win, n_loss, n_draw, ts, time.time() - t0,
            )

        table.flush()
    finally:
        table.close()
        _fh.close()


def solve_3_3(out_dir: Path, verbose: bool = True) -> None:
    """Convenience wrapper: solve the (3,3) base case."""
    solve_table(3, 3, {}, _wdl_path(out_dir, 3, 3), verbose=verbose)


# ── Table file I/O ─────────────────────────────────────────────────────────────

def _wdl_path(out_dir: Path, nW: int, nB: int) -> Path:
    return out_dir / f"endgame_{nW}_{nB}.wdl"


def _load_table(out_dir: Path, nW: int, nB: int) -> bytes | None:
    p = _wdl_path(out_dir, nW, nB)
    if not p.exists():
        return None
    expected_bytes = (_table_size(nW, nB) + 3) >> 2
    data = p.read_bytes()
    if len(data) != expected_bytes:
        logger.warning(
            "(%d,%d) Size mismatch: %s has %d bytes (expected %d) — skipping.",
            nW, nB, p, len(data), expected_bytes,
        )
        return None
    return data


# ── Build schedule ─────────────────────────────────────────────────────────────

_ALL_TABLES: list[tuple[int, int]] = [
    (nW, nB)
    for s in range(6, 12)
    for nW in range(3, s)
    for nB in [s - nW]
    if nB >= 3
]


def _sub_tables_needed(nW: int, nB: int) -> list[tuple[int, int]]:
    deps = []
    if nB - 1 >= 3:
        deps.append((nW, nB - 1))
    if nW - 1 >= 3:
        deps.append((nW - 1, nB))
    return deps


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build NMM retrograde endgame WDL tables."
    )
    ap.add_argument(
        "--out-dir", default="data/endgame",
        help="Directory to write endgame_*.wdl files (default: data/endgame)",
    )
    ap.add_argument("--nW", type=int, help="White piece count for a single table build")
    ap.add_argument("--nB", type=int, help="Black piece count for a single table build")
    ap.add_argument(
        "--build-all", action="store_true",
        help="Build all tables in dependency order",
    )
    ap.add_argument(
        "--max-sum", type=int, default=11,
        help="Maximum nW+nB to build when using --build-all (default: 11)",
    )
    ap.add_argument(
        "--skip-existing", action="store_true",
        help="Skip tables whose .wdl file already exists on disk",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress per-pass logging")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    if args.build_all:
        schedule = [
            (nW, nB) for (nW, nB) in _ALL_TABLES if nW + nB <= args.max_sum
        ]
    elif args.nW is not None and args.nB is not None:
        if args.nW < 3 or args.nB < 3:
            ap.error("--nW and --nB must each be ≥ 3")
        schedule = [(args.nW, args.nB)]
    else:
        ap.error("Specify --build-all or both --nW and --nB")

    loaded: dict[tuple[int, int], bytes] = {}

    for nW, nB in schedule:
        wdl_path = _wdl_path(out_dir, nW, nB)
        if args.skip_existing and wdl_path.exists():
            logger.info("(%d,%d) Already exists — skipping.", nW, nB)
            data = _load_table(out_dir, nW, nB)
            if data is not None:
                loaded[(nW, nB)] = data
            continue

        sub_tables: dict[tuple[int, int], bytes] = {}
        for dep_nw, dep_nb in _sub_tables_needed(nW, nB):
            if (dep_nw, dep_nb) in loaded:
                sub_tables[(dep_nw, dep_nb)] = loaded[(dep_nw, dep_nb)]
            else:
                data = _load_table(out_dir, dep_nw, dep_nb)
                if data is not None:
                    sub_tables[(dep_nw, dep_nb)] = data
                    loaded[(dep_nw, dep_nb)] = data
                else:
                    logger.warning(
                        "(%d,%d) Sub-table (%d,%d) not found — capture outcomes "
                        "into that table will be treated as UNKNOWN.",
                        nW, nB, dep_nw, dep_nb,
                    )

        logger.info(
            "Building (%d,%d): %d positions, %.1f MB table",
            nW, nB,
            _table_size(nW, nB),
            (_table_size(nW, nB) + 3) / 4 / 1024 / 1024,
        )
        solve_table(nW, nB, sub_tables, wdl_path, verbose=verbose)
        logger.info("Wrote %s (%d bytes)", wdl_path, wdl_path.stat().st_size)
        data = _load_table(out_dir, nW, nB)
        if data is not None:
            loaded[(nW, nB)] = data

        remaining_schedule = set(schedule[schedule.index((nW, nB)) + 1:])
        still_needed = set()
        for rnW, rnB in remaining_schedule:
            for dep in _sub_tables_needed(rnW, rnB):
                still_needed.add(dep)
        for key in list(loaded.keys()):
            if key not in remaining_schedule and key not in still_needed:
                del loaded[key]


if __name__ == "__main__":
    main()
