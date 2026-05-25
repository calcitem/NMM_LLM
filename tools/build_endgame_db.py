#!/usr/bin/env python3
"""tools/build_endgame_db.py — Retrograde endgame solver for NMM (3,3) positions.

Usage
-----
    python tools/build_endgame_db.py [--out-dir DIR] [--verbose]

Builds ``endgame_3_3.wdl`` in the output directory (default: ``data/endgame/``).
The file is consumed at query time by ``ai/endgame_solved_db.EndgameSolvedDB``.

Algorithm
---------
All C(24,3)×C(21,3)×2 = 5,383,840 (3,3) fly-phase positions are enumerated via
the combinatorial index.  Positions where any fly move closes a mill are
immediately marked WIN (the capturing player reduces the opponent to 2 pieces,
which is an instant loss under NMM rules).  A fixed-point forward pass then
propagates:

  WIN  — has at least one move leading to a terminal-WIN or a LOSS-labelled
          successor (opponent is to move and is in a losing position).
  LOSS — every move leads to a WIN-labelled successor for the opponent.

Positions remaining UNKNOWN after convergence are in drawn cycles → DRAW.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import logging
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

from game.board import MILLS, POSITIONS

combo_unrank = _esdb_mod.combo_unrank
TABLE_SIZE_3_3 = _esdb_mod.TABLE_SIZE_3_3
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

# ── Board constants ────────────────────────────────────────────────────────────

_POS_TO_IDX: dict[str, int] = {pos: i for i, pos in enumerate(POSITIONS)}
_N = 24
_ALL_MASK = (1 << _N) - 1

# ── Mill bitmasks ──────────────────────────────────────────────────────────────
# For each square i, list of mill bitmasks that include i.
_MILL_MASKS_FOR: list[list[int]] = [[] for _ in range(_N)]
for _mill in MILLS:
    _mask = 0
    for _p in _mill:
        _mask |= 1 << _POS_TO_IDX[_p]
    for _p in _mill:
        _MILL_MASKS_FOR[_POS_TO_IDX[_p]].append(_mask)


def _closes_mill(piece_mask: int, to_idx: int) -> bool:
    """True if adding piece at to_idx closes a mill given current piece_mask."""
    for mm in _MILL_MASKS_FOR[to_idx]:
        if (piece_mask & mm) == mm:
            return True
    return False


# ── Fast position ID encoding (index-level, avoids POSITIONS name lookups) ────

_C21_3 = comb(21, 3)   # 1330
_C21_3_x2 = _C21_3 * 2  # 2660


def _encode(w_sorted: list[int], b_sorted: list[int], turn_bit: int) -> int:
    """Fast encode_position_id operating on sorted index lists + turn bit."""
    # white_rank
    wr = comb(w_sorted[0], 1) + comb(w_sorted[1], 2) + comb(w_sorted[2], 3)
    # remaining squares (not occupied by white), sorted
    w_set = set(w_sorted)
    remaining = [i for i in range(_N) if i not in w_set]
    # remap black indices through remaining
    b0 = remaining.index(b_sorted[0])
    b1 = remaining.index(b_sorted[1])
    b2 = remaining.index(b_sorted[2])
    br = comb(b0, 1) + comb(b1, 2) + comb(b2, 3)
    return wr * _C21_3_x2 + br * 2 + turn_bit


def _decode(pos_id: int) -> tuple[list[int], list[int], int]:
    """Return (w_sorted, b_sorted, turn_bit) from pos_id."""
    turn_bit = pos_id & 1
    rem = pos_id >> 1
    br = rem % _C21_3
    wr = rem // _C21_3
    w = combo_unrank(wr, 3, _N)
    w_set = set(w)
    remaining = [i for i in range(_N) if i not in w_set]
    b_remapped = combo_unrank(br, 3, 21)
    b = sorted(remaining[j] for j in b_remapped)
    return w, b, turn_bit


# ── Fly-move successor generator ───────────────────────────────────────────────

def _fly_successors(
    mover: list[int], other: list[int], next_turn_bit: int
) -> list[tuple[bool, int]]:
    """Return (is_terminal_win, succ_id) for each legal fly move.

    is_terminal_win=True means the move closes a mill (opponent reduced to 2 pieces).
    succ_id is only valid when is_terminal_win=False.
    """
    occupied_mask = 0
    for i in mover + other:
        occupied_mask |= 1 << i
    empty_mask = _ALL_MASK & ~occupied_mask
    results = []
    for fi, from_idx in enumerate(mover):
        new_mover_base = [mover[j] for j in range(3) if j != fi]
        from_bit = 1 << from_idx
        # After picking up the piece, from_idx becomes empty too.
        avail = empty_mask | from_bit
        bits = avail
        while bits:
            lb = bits & (-bits)
            bits ^= lb
            to_idx = lb.bit_length() - 1
            if to_idx == from_idx:
                continue
            new_mover = sorted(new_mover_base + [to_idx])
            new_mask = (occupied_mask & ~from_bit) | lb
            mover_mask = 0
            for i in new_mover:
                mover_mask |= 1 << i
            if _closes_mill(mover_mask, to_idx):
                results.append((True, -1))
            else:
                if next_turn_bit == 1:  # next is B, so current was W → new_mover=W, other=B
                    succ_id = _encode(new_mover, other, next_turn_bit)
                else:  # next is W, so current was B → new_mover=B, other=W
                    succ_id = _encode(other, new_mover, next_turn_bit)
                results.append((False, succ_id))
    return results


# ── Solver ─────────────────────────────────────────────────────────────────────

def solve_3_3(out_dir: Path, verbose: bool = True) -> bytearray:
    """Solve all (3,3) positions.  Returns the WDL bytearray."""
    n_bytes = (TABLE_SIZE_3_3 + 3) >> 2
    table = bytearray(n_bytes)
    t0 = time.time()

    # Pass 0 — mark positions where any fly move closes a mill as WIN.
    n_win0 = 0
    for pos_id in range(TABLE_SIZE_3_3):
        w, b, turn_bit = _decode(pos_id)
        mover = w if turn_bit == 0 else b
        other = b if turn_bit == 0 else w
        mover_mask = sum(1 << i for i in mover)
        occupied_mask = mover_mask | sum(1 << i for i in other)
        empty_mask = _ALL_MASK & ~occupied_mask
        found = False
        for fi, from_idx in enumerate(mover):
            new_base = [mover[j] for j in range(3) if j != fi]
            from_bit = 1 << from_idx
            avail = empty_mask | from_bit
            bits = avail
            while bits and not found:
                lb = bits & (-bits)
                bits ^= lb
                to_idx = lb.bit_length() - 1
                if to_idx == from_idx:
                    continue
                new_mask = (mover_mask & ~from_bit) | lb
                if _closes_mill(new_mask, to_idx):
                    found = True
            if found:
                break
        if found:
            set_wdl(table, pos_id, WDL_WIN)
            n_win0 += 1
    if verbose:
        logger.info("Pass 0: %d immediate WINs (%.1fs)", n_win0, time.time() - t0)

    # Iterative forward passes.
    for pass_num in range(1, 30):
        changed = 0
        tp = time.time()
        for pos_id in range(TABLE_SIZE_3_3):
            if get_wdl(table, pos_id) != WDL_UNKNOWN:
                continue
            w, b, turn_bit = _decode(pos_id)
            mover = w if turn_bit == 0 else b
            other = b if turn_bit == 0 else w
            next_bit = 1 - turn_bit
            succs = _fly_successors(mover, other, next_bit)

            has_winning_move = False
            all_opponent_wins = True
            for is_terminal, succ_id in succs:
                if is_terminal:
                    has_winning_move = True
                    all_opponent_wins = False
                else:
                    sv = get_wdl(table, succ_id)
                    if sv == WDL_LOSS:
                        has_winning_move = True
                        all_opponent_wins = False
                    elif sv == WDL_WIN:
                        pass  # opponent wins from succ
                    else:
                        all_opponent_wins = False

            if has_winning_move:
                set_wdl(table, pos_id, WDL_WIN)
                changed += 1
            elif all_opponent_wins:
                set_wdl(table, pos_id, WDL_LOSS)
                changed += 1

        if verbose:
            logger.info("Pass %d: %d newly resolved (%.1fs)", pass_num, changed, time.time() - tp)
        if changed == 0:
            break

    # Mark remaining UNKNOWN as DRAW.
    n_draw = 0
    for pos_id in range(TABLE_SIZE_3_3):
        if get_wdl(table, pos_id) == WDL_UNKNOWN:
            set_wdl(table, pos_id, WDL_DRAW)
            n_draw += 1

    n_win = sum(1 for i in range(TABLE_SIZE_3_3) if get_wdl(table, i) == WDL_WIN)
    n_loss = sum(1 for i in range(TABLE_SIZE_3_3) if get_wdl(table, i) == WDL_LOSS)
    if verbose:
        logger.info(
            "Solved: %d WIN  %d LOSS  %d DRAW  (total %d, %.1fs)",
            n_win, n_loss, n_draw, TABLE_SIZE_3_3, time.time() - t0,
        )
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="Build NMM retrograde endgame WDL table.")
    ap.add_argument("--out-dir", default="data/endgame",
                    help="Directory to write endgame_3_3.wdl (default: data/endgame)")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wdl_path = out_dir / "endgame_3_3.wdl"

    table = solve_3_3(out_dir, verbose=not args.quiet)
    wdl_path.write_bytes(bytes(table))
    logger.info("Wrote %s (%d bytes)", wdl_path, len(table))


if __name__ == "__main__":
    main()
