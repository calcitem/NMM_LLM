#!/usr/bin/env python3
"""Measure actual search depth reached by iterative deepening at various budgets.

Reports: depth completed, nodes searched, wall time, and the chosen move with
its raw leaf eval — for placement, movement, and fly positions.

Usage:
    .venv/bin/python tools/diagnose_depth.py [--diff 1-10] [--phase all|place|move|fly]
"""
from __future__ import annotations

import sys
import time
import math
import pathlib
import argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves
from ai.game_ai import GameAI, _SearchAbort, _order_moves
import ai.game_ai as _gai_mod
from ai.heuristics import evaluate_v2, evaluate, INF


# ── Representative positions ────────────────────────────────────────────────

def _place_position() -> BoardState:
    """Midway through placement — 5 pieces each placed."""
    positions = {
        "d7": "W", "a4": "W", "g4": "W", "d6": "W", "f6": "W",
        "a7": "B", "d1": "B", "g1": "B", "b4": "B", "f4": "B",
    }
    # from_setup with phase="place" sets pieces_placed = on-board count (5 each)
    return BoardState.from_setup(positions, turn="W", phase="place")


def _move_position() -> BoardState:
    """Typical midgame movement position — both sides fully placed."""
    positions = {
        "a7": "W", "d7": "W", "g7": "W",
        "a4": "W", "b4": "W",
        "d6": "W", "f6": "W", "g4": "W", "e5": "W",
        "a1": "B", "d1": "B", "g1": "B",
        "c3": "B", "d3": "B",
        "b6": "B", "f2": "B", "d2": "B", "c5": "B",
    }
    return BoardState.from_setup(positions, turn="W", phase="move")


def _fly_position() -> BoardState:
    """Fly-phase position: W has 3 pieces, B has 4."""
    positions = {
        "d7": "W", "d5": "W", "d3": "W",
        "a7": "B", "g7": "B", "a1": "B", "g1": "B",
    }
    return BoardState.from_setup(positions, turn="W", phase="fly")


POSITIONS = {
    "place": ("Placement (5+5 placed)", _place_position),
    "move":  ("Movement (9+9, complex)", _move_position),
    "fly":   ("Fly phase (W=3, B=4)", _fly_position),
}


# ── Instrumented iterative deepening ────────────────────────────────────────

def _run_timed(ai: GameAI, board: BoardState, time_budget: float, max_depth: int):
    """Run one choose_move() call and return (move, depth_completed, nodes, elapsed)."""
    import random
    from ai.heuristics import clear_eval_cache

    ai._force_stop = False
    ai._deadline = math.inf
    ai.last_thinking = ""
    ai._tt.clear()
    ai._killers = [[None, None] for _ in range(32)]
    ai._history = {}
    ai._trajectory_db = None
    ai._game_notations = []
    ai._trajectory_line = []
    ai._db_active_this_move = True   # always probe DB in this diagnostic
    ai._nodes = 0
    ai.suppress_fork_variety = False

    moves = get_all_legal_moves(board)
    if not moves:
        return {}, 0, 0, 0.0

    _ASP_MARGIN = 175
    ai._deadline = time.time() + time_budget
    clear_eval_cache()

    best_move = moves[0]
    prev_score = None
    depth_done = 1
    t0 = time.time()

    for depth in range(2, max_depth + 1):
        if time.time() >= ai._deadline:
            break
        try:
            move, score = ai._root_search(board, depth, top_n=1, moves=moves)
            best_move = move
            prev_score = score
            depth_done = depth
        except _SearchAbort:
            break

    elapsed = time.time() - t0
    return best_move, depth_done, ai._nodes, elapsed


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["all", "place", "move", "fly"])
    parser.add_argument("--diff", type=int, default=None,
                        help="Single difficulty level to test (default: test several)")
    parser.add_argument("--budgets", default=None,
                        help="Comma-separated time budgets in seconds (overrides difficulty)")
    args = parser.parse_args()

    phases = list(POSITIONS.items()) if args.phase == "all" else [(args.phase, POSITIONS[args.phase])]

    if args.budgets:
        budgets = [float(x) for x in args.budgets.split(",")]
        diff_list = [None] * len(budgets)
    elif args.diff is not None:
        from ai.game_ai import _TIME_LIMIT
        budgets = [_TIME_LIMIT.get(args.diff, 10.0)]
        diff_list = [args.diff]
    else:
        from ai.game_ai import _TIME_LIMIT
        diff_list = [3, 5, 6, 7, 8]
        budgets = [_TIME_LIMIT[d] for d in diff_list]

    for phase_key, (phase_label, make_board) in phases:
        print(f"\n{'='*66}")
        print(f"  {phase_label}")
        print(f"{'='*66}")
        print(f"  {'Diff':>5}  {'Budget':>8}  {'Depth':>6}  {'Nodes':>10}  {'Time':>7}  Move / eval")
        print(f"  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*20}")

        board = make_board()

        for budget, diff in zip(budgets, diff_list):
            ai = GameAI(color="W", difficulty=diff if diff else 5)
            # use_v2_heuristics already True by default
            ai.max_search_depth = 22

            move, depth, nodes, elapsed = _run_timed(ai, board, budget, max_depth=22)

            # Leaf eval of the chosen move's resulting position
            if move:
                after = board.apply_move(move)
                ev2 = evaluate_v2(after, "B", weights=ai._weights)   # from opp's POV
                ev1 = evaluate(after, "B")
                notation = f"{move.get('from','?')}→{move['to']}"
                cap = f" x{move['capture']}" if move.get('capture') else ""
                eval_str = f"v2={-ev2:+6d} v1={-ev1:+6d}"
            else:
                notation = "(no move)"
                eval_str = ""

            diff_label = f"L{diff}" if diff else "?"
            print(f"  {diff_label:>5}  {budget:>7.1f}s  {depth:>6}  {nodes:>10,}  {elapsed:>6.2f}s  {notation}{cap}  {eval_str}")

    print()


if __name__ == "__main__":
    main()
