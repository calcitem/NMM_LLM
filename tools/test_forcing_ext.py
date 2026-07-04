"""tools/test_forcing_ext.py — Verify Phase-2 forcing qsearch extension fires.

Plays N games (AI vs AI, difficulty 3) and counts how many times the Phase-2
qsearch extension (forced-block / reachable two-config) fires across all moves.
Reports per-game and total statistics.

Usage:
    .venv/bin/python tools/test_forcing_ext.py [--games 100] [--diff 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nmm_core as _rc
from game.game_engine import GameEngine
from ai.game_ai import GameAI


MAX_MOVES = 300


def make_ai(color: str, diff: int) -> GameAI:
    ai = GameAI(color=color, difficulty=diff)
    ai.use_v2_heuristics = True
    return ai


def play_game(white_ai: GameAI, black_ai: GameAI) -> str | None:
    engine = GameEngine(human_color="W")
    move_count = 0
    while not engine.finished and move_count < MAX_MOVES:
        board = engine.board
        ai = white_ai if board.turn == "W" else black_ai
        move = ai.choose_move(board)
        if not move:
            break
        engine.apply_move(move)
        move_count += 1
    return engine.winner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--diff",  type=int, default=3)
    args = ap.parse_args()

    n_games = args.games
    diff    = args.diff

    print(f"Phase-2 forcing extension fire test — {n_games} games, difficulty {diff}")
    print("-" * 60)

    _rc.py_reset_forcing_ext_count()

    games_with_ext = 0
    total_ext      = 0
    results        = []

    for game_idx in range(n_games):
        v2_is_white = (game_idx % 2 == 0)
        white_ai = make_ai("W", diff)
        black_ai = make_ai("B", diff)

        before = _rc.py_get_forcing_ext_count()
        winner = play_game(white_ai, black_ai)
        after  = _rc.py_get_forcing_ext_count()

        game_ext = after - before
        total_ext += game_ext
        if game_ext > 0:
            games_with_ext += 1

        results.append(game_ext)
        print(
            f"Game {game_idx+1:>3}/{n_games}  "
            f"winner={winner or 'draw':<4}  "
            f"forcing_ext={game_ext:>6}  "
            f"total={total_ext:>8}",
            flush=True,
        )

    print()
    print("=" * 60)
    print(f"RESULT  {n_games} games at difficulty {diff}")
    print(f"  Games with ≥1 forcing extension : {games_with_ext}/{n_games} "
          f"({100*games_with_ext/n_games:.0f}%)")
    print(f"  Total forcing extensions fired  : {total_ext:,}")
    print(f"  Average per game                : {total_ext/n_games:.1f}")
    if results:
        print(f"  Min / Max per game              : {min(results)} / {max(results)}")
    print("=" * 60)

    if total_ext == 0:
        print("\nFAIL — extension never fired. Check qsearch wiring.")
        sys.exit(1)
    else:
        print("\nPASS — Phase-2 forcing extension is live and firing.")


if __name__ == "__main__":
    main()
