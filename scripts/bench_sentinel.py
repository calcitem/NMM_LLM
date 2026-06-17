"""scripts/bench_sentinel.py — headless AI vs AI benchmark.

Runs N games between two GameAI configurations and reports win/draw/loss rates.
Use this to measure whether sentinel and/or value_net improve the heuristic engine.

Usage examples
--------------
# Baseline vs Baseline (sanity check — should be near 50/50)
python scripts/bench_sentinel.py --games 200 --difficulty 4

# Sentinel (score_adjust) vs Baseline
python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-sentinel score_adjust

# Sentinel + value_net vs Baseline
python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-sentinel score_adjust --white-value-net

# Sentinel vs Sentinel (should be ~50/50)
python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-sentinel score_adjust --black-sentinel score_adjust

Colours alternate first-mover across games to cancel first-move advantage.
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import BoardState
from ai.game_ai import GameAI

_SENTINEL_CKPT = "learned_ai/sentinel/checkpoints/best.pt"
_VALUE_NET_PATH = "data/value_net.npz"


def _load_sentinel():
    try:
        from learned_ai.sentinel.infer import load_advisor
        advisor = load_advisor(_SENTINEL_CKPT)
        if advisor:
            print(f"  Sentinel loaded: {_SENTINEL_CKPT}")
        else:
            print("  Sentinel load returned None.")
        return advisor
    except Exception as e:
        print(f"  Sentinel load failed: {e}")
        return None


def _load_value_net():
    try:
        from ai.value_net import ValueNet
        vn = ValueNet.load(_VALUE_NET_PATH)
        print(f"  Value net loaded: {_VALUE_NET_PATH}")
        return vn
    except Exception as e:
        print(f"  Value net load failed: {e}")
        return None


def _make_ai(color: str, difficulty: int, sentinel=None, sentinel_mode: str = "advisory",
             value_net=None, time_budget: float = 0.25, vn_blend: int = 0) -> GameAI:
    # override_time_budget also bypasses the 2s early-game floor so benchmark runs fast.
    from ai.heuristics import HeuristicWeights
    weights = HeuristicWeights(value_net_blend=vn_blend) if vn_blend else None
    ai = GameAI(color=color, difficulty=difficulty, value_net=value_net,
                weights=weights, override_time_budget=time_budget)
    if sentinel is not None:
        ai.set_sentinel(sentinel, mode=sentinel_mode)
    return ai


def play_game(white_ai: GameAI, black_ai: GameAI, max_plies: int = 400) -> Optional[str]:
    """Return 'W', 'B', or None (draw/stalemate)."""
    from game.game_engine import GameEngine
    from game.rules import is_terminal

    engine = GameEngine(human_color=None)  # no human — both sides AI
    for _ in range(max_plies):
        if engine.winner is not None:
            return engine.winner
        terminal, winner = is_terminal(engine.board)
        if terminal:
            return winner
        color = engine.board.turn
        ai = white_ai if color == "W" else black_ai
        try:
            move = ai.choose_move(engine.board)
        except Exception:
            return None
        if move is None:
            return "B" if color == "W" else "W"
        try:
            engine.apply_move(move)
        except Exception:
            return None

    return None  # draw by length


def main() -> int:
    p = argparse.ArgumentParser(description="Headless sentinel/value-net benchmark")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--difficulty", type=int, default=4)
    p.add_argument("--white-sentinel", default=None,
                   choices=["advisory", "score_adjust", "reconsider"],
                   help="Sentinel mode for White; omit for pure heuristic")
    p.add_argument("--black-sentinel", default=None,
                   choices=["advisory", "score_adjust", "reconsider"])
    p.add_argument("--white-value-net", action="store_true")
    p.add_argument("--black-value-net", action="store_true")
    p.add_argument("--vn-blend", type=int, default=0,
                   help="value_net_blend %% (0=off; e.g. 80 blends 80%% VN into leaf eval)")
    p.add_argument("--time-budget", type=float, default=0.25,
                   help="Seconds per move (overrides difficulty time; default 0.25s)")
    args = p.parse_args()

    need_sentinel = args.white_sentinel or args.black_sentinel
    need_vn = args.white_value_net or args.black_value_net

    print("Loading components...")
    sentinel = _load_sentinel() if need_sentinel else None
    value_net = _load_value_net() if need_vn else None
    print()

    white_label = f"White[d{args.difficulty}"
    black_label = f"Black[d{args.difficulty}"
    if args.white_sentinel:
        white_label += f"+sentinel:{args.white_sentinel}"
    if args.white_value_net:
        white_label += f"+vn{args.vn_blend}%"
    white_label += "]"
    if args.black_sentinel:
        black_label += f"+sentinel:{args.black_sentinel}"
    if args.black_value_net:
        black_label += f"+vn{args.vn_blend}%"
    black_label += "]"

    print(f"Match: {white_label} vs {black_label}")
    print(f"Games:  {args.games}  (colours alternate first mover)\n")

    # Track by CONFIG (A=enhanced, B=baseline), not by colour.
    # Even: A plays White, B plays Black.  Odd: A plays Black, B plays White.
    results = {"A": 0, "B": 0, "draw": 0}
    t0 = time.time()

    for g in range(args.games):
        if g % 2 == 0:
            w_sentinel = sentinel if args.white_sentinel else None
            w_mode = args.white_sentinel or "advisory"
            b_sentinel = sentinel if args.black_sentinel else None
            b_mode = args.black_sentinel or "advisory"
            w_vn = value_net if args.white_value_net else None
            b_vn = value_net if args.black_value_net else None
            a_color = "W"
        else:
            w_sentinel = sentinel if args.black_sentinel else None
            w_mode = args.black_sentinel or "advisory"
            b_sentinel = sentinel if args.white_sentinel else None
            b_mode = args.white_sentinel or "advisory"
            w_vn = value_net if args.black_value_net else None
            b_vn = value_net if args.white_value_net else None
            a_color = "B"

        w_blend = args.vn_blend if args.white_value_net else 0
        b_blend = args.vn_blend if args.black_value_net else 0
        if g % 2 != 0:  # colors swapped
            w_blend, b_blend = (args.vn_blend if args.black_value_net else 0), (args.vn_blend if args.white_value_net else 0)
        white_ai = _make_ai("W", args.difficulty, w_sentinel, w_mode, w_vn, args.time_budget, w_blend)
        black_ai = _make_ai("B", args.difficulty, b_sentinel, b_mode, b_vn, args.time_budget, b_blend)

        winner = play_game(white_ai, black_ai)
        if winner is None:
            results["draw"] += 1
        elif winner == a_color:
            results["A"] += 1
        else:
            results["B"] += 1

        elapsed = time.time() - t0
        done = g + 1
        rate = done / elapsed
        eta = (args.games - done) / rate if rate > 0 else 0
        print(f"\r  {done}/{args.games}  A:{results['A']}  B:{results['B']}  "
              f"D:{results['draw']}  {rate:.1f} g/s  ETA {eta:.0f}s    ", end="", flush=True)

    print()
    total = args.games
    a_rate = 100 * results["A"] / total
    b_rate = 100 * results["B"] / total
    d_rate = 100 * results["draw"] / total
    edge = a_rate - b_rate
    print(f"\n{'='*62}")
    print(f"  RESULTS after {total} games  ({time.time()-t0:.0f}s)")
    print(f"{'='*62}")
    print(f"  Config A: {white_label}")
    print(f"  Config B: {black_label}")
    print(f"  (each config plays White in half the games, Black in the other half)")
    print(f"{'='*62}")
    print(f"  A wins : {results['A']:4d}  ({a_rate:.1f}%)")
    print(f"  B wins : {results['B']:4d}  ({b_rate:.1f}%)")
    print(f"  Draws  : {results['draw']:4d}  ({d_rate:.1f}%)")
    print(f"  A edge : {edge:+.1f}pp  ({'A better' if edge > 2 else 'B better' if edge < -2 else 'roughly equal'})")
    print(f"{'='*62}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
