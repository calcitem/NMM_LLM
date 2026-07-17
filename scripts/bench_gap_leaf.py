"""scripts/bench_gap_leaf.py — gap-net leaf-side ablation.

Runs a 4-way head-to-head to answer: where should the gap-net blunder-zone
correction be applied at α-β leaves?

  A = ai_side   — current default: bonus only when it's the AI's turn
                  (the historical V3a wiring)
  B = opp_side  — "set traps": bonus when it's the opponent's turn
                  (the Sanmill developer's hypothesis)
  C = both      — additive on both sides
  D = off       — no gap correction (control)

Opponent for every config is a **value-net-blend heuristic** (VN blend 20%),
which is more human-like because the VN was trained on human games — this makes
the ablation informative for actual gameplay against humans.

Reports per config:
  * W / D / L
  * Score% = (W + 0.5·D) / n
  * P(true Score% > 50%) — Sanmill's superiority probability

Streams results to `data/bench/gap_leaf_<timestamp>.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import is_terminal, get_all_legal_moves
from learned_ai.training.advance_stats import superiority_probability, score_proportion

MODES = ["ai_side", "opp_side", "both", "off"]
MODE_LABEL = {
    "ai_side":  "gap: AI-side leaves only (current default)",
    "opp_side": "gap: opponent-side leaves only (set-traps)",
    "both":     "gap: both sides (additive)",
    "off":      "gap: OFF (control)",
}


def _load_components(sentinel_path: str, value_net_path: str, gap_net_path: str, malom_path: str):
    from ai.value_net import ValueNet
    from ai.gap_net import GapNet
    from learned_ai.sentinel.infer import load_advisor
    from learned_ai.sentinel.db_teacher import ExternalSolvedDB

    sentinel = None
    try:
        sentinel = load_advisor(sentinel_path)
        if sentinel is not None and not sentinel.is_loaded():
            sentinel = None
    except Exception as e:
        print(f"  sentinel load failed: {e}")

    value_net = None
    try:
        value_net = ValueNet.load(value_net_path)
    except Exception as e:
        print(f"  value net load failed: {e}")

    gap_net = None
    try:
        gap_net = GapNet.load(gap_net_path)
    except Exception as e:
        print(f"  gap net load failed: {e}")

    malom_db = None
    try:
        malom_db = ExternalSolvedDB(malom_path)
        if not malom_db.is_available():
            malom_db = None
    except Exception as e:
        print(f"  malom load failed: {e}")

    print(f"  sentinel={'OK' if sentinel is not None else 'missing'}, "
          f"value_net={'OK' if value_net is not None else 'missing'}, "
          f"gap_net={'OK' if gap_net is not None else 'missing'}, "
          f"malom={'OK' if malom_db is not None else 'missing'}")
    return sentinel, value_net, gap_net, malom_db


def _make_candidate_ai(color: str, difficulty: int, time_budget: float, mode: str,
                       sentinel, value_net, gap_net, malom_db):
    """Candidate: heuristic + VN + gap (mode-controlled leaf side) + sentinel."""
    from ai.game_ai import GameAI
    from ai.heuristics import HeuristicWeights

    weights = HeuristicWeights(value_net_blend=20)   # VN blend 20% — more human-like
    ai = GameAI(
        color=color,
        difficulty=difficulty,
        value_net=value_net,
        gap_net=gap_net,
        weights=weights,
        malom_db=malom_db,
        override_time_budget=time_budget,
    )
    if sentinel is not None:
        ai.set_sentinel(sentinel, mode="score_adjust")
        ai._sentinel_activation_prob = 0.20
    ai.gap_net_leaf_mode = mode
    return ai


def _make_opponent_ai(color: str, difficulty: int, time_budget: float,
                      sentinel, value_net, malom_db):
    """Opponent: value-net-blend heuristic (more human) — no gap net, no sentinel.

    Sentinel is deliberately excluded because it suppresses exactly the blunders
    the gap-net leaf correction is meant to exploit.  Including it would
    contaminate the ablation signal: opp_side and both modes would both look
    weaker not because the hypothesis is wrong but because the opponent no
    longer makes blunders to steer into.
    """
    from ai.game_ai import GameAI
    from ai.heuristics import HeuristicWeights

    weights = HeuristicWeights(value_net_blend=20)
    ai = GameAI(
        color=color,
        difficulty=difficulty,
        value_net=value_net,
        gap_net=None,
        weights=weights,
        malom_db=malom_db,
        override_time_budget=time_budget,
    )
    # NB: intentionally NOT calling ai.set_sentinel(...).
    return ai


def _play_one(candidate_ai, opponent_ai, candidate_color: str, max_plies: int = 400) -> Optional[str]:
    board = BoardState.new_game()
    for _ in range(max_plies):
        term, winner = is_terminal(board)
        if term:
            return winner
        legal = get_all_legal_moves(board)
        if not legal:
            return "B" if board.turn == "W" else "W"
        if board.turn == candidate_color:
            try:
                move = candidate_ai.choose_move(board)
            except Exception:
                return "B" if board.turn == "W" else "W"
        else:
            try:
                move = opponent_ai.choose_move(board)
            except Exception:
                return candidate_color
        if not move:
            return "B" if board.turn == "W" else "W"
        board = board.apply_move(move)
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Gap-net leaf-side ablation vs VN-blend heuristic.")
    p.add_argument("--games",         type=int,   default=40,
                   help="Games per mode (default 40; alternates colours).")
    p.add_argument("--difficulty",    type=int,   default=5)
    p.add_argument("--time-budget",   type=float, default=1.0,
                   help="Per-move budget seconds (default 1.0). Long enough for meaningful AB.")
    p.add_argument("--max-plies",     type=int,   default=400)
    p.add_argument("--sentinel-path", default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"))
    p.add_argument("--value-net-path", default=str(_ROOT / "data" / "value_net.npz"))
    p.add_argument("--gap-net-path",  default=str(_ROOT / "data" / "gap_net.npz"))
    p.add_argument("--malom-path",    default="/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted")
    p.add_argument("--modes",         default=",".join(MODES),
                   help="Comma-separated subset of modes to test.")
    p.add_argument("--out-dir",       default=str(_ROOT / "data" / "bench"))
    p.add_argument("--quiet",         action="store_true")
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODES]
    if not modes:
        print("ERROR: no valid modes", file=sys.stderr)
        return 2

    print("Loading components…")
    sentinel, value_net, gap_net, malom_db = _load_components(
        args.sentinel_path, args.value_net_path, args.gap_net_path, args.malom_path,
    )
    if gap_net is None:
        print("ERROR: gap_net required for ablation.", file=sys.stderr)
        return 1
    print()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"gap_leaf_{ts}.jsonl"
    print(f"Streaming per-game rows to {out_path}")
    print()

    print(f"Config: diff={args.difficulty}, time_budget={args.time_budget}s, games={args.games} per mode")
    print(f"Opponent: heuristic + VN blend 20% (no sentinel, no gap net) — keeps blunder rate intact")
    print()

    hdr = f"{'Mode':<48} {'W':>4} {'D':>4} {'L':>4}  {'Score%':>7}  {'P(>50%)':>8}  {'s/gm':>5}"
    print(hdr)
    print("─" * len(hdr))

    all_rows: list[dict] = []
    bench_t0 = time.time()

    for mode in modes:
        wins = draws = losses = 0
        t0 = time.time()
        for g in range(args.games):
            cand_color = "W" if g % 2 == 0 else "B"
            opp_color  = "B" if cand_color == "W" else "W"
            cand = _make_candidate_ai(cand_color, args.difficulty, args.time_budget, mode,
                                      sentinel, value_net, gap_net, malom_db)
            opp  = _make_opponent_ai(opp_color, args.difficulty, args.time_budget,
                                     sentinel, value_net, malom_db)
            winner = _play_one(cand, opp, cand_color, args.max_plies)
            if winner is None:
                draws += 1; oc = "D"
            elif winner == cand_color:
                wins += 1;  oc = "W"
            else:
                losses += 1; oc = "L"
            row = {"mode": mode, "game": g + 1, "cand_color": cand_color, "outcome": oc,
                   "difficulty": args.difficulty, "time_budget": args.time_budget,
                   "timestamp": datetime.now().isoformat(timespec="seconds")}
            all_rows.append(row)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if not args.quiet:
                ch = {"W": "+", "L": "-", "D": "="}[oc]
                print(f"  g{g+1:02d}({cand_color}):{ch}", end="", flush=True)
                if (g + 1) % 10 == 0:
                    print("", flush=True)
        if not args.quiet:
            print()
        elapsed = time.time() - t0
        p_hat = score_proportion(wins, draws, losses)
        p_sup = superiority_probability(wins, draws, losses, target=0.50)
        print(f"{MODE_LABEL[mode]:<48} "
              f"{wins:>4} {draws:>4} {losses:>4}  "
              f"{p_hat*100:>6.1f}%  {p_sup:>7.3f}  "
              f"{elapsed/max(args.games,1):>4.1f}s")

    # ── summary ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - bench_t0
    print()
    print(f"Done in {total_elapsed/60:.1f} min ({total_elapsed:.0f} s).")
    print()
    print("## Summary")
    print()
    print("| Mode | W | D | L | Score% | P(true > 50%) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for mode in modes:
        rows = [r for r in all_rows if r["mode"] == mode]
        w = sum(1 for r in rows if r["outcome"] == "W")
        d = sum(1 for r in rows if r["outcome"] == "D")
        l = sum(1 for r in rows if r["outcome"] == "L")
        p_hat = score_proportion(w, d, l)
        p_sup = superiority_probability(w, d, l, target=0.50)
        print(f"| {MODE_LABEL[mode]} | {w} | {d} | {l} | {p_hat*100:.1f}% | {p_sup:.3f} |")

    print()
    print(f"Rows: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
