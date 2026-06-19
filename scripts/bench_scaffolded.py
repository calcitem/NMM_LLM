"""scripts/bench_scaffolded.py — headless benchmark for the scaffolded meta-policy.

Tests ScaffoldedAgent against a matrix of opponent configurations:
  * raw heuristic (various difficulties)
  * heuristic + sentinel (score_adjust)
  * heuristic + value net (vn_blend=80%)
  * heuristic + sentinel + value net  (the full deployed stack)

Colours alternate first-mover every game to cancel first-move advantage.
Results track by AGENT CONFIG, not by colour.

Usage examples
--------------
# Quick sanity check after Stage 1 imitation training (5 games, diff 2 only)
.venv/bin/python scripts/bench_scaffolded.py \\
    --checkpoint learned_ai/checkpoints/scaffolded/s1/best.pt \\
    --games 5 --difficulties 2

# Full benchmark after Stage 2 (40 games, diff 2-4, all opponent configs)
.venv/bin/python scripts/bench_scaffolded.py \\
    --checkpoint learned_ai/checkpoints/scaffolded/s2/best.pt \\
    --games 40 --difficulties 2,3,4

# Compare two checkpoints (s1 vs s2) against the same diff 3 opponent
.venv/bin/python scripts/bench_scaffolded.py \\
    --checkpoint learned_ai/checkpoints/scaffolded/s2/best.pt \\
    --compare   learned_ai/checkpoints/scaffolded/s1/best.pt \\
    --games 40 --difficulties 3

# Only test against full stack (sentinel + vn80%) at diff 4
.venv/bin/python scripts/bench_scaffolded.py \\
    --checkpoint learned_ai/checkpoints/scaffolded/s2/best.pt \\
    --games 40 --difficulties 4 --opponents full

Available --opponents values (comma-separated):
  raw        heuristic only (no sentinel, no value net)
  sentinel   heuristic + sentinel score_adjust
  vn         heuristic + value_net blend 80%
  full       heuristic + sentinel + value_net 80%

Default: raw,sentinel,vn,full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import is_terminal, get_all_legal_moves
from learned_ai.agents.scaffolded_agent import ScaffoldedAgent
from learned_ai.sentinel.infer import load_advisor

_SENTINEL_CKPT = str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt")
_VALUE_NET_PATH = str(_ROOT / "data" / "value_net.npz")


# ── component loaders ──────────────────────────────────────────────────────────

def _load_sentinel(path: str):
    try:
        adv = load_advisor(path)
        if adv and adv.is_loaded():
            print(f"  Sentinel loaded: {path}")
            return adv
        print("  Sentinel load returned None — opponent sentinel unavailable")
        return None
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
        print(f"  Value net load failed ({e}) — vn configs will be skipped")
        return None


def _make_heuristic_ai(color: str, difficulty: int, sentinel=None, value_net=None,
                       vn_blend: int = 80, time_budget: float = 0.25):
    from ai.game_ai import GameAI
    from ai.heuristics import HeuristicWeights
    weights = HeuristicWeights(value_net_blend=vn_blend) if (value_net and vn_blend) else None
    ai = GameAI(
        color=color,
        difficulty=difficulty,
        value_net=value_net if value_net else None,
        weights=weights,
        override_time_budget=time_budget,
    )
    if sentinel is not None:
        ai.set_sentinel(sentinel, mode="score_adjust")
    return ai


# ── single game runner ─────────────────────────────────────────────────────────

def _play_one(scaffolded, opponent_ai, scaffolded_color: str, max_plies: int = 400) -> Optional[str]:
    """Play one game. Returns 'W', 'B', or None (draw)."""
    board = BoardState.new_game()
    for _ in range(max_plies):
        terminal, winner = is_terminal(board)
        if terminal:
            return winner
        legal = get_all_legal_moves(board)
        if not legal:
            return "B" if board.turn == "W" else "W"

        if board.turn == scaffolded_color:
            move = scaffolded.choose_move(board)
        else:
            try:
                move = opponent_ai.choose_move(board)
            except Exception:
                return scaffolded_color  # opponent crash = scaffolded wins

        if not move:
            return "B" if board.turn == "W" else "W"

        try:
            board = board.apply_move(move)
        except Exception:
            return None

    return None  # draw by length


# ── match runner ───────────────────────────────────────────────────────────────

def _run_match(
    scaffolded: ScaffoldedAgent,
    opponent_label: str,
    difficulty: int,
    sentinel_for_opp,
    vn_for_opp,
    games: int,
    time_budget: float,
    max_plies: int,
) -> dict:
    """Run one matchup (N games, alternating colours). Returns result dict."""
    wins = draws = losses = 0

    for g in range(games):
        scaffolded_color = "W" if g % 2 == 0 else "B"
        opp_color = "B" if scaffolded_color == "W" else "W"

        opponent_ai = _make_heuristic_ai(
            color=opp_color,
            difficulty=difficulty,
            sentinel=sentinel_for_opp,
            value_net=vn_for_opp,
            time_budget=time_budget,
        )

        winner = _play_one(scaffolded, opponent_ai, scaffolded_color, max_plies)

        if winner is None:
            draws += 1
        elif winner == scaffolded_color:
            wins += 1
        else:
            losses += 1

    total = max(wins + draws + losses, 1)
    return {
        "opponent": opponent_label,
        "difficulty": difficulty,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total,
        "draw_rate": draws / total,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark ScaffoldedAgent vs heuristic configs")
    p.add_argument("--checkpoint", required=True,
                   help="ScaffoldedAgent checkpoint (e.g. checkpoints/scaffolded/s2/best.pt)")
    p.add_argument("--compare", default="",
                   help="Optional second checkpoint to run in the same table")
    p.add_argument("--games", type=int, default=40,
                   help="Games per matchup (default 40; use 10 for quick smoke test)")
    p.add_argument("--difficulties", default="2,3,4",
                   help="Comma-separated difficulty levels to test (default 2,3,4)")
    p.add_argument("--opponents", default="raw,sentinel,vn,full",
                   help="Which opponent configs: raw,sentinel,vn,full (comma-separated)")
    p.add_argument("--sentinel-path", default=_SENTINEL_CKPT,
                   help="Sentinel checkpoint for OPPONENT (and --agent-sentinel if requested)")
    p.add_argument("--agent-sentinel", action="store_true",
                   help="Also give the scaffolded agent a sentinel (recommended)")
    p.add_argument("--time-budget", type=float, default=0.25,
                   help="Seconds per opponent move (default 0.25)")
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--out", default="",
                   help="Optional JSON output path for results")
    args = p.parse_args()

    difficulties = [int(d) for d in args.difficulties.split(",")]
    opp_configs   = [s.strip() for s in args.opponents.split(",")]

    # ── load components ────────────────────────────────────────────────────────
    print("\nLoading components...")
    sentinel = _load_sentinel(args.sentinel_path)
    value_net = _load_value_net()
    print()

    opp_sentinels = {"raw": None, "sentinel": sentinel, "vn": None,    "full": sentinel}
    opp_vns       = {"raw": None, "sentinel": None,     "vn": value_net, "full": value_net}
    opp_labels    = {
        "raw":      "heuristic",
        "sentinel": "heuristic+sentinel",
        "vn":       "heuristic+vn80%",
        "full":     "heuristic+sentinel+vn80%",
    }

    # ── load scaffolded agents ─────────────────────────────────────────────────
    checkpoints = [(args.checkpoint, "agent")]
    if args.compare:
        checkpoints.append((args.compare, "compare"))

    def _make_agent(ckpt_path: str) -> ScaffoldedAgent:
        agent_sentinel = sentinel if args.agent_sentinel else None
        return ScaffoldedAgent(
            color="W",
            checkpoint_path=ckpt_path,
            sentinel_advisor=agent_sentinel,
            mode="argmax",
        )

    all_results: list[dict] = []

    for ckpt_path, ckpt_label in checkpoints:
        ckpt_name = Path(ckpt_path).parent.name + "/" + Path(ckpt_path).name
        print(f"\n{'='*60}")
        print(f"  Agent: {ckpt_name}  ({'with sentinel' if args.agent_sentinel else 'no sentinel'})")
        print(f"{'='*60}")

        # Build the agent once per checkpoint
        try:
            agent = _make_agent(ckpt_path)
        except Exception as e:
            print(f"  Failed to load {ckpt_path}: {e}")
            continue

        header = f"{'Opponent':<28} {'D':>2}  {'W':>5} {'D':>5} {'L':>5}  {'WR%':>6}  {'DR%':>6}"
        print(header)
        print("─" * len(header))

        for diff in difficulties:
            for cfg in opp_configs:
                if cfg not in opp_labels:
                    continue
                # Skip vn/full if value_net not available
                if "vn" in cfg and value_net is None:
                    print(f"  {'[skipped — value_net unavailable]':<28}  d{diff}")
                    continue

                t0 = time.time()
                result = _run_match(
                    scaffolded=agent,
                    opponent_label=opp_labels[cfg],
                    difficulty=diff,
                    sentinel_for_opp=opp_sentinels[cfg],
                    vn_for_opp=opp_vns[cfg],
                    games=args.games,
                    time_budget=args.time_budget,
                    max_plies=args.max_plies,
                )
                elapsed = time.time() - t0
                result["checkpoint"] = ckpt_path
                result["ckpt_label"] = ckpt_label
                result["agent_sentinel"] = args.agent_sentinel
                result["elapsed_s"] = round(elapsed, 1)
                all_results.append(result)

                label = opp_labels[cfg]
                print(
                    f"  {label:<26} {diff:>2}  "
                    f"{result['wins']:>5} {result['draws']:>5} {result['losses']:>5}  "
                    f"{result['win_rate']:>5.1%}  {result['draw_rate']:>5.1%}  "
                    f"({elapsed:.0f}s)"
                )

        print()

    # ── summary ────────────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
