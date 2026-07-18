"""scripts/bench_scaffolded.py — headless benchmark for the three v2 phase specialists.

Runs the **SpecialistRouter** (opening + midgame + endgame specialists, routed by
phase) as ONE player, versus a matrix of heuristic-opponent configurations at
multiple difficulty levels.  Colours alternate every game to cancel first-mover
advantage.  Designed to run overnight.

Opponent configurations (default set)
-------------------------------------
  raw       GameAI only (no sentinel / vn / gap net)
  sentinel  GameAI + sentinel (score_adjust, 20% intervention probability)
  vn        GameAI + value_net blend 20%
  gap       GameAI + gap net (blunder-zone exploitation)
  sv        GameAI + sentinel + value_net
  full      GameAI + sentinel + value_net + gap net
  deep      GameAI full stack + extended max_search_depth (25) — heaviest tuning

All opponent configs share the same tuning: value_net_blend = 20, sentinel
intervention probability = 20% when sentinel is enabled.

Router (specialist AI) uses sentinel + value_net + gap_net + human_db + Malom DB at inference.

Time budget
-----------
By default, each heuristic move gets the SAME per-difficulty time budget the
game uses in real play (see ``GameAI._iterative_deepen`` cap table):

  diff 1-5 : 15 s    diff 6 : 30 s    diff 7 : 45 s    diff 8-10 : 60 s
  (further reduced to 3 s / 10 s during early placement — first 2 pieces / ≤4)

Pass ``--time-budget SECONDS`` to override with a flat cap (useful for a
fast smoke test — the game-native caps are slow because they mirror real play).

Usage
-----
# Full overnight sweep at game-native per-difficulty budgets
.venv/bin/python scripts/bench_scaffolded.py --games 40 --difficulties 3,5,7,9

# Quick 10-game sanity check at diff 5, capped at 2 s / move
.venv/bin/python scripts/bench_scaffolded.py --games 10 --difficulties 5 \\
    --opponents raw,full --time-budget 2.0

# Deeper specialist lookahead + game-native budgets
.venv/bin/python scripts/bench_scaffolded.py --games 40 --difficulties 5,7,9 \\
    --specialist-ply-depth 25

Results
-------
Streamed to `data/bench/scaffolded_v2_<timestamp>.jsonl` (one row per matchup)
plus a final markdown table printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import is_terminal, get_all_legal_moves
from learned_ai.agents.specialist_router import load_specialist_router, load_generalist
from learned_ai.sentinel.infer import load_advisor

_SENTINEL_CKPT   = str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt")
_VALUE_NET_PATH  = str(_ROOT / "data" / "value_net.npz")
_GAP_NET_PATH    = str(_ROOT / "data" / "gap_net.npz")
_MALOM_DEFAULT   = "/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"


# ── component loaders ─────────────────────────────────────────────────────────

def _load_sentinel(path: str):
    try:
        adv = load_advisor(path)
        if adv and adv.is_loaded():
            print(f"  Sentinel loaded from {path}")
            return adv
        print("  Sentinel unavailable (loader returned None)")
        return None
    except Exception as e:
        print(f"  Sentinel load failed: {e}")
        return None


def _load_value_net(path: str):
    try:
        from ai.value_net import ValueNet
        vn = ValueNet.load(path)
        print(f"  Value net loaded from {path}")
        return vn
    except Exception as e:
        print(f"  Value net load failed: {e}")
        return None


def _load_gap_net(path: str):
    try:
        from ai.gap_net import GapNet
        gn = GapNet.load(path)
        print(f"  Gap net loaded from {path}")
        return gn
    except Exception as e:
        print(f"  Gap net load failed: {e}")
        return None


def _load_malom(path: str):
    try:
        from learned_ai.sentinel.db_teacher import ExternalSolvedDB
        db = ExternalSolvedDB(path)
        if db.is_available():
            print(f"  Malom perfect DB loaded from {path}")
            return db
        print(f"  Malom DB path exists but not available: {path}")
        return None
    except Exception as e:
        print(f"  Malom DB load failed: {e}")
        return None


# ── game-native per-difficulty time cap ──────────────────────────────────────

def _game_time_cap_for_diff(difficulty: int) -> float:
    """Return the per-move cap GameAI uses in real play at this difficulty.
    Mirrors the table in ``GameAI._iterative_deepen`` (excludes early-placement
    reductions to 3 s / 10 s, which GameAI applies internally per-position).

      diff 1-5 : 15 s
      diff 6   : 30 s
      diff 7   : 45 s
      diff 8+  : 60 s
    """
    d = max(1, min(10, int(difficulty)))
    if d >= 8: return 60.0
    if d == 7: return 45.0
    if d == 6: return 30.0
    return 15.0


# ── opponent factory ─────────────────────────────────────────────────────────

def _make_opp_factory(
    config: str,
    sentinel,
    value_net,
    gap_net,
    malom_db,
) -> Callable[[str, int, float], "object"]:
    """Return a factory that builds a fresh GameAI for a matchup."""
    from ai.game_ai import GameAI
    from ai.heuristics import HeuristicWeights

    use_sentinel = config in ("sentinel", "sv", "full", "deep")
    use_vn       = config in ("vn", "sv", "full", "deep")
    use_gap      = config in ("gap", "full", "deep")
    deep_search  = (config == "deep")

    def factory(color: str, difficulty: int, time_budget: Optional[float]):
        # time_budget: None or ≤ 0 → let GameAI use its native per-difficulty caps
        # (15/30/45/60 s, with 3/10 s early-placement reductions).  A positive
        # value overrides all of that with a flat per-move cap.
        override = None if (time_budget is None or time_budget <= 0) else float(time_budget)
        # Global tuning: value_net_blend = 20%, sentinel intervention prob = 20%.
        weights = HeuristicWeights(value_net_blend=20) if use_vn else None
        ai = GameAI(
            color=color,
            difficulty=difficulty,
            value_net=(value_net if use_vn else None),
            gap_net=(gap_net if use_gap else None),
            weights=weights,
            malom_db=malom_db,
            override_time_budget=override,
        )
        if use_sentinel and sentinel is not None:
            ai.set_sentinel(sentinel, mode="score_adjust")
            ai._sentinel_activation_prob = 0.20   # sentinel intervenes on 20% of decisions
        if deep_search:
            ai.max_search_depth = 25   # allow iterative deepening deeper (still bounded by time)
        return ai

    return factory


CONFIG_LABELS = {
    "raw":      "GameAI only",
    "sentinel": "GameAI + sentinel20%",
    "vn":       "GameAI + vn20%",
    "gap":      "GameAI + gap_net",
    "sv":       "GameAI + sent + vn",
    "full":     "GameAI + sent + vn + gap",
    "deep":     "GameAI full + depth25",
}


# ── router move-picker ────────────────────────────────────────────────────────

def _router_choose(router, board: BoardState) -> Optional[dict]:
    """Ask the SpecialistRouter for its argmax move at ``board``."""
    legal = get_all_legal_moves(board)
    if not legal:
        return None
    candidates = [
        {"from": m.get("from"), "to": m.get("to"), "capture": m.get("capture")}
        for m in legal
    ]
    probs = router.score_moves(board, candidates, board.turn)
    if not probs:
        # Fallback — should be rare; router covers all phases
        return legal[0]
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return legal[best_idx]


# ── one-game runner ───────────────────────────────────────────────────────────

def _play_one(router, opponent_ai, router_color: str, max_plies: int = 400) -> Optional[str]:
    """Return 'W', 'B', or None (draw)."""
    board = BoardState.new_game()
    for _ in range(max_plies):
        terminal, winner = is_terminal(board)
        if terminal:
            return winner
        legal = get_all_legal_moves(board)
        if not legal:
            return "B" if board.turn == "W" else "W"

        if board.turn == router_color:
            move = _router_choose(router, board)
        else:
            try:
                move = opponent_ai.choose_move(board)
            except Exception:
                return router_color   # opponent crash counts as router win

        if not move:
            return "B" if board.turn == "W" else "W"
        try:
            board = board.apply_move(move)
        except Exception:
            return None
    return None


# ── matchup runner ────────────────────────────────────────────────────────────

def _run_matchup(
    router,
    opp_factory,
    label: str,
    difficulty: int,
    games: int,
    time_budget: float,
    max_plies: int,
    on_game_done: Optional[Callable[[int, str, str], None]] = None,
) -> dict:
    """Play N games, alternating colours.  Returns aggregate row."""
    wins = draws = losses = 0
    t0 = time.time()
    for g in range(games):
        router_color = "W" if g % 2 == 0 else "B"
        opp_color    = "B" if router_color == "W" else "W"
        opp = opp_factory(color=opp_color, difficulty=difficulty, time_budget=time_budget)
        winner = _play_one(router, opp, router_color, max_plies)
        outcome = ("W" if winner == router_color else
                   "L" if winner is not None    else "D")
        if outcome == "W":  wins   += 1
        elif outcome == "L": losses += 1
        else:               draws  += 1
        if on_game_done is not None:
            on_game_done(g + 1, router_color, outcome)
    total = max(wins + draws + losses, 1)
    elapsed = time.time() - t0
    score = (wins + 0.5 * draws) / total   # standard "points" (draw = ½ win)
    return {
        "opponent": label,
        "difficulty": difficulty,
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "score": score,
        "elapsed_s": round(elapsed, 1),
        "avg_s_per_game": round(elapsed / total, 2),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark v2 SpecialistRouter vs heuristic configs.")
    p.add_argument("--games",           type=int,   default=40,
                   help="Games per matchup (default 40).")
    p.add_argument("--difficulties",    default="3,5,7,9",
                   help="Comma-separated GameAI difficulties (1-10; default '3,5,7,9').")
    p.add_argument("--opponents",       default="raw,sentinel,vn,gap,sv,full,deep",
                   help=f"Configs (choose from {','.join(CONFIG_LABELS)}).")
    p.add_argument("--time-budget",     type=float, default=-1.0,
                   help="Per-move time budget for the heuristic opponent (seconds). "
                        "Default (-1) uses the game's per-difficulty caps "
                        "(15/30/45/60 s at diff 1-5/6/7/8+, with 3-10 s early-placement reductions). "
                        "Pass a positive value for a flat override.")
    p.add_argument("--agent",               default="specialist",
                   choices=["specialist", "generalist"],
                   help="Which learned AI to bench: 'specialist' (phase-routed, default) or 'generalist' (s_gen_v2).")
    p.add_argument("--specialist-ply-depth", type=int, default=12,
                   help="LookaheadAdvisor ply depth for the specialists (default 12, matches V4 training).")
    p.add_argument("--max-plies",       type=int,   default=400)
    p.add_argument("--sentinel-path",   default=_SENTINEL_CKPT)
    p.add_argument("--value-net-path",  default=_VALUE_NET_PATH)
    p.add_argument("--gap-net-path",    default=_GAP_NET_PATH)
    p.add_argument("--malom-path",      default=_MALOM_DEFAULT,
                   help=f"Malom perfect DB dir (default {_MALOM_DEFAULT}).")
    p.add_argument("--out-dir",         default=str(_ROOT / "data" / "bench"))
    p.add_argument("--quiet",           action="store_true",
                   help="Suppress per-game progress dots.")
    args = p.parse_args()

    try:
        difficulties = [int(d) for d in args.difficulties.split(",") if d.strip()]
    except ValueError:
        print("ERROR: --difficulties must be a comma-separated list of integers", file=sys.stderr)
        return 2
    configs = [c.strip() for c in args.opponents.split(",") if c.strip() in CONFIG_LABELS]
    if not configs:
        print(f"ERROR: no valid --opponents; choose from {','.join(CONFIG_LABELS)}", file=sys.stderr)
        return 2

    # ── components ────────────────────────────────────────────────────────────
    print("Loading components…")
    sentinel  = _load_sentinel(args.sentinel_path)
    value_net = _load_value_net(args.value_net_path)
    gap_net   = _load_gap_net(args.gap_net_path)
    malom_db  = _load_malom(args.malom_path)
    human_db  = None
    _hdb_path = _ROOT / "data" / "human_db.sqlite"
    if _hdb_path.exists():
        try:
            from ai.human_db import HumanDB as _HumanDB
            human_db = _HumanDB(_hdb_path)
            print(f"  HumanDB loaded: {human_db.game_count} games")
        except Exception as _e:
            print(f"  HumanDB load failed: {_e}")
    else:
        print("  HumanDB: not found (human_norm features will be 0.5)")
    print()

    if args.agent == "generalist":
        print("Loading GeneralistAgent (s_gen_v2)…")
        router = load_generalist(
            sentinel_advisor=sentinel,
            human_db=human_db,
            value_net=value_net,
            gap_net=gap_net,
            ply_depth=args.specialist_ply_depth,
        )
        if router is None:
            print("ERROR: s_gen_v2/best.pt not found. Train first or check paths.", file=sys.stderr)
            return 1
        print(f"  generalist=OK  ply_depth={args.specialist_ply_depth}")
    else:
        print("Loading SpecialistRouter (v2 phase specialists)…")
        router = load_specialist_router(
            sentinel_advisor=sentinel,
            human_db=human_db,
            db=malom_db,
            value_net=value_net,
            gap_net=gap_net,
            ply_depth=args.specialist_ply_depth,
        )
        if router is None:
            print("ERROR: no v2 specialist checkpoints found. Train first or check paths.", file=sys.stderr)
            return 1
        print(f"  open={'OK' if router._spec_open else 'missing'}  "
              f"mid={'OK' if router._spec_mid else 'missing'}  "
              f"end={'OK' if router._spec_end else 'missing'}  "
              f"ply_depth={args.specialist_ply_depth}")

    print()

    # ── plan ──────────────────────────────────────────────────────────────────
    total_matchups = len(difficulties) * len(configs)
    total_games    = total_matchups * args.games
    print(f"Benchmark plan: {len(configs)} configs × {len(difficulties)} difficulties × "
          f"{args.games} games = {total_matchups} matchups, {total_games} games total.")
    if args.time_budget is None or args.time_budget <= 0:
        cap_report = ", ".join(f"d{d}={_game_time_cap_for_diff(d):.0f}s" for d in difficulties)
        print(f"Heuristic time budget: game-native per-difficulty ({cap_report}). "
              f"Early-placement reductions (3 s / 10 s) applied automatically by GameAI.")
    else:
        print(f"Heuristic time budget: flat {args.time_budget:.1f} s per move (overriding game defaults).")
    print()

    # ── output ────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"scaffolded_{args.agent}_{ts}.jsonl"
    print(f"Streaming results to {out_path}")
    print()

    header = f"{'Opponent':<26} {'D':>2}  {'W':>4} {'D':>4} {'L':>4}  {'WR':>6} {'DR':>6} {'Score':>6}  {'s/gm':>5}"
    print(header)
    print("─" * len(header))

    results: list[dict] = []
    matchup_idx = 0
    bench_t0 = time.time()

    for diff in difficulties:
        for cfg in configs:
            matchup_idx += 1
            # ETA based on elapsed
            elapsed = time.time() - bench_t0
            eta_remaining = ""
            if matchup_idx > 1:
                per = elapsed / (matchup_idx - 1)
                remaining_matchups = total_matchups - (matchup_idx - 1)
                secs = per * remaining_matchups
                mins = int(secs // 60); hrs = mins // 60
                eta_remaining = f"  [ETA {hrs}h{mins%60:02d}m]" if hrs else f"  [ETA {mins}m]"

            print(f"\n… Matchup {matchup_idx}/{total_matchups}: "
                  f"router vs {CONFIG_LABELS[cfg]} @ diff{diff}{eta_remaining}", flush=True)

            factory = _make_opp_factory(cfg, sentinel, value_net, gap_net, malom_db)

            def _dot(g_idx: int, rc: str, outcome: str):
                if args.quiet: return
                ch = {"W": "+", "L": "-", "D": "="}[outcome]
                print(f"  g{g_idx:02d}({rc}):{ch}", end="", flush=True)
                if g_idx % 10 == 0:
                    print("", flush=True)

            row = _run_matchup(
                router=router,
                opp_factory=factory,
                label=CONFIG_LABELS[cfg],
                difficulty=diff,
                games=args.games,
                time_budget=args.time_budget,
                max_plies=args.max_plies,
                on_game_done=_dot,
            )
            row["config"] = cfg
            # Record the effective budget: either the flat override or the
            # game-native per-difficulty cap.
            eff_budget = (args.time_budget if args.time_budget is not None and args.time_budget > 0
                          else _game_time_cap_for_diff(diff))
            row["time_budget_s"] = eff_budget
            row["time_budget_mode"] = ("flat_override"
                                       if args.time_budget is not None and args.time_budget > 0
                                       else "game_native_per_diff")
            row["specialist_ply_depth"] = args.specialist_ply_depth
            row["timestamp"] = datetime.now().isoformat(timespec="seconds")
            results.append(row)

            if not args.quiet:
                print()   # newline after per-game dots

            print(f"{CONFIG_LABELS[cfg]:<26} {diff:>2}  "
                  f"{row['wins']:>4} {row['draws']:>4} {row['losses']:>4}  "
                  f"{row['win_rate']:>5.1%} {row['draw_rate']:>5.1%} "
                  f"{row['score']:>5.1%}  {row['avg_s_per_game']:>4}s")

            # Stream to disk after each matchup (robust to overnight interruption).
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    # ── final markdown table ─────────────────────────────────────────────────
    total_elapsed = time.time() - bench_t0
    print()
    print(f"Done in {total_elapsed/3600:.2f} h ({total_elapsed:.0f}s).")
    print()
    print("## Summary (score = wins + ½·draws / games)")
    print()
    diffs_sorted = sorted(set(r["difficulty"] for r in results))
    print("| Config | " + " | ".join(f"d{d} score" for d in diffs_sorted) + " |")
    print("| --- | " + " | ".join(["---"] * len(diffs_sorted)) + " |")
    for cfg in configs:
        label = CONFIG_LABELS[cfg]
        row_cells = []
        for d in diffs_sorted:
            match = next((r for r in results if r["config"] == cfg and r["difficulty"] == d), None)
            if match is None:
                row_cells.append("—")
            else:
                row_cells.append(f"{match['score']:.1%} ({match['wins']}/{match['draws']}/{match['losses']})")
        print(f"| {label} | " + " | ".join(row_cells) + " |")

    print()
    print(f"Full per-matchup results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
