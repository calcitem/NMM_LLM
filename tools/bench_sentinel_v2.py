"""tools/bench_sentinel_v2.py — Round-robin: base vs old sentinel vs new sentinel.

5 configs:
  Base       — pure heuristics, no sentinel
  OldS20     — old sentinel (best.pt),    score_adjust, min_gap=0.20
  OldS30     — old sentinel (best.pt),    score_adjust, min_gap=0.30
  NewS20     — new sentinel (v2/best.pt), score_adjust, min_gap=0.20
  NewS30     — new sentinel (v2/best.pt), score_adjust, min_gap=0.30

Round-robin: C(5,2) = 10 pairs × games_per_pair (default 10, must be even).
Colours alternate each game within a pair.

Old sentinel path:  learned_ai/sentinel/checkpoints/best.pt
New sentinel path:  learned_ai/sentinel/checkpoints/v2/best.pt  (override with --new-ckpt)

Usage:
    .venv/bin/python tools/bench_sentinel_v2.py
    .venv/bin/python tools/bench_sentinel_v2.py --diff 5 --budget 3.0 --games-per-pair 20
    .venv/bin/python tools/bench_sentinel_v2.py --new-ckpt learned_ai/sentinel/checkpoints/v2/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.game_engine import GameEngine
from ai.game_ai import GameAI
from ai.heuristics import HeuristicWeights


# ── Load sentinels ────────────────────────────────────────────────────────────

def _load_sentinel(path: Path, label: str):
    try:
        from learned_ai.sentinel.infer import load_advisor
        from learned_ai.sentinel.config import load_config as _load_cfg
        cfg = _load_cfg()
        advisor = load_advisor(str(path), cfg)
        if advisor is not None:
            print(f"[init] Loaded {label} from {path.relative_to(ROOT)}")
            return advisor
        print(f"[init] load_advisor returned None for {label} — configs using it will be skipped")
        return None
    except Exception as e:
        print(f"[init] Could not load {label}: {e} — configs using it will be skipped")
        return None


# Paths set at arg-parse time; sentinel objects loaded after args are parsed.
_old_sentinel = None
_new_sentinel = None


# ── Config definition ─────────────────────────────────────────────────────────

@dataclass
class Config:
    name:      str
    sentinel:  object        # SentinelAdvisor | None
    min_gap:   float         # sentinel minimum opportunity gap to intervene

    wins:   int = field(default=0, repr=False)
    draws:  int = field(default=0, repr=False)
    losses: int = field(default=0, repr=False)

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def pct(self) -> float:
        return (self.points / self.games * 100) if self.games else 0.0

    @property
    def available(self) -> bool:
        return self.sentinel is None or self.sentinel is not None


def _build_configs(old, new) -> list[Config]:
    configs = [Config("Base", sentinel=None, min_gap=0.0)]
    if old is not None:
        configs += [
            Config("OldS20", sentinel=old, min_gap=0.20),
            Config("OldS30", sentinel=old, min_gap=0.30),
        ]
    else:
        print("[warn] Old sentinel unavailable — OldS20/OldS30 skipped")
    if new is not None:
        configs += [
            Config("NewS20", sentinel=new, min_gap=0.20),
            Config("NewS30", sentinel=new, min_gap=0.30),
        ]
    else:
        print("[warn] New sentinel unavailable — NewS20/NewS30 skipped")
    return configs


# ── Engine factory ────────────────────────────────────────────────────────────

def make_ai(color: str, cfg: Config, difficulty: int, budget: float) -> GameAI:
    weights = HeuristicWeights()
    ai = GameAI(
        color=color,
        difficulty=difficulty,
        weights=weights,
        override_time_budget=budget,
    )
    if cfg.sentinel is not None:
        ai.set_sentinel(cfg.sentinel, mode="score_adjust")
        ai._sentinel_min_gap = cfg.min_gap
    return ai


# ── Game runner ───────────────────────────────────────────────────────────────

MAX_MOVES = 300

def play_game(white_ai: GameAI, black_ai: GameAI) -> Optional[str]:
    """Return 'W', 'B', or None (draw / max-moves)."""
    engine = GameEngine(human_color="W")
    move_count = 0
    while not engine.finished and move_count < MAX_MOVES:
        ai = white_ai if engine.board.turn == "W" else black_ai
        move = ai.choose_move(engine.board)
        if not move:
            break
        engine.apply_move(move)
        move_count += 1
    return engine.winner


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_standings(configs: list[Config]) -> None:
    active = [c for c in configs if c.games > 0]
    if not active:
        return
    ranked = sorted(active, key=lambda c: c.points, reverse=True)
    nw = max(len(c.name) for c in configs)
    print()
    print(f"  {'Config':<{nw}}  {'Pts':>6}  {'W':>4}  {'D':>4}  {'L':>4}  {'G':>4}  {'%':>6}")
    print(f"  {'-'*nw}  {'------':>6}  {'----':>4}  {'----':>4}  {'----':>4}  {'----':>4}  {'------':>6}")
    for c in ranked:
        print(f"  {c.name:<{nw}}  {c.points:>6.1f}  {c.wins:>4}  {c.draws:>4}  {c.losses:>4}  {c.games:>4}  {c.pct:>5.1f}%")
    print()


def _print_matrix(configs: list[Config], matrix: list[list[dict]]) -> None:
    nw = max(len(c.name) for c in configs)
    cw = 10
    print(f"  {'':>{nw}}  " + "  ".join(f"{c.name:^{cw}}" for c in configs))
    for i, ci in enumerate(configs):
        cells = []
        for j in range(len(configs)):
            if i == j:
                cells.append(f"{'—':^{cw}}")
            else:
                r = matrix[i][j]
                cells.append(f"{r['w']:>2}W {r['d']:>2}D {r['l']:>2}L")
        print(f"  {ci.name:>{nw}}  " + "  ".join(cells))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Round-robin: base vs old sentinel vs new sentinel (v2 heuristics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--diff",           type=int,   default=4)
    ap.add_argument("--budget",         type=float, default=3.0,
                    help="Per-move time budget (seconds)")
    ap.add_argument("--games-per-pair", type=int,   default=10,
                    help="Games per config pair (must be even)")
    ap.add_argument("--old-ckpt",       type=Path,
                    default=ROOT / "learned_ai/sentinel/checkpoints/best.pt",
                    help="Old sentinel checkpoint")
    ap.add_argument("--new-ckpt",       type=Path,
                    default=ROOT / "learned_ai/sentinel/checkpoints/v2/best.pt",
                    help="New sentinel checkpoint (trained with evaluate_v2)")
    ap.add_argument("--out",            type=str,
                    default=str(ROOT / "eval_results_sentinel_v2.json"),
                    help="JSON output path")
    args = ap.parse_args()

    if args.games_per_pair % 2 != 0:
        sys.exit("--games-per-pair must be even")

    global _old_sentinel, _new_sentinel
    _old_sentinel = _load_sentinel(args.old_ckpt, "old sentinel") if args.old_ckpt.exists() else None
    _new_sentinel = _load_sentinel(args.new_ckpt, "new sentinel") if args.new_ckpt.exists() else None

    if _old_sentinel is None and not args.old_ckpt.exists():
        print(f"[warn] Old checkpoint not found: {args.old_ckpt}")
    if _new_sentinel is None and not args.new_ckpt.exists():
        print(f"[warn] New checkpoint not found: {args.new_ckpt}")

    configs = _build_configs(_old_sentinel, _new_sentinel)

    if len(configs) < 2:
        sys.exit("Need at least 2 configs. Check that sentinel checkpoints exist.")

    n     = len(configs)
    pairs = list(combinations(range(n), 2))
    gpp   = args.games_per_pair
    total = len(pairs) * gpp

    print(f"\nRound-robin: {n} configs  ×  {len(pairs)} pairs  ×  {gpp} games  =  {total} games")
    print(f"Difficulty {args.diff},  budget {args.budget:.1f}s/move\n")
    print("Configs:")
    for c in configs:
        sent_tag = "no sentinel" if c.sentinel is None else (
            f"old sentinel min_gap={c.min_gap:.0%}" if c.sentinel is _old_sentinel
            else f"new sentinel min_gap={c.min_gap:.0%}"
        )
        print(f"  {c.name:<10}  {sent_tag}")
    print()

    matrix   = [[{"w": 0, "d": 0, "l": 0} for _ in range(n)] for _ in range(n)]
    game_log: list[dict] = []
    game_num  = 0
    t_start   = time.time()

    for i, j in pairs:
        ci, cj = configs[i], configs[j]
        for k in range(gpp):
            i_is_white   = (k % 2 == 0)
            w_cfg, b_cfg = (ci, cj) if i_is_white else (cj, ci)
            w_idx, b_idx = (i,  j)  if i_is_white else (j,  i)

            white_ai = make_ai("W", w_cfg, args.diff, args.budget)
            black_ai = make_ai("B", b_cfg, args.diff, args.budget)

            g_start   = time.time()
            winner    = play_game(white_ai, black_ai)
            g_elapsed = time.time() - g_start
            game_num += 1

            if winner == "W":
                win_idx, lose_idx = w_idx, b_idx
            elif winner == "B":
                win_idx, lose_idx = b_idx, w_idx
            else:
                win_idx = lose_idx = None

            if win_idx is not None:
                configs[win_idx].wins    += 1
                configs[lose_idx].losses += 1
                matrix[win_idx][lose_idx]["w"] += 1
                matrix[lose_idx][win_idx]["l"] += 1
            else:
                configs[i].draws += 1
                configs[j].draws += 1
                matrix[i][j]["d"] += 1
                matrix[j][i]["d"] += 1

            if winner == ("W" if i_is_white else "B"):
                outcome_str = f"{ci.name} wins"
            elif winner == ("B" if i_is_white else "W"):
                outcome_str = f"{cj.name} wins"
            else:
                outcome_str = "draw"

            elapsed = time.time() - t_start
            eta     = (elapsed / game_num) * (total - game_num) if game_num < total else 0
            print(
                f"[{game_num:>3}/{total}]  {ci.name} vs {cj.name}"
                f"  #{k+1}/{gpp}  W={w_cfg.name}"
                f"  winner={'draw' if winner is None else winner}"
                f"  → {outcome_str}"
                f"  {g_elapsed:.0f}s  ETA {eta/60:.0f}m",
                flush=True,
            )

            game_log.append({
                "game":      game_num,
                "white":     w_cfg.name,
                "black":     b_cfg.name,
                "winner":    winner,
                "elapsed_s": round(g_elapsed, 1),
            })

        _print_standings(configs)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print("=" * 70)
    print(f"FINAL STANDINGS — {total} games  diff={args.diff}  budget={args.budget:.1f}s")
    print("=" * 70)
    _print_standings(configs)
    print("HEAD-TO-HEAD MATRIX  (row vs col: W D L from row's perspective)")
    _print_matrix(configs, matrix)
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f}h)\n")

    # ── JSON output ───────────────────────────────────────────────────────────
    out_path = Path(args.out)
    standings_data = sorted(
        [
            {
                "name":     c.name,
                "sentinel": (None if c.sentinel is None else
                             ("old" if c.sentinel is _old_sentinel else "new")),
                "min_gap":  c.min_gap,
                "wins":     c.wins,
                "draws":    c.draws,
                "losses":   c.losses,
                "games":    c.games,
                "points":   c.points,
                "pct":      round(c.pct, 1),
            }
            for c in configs
        ],
        key=lambda x: x["points"],
        reverse=True,
    )
    out_path.write_text(json.dumps({
        "difficulty":     args.diff,
        "budget_s":       args.budget,
        "games_per_pair": gpp,
        "total_games":    total,
        "total_time_min": round(total_time / 60, 1),
        "old_ckpt":       str(args.old_ckpt),
        "new_ckpt":       str(args.new_ckpt),
        "standings":      standings_data,
        "games":          game_log,
    }, indent=2))
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
