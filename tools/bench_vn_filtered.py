"""tools/bench_vn_filtered.py — Round-robin: base heuristics vs all value net variants.

Up to 7 configs (missing nets are skipped automatically):
  Base     — pure heuristics, no value net
  OldVN30  — data/value_net.npz,                vn_blend=30  (AI self-play trained)
  OldVN60  — data/value_net.npz,                vn_blend=60
  FiltVN30 — data/value_net_human_filtered.npz, vn_blend=30  (top-25% Elo, decisive-only)
  FiltVN60 — data/value_net_human_filtered.npz, vn_blend=60
  V2VN30   — data/value_net_human_v2.npz,       vn_blend=30  (all games + placement blend)
  V2VN60   — data/value_net_human_v2.npz,       vn_blend=60

Round-robin: C(n,2) pairs × games_per_pair (default 10, must be even).
Colours alternate each game within a pair.

Usage:
    .venv/bin/python tools/bench_vn_filtered.py
    .venv/bin/python tools/bench_vn_filtered.py --diff 5 --budget 3.0 --games-per-pair 20
    .venv/bin/python tools/bench_vn_filtered.py --out results_vn_filtered.json
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
from ai.value_net import ValueNet


# ── Load value nets ───────────────────────────────────────────────────────────

def _load_net(rel: str) -> Optional[ValueNet]:
    path = ROOT / rel
    net = ValueNet.load_if_exists(path)
    if net is not None:
        print(f"[init] Loaded {rel}")
    else:
        print(f"[init] NOT FOUND: {rel}  — configs using this net will be skipped")
    return net


_old_net  = _load_net("data/value_net.npz")
_filt_net = _load_net("data/value_net_human_filtered.npz")
_v2_net   = _load_net("data/value_net_human_v2.npz")


# ── Config definition ─────────────────────────────────────────────────────────

@dataclass
class Config:
    name:     str
    vn_blend: int
    net:      Optional[ValueNet]

    # Accumulated results
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
        """False if the required net is missing."""
        return self.vn_blend == 0 or self.net is not None


CONFIGS: list[Config] = [
    Config("Base",     vn_blend=0,  net=None),
    Config("OldVN30",  vn_blend=30, net=_old_net),
    Config("OldVN60",  vn_blend=60, net=_old_net),
    Config("FiltVN30", vn_blend=30, net=_filt_net),
    Config("FiltVN60", vn_blend=60, net=_filt_net),
    Config("V2VN30",   vn_blend=30, net=_v2_net),
    Config("V2VN60",   vn_blend=60, net=_v2_net),
]


# ── Engine factory ────────────────────────────────────────────────────────────

def make_ai(color: str, cfg: Config, difficulty: int, budget: float) -> GameAI:
    weights = HeuristicWeights(value_net_blend=cfg.vn_blend)
    return GameAI(
        color=color,
        difficulty=difficulty,
        weights=weights,
        override_time_budget=budget,
        value_net=cfg.net if cfg.vn_blend > 0 else None,
    )


# ── Game runner ───────────────────────────────────────────────────────────────

MAX_MOVES = 300

def play_game(white_ai: GameAI, black_ai: GameAI) -> Optional[str]:
    """Return 'W', 'B', or None (draw / max-moves reached)."""
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
    nw  = max(len(c.name) for c in configs)
    cw  = 10  # " WW-DD-LL "
    header = f"  {'':>{nw}}  " + "  ".join(f"{c.name:^{cw}}" for c in configs)
    print(header)
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
        description="Round-robin: base heuristics vs old/new value nets at 30% & 60%",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--diff",           type=int,   default=4,
                    help="Difficulty level")
    ap.add_argument("--budget",         type=float, default=3.0,
                    help="Per-move time budget (seconds)")
    ap.add_argument("--games-per-pair", type=int,   default=10,
                    help="Games per config pair (must be even)")
    ap.add_argument("--out",            type=str,
                    default=str(ROOT / "eval_results_vn_filtered.json"),
                    help="JSON output path")
    args = ap.parse_args()

    if args.games_per_pair % 2 != 0:
        sys.exit("--games-per-pair must be even (balanced colour allocation)")

    # Drop configs whose net is missing
    configs = [c for c in CONFIGS if c.available]
    skipped = [c.name for c in CONFIGS if not c.available]
    if skipped:
        print(f"[warn] Skipping unavailable configs: {', '.join(skipped)}")

    if len(configs) < 2:
        sys.exit("Need at least 2 available configs to run a tournament.")

    n    = len(configs)
    pairs = list(combinations(range(n), 2))
    gpp   = args.games_per_pair
    total = len(pairs) * gpp

    print(f"\nRound-robin: {n} configs  ×  {len(pairs)} pairs  ×  {gpp} games  =  {total} games")
    print(f"Difficulty {args.diff},  budget {args.budget:.1f}s/move\n")
    print("Configs:")
    for c in configs:
        net_tag = (
            "no net"                        if c.net is None      else
            "value_net.npz"                 if c.net is _old_net  else
            "value_net_human_filtered.npz"  if c.net is _filt_net else
            "value_net_human_v2.npz"
        )
        print(f"  {c.name:<10}  vn_blend={c.vn_blend}%  net={net_tag}")
    print()

    matrix   = [[{"w": 0, "d": 0, "l": 0} for _ in range(n)] for _ in range(n)]
    game_log: list[dict] = []
    game_num  = 0
    t_start   = time.time()

    for i, j in pairs:
        ci, cj = configs[i], configs[j]
        for k in range(gpp):
            i_is_white  = (k % 2 == 0)
            w_cfg, b_cfg = (ci, cj) if i_is_white else (cj, ci)
            w_idx, b_idx = (i,  j)  if i_is_white else (j,  i)

            white_ai = make_ai("W", w_cfg, args.diff, args.budget)
            black_ai = make_ai("B", b_cfg, args.diff, args.budget)

            g_start = time.time()
            winner  = play_game(white_ai, black_ai)
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
                "vn_blend": c.vn_blend,
                "net":      (None       if c.net is None      else
                             "old"       if c.net is _old_net  else
                             "filtered"  if c.net is _filt_net else
                             "v2"),
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
    summary = {
        "difficulty":     args.diff,
        "budget_s":       args.budget,
        "games_per_pair": gpp,
        "total_games":    total,
        "total_time_min": round(total_time / 60, 1),
        "standings":      standings_data,
        "games":          game_log,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
