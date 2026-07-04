"""tools/bench_tournament.py — round-robin tournament: sentinel × value-net configs.

7 configurations:
  Base  — no sentinel, VN blend=0   (pure engine baseline)
  S0    — sentinel score_adjust, min_gap=0.00, VN=0  (fires on any quality gain)
  S10   — sentinel score_adjust, min_gap=0.10, VN=0
  S20   — sentinel score_adjust, min_gap=0.20, VN=0
  S30   — sentinel score_adjust, min_gap=0.30, VN=0  (fires only on large gaps)
  VN30  — no sentinel, VN blend=30
  VN60  — no sentinel, VN blend=60

Round-robin: C(7,2) = 21 pairs × 10 games/pair = 210 games total.
Each pair alternates colour each game (5 games each colour per pair).

Usage:
    .venv/bin/python tools/bench_tournament.py [--diff 4] [--budget 3.0] [--games-per-pair 10]
    .venv/bin/python tools/bench_tournament.py --diff 6 --budget 3.0 --games-per-pair 6
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

from game.board import BoardState
from game.game_engine import GameEngine
from ai.game_ai import GameAI
from ai.heuristics import HeuristicWeights
from ai.value_net import ValueNet


# ── One-time resource loading ─────────────────────────────────────────────────

_value_net: Optional[ValueNet] = ValueNet.load_if_exists(ROOT / "data" / "value_net.npz")
if _value_net is not None:
    print(f"[init] ValueNet loaded from data/value_net.npz")
else:
    print("[init] ValueNet not found — VN configs will be skipped")

_sentinel_advisor = None
try:
    from learned_ai.sentinel.infer import load_advisor
    from learned_ai.sentinel.config import load_config as _load_sentinel_cfg
    _scfg = _load_sentinel_cfg()
    _ckpt = ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"
    if _ckpt.exists():
        _sentinel_advisor = load_advisor(str(_ckpt), _scfg)
        print(f"[init] Sentinel loaded from {_ckpt}")
    else:
        print(f"[init] Sentinel checkpoint not found at {_ckpt} — S* configs will run without sentinel")
except Exception as _e:
    print(f"[init] Sentinel unavailable ({_e}) — S* configs will run without sentinel")


# ── Config definition ─────────────────────────────────────────────────────────

@dataclass
class TournConfig:
    name: str
    sentinel_gap: Optional[float]  # None = disabled; float = _sentinel_min_gap
    vn_blend: int                  # 0..100

    # Populated at runtime:
    wins: int = field(default=0, repr=False)
    draws: int = field(default=0, repr=False)
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
    def label(self) -> str:
        parts = []
        if self.sentinel_gap is not None:
            parts.append(f"S{int(self.sentinel_gap * 100):02d}")
        if self.vn_blend:
            parts.append(f"VN{self.vn_blend}")
        return self.name


CONFIGS: list[TournConfig] = [
    TournConfig("Base", sentinel_gap=None,  vn_blend=0),
    TournConfig("S0",   sentinel_gap=0.00,  vn_blend=0),
    TournConfig("S10",  sentinel_gap=0.10,  vn_blend=0),
    TournConfig("S20",  sentinel_gap=0.20,  vn_blend=0),
    TournConfig("S30",  sentinel_gap=0.30,  vn_blend=0),
    TournConfig("VN30", sentinel_gap=None,  vn_blend=30),
    TournConfig("VN60", sentinel_gap=None,  vn_blend=60),
]


# ── Engine factory ────────────────────────────────────────────────────────────

def make_ai(color: str, cfg: TournConfig, difficulty: int, budget: float) -> GameAI:
    vnet = _value_net if (cfg.vn_blend > 0 and _value_net is not None) else None
    weights = HeuristicWeights(value_net_blend=cfg.vn_blend)
    ai = GameAI(
        color=color,
        difficulty=difficulty,
        weights=weights,
        override_time_budget=budget,
        value_net=vnet,
    )
    if cfg.sentinel_gap is not None and _sentinel_advisor is not None:
        ai.set_sentinel(_sentinel_advisor, mode="score_adjust")
        ai._sentinel_min_gap = cfg.sentinel_gap
    ai._search_label = cfg.name  # printed in search output
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


# ── Head-to-head matrix ───────────────────────────────────────────────────────

def _empty_matrix(n: int) -> list[list[dict]]:
    return [[{"w": 0, "d": 0, "l": 0} for _ in range(n)] for _ in range(n)]


# ── Main ──────────────────────────────────────────────────────────────────────

def _print_standings(configs: list[TournConfig]) -> None:
    ranked = sorted(configs, key=lambda c: c.points, reverse=True)
    name_w = max(len(c.name) for c in configs)
    print()
    print(f"  {'Config':<{name_w}}  {'Pts':>6}  {'W':>4}  {'D':>4}  {'L':>4}  {'G':>4}  {'%':>6}")
    print(f"  {'-'*name_w}  {'------':>6}  {'----':>4}  {'----':>4}  {'----':>4}  {'----':>4}  {'------':>6}")
    for c in ranked:
        if c.games == 0:
            continue
        print(f"  {c.name:<{name_w}}  {c.points:>6.1f}  {c.wins:>4}  {c.draws:>4}  {c.losses:>4}  {c.games:>4}  {c.pct:>5.1f}%")
    print()


def _print_matrix(configs: list[TournConfig], matrix: list[list[dict]]) -> None:
    n = len(configs)
    name_w = max(len(c.name) for c in configs)
    col_w = 9  # "WW-DD-LL"
    header = f"  {'':>{name_w}}  " + "  ".join(f"{c.name:^{col_w}}" for c in configs)
    print(header)
    for i, ci in enumerate(configs):
        row = f"  {ci.name:>{name_w}}  "
        cells = []
        for j, cj in enumerate(configs):
            if i == j:
                cells.append(f"{'—':^{col_w}}")
            else:
                r = matrix[i][j]
                cells.append(f"{r['w']:>2}W-{r['d']:>2}D-{r['l']:>2}L")
        row += "  ".join(cells)
        print(row)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sentinel × VN round-robin tournament")
    ap.add_argument("--diff",           type=int,   default=4,    help="difficulty (default 4)")
    ap.add_argument("--budget",         type=float, default=3.0,  help="per-move time budget seconds (default 3.0)")
    ap.add_argument("--games-per-pair", type=int,   default=10,   help="games per config pair (default 10, must be even)")
    ap.add_argument("--out",            type=str,   default=str(ROOT / "eval_results.json"))
    args = ap.parse_args()

    gpp = args.games_per_pair
    if gpp % 2 != 0:
        print("--games-per-pair must be even (for balanced colour allocation)")
        sys.exit(1)

    configs = CONFIGS
    n = len(configs)
    pairs = list(combinations(range(n), 2))
    total_games = len(pairs) * gpp

    print(f"\nTournament: {n} configs, {len(pairs)} pairs, {gpp} games/pair = {total_games} games")
    print(f"Difficulty {args.diff}, budget {args.budget:.1f}s/move")
    print("Configs:")
    for c in configs:
        sentinel_str = f"sentinel score_adjust min_gap={c.sentinel_gap:.0%}" if c.sentinel_gap is not None else "no sentinel"
        vn_str = f"VN blend={c.vn_blend}%" if c.vn_blend else "VN off"
        print(f"  {c.name:<8} {sentinel_str}, {vn_str}")
    print()

    matrix = _empty_matrix(n)
    game_log: list[dict] = []
    game_num = 0
    t_start = time.time()

    for i, j in pairs:
        ci, cj = configs[i], configs[j]
        for k in range(gpp):
            # Alternate who plays White
            i_is_white = (k % 2 == 0)
            w_cfg, b_cfg = (ci, cj) if i_is_white else (cj, ci)
            w_idx, b_idx = (i, j) if i_is_white else (j, i)

            white_ai = make_ai("W", w_cfg, args.diff, args.budget)
            black_ai = make_ai("B", b_cfg, args.diff, args.budget)

            g_start = time.time()
            winner = play_game(white_ai, black_ai)
            g_elapsed = time.time() - g_start
            game_num += 1

            # Record result from perspective of config i
            if winner == "W":
                win_idx, lose_idx = w_idx, b_idx
            elif winner == "B":
                win_idx, lose_idx = b_idx, w_idx
            else:
                win_idx = lose_idx = None  # draw

            if win_idx is not None:
                configs[win_idx].wins   += 1
                configs[lose_idx].losses += 1
                matrix[win_idx][lose_idx]["w"] += 1
                matrix[lose_idx][win_idx]["l"] += 1
            else:
                configs[i].draws += 1
                configs[j].draws += 1
                matrix[i][j]["d"] += 1
                matrix[j][i]["d"] += 1

            # Determine outcome label
            if winner == ("W" if i_is_white else "B"):
                outcome = f"{ci.name} wins"
            elif winner == ("B" if i_is_white else "W"):
                outcome = f"{cj.name} wins"
            else:
                outcome = "draw"

            elapsed = time.time() - t_start
            eta = (elapsed / game_num) * (total_games - game_num)
            print(
                f"[{game_num:>3}/{total_games}] {ci.name} vs {cj.name}  "
                f"(#{k+1}/{gpp})  W={w_cfg.name}  "
                f"winner={'draw' if winner is None else winner:<4}  "
                f"{outcome}  {g_elapsed:.0f}s  ETA {eta/60:.0f}m",
                flush=True,
            )

            game_log.append({
                "game": game_num,
                "white": w_cfg.name,
                "black": b_cfg.name,
                "winner": winner,
                "elapsed_s": round(g_elapsed, 1),
            })

        # Print standings after each pair
        _print_standings(configs)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print("=" * 70)
    print(f"FINAL STANDINGS — {total_games} games, diff={args.diff}, budget={args.budget:.1f}s")
    print("=" * 70)
    _print_standings(configs)

    print("HEAD-TO-HEAD MATRIX (row vs col: W-D-L from row's perspective)")
    _print_matrix(configs, matrix)

    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f}h)")

    # ── JSON output ───────────────────────────────────────────────────────────
    out_path = Path(args.out)
    standings = sorted(
        [
            {
                "name": c.name,
                "sentinel_gap": c.sentinel_gap,
                "vn_blend": c.vn_blend,
                "wins": c.wins,
                "draws": c.draws,
                "losses": c.losses,
                "games": c.games,
                "points": c.points,
                "pct": round(c.pct, 1),
            }
            for c in configs
        ],
        key=lambda x: x["points"],
        reverse=True,
    )
    summary = {
        "difficulty": args.diff,
        "budget_s": args.budget,
        "games_per_pair": gpp,
        "total_games": total_games,
        "total_time_min": round(total_time / 60, 1),
        "standings": standings,
        "games": game_log,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
