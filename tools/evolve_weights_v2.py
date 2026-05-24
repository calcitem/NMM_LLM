"""tools/evolve_weights_v2.py — Per-personality era-aware (1+1)-ES.

Evolves each personality's weight overrides independently using the same
era-aware (1+1)-ES algorithm as evolve_weights.py.

Architecture note
-----------------
Personalities are thin overrides on top of best.json (matching the server's
runtime merge order in web/app.py):

    HeuristicWeights defaults  ←  best.json  ←  personality overrides

Only the fields already present in a personality's JSON file are mutated.
All other fields inherit from best.json as they would at runtime.
make_mistakes and opening_adherence are never mutated so each personality
keeps its behavioural character throughout evolution.

Outputs per personality
-----------------------
  data/personalities/{name}.json          — updated on every promotion
  data/weights/personalities/{name}_log.jsonl
  data/weights/personalities/{name}_checkpoint_gen{N:04d}.json

Usage
-----
  python tools/evolve_weights_v2.py [options]

Examples
--------
  # All personalities, 30 gens each, 4 workers
  python tools/evolve_weights_v2.py --generations 30 --parallel 4

  # Selected personalities only
  python tools/evolve_weights_v2.py --personalities aggressive,defensive --generations 50

  # Quick test run
  python tools/evolve_weights_v2.py --generations 10 --games-per-gen 12 --parallel 4

  # Ben's full run
  python tools/evolve_weights_v2.py --generations 100 --parallel 8 --games-per-gen 32 \\
      --difficulty 7 --era-size 10 --bias-strength 0.3 --era-top-k 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ai.heuristics import HeuristicWeights

PERSONALITIES_DIR = ROOT / "data" / "personalities"
WEIGHTS_DIR       = ROOT / "data" / "weights" / "personalities"

# Never mutated — these define each personality's behavioural character
_FIXED_FIELDS = {"make_mistakes", "opening_adherence"}

# Personalities skipped by default (user-curated, not evolved)
_DEFAULT_SKIP = {"custom"}

_WEIGHT_MIN = 1
_WEIGHT_MAX = 2000

# Rechenberg 1/5 rule multipliers
_SIGMA_UP   = 1.22
_SIGMA_DOWN = 0.82
_SIGMA_MIN  = 0.02
_SIGMA_MAX  = 0.50


# ── Weight helpers ────────────────────────────────────────────────────────────

def weights_to_dict(w: HeuristicWeights) -> dict:
    return asdict(w)


def weights_from_dict(d: dict) -> HeuristicWeights:
    known = {f.name for f in fields(HeuristicWeights)}
    return HeuristicWeights(**{k: v for k, v in d.items() if k in known})


def load_best() -> HeuristicWeights:
    path = ROOT / "data" / "weights" / "best.json"
    if path.exists():
        try:
            return weights_from_dict(json.loads(path.read_text()))
        except Exception:
            pass
    return HeuristicWeights()


def load_personality(name: str) -> dict | None:
    path = PERSONALITIES_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_personality(name: str, overrides: dict) -> None:
    path = PERSONALITIES_DIR / f"{name}.json"
    path.write_text(json.dumps(overrides, indent=2))


def save_checkpoint(name: str, overrides: dict, gen: int) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / f"{name}_checkpoint_gen{gen:04d}.json"
    path.write_text(json.dumps(overrides, indent=2))


def personality_tunable(p_dict: dict) -> list[str]:
    """Fields present in the personality file that are eligible for mutation."""
    return [k for k in p_dict if k not in _FIXED_FIELDS]


def merged_weights(best: HeuristicWeights, p_overrides: dict) -> HeuristicWeights:
    """Reproduce the server merge: defaults ← best.json ← personality overrides."""
    d = weights_to_dict(best)
    d.update(p_overrides)
    return weights_from_dict(d)


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(
    p_overrides: dict,
    tunable: list[str],
    sigma: float,
    rng: random.Random,
    bias: dict[str, float] | None = None,
    bias_strength: float = 0.0,
) -> dict:
    """
    Return a mutated copy of p_overrides, touching only tunable fields.
    All other keys (including fixed fields) are passed through unchanged.
    """
    result = dict(p_overrides)
    for name in tunable:
        val   = result[name]
        noise = rng.gauss(0, max(1.0, abs(val) * sigma))
        nudge = (bias[name] * bias_strength) if (bias and name in bias) else 0.0
        result[name] = max(_WEIGHT_MIN, min(_WEIGHT_MAX, int(round(val + noise + nudge))))
    return result


# ── Warm restart ──────────────────────────────────────────────────────────────

def warm_restart(
    baseline_overrides: dict,
    era_best_overrides: dict,
    tunable: list[str],
    blend: float,
) -> dict:
    result = dict(baseline_overrides)
    for name in tunable:
        mixed = (1.0 - blend) * baseline_overrides[name] + blend * era_best_overrides[name]
        result[name] = max(_WEIGHT_MIN, min(_WEIGHT_MAX, int(round(mixed))))
    return result


# ── Era helpers ───────────────────────────────────────────────────────────────

def compute_era_bias(
    baseline_overrides: dict,
    candidates: list[dict],
    tunable: list[str],
    top_k: int,
) -> dict[str, float]:
    top = sorted(candidates, key=lambda c: c["win_rate"], reverse=True)[:top_k]
    if not top:
        return {}
    return {
        name: sum(c["overrides"][name] - baseline_overrides[name] for c in top) / len(top)
        for name in tunable
    }


def adapt_sigma(sigma: float, success_count: int, era_size: int) -> float:
    rate = success_count / max(1, era_size)
    if rate > 0.2:
        new_sigma = sigma * _SIGMA_UP
    elif rate < 0.2:
        new_sigma = sigma * _SIGMA_DOWN
    else:
        new_sigma = sigma
    return max(_SIGMA_MIN, min(_SIGMA_MAX, new_sigma))


# ── Single game (subprocess-safe) ─────────────────────────────────────────────

def _play_one_game(white_w_dict: dict, black_w_dict: dict, difficulty: int) -> str | None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from collections import Counter
    from ai.heuristics import HeuristicWeights
    from ai.game_ai import GameAI
    from game.game_engine import GameEngine
    from dataclasses import fields as dc_fields

    def _from_dict(d):
        known = {f.name for f in dc_fields(HeuristicWeights)}
        return HeuristicWeights(**{k: v for k, v in d.items() if k in known})

    engine   = GameEngine(human_color="W")
    white_ai = GameAI(color="W", difficulty=difficulty, weights=_from_dict(white_w_dict), blunder_probability=0.0)
    black_ai = GameAI(color="B", difficulty=difficulty, weights=_from_dict(black_w_dict), blunder_probability=0.0)

    fen_counts: Counter = Counter()
    moves_since_capture = 0
    move_count = 0

    while not engine.finished and move_count < 300:
        board = engine.board
        fen   = board.to_fen_string()
        fen_counts[fen] += 1
        if fen_counts[fen] >= 3 or moves_since_capture >= 100:
            break

        ai   = white_ai if board.turn == "W" else black_ai
        move = ai.choose_move(board, top_n=2, fast_early_game=True)

        pieces_before = sum(1 for v in board.positions.values() if v)
        engine.apply_move(move)
        pieces_after  = sum(1 for v in engine.board.positions.values() if v)

        moves_since_capture = 0 if pieces_after < pieces_before else moves_since_capture + 1
        move_count += 1

    return engine.winner


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    cand_weights: HeuristicWeights,
    base_weights: HeuristicWeights,
    games: int,
    difficulty: int,
    n_workers: int,
) -> float:
    half   = games // 2
    cand_d = weights_to_dict(cand_weights)
    base_d = weights_to_dict(base_weights)

    tasks: list[tuple[dict, dict, bool]] = (
        [(cand_d, base_d, True)]  * half +
        [(base_d, cand_d, False)] * (games - half)
    )

    results: list[tuple[str | None, bool]] = []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {
                pool.submit(_play_one_game, w, b, difficulty): cand_is_w
                for w, b, cand_is_w in tasks
            }
            for fut, cand_is_w in futs.items():
                results.append((fut.result(), cand_is_w))
    else:
        for w, b, cand_is_w in tasks:
            results.append((_play_one_game(w, b, difficulty), cand_is_w))

    wins = 0.0
    for winner, cand_is_white in results:
        if winner is None:
            wins += 0.5
        elif (winner == "W") == cand_is_white:
            wins += 1.0
    return wins / games


# ── Delta summary ─────────────────────────────────────────────────────────────

def _delta_summary(cand: dict, base: dict, tunable: list[str], n: int = 6) -> str:
    deltas = [
        f"{name}:{cand[name] - base[name]:+d}"
        for name in tunable
        if abs(cand[name] - base[name]) > 5
    ]
    return "  ".join(deltas[:n]) or "(small deltas)"


# ── Per-personality evolution ─────────────────────────────────────────────────

def evolve_personality(
    name: str,
    p_overrides: dict,
    best: HeuristicWeights,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict:
    """
    Run era-aware (1+1)-ES on one personality.  Returns the best overrides found.
    """
    tunable  = personality_tunable(p_overrides)
    games    = args.games_per_gen + (args.games_per_gen % 2)
    log_path = WEIGHTS_DIR / f"{name}_log.jsonl"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    sigma          = args.sigma
    baseline_ov    = dict(p_overrides)
    best_ov        = dict(p_overrides)
    best_rate      = 0.5
    promotions     = 0

    era_candidates: list[dict] = []
    era_bias:       dict[str, float] = {}
    era_successes   = 0
    era_number      = 1

    print(f"\n  [{name}] tunable fields: {len(tunable)}  |  games/gen: {games}  |  gens: {args.generations}")

    for gen in range(1, args.generations + 1):
        t0       = time.perf_counter()
        cand_ov  = mutate(baseline_ov, tunable, sigma, rng,
                          bias=era_bias, bias_strength=args.bias_strength)

        base_full = merged_weights(best, baseline_ov)
        cand_full = merged_weights(best, cand_ov)
        win_rate  = evaluate(cand_full, base_full, games, args.difficulty, args.parallel)
        elapsed   = time.perf_counter() - t0

        promoted = win_rate >= args.threshold
        tag      = "PROMOTED" if promoted else "rejected"
        print(f"    Gen {gen:3d}/{args.generations}  wr={win_rate:.3f}  σ={sigma:.3f}"
              f"  era={era_number}  [{tag}]  {elapsed:.1f}s")

        era_candidates.append({"gen": gen, "win_rate": win_rate, "overrides": cand_ov})
        if promoted:
            era_successes += 1

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "gen":       gen,
                "era":       era_number,
                "win_rate":  round(win_rate, 4),
                "promoted":  promoted,
                "sigma":     round(sigma, 4),
                "elapsed_s": round(elapsed, 1),
                "overrides": cand_ov,
            }) + "\n")

        if promoted:
            promotions  += 1
            baseline_ov  = cand_ov
            print(f"          ↳ {_delta_summary(cand_ov, best_ov, tunable)}")
            save_personality(name, baseline_ov)
            save_checkpoint(name, baseline_ov, gen)
            if win_rate > best_rate:
                best_rate = win_rate
                best_ov   = dict(cand_ov)

        # ── Era boundary ──────────────────────────────────────────────────────
        if gen % args.era_size == 0 or gen == args.generations:
            old_sigma = sigma
            sigma = adapt_sigma(sigma, era_successes, len(era_candidates))

            if era_candidates:
                top_era    = max(era_candidates, key=lambda c: c["win_rate"])
                sigma_arrow = "↑" if sigma > old_sigma else ("↓" if sigma < old_sigma else "→")
                print(f"\n    ── Era {era_number} ({len(era_candidates)} gens)  "
                      f"best_wr={top_era['win_rate']:.3f}  "
                      f"promotions={era_successes}  "
                      f"σ {old_sigma:.3f}→{sigma:.3f}{sigma_arrow}")

                if era_successes == 0:
                    baseline_ov = warm_restart(
                        baseline_ov, top_era["overrides"], tunable, blend=args.warm_blend
                    )
                    print(f"    warm-restart blend={args.warm_blend:.0%} toward era best")

                era_bias = compute_era_bias(baseline_ov, era_candidates, tunable, top_k=args.era_top_k)
                print()

            era_candidates = []
            era_successes  = 0
            era_number    += 1

    print(f"  [{name}] done — {promotions}/{args.generations} promotions  best_wr={best_rate:.3f}")
    return best_ov


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve per-personality weights via era-aware (1+1)-ES"
    )
    parser.add_argument("--personalities", default="",
                        help="Comma-separated list of personalities to train "
                             "(default: all except custom)")
    parser.add_argument("--skip",          default="custom",
                        help="Comma-separated personalities to skip (default: custom)")
    parser.add_argument("--generations",   type=int,   default=30,
                        help="Generations per personality (default: 30)")
    parser.add_argument("--games-per-gen", type=int,   default=20,
                        help="Games per evaluation, rounded to even (default: 20)")
    parser.add_argument("--difficulty",    type=int,   default=5,
                        help="Search difficulty 1–10 (default: 5)")
    parser.add_argument("--sigma",         type=float, default=0.12,
                        help="Initial Gaussian noise relative to weight value (default: 0.12)")
    parser.add_argument("--threshold",     type=float, default=0.55,
                        help="Win rate required to promote (default: 0.55)")
    parser.add_argument("--parallel",      type=int,   default=4,
                        help="Parallel game workers (default: 4)")
    parser.add_argument("--era-size",      type=int,   default=5,
                        help="Generations per era for sigma adaptation (default: 5)")
    parser.add_argument("--era-top-k",     type=int,   default=3,
                        help="Top-K era candidates for directional bias (default: 3)")
    parser.add_argument("--bias-strength", type=float, default=0.3,
                        help="Fraction of era bias added to mutations (default: 0.3)")
    parser.add_argument("--warm-blend",    type=float, default=0.25,
                        help="Blend toward era best on failed era (default: 0.25)")
    parser.add_argument("--seed",          type=int,   default=None,
                        help="RNG seed for reproducibility")
    args = parser.parse_args()

    rng  = random.Random(args.seed)
    best = load_best()

    # Resolve personality list
    skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
    if args.personalities:
        names = [n.strip() for n in args.personalities.split(",") if n.strip()]
    else:
        names = sorted(
            p.stem for p in PERSONALITIES_DIR.glob("*.json")
            if p.stem not in skip_set
        )

    if not names:
        print("No personalities found to evolve.")
        sys.exit(1)

    # Pre-load and validate
    personality_data: dict[str, dict] = {}
    for name in names:
        p = load_personality(name)
        if p is None:
            print(f"  WARNING: {name}.json not found — skipping.")
            continue
        tunable = personality_tunable(p)
        if not tunable:
            print(f"  WARNING: {name} has no tunable fields — skipping.")
            continue
        personality_data[name] = p

    if not personality_data:
        print("No valid personalities to evolve.")
        sys.exit(1)

    total_gens = args.generations * len(personality_data)
    games_each = args.games_per_gen + (args.games_per_gen % 2)

    print(f"\nNine Men's Morris — Per-Personality Weight Evolution")
    print(f"  Personalities : {', '.join(personality_data)}")
    print(f"  Generations   : {args.generations} per personality  ({total_gens} total)")
    print(f"  Games/gen     : {games_each}  ({args.parallel} workers, diff {args.difficulty})")
    print(f"  Sigma         : {args.sigma:.0%}  |  Threshold: {args.threshold:.0%}")
    print(f"  Era size      : {args.era_size}  |  Top-K: {args.era_top_k}"
          f"  |  Bias: {args.bias_strength:.0%}  |  Warm blend: {args.warm_blend:.0%}")
    print(f"  best.json     : {'loaded' if (ROOT / 'data/weights/best.json').exists() else 'not found (using defaults)'}")
    print()

    results: dict[str, tuple[dict, float]] = {}

    for name, p_overrides in personality_data.items():
        print(f"{'='*60}")
        print(f"  Evolving: {name}")
        print(f"{'='*60}")
        best_ov = evolve_personality(name, p_overrides, best, args, rng)
        save_personality(name, best_ov)
        results[name] = best_ov
        print(f"  Saved → {PERSONALITIES_DIR / f'{name}.json'}")

    print(f"\n{'='*60}")
    print(f"  All done.")
    for name in results:
        print(f"    {name:20s} → data/personalities/{name}.json")
    print()
    print("  Restart the web server to pick up updated personalities.")
    print()


if __name__ == "__main__":
    main()
