"""tools/evolve_weights.py — Era-aware (1+1) Evolution Strategy for HeuristicWeights.

Runs headless self-play to hill-climb the AI's heuristic weights.  Each
generation mutates the baseline by Gaussian noise, evaluates the candidate
against the baseline in N games, and promotes if the candidate win rate
exceeds the threshold.

Era system (--era-size, default 5):
  Every ERA_SIZE generations the script pauses and reviews all candidates
  from that era.  It picks the top-K performers (by win rate), computes a
  per-field directional bias from them, and adapts sigma using the 1/5
  success rule.  The next era then starts from a "warm restart" blending
  the current baseline with the era's best candidate, giving evolution a
  kick-start toward directions that were already locally promising.

Sigma adaptation (1/5 success rule, Rechenberg 1973):
  - If success rate > 1/5 over the era  → increase sigma (multiply by 1.22)
  - If success rate < 1/5               → decrease sigma (multiply by 0.82)
  - Otherwise                           → leave sigma unchanged
  This is the classical method proven to maintain progress throughout a run
  and prevent premature convergence [Rechenberg 1973, Auger 2009].

Directional bias:
  From the top-K era candidates (by win rate), compute the mean signed delta
  per field vs. the baseline.  The next era's mutations are nudged in that
  direction by a configurable fraction (--bias-strength, default 0.3).

Results are saved to data/weights/:
  best.json                — promoted weights after every generation that passes
  checkpoint_gen<N>.json   — snapshot at every promotion
  era_best_gen<N>.json     — era top-performer snapshot (even if not promoted)
  evolution_log.jsonl      — one JSON line per generation (win rate, promoted,
                             sigma, era, bias_nudge, etc.)

Usage:
  python tools/evolve_weights.py [options]

Examples:
  # Quick run — 20 generations, 16 games each, 4 parallel workers
  python tools/evolve_weights.py --generations 20 --games-per-gen 16 --parallel 4

  # Continue from best known weights
  python tools/evolve_weights.py --generations 50 --from-best --parallel 4

  # Slower but stronger evaluation (difficulty 6, 30 games)
  python tools/evolve_weights.py --difficulty 6 --games-per-gen 30 --parallel 6

  # Deep overnight evolution with era system
  python tools/evolve_weights.py --generations 100 --from-best --parallel 8 \\
      --games-per-gen 32 --difficulty 7 --era-size 5 --bias-strength 0.3

  # Large era for more stable sigma adaptation
  python tools/evolve_weights.py --generations 100 --from-best --parallel 8 \\
      --games-per-gen 16 --difficulty 7 --era-size 10 --era-top-k 3

	# Ben settings
	python tools/evolve_weights_v2.py --generations 100 --from-best --parallel 8  --games-per-gen 32 --difficulty 7 
	--era-size 10 --bias-strength 0.3 --era-top-k 3


"""
from __future__ import annotations

import argparse
import json
import math
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

WEIGHTS_DIR = ROOT / "data" / "weights"

# Behaviour knobs — excluded from evolution (they control style, not strength)
_FIXED_FIELDS = {"make_mistakes", "opening_adherence"}

# Hard bounds for evolved weights (prevents degenerate extremes)
_WEIGHT_MIN = 1
_WEIGHT_MAX = 2000

# 1/5 success rule multipliers (Rechenberg 1973)
_SIGMA_UP   = 1.22   # multiply sigma when success rate > 0.2
_SIGMA_DOWN = 0.82   # multiply sigma when success rate < 0.2
_SIGMA_MIN  = 0.02
_SIGMA_MAX  = 0.50


# ── Weight serialisation ───────────────────────────────────────────────────────

def weights_to_dict(w: HeuristicWeights) -> dict:
    return asdict(w)


def weights_from_dict(d: dict) -> HeuristicWeights:
    known = {f.name for f in fields(HeuristicWeights)}
    return HeuristicWeights(**{k: v for k, v in d.items() if k in known})


def tunable_fields() -> list[str]:
    return [f.name for f in fields(HeuristicWeights) if f.name not in _FIXED_FIELDS]


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(
    weights: HeuristicWeights,
    sigma: float,
    rng: random.Random,
    bias: dict[str, float] | None = None,
    bias_strength: float = 0.0,
) -> HeuristicWeights:
    """
    Return a new HeuristicWeights perturbed by Gaussian noise.

    bias:          per-field mean delta from top-K era candidates (optional)
    bias_strength: fraction [0, 1] of the bias nudge added on top of noise
    """
    d = weights_to_dict(weights)
    for name in tunable_fields():
        val  = d[name]
        noise = rng.gauss(0, max(1.0, abs(val) * sigma))
        nudge = (bias[name] * bias_strength) if (bias and name in bias) else 0.0
        d[name] = max(_WEIGHT_MIN, min(_WEIGHT_MAX, int(round(val + noise + nudge))))
    return weights_from_dict(d)


# ── Warm restart: blend baseline toward era best ──────────────────────────────

def warm_restart(
    baseline: HeuristicWeights,
    era_best: HeuristicWeights,
    blend: float,
) -> HeuristicWeights:
    """
    Return a new weight set that is (1-blend)*baseline + blend*era_best.
    blend=0.0 → pure baseline, blend=1.0 → pure era_best.
    Only applied when the era produced no promotion, to nudge baseline
    toward a direction that was demonstrably better (even if sub-threshold).
    """
    bd = weights_to_dict(baseline)
    ed = weights_to_dict(era_best)
    result = {}
    for name in tunable_fields():
        mixed = (1.0 - blend) * bd[name] + blend * ed[name]
        result[name] = max(_WEIGHT_MIN, min(_WEIGHT_MAX, int(round(mixed))))
    # Keep fixed fields from baseline
    for name in _FIXED_FIELDS:
        result[name] = bd[name]
    return weights_from_dict(result)


# ── Directional bias from era top-K ──────────────────────────────────────────

def compute_era_bias(
    baseline: HeuristicWeights,
    candidates: list[dict],          # list of {"weights": dict, "win_rate": float}
    top_k: int,
) -> dict[str, float]:
    """
    Compute the mean signed delta per field between the top-K candidates
    and the baseline.  Used to nudge the next era's mutations in a
    promising direction even when no candidate crossed the threshold.
    """
    sorted_cands = sorted(candidates, key=lambda c: c["win_rate"], reverse=True)
    top = sorted_cands[:top_k]
    if not top:
        return {}
    bd = weights_to_dict(baseline)
    bias: dict[str, float] = {}
    for name in tunable_fields():
        deltas = [c["weights"][name] - bd[name] for c in top]
        bias[name] = sum(deltas) / len(deltas)
    return bias


# ── 1/5 success rule sigma adaptation ────────────────────────────────────────

def adapt_sigma(sigma: float, success_count: int, era_size: int) -> float:
    """
    Classic Rechenberg 1/5 rule:
      success_rate > 1/5 → increase sigma
      success_rate < 1/5 → decrease sigma
    """
    rate = success_count / max(1, era_size)
    if rate > 0.2:
        new_sigma = sigma * _SIGMA_UP
    elif rate < 0.2:
        new_sigma = sigma * _SIGMA_DOWN
    else:
        new_sigma = sigma
    return max(_SIGMA_MIN, min(_SIGMA_MAX, new_sigma))


# ── Single game (runs in subprocess via ProcessPoolExecutor) ──────────────────

def _play_one_game(white_w_dict: dict, black_w_dict: dict, difficulty: int) -> str | None:
    """Play one complete fast game. Returns winner 'W', 'B', or None (draw)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from collections import Counter
    from ai.heuristics import HeuristicWeights
    from ai.game_ai import GameAI
    from game.game_engine import GameEngine

    white_w = weights_from_dict(white_w_dict)
    black_w = weights_from_dict(black_w_dict)

    engine   = GameEngine(human_color="W")
    white_ai = GameAI(color="W", difficulty=difficulty, weights=white_w, blunder_probability=0.0)
    black_ai = GameAI(color="B", difficulty=difficulty, weights=black_w, blunder_probability=0.0)

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
    candidate: HeuristicWeights,
    baseline: HeuristicWeights,
    games: int,
    difficulty: int,
    n_workers: int,
) -> float:
    """
    Play `games` games with colours swapped symmetrically.
    Returns candidate win rate in [0, 1]; draws count as 0.5.
    """
    half = games // 2
    cand_d = weights_to_dict(candidate)
    base_d = weights_to_dict(baseline)

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


# ── Persistence ───────────────────────────────────────────────────────────────

def load_best() -> HeuristicWeights | None:
    path = WEIGHTS_DIR / "best.json"
    if path.exists():
        try:
            return weights_from_dict(json.loads(path.read_text()))
        except Exception:
            return None
    return None


def save(w: HeuristicWeights, name: str) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    (WEIGHTS_DIR / f"{name}.json").write_text(json.dumps(weights_to_dict(w), indent=2))


# ── Delta summary ─────────────────────────────────────────────────────────────

def _delta_summary(candidate: HeuristicWeights, baseline: HeuristicWeights, n: int = 6) -> str:
    cd = weights_to_dict(candidate)
    bd = weights_to_dict(baseline)
    deltas = []
    for name in tunable_fields():
        diff = cd[name] - bd[name]
        if abs(diff) > 5:
            deltas.append(f"{name}:{diff:+d}")
    return "  ".join(deltas[:n]) or "(small deltas)"


# ── Era summary ───────────────────────────────────────────────────────────────

def _era_summary(
    era_candidates: list[dict],
    threshold: float,
    new_sigma: float,
    old_sigma: float,
    warm_blend: float,
    promoted_this_era: int,
) -> None:
    n = len(era_candidates)
    if not n:
        return
    rates = [c["win_rate"] for c in era_candidates]
    best_rate = max(rates)
    mean_rate = sum(rates) / n
    sigma_arrow = "↑" if new_sigma > old_sigma else ("↓" if new_sigma < old_sigma else "→")

    print()
    print(f"  ── Era review ({n} gens) ─────────────────────────────────────────")
    print(f"     Win rates  : mean={mean_rate:.3f}  best={best_rate:.3f}  threshold={threshold:.2f}")
    print(f"     Promotions : {promoted_this_era}")
    sigma_str = f"{old_sigma:.3f} → {new_sigma:.3f} {sigma_arrow}"
    print(f"     Sigma      : {sigma_str}  (1/5 rule)")
    if promoted_this_era == 0:
        print(f"     Warm restart blend={warm_blend:.0%} toward era best")
    print(f"  ─────────────────────────────────────────────────────────────────")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve HeuristicWeights via era-aware (1+1)-ES with adaptive sigma"
    )
    parser.add_argument("--generations",   type=int,   default=30,   help="Total number of generations")
    parser.add_argument("--games-per-gen", type=int,   default=20,   help="Games per evaluation (rounded to even)")
    parser.add_argument("--difficulty",    type=int,   default=5,    help="Search difficulty for both engines (1–10)")
    parser.add_argument("--sigma",         type=float, default=0.12, help="Initial Gaussian noise relative to weight value (0–1)")
    parser.add_argument("--threshold",     type=float, default=0.55, help="Win rate required to promote (0.5–1)")
    parser.add_argument("--parallel",      type=int,   default=4,    help="Parallel game workers")
    parser.add_argument("--from-best",     action="store_true",      help="Start from data/weights/best.json if present")
    parser.add_argument("--seed",          type=int,   default=None, help="RNG seed for reproducibility")
    # Era options
    parser.add_argument("--era-size",      type=int,   default=5,    help="Generations per era (sigma adaptation + warm restart)")
    parser.add_argument("--era-top-k",     type=int,   default=3,    help="Top-K era candidates used to compute directional bias")
    parser.add_argument("--bias-strength", type=float, default=0.3,  help="Fraction of era directional bias added to next-era mutations (0–1)")
    parser.add_argument("--warm-blend",    type=float, default=0.25, help="Blend fraction toward era best on failed-era warm restart (0–1)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.from_best:
        loaded = load_best()
        baseline = loaded or HeuristicWeights()
        source   = "data/weights/best.json" if loaded else "coded defaults (best.json not found)"
    else:
        baseline = HeuristicWeights()
        source   = "coded defaults"

    games    = args.games_per_gen + (args.games_per_gen % 2)
    log_path = WEIGHTS_DIR / "evolution_log.jsonl"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    sigma = args.sigma

    print(f"\nNine Men's Morris — Era-Aware Weight Evolution  ({args.difficulty})")
    print(f"  Generations : {args.generations}")
    print(f"  Games/gen   : {games}  ({args.parallel} workers, diff {args.difficulty})")
    print(f"  Sigma       : {sigma:.0%}  (adaptive via 1/5 rule)  |  Threshold: {args.threshold:.0%}")
    print(f"  Era size    : {args.era_size} gens  |  Top-K bias: {args.era_top_k}  |  Bias strength: {args.bias_strength:.0%}")
    print(f"  Warm blend  : {args.warm_blend:.0%} (applied to baseline when era has no promotions)")
    print(f"  Baseline    : {source}")
    print()

    promotions       = 0
    best_rate        = 0.5
    best_ever        = deepcopy(baseline)

    # Era state
    era_candidates: list[dict] = []
    era_bias: dict[str, float] = {}
    era_successes   = 0
    era_number      = 1

    for gen in range(1, args.generations + 1):
        t0        = time.perf_counter()
        candidate = mutate(baseline, sigma, rng, bias=era_bias, bias_strength=args.bias_strength)
        win_rate  = evaluate(candidate, baseline, games, args.difficulty, args.parallel)
        elapsed   = time.perf_counter() - t0

        promoted = win_rate >= args.threshold
        tag      = "PROMOTED" if promoted else "rejected"
        print(f"  Gen {gen:3d}/{args.generations}  wr={win_rate:.3f}  σ={sigma:.3f}  era={era_number}  [{tag}]  {elapsed:.1f}s")

        # Track era candidates
        era_candidates.append({
            "gen":      gen,
            "win_rate": win_rate,
            "weights":  weights_to_dict(candidate),
        })
        if promoted:
            era_successes += 1

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "gen":         gen,
                "era":         era_number,
                "win_rate":    round(win_rate, 4),
                "promoted":    promoted,
                "sigma":       round(sigma, 4),
                "elapsed_s":   round(elapsed, 1),
                "weights":     weights_to_dict(candidate),
            }) + "\n")

        if promoted:
            promotions += 1
            baseline    = candidate
            print(f"          ↳ {_delta_summary(candidate, best_ever)}")
            save(baseline, "best")
            save(baseline, f"checkpoint_gen{gen:04d}")
            if win_rate > best_rate:
                best_rate = win_rate
                best_ever = deepcopy(candidate)

        # ── Era boundary ─────────────────────────────────────────────────────
        if gen % args.era_size == 0 or gen == args.generations:
            old_sigma = sigma

            # 1. Adapt sigma via 1/5 success rule
            sigma = adapt_sigma(sigma, era_successes, len(era_candidates))

            # 2. Save era best (even if not promoted)
            if era_candidates:
                best_era_cand = max(era_candidates, key=lambda c: c["win_rate"])
                era_best_w    = weights_from_dict(best_era_cand["weights"])
                save(era_best_w, f"era_best_gen{gen:04d}")

                # 3. Warm restart: blend baseline toward era best if no promotion this era
                if era_successes == 0:
                    baseline = warm_restart(baseline, era_best_w, blend=args.warm_blend)

                # 4. Compute directional bias for next era
                era_bias = compute_era_bias(baseline, era_candidates, top_k=args.era_top_k)

            _era_summary(
                era_candidates,
                args.threshold,
                sigma,
                old_sigma,
                args.warm_blend,
                era_successes,
            )

            # Reset era state
            era_candidates = []
            era_successes  = 0
            era_number    += 1

    print(f"Done — {promotions}/{args.generations} promotions.")
    print(f"Best win rate: {best_rate:.3f}")
    save(best_ever, "best")
    print(f"Saved → {WEIGHTS_DIR / 'best.json'}")
    print()
    print("To activate evolved weights in the web server, restart the server.")
    print("The server auto-loads data/weights/best.json on startup.")


if __name__ == "__main__":
    main()
