"""tools/evolve_weights.py — (1+1) Evolution Strategy for HeuristicWeights.

Runs headless self-play to hill-climb the AI's heuristic weights.  Each
generation mutates the baseline by Gaussian noise, evaluates the candidate
against the baseline in N games, and promotes if the candidate win rate
exceeds the threshold.

Results are saved to data/weights/:
  best.json              — promoted weights after every generation that passes
  checkpoint_gen<N>.json — snapshot at every promotion
  evolution_log.jsonl    — one JSON line per generation (win rate, promoted, etc.)

Usage:
  python tools/evolve_weights.py [options]

Examples:
  # Quick run — 20 generations, 16 games each, 4 parallel workers
  python tools/evolve_weights.py --generations 20 --games-per-gen 16 --parallel 4

  # Continue from best known weights
  python tools/evolve_weights.py --generations 50 --from-best --parallel 4

  # Slower but stronger evaluation (difficulty 6, 30 games)
  python tools/evolve_weights.py --difficulty 6 --games-per-gen 30 --parallel 6

# Deep evolution
eg python tools/evolve_weights.py --generations 100 --from-best --parallel 8 --games-per-gen 32 --difficulty 7

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
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ai.heuristics import HeuristicWeights

WEIGHTS_DIR = ROOT / "data" / "weights"

# Behaviour knobs — excluded from evolution (they control style, not strength)
_FIXED_FIELDS = {"make_mistakes", "opening_adherence"}

# Hard bounds for evolved weights (prevents degenerate extremes)
_WEIGHT_MIN = 1
_WEIGHT_MAX = 2000


# ── Weight serialisation ───────────────────────────────────────────────────────

def weights_to_dict(w: HeuristicWeights) -> dict:
    return asdict(w)


def weights_from_dict(d: dict) -> HeuristicWeights:
    known = {f.name for f in fields(HeuristicWeights)}
    return HeuristicWeights(**{k: v for k, v in d.items() if k in known})


def tunable_fields() -> list[str]:
    return [f.name for f in fields(HeuristicWeights) if f.name not in _FIXED_FIELDS]


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(weights: HeuristicWeights, sigma: float, rng: random.Random) -> HeuristicWeights:
    """Return a new HeuristicWeights with each tunable field perturbed by sigma * value."""
    d = weights_to_dict(weights)
    for name in tunable_fields():
        val = d[name]
        noise = rng.gauss(0, max(1.0, abs(val) * sigma))
        d[name] = max(_WEIGHT_MIN, min(_WEIGHT_MAX, int(round(val + noise))))
    return weights_from_dict(d)


# ── Single game (runs in subprocess via ProcessPoolExecutor) ──────────────────

def _play_one_game(white_w_dict: dict, black_w_dict: dict, difficulty: int) -> str | None:
    """
    Play one complete fast game. Returns winner 'W', 'B', or None (draw).
    Must be a top-level function so ProcessPoolExecutor can pickle it.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from collections import Counter
    from ai.heuristics import HeuristicWeights
    from ai.game_ai import GameAI
    from game.game_engine import GameEngine
    from game.rules import get_game_phase

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

    return engine.winner  # 'W', 'B', or None


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
    # First half: candidate=White, baseline=Black
    # Second half: baseline=White, candidate=Black
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

def _delta_summary(candidate: HeuristicWeights, baseline: HeuristicWeights) -> str:
    cd = weights_to_dict(candidate)
    bd = weights_to_dict(baseline)
    deltas = []
    for name in tunable_fields():
        diff = cd[name] - bd[name]
        if abs(diff) > 5:
            deltas.append(f"{name}:{diff:+d}")
    return "  ".join(deltas[:6]) or "(small deltas)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve HeuristicWeights via (1+1)-ES self-play hill-climbing"
    )
    parser.add_argument("--generations",   type=int,   default=30,   help="Number of evolution generations")
    parser.add_argument("--games-per-gen", type=int,   default=20,   help="Games per evaluation (rounded to even)")
    parser.add_argument("--difficulty",    type=int,   default=5,    help="Search difficulty for both engines (1–10)")
    parser.add_argument("--sigma",         type=float, default=0.12, help="Gaussian noise relative to weight value (0–1)")
    parser.add_argument("--threshold",     type=float, default=0.55, help="Candidate win rate required to promote (0.5–1)")
    parser.add_argument("--parallel",      type=int,   default=4,    help="Parallel game workers")
    parser.add_argument("--from-best",     action="store_true",      help="Start from data/weights/best.json if present")
    parser.add_argument("--seed",          type=int,   default=None, help="RNG seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.from_best:
        loaded = load_best()
        baseline = loaded or HeuristicWeights()
        source   = f"data/weights/best.json" if loaded else "coded defaults (best.json not found)"
    else:
        baseline = HeuristicWeights()
        source   = "coded defaults"

    games = args.games_per_gen + (args.games_per_gen % 2)  # ensure even
    log_path = WEIGHTS_DIR / "evolution_log.jsonl"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nNine Men's Morris — Weight Evolution  ({args.difficulty})")
    print(f"  Generations : {args.generations}")
    print(f"  Games/gen   : {games}  ({args.parallel} workers, diff {args.difficulty})")
    print(f"  Sigma       : {args.sigma:.0%}  |  Threshold: {args.threshold:.0%}")
    print(f"  Baseline    : {source}")
    print()

    promotions  = 0
    best_rate   = 0.5
    best_ever   = deepcopy(baseline)

    for gen in range(1, args.generations + 1):
        t0        = time.perf_counter()
        candidate = mutate(baseline, args.sigma, rng)
        win_rate  = evaluate(candidate, baseline, games, args.difficulty, args.parallel)
        elapsed   = time.perf_counter() - t0

        promoted = win_rate >= args.threshold
        tag      = "PROMOTED" if promoted else "rejected"
        print(f"  Gen {gen:3d}/{args.generations}  wr={win_rate:.3f}  [{tag}]  {elapsed:.1f}s")

        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "gen":       gen,
                "win_rate":  round(win_rate, 4),
                "promoted":  promoted,
                "elapsed_s": round(elapsed, 1),
                "weights":   weights_to_dict(candidate),
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

    print(f"\nDone — {promotions}/{args.generations} promotions.")
    print(f"Best win rate: {best_rate:.3f}")
    save(best_ever, "best")
    print(f"Saved → {WEIGHTS_DIR / 'best.json'}")
    print()
    print("To activate evolved weights in the web server, restart the server.")
    print("The server auto-loads data/weights/best.json on startup.")


if __name__ == "__main__":
    main()
