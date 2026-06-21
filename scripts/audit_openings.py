"""scripts/audit_openings.py — Audit opening book for color structural advantage.

For each opening line in book_openings.json, openings.json, and
learned_openings.json, plays out the line, evaluates the resulting board
position with the heuristic evaluator, and optionally simulates N games
from that position.  Writes a "favored_side" tag to each opening entry:

    "W"      — position structurally advantages White
    "B"      — position structurally advantages Black
    "equal"  — within threshold; neither side has a clear edge
    "unknown"— line is empty or could not be evaluated

The field is handled by the Opening dataclass (favored_side, default "unknown")
and is preserved through normal OpeningBook save/load cycles.  Game code does not
yet read it — integration happens in a later pass.

NOTE: This script writes to book_openings.json (normally read-only at runtime).
That restriction applies to the game process only; this script is a maintenance
tool and intentionally updates the canonical source so the tag persists if
openings.json is ever re-seeded from the book.

Usage
-----
    .venv/bin/python scripts/audit_openings.py [options]

Options
-------
  --games N       Simulate N games from each end position (default 0 = eval only)
  --diff D        Heuristic difficulty for simulation games (default 3)
  --threshold T   Eval margin for W/B vs equal (default 0.06, range [−1,1])
  --sim-margin M  Win-rate margin for sim-based classification (default 0.08)
  --dry-run       Print report; do not write files
  --seed N        RNG seed for simulation (default 42)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal


# ── Lazy heuristic evaluator (same pattern as scaffolded_encoder) ─────────────

_evaluate_fn = None


def _heuristic_eval(board, player: str) -> float:
    global _evaluate_fn
    if _evaluate_fn is None:
        import importlib.util, os, types
        if "ai" not in sys.modules:
            ai_pkg = types.ModuleType("ai")
            ai_pkg.__path__ = [str(_ROOT / "ai")]
            sys.modules["ai"] = ai_pkg
        spec = importlib.util.spec_from_file_location(
            "ai.heuristics", str(_ROOT / "ai" / "heuristics.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ai.heuristics"] = mod
        spec.loader.exec_module(mod)
        _evaluate_fn = mod.evaluate
    try:
        return float(_evaluate_fn(board, player, strength_mode=True))
    except Exception:
        return 0.0


# ── Opening line playback ─────────────────────────────────────────────────────

def apply_opening_line(line_moves: list[str]) -> tuple[Optional[BoardState], int]:
    """Apply alternating placement moves from the opening line.

    Returns (board_after, plies_applied).  On failure returns (None, plies_applied).
    When a placement forms a mill the engine requires a capture; we pick the first
    legal capture (arbitrary but deterministic for audit purposes).
    """
    board = BoardState.new_game()
    for i, sq in enumerate(line_moves):
        legal = get_all_legal_moves(board)
        mv = next((m for m in legal if m.get("to") == sq), None)
        if mv is None:
            return board, i   # couldn't apply; return last valid state
        try:
            board = board.apply_move(mv)
        except Exception:
            return None, i
    return board, len(line_moves)


# ── Game simulation ───────────────────────────────────────────────────────────

def _sim_games(board: BoardState, n: int, difficulty: int, rng: random.Random) -> dict:
    """Play n heuristic vs heuristic games from board; return {"W":, "B":, "D":}."""
    from learned_ai.agents.heuristic_agent import HeuristicAgent, GameAI as _GA

    TIME_BUDGET = 0.05
    MAX_PLY     = 300
    results = {"W": 0, "B": 0, "D": 0}

    for _ in range(n):
        b = board
        ply = 0
        ai_w = HeuristicAgent(
            color="W", difficulty=difficulty,
            game_ai=_GA(color="W", difficulty=difficulty, override_time_budget=TIME_BUDGET),
        )
        ai_b = HeuristicAgent(
            color="B", difficulty=difficulty,
            game_ai=_GA(color="B", difficulty=difficulty, override_time_budget=TIME_BUDGET),
        )
        winner = None
        while ply < MAX_PLY:
            terminal, w = is_terminal(b)
            if terminal:
                winner = w
                break
            agent = ai_w if b.turn == "W" else ai_b
            mv = agent.choose_move(b, top_n=2)
            if not mv:
                break
            b = b.apply_move(mv)
            ply += 1
        if winner == "W":
            results["W"] += 1
        elif winner == "B":
            results["B"] += 1
        else:
            results["D"] += 1

    return results


# ── Classification ────────────────────────────────────────────────────────────

def classify_from_eval(eval_w: float, threshold: float) -> str:
    """Classify favored_side from heuristic eval (White's perspective)."""
    if eval_w > threshold:
        return "W"
    if eval_w < -threshold:
        return "B"
    return "equal"


def classify_from_sim(sim: dict, margin: float) -> str:
    """Classify favored_side from simulation win rates."""
    total = sum(sim.values())
    if total == 0:
        return "equal"
    w_rate = sim["W"] / total
    b_rate = sim["B"] / total
    if w_rate - b_rate > margin:
        return "W"
    if b_rate - w_rate > margin:
        return "B"
    return "equal"


# ── Per-opening audit ─────────────────────────────────────────────────────────

def audit_opening(entry: dict, args: argparse.Namespace, rng: random.Random) -> dict:
    """Evaluate one opening dict; return enriched dict with favored_side."""
    line = entry.get("line_moves", [])
    oid  = entry.get("opening_id", "?")

    if not line:
        entry["favored_side"] = "unknown"
        return entry

    board, plies = apply_opening_line(line)
    if board is None:
        print(f"  {oid}: line failed at ply {plies} — marking unknown")
        entry["favored_side"] = "unknown"
        return entry

    # Static heuristic eval (from White's perspective)
    eval_w = _heuristic_eval(board, "W")
    favored = classify_from_eval(eval_w, args.threshold)

    sim_str = ""
    if args.games > 0:
        sim = _sim_games(board, args.games, args.diff, rng)
        sim_favored = classify_from_sim(sim, args.sim_margin)
        total = sum(sim.values())
        w_pct = 100 * sim["W"] / total if total else 0
        b_pct = 100 * sim["B"] / total if total else 0
        d_pct = 100 * sim["D"] / total if total else 0
        sim_str = f"  sim: W={w_pct:.0f}% B={b_pct:.0f}% D={d_pct:.0f}% → {sim_favored}"
        # Simulation overrides eval when available — empirical beats static
        favored = sim_favored

    name  = entry.get("name", oid)[:38]
    side  = entry.get("side", "?")
    stats = entry.get("outcome_stats", {})
    stat_str = f"W:{stats.get('W',0)} B:{stats.get('B',0)} D:{stats.get('D',0)}"
    print(
        f"  {oid[:32]:<32}  side={side}  eval={eval_w:+.3f}  → {favored:<5}  "
        f"plies={plies:<3}  {stat_str}{sim_str}"
    )

    entry["favored_side"] = favored
    return entry


# ── File I/O ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_json(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    book_path    = _ROOT / "data" / "openings" / "book_openings.json"
    mutable_path = _ROOT / "data" / "openings" / "openings.json"
    learned_path = _ROOT / "data" / "openings" / "learned_openings.json"

    book_data    = load_json(book_path)
    mutable_data = load_json(mutable_path)
    learned_data = load_json(learned_path)

    # Index mutable openings by opening_id for quick update
    mutable_by_id = {e.get("opening_id"): e for e in mutable_data}

    totals = {"W": 0, "B": 0, "equal": 0, "unknown": 0}

    # ── Book openings ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BOOK OPENINGS  ({len(book_data)} entries from {book_path.name})")
    print(f"{'='*70}")
    for entry in book_data:
        entry = audit_opening(entry, args, rng)
        totals[entry["favored_side"]] = totals.get(entry["favored_side"], 0) + 1
        # Mirror tag into mutable copy
        oid = entry.get("opening_id")
        if oid and oid in mutable_by_id:
            mutable_by_id[oid]["favored_side"] = entry["favored_side"]

    # ── Learned openings ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"LEARNED OPENINGS  ({len(learned_data)} entries from {learned_path.name})")
    print(f"{'='*70}")
    for entry in learned_data:
        entry = audit_opening(entry, args, rng)
        totals[entry["favored_side"]] = totals.get(entry["favored_side"], 0) + 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(totals.values())
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  W-favored : {totals.get('W', 0):3d}  ({100*totals.get('W',0)/total:.0f}%)")
    print(f"  B-favored : {totals.get('B', 0):3d}  ({100*totals.get('B',0)/total:.0f}%)")
    print(f"  Equal     : {totals.get('equal', 0):3d}  ({100*totals.get('equal',0)/total:.0f}%)")
    print(f"  Unknown   : {totals.get('unknown', 0):3d}  ({100*totals.get('unknown',0)/total:.0f}%)")

    if args.dry_run:
        print("\n[dry-run] Files not written.")
        return

    # ── Write files ───────────────────────────────────────────────────────────
    # book_openings.json normally never written at runtime, but this is a
    # maintenance tool that explicitly updates the canonical source so the tag
    # survives re-seeding of openings.json.
    save_json(book_path, book_data)
    print(f"\nWrote {book_path}")

    save_json(mutable_path, mutable_data)
    print(f"Wrote {mutable_path}")

    save_json(learned_path, learned_data)
    print(f"Wrote {learned_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Audit opening book for color advantage")
    p.add_argument("--games",      type=int,   default=0,    help="Games to simulate per opening (0 = eval only)")
    p.add_argument("--diff",       type=int,   default=3,    help="Heuristic difficulty for simulation")
    p.add_argument("--threshold",  type=float, default=0.06, help="Eval margin for W/B vs equal")
    p.add_argument("--sim-margin", type=float, default=0.08, help="Win-rate margin for sim classification")
    p.add_argument("--dry-run",    action="store_true",      help="Print report without writing files")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
