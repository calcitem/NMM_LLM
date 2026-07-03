"""
tools/bench_gating.py — benchmark per-position cost of three evaluation
strategies for frequency-gated opponent move pruning.

For each of N random move-phase positions from data/human_db.sqlite, measures:
  a) Static eval  — evaluate_v2 applied to each successor board (Rust)
  b) Value net    — value_net.predict applied to each successor board (PyTorch)
  c) Sentinel     — sentinel.advise batched over all candidates (PyTorch)

Usage:
    .venv/bin/python tools/bench_gating.py [--n 100] [--phase move]
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves


# ── Helpers ──────────────────────────────────────────────────────────────────

def state_key_to_board(state_key: str) -> BoardState | None:
    """Convert a human_db state_key back to a BoardState."""
    try:
        parts = state_key.split("|")
        # format: canon|turn|phase|placed_w|placed_b|on_w|on_b
        canon, turn = parts[0], parts[1]
        placed_w, placed_b = int(parts[3]), int(parts[4])
        on_w, on_b = int(parts[5]), int(parts[6])
        positions = {POSITIONS[i]: (canon[i] if canon[i] != "." else "") for i in range(24)}
        from game.board import hash_board
        b = BoardState(
            positions=positions,
            turn=turn,
            pieces_on_board={"W": on_w, "B": on_b},
            pieces_placed={"W": placed_w, "B": placed_b},
            pieces_captured={"W": placed_b - on_b, "B": placed_w - on_w},
            hash_key=0,
        )
        b.hash_key = hash_board(b)
        return b
    except Exception:
        return None


def sample_positions(db_path: Path, n: int, phase_filter: str) -> list[BoardState]:
    """Sample n positions from the DB matching phase_filter."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Filter: phase in state_key, at least 5 games seen, not trivial
    pattern = f"%|{phase_filter}|%"
    cur.execute(
        "SELECT state_key FROM positions "
        "WHERE total_games >= 5 AND state_key LIKE ? "
        "ORDER BY RANDOM() LIMIT ?",
        (pattern, n * 3),  # oversample to account for parse failures
    )
    rows = cur.fetchall()
    conn.close()

    boards = []
    for (sk,) in rows:
        b = state_key_to_board(sk)
        if b is None:
            continue
        moves = get_all_legal_moves(b)
        if len(moves) < 3:
            continue  # not interesting enough
        boards.append(b)
        if len(boards) >= n:
            break

    print(f"Sampled {len(boards)} {phase_filter}-phase positions with ≥3 legal moves")
    return boards


# ── Benchmark runners ─────────────────────────────────────────────────────────

def bench_static_eval(boards: list[BoardState]) -> dict:
    """Time static evaluate_v2 across all legal successors of each position."""
    from ai.heuristics import evaluate_v2

    times = []
    move_counts = []
    for b in boards:
        moves = get_all_legal_moves(b)
        move_counts.append(len(moves))
        t0 = time.perf_counter()
        for mv in moves:
            succ = b.apply_move(mv)
            evaluate_v2(succ, b.turn)
        times.append(time.perf_counter() - t0)

    total_moves = sum(move_counts)
    total_time = sum(times)
    return {
        "name": "Static eval (evaluate_v2)",
        "positions": len(boards),
        "total_moves": total_moves,
        "avg_moves_per_pos": mean(move_counts),
        "total_time_ms": total_time * 1000,
        "avg_time_per_pos_ms": mean(times) * 1000,
        "median_time_per_pos_ms": median(times) * 1000,
        "avg_time_per_move_us": (total_time / total_moves) * 1_000_000,
    }


def bench_value_net(boards: list[BoardState], value_net) -> dict:
    """Time value_net.predict across all legal successors of each position."""
    times = []
    move_counts = []
    for b in boards:
        moves = get_all_legal_moves(b)
        move_counts.append(len(moves))
        t0 = time.perf_counter()
        for mv in moves:
            succ = b.apply_move(mv)
            value_net.predict(succ, b.turn)
        times.append(time.perf_counter() - t0)

    total_moves = sum(move_counts)
    total_time = sum(times)
    return {
        "name": "Value net (per-move predict)",
        "positions": len(boards),
        "total_moves": total_moves,
        "avg_moves_per_pos": mean(move_counts),
        "total_time_ms": total_time * 1000,
        "avg_time_per_pos_ms": mean(times) * 1000,
        "median_time_per_pos_ms": median(times) * 1000,
        "avg_time_per_move_us": (total_time / total_moves) * 1_000_000,
    }


def bench_sentinel(boards: list[BoardState], sentinel) -> dict:
    """Time sentinel.advise (batched) across all legal moves of each position."""
    times = []
    move_counts = []
    for b in boards:
        moves = get_all_legal_moves(b)
        candidates = [{"from": m.get("from"), "to": m["to"], "capture": m.get("capture")}
                      for m in moves]
        move_counts.append(len(candidates))
        t0 = time.perf_counter()
        sentinel.advise(b, candidates, b.turn)
        times.append(time.perf_counter() - t0)

    total_moves = sum(move_counts)
    total_time = sum(times)
    return {
        "name": "Sentinel (batched advise)",
        "positions": len(boards),
        "total_moves": total_moves,
        "avg_moves_per_pos": mean(move_counts),
        "total_time_ms": total_time * 1000,
        "avg_time_per_pos_ms": mean(times) * 1000,
        "median_time_per_pos_ms": median(times) * 1000,
        "avg_time_per_move_us": (total_time / total_moves) * 1_000_000,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_result(r: dict) -> None:
    print(f"\n{'─'*55}")
    print(f"  {r['name']}")
    print(f"{'─'*55}")
    print(f"  Positions:              {r['positions']}")
    print(f"  Total moves evaluated:  {r['total_moves']}")
    print(f"  Avg moves/position:     {r['avg_moves_per_pos']:.1f}")
    print(f"  Total time:             {r['total_time_ms']:.1f} ms")
    print(f"  Avg time/position:      {r['avg_time_per_pos_ms']:.3f} ms")
    print(f"  Median time/position:   {r['median_time_per_pos_ms']:.3f} ms")
    print(f"  Avg time/move:          {r['avg_time_per_move_us']:.1f} µs")


def print_pruning_impact(results: list[dict]) -> None:
    """Estimate how gating cost compares to search node budget."""
    print(f"\n{'═'*55}")
    print("  PRUNING OVERHEAD ESTIMATE")
    print(f"  (at remaining depth=5, ~50k nodes examined at that ply)")
    print(f"{'═'*55}")
    nodes_at_ply = 50_000
    for r in results:
        cost_per_move_ms = r["avg_time_per_move_us"] / 1000
        # At each opponent node we evaluate all moves, then prune some.
        # Overhead = nodes_at_ply × avg_moves × cost_per_move
        avg_moves = r["avg_moves_per_pos"]
        total_overhead_ms = nodes_at_ply * avg_moves * cost_per_move_ms
        print(f"  {r['name'][:35]:35s}: {total_overhead_ms/1000:.1f}s overhead at 50k nodes")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of positions to sample")
    parser.add_argument("--phase", default="move", choices=["move", "place"],
                        help="Game phase to sample from (default: move)")
    parser.add_argument("--no-vn", action="store_true", help="Skip value net benchmark")
    parser.add_argument("--no-sentinel", action="store_true", help="Skip sentinel benchmark")
    args = parser.parse_args()

    db_path = ROOT / "data" / "human_db.sqlite"
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        sys.exit(1)

    print(f"Sampling {args.n} {args.phase}-phase positions from {db_path}…")
    boards = sample_positions(db_path, args.n, args.phase)
    if not boards:
        print("No positions found. Exiting.")
        sys.exit(1)

    results = []

    # ── A: Static eval ───────────────────────────────────────────────────────
    print("\nBenchmarking static eval…")
    r_static = bench_static_eval(boards)
    print_result(r_static)
    results.append(r_static)

    # ── B: Value net ─────────────────────────────────────────────────────────
    if not args.no_vn:
        vn_path = ROOT / "data" / "value_net.npz"
        if not vn_path.exists():
            print("\nValue net not found — skipping.")
        else:
            from ai.value_net import ValueNet
            value_net = ValueNet.load_if_exists(vn_path)
            if value_net is None:
                print("\nValue net failed to load — skipping.")
            else:
                print("\nBenchmarking value net…")
                # Warmup
                b0 = boards[0]
                for mv in get_all_legal_moves(b0)[:2]:
                    value_net.predict(b0.apply_move(mv), b0.turn)
                r_vn = bench_value_net(boards, value_net)
                print_result(r_vn)
                results.append(r_vn)

    # ── C: Sentinel ──────────────────────────────────────────────────────────
    if not args.no_sentinel:
        try:
            from learned_ai.sentinel.infer import load_advisor
            from learned_ai.sentinel.config import load_config
            cfg = load_config()
            ckpt = ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"
            if not ckpt.exists():
                print("\nSentinel checkpoint not found — skipping.")
            else:
                sentinel = load_advisor(str(ckpt), cfg)
                if sentinel is None or not sentinel.is_loaded():
                    print("\nSentinel failed to load — skipping.")
                else:
                    print("\nBenchmarking sentinel…")
                    # Warmup
                    b0 = boards[0]
                    cands = [{"from": m.get("from"), "to": m["to"], "capture": m.get("capture")}
                             for m in get_all_legal_moves(b0)]
                    sentinel.advise(b0, cands, b0.turn)
                    r_sent = bench_sentinel(boards, sentinel)
                    print_result(r_sent)
                    results.append(r_sent)
        except Exception as e:
            print(f"\nSentinel benchmark failed: {e}")

    if len(results) > 1:
        print_pruning_impact(results)

    print(f"\n{'═'*55}")
    print("  SUMMARY — avg µs per move evaluation")
    print(f"{'═'*55}")
    for r in results:
        bar_len = int(r["avg_time_per_move_us"] / 10)
        bar = "█" * min(bar_len, 50)
        print(f"  {r['name'][:30]:30s}  {r['avg_time_per_move_us']:8.1f} µs  {bar}")
    print()


if __name__ == "__main__":
    main()
