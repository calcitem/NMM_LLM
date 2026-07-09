"""scripts/train_vn_trajectory.py — Train phase-specific ValueNets via Malom trajectories.

Three separate nets are trained (placement / movement / fly) and saved as:
  data/value_net_phase_place.npz
  data/value_net_phase_move.npz
  data/value_net_phase_fly.npz

These are loaded at inference by PhaseValueNet, which dispatches predict() calls
to the correct sub-net based on get_game_phase(board, color).

Reward signal (Option B — Malom-weighted composite):
  malom_sign = +1 if Malom says position is W, −1 if L, 0 if D
  best_composite = max over legal moves of:
      0.6 × sentinel_score + 0.4 × normalised_heuristic_score
  y = malom_sign × best_composite   ∈ [−1, 1]

This blends the Malom oracle (winning/losing trajectory signal) with the sentinel +
heuristic composite (same signals used by the GAP net), without needing pre-built
GAP net data.  Winning positions that also look strong per sentinel/heuristics get
large positive labels; losing positions get large negative labels.

Winner's moves are randomly sampled from ALL winning successors (outcome='L'),
not just the single highest-DTW move, so the net learns to recognise any winning
move rather than only the optimal path.

Usage:
    .venv/bin/python scripts/train_vn_trajectory.py
    .venv/bin/python scripts/train_vn_trajectory.py --n-starts 10000 --traj-depth 40
    .venv/bin/python scripts/train_vn_trajectory.py --phase move
    .venv/bin/python scripts/train_vn_trajectory.py \\
        --epochs 0 --bench-accuracy 2000 --bench-games 50
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves, is_terminal, get_game_phase
from ai.value_net import ValueNet, PhaseValueNet, board_to_features
from ai.trajectory_db import make_board_state_key

_DB_PATH       = ROOT / "data" / "human_db.sqlite"
_OUT_BASE      = ROOT / "data" / "value_net_phase"   # → _place.npz, _move.npz, _fly.npz
_SENTINEL_PATH = ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"
_MALOM_PATH    = Path("/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted")

_SENTINEL_WEIGHT = 0.6   # matches build_gap_dataset.py
ALL_PHASES       = ("place", "move", "fly")
_MALOM_SIGN      = {"W": 1.0, "L": -1.0, "D": 0.0}


# ── Phase detection from feature vector ───────────────────────────────────────

def phase_from_features(x: np.ndarray) -> str:
    """Derive game phase from a 79-dim feature vector (no board needed).

    Relies on metadata fields encoded by board_to_features():
      x[73] = own_placed / 9   x[75] = own_on_board / 9
    """
    own_placed   = round(float(x[73]) * 9)
    own_on_board = round(float(x[75]) * 9)
    if own_placed < 9:
        return "place"
    if own_on_board <= 3:
        return "fly"
    return "move"


# ── Composite quality (sentinel + heuristic) ──────────────────────────────────

def _best_composite(board: BoardState, legal: list[dict], sentinel_advisor) -> float:
    """Compute composite quality of the best available move from board.

    Mirrors build_gap_dataset._score_moves() exactly:
      composite = 0.6 × sentinel_q + 0.4 × heuristic_q_norm  ∈ [0, 1]

    Returns the MAX composite across all legal moves (position strength).
    Falls back to 0.5 if neither sentinel nor heuristic is available.
    """
    from ai.heuristics import evaluate_v2

    if not legal:
        return 0.5

    color = board.turn

    # Heuristic scores (evaluate successor from current player's view)
    h_scores: list[float] = []
    for m in legal:
        try:
            succ = board.apply_move(m)
            h_scores.append(float(evaluate_v2(succ, color)))
        except Exception:
            h_scores.append(0.0)

    h_min, h_max = min(h_scores), max(h_scores)
    span = h_max - h_min
    if span < 1e-6:
        h_norms = [0.5] * len(legal)
    else:
        h_norms = [(h - h_min) / span for h in h_scores]

    # Sentinel scores
    s_scores = [0.5] * len(legal)
    if sentinel_advisor is not None:
        try:
            advice = sentinel_advisor.advise(board, legal, color, played_move_idx=0)
            if advice is not None and len(advice.move_scores) == len(legal):
                s_scores = list(advice.move_scores)
        except Exception:
            pass

    best = max(
        _SENTINEL_WEIGHT * s_scores[i] + (1.0 - _SENTINEL_WEIGHT) * h_norms[i]
        for i in range(len(legal))
    )
    return float(best)


# ── Board reconstruction from state_key ───────────────────────────────────────

def state_key_to_board(state_key: str) -> Optional[BoardState]:
    parts = state_key.split("|")
    if len(parts) < 7:
        return None
    canon, turn = parts[0], parts[1]
    if len(canon) != 24 or turn not in ("W", "B"):
        return None
    try:
        placed_w, placed_b = int(parts[3]), int(parts[4])
        on_w,     on_b     = int(parts[5]), int(parts[6])
    except (ValueError, IndexError):
        return None

    positions = {POSITIONS[i]: ("" if canon[i] == "." else canon[i]) for i in range(24)}
    if sum(1 for v in positions.values() if v == "W") != on_w:
        return None
    if sum(1 for v in positions.values() if v == "B") != on_b:
        return None

    return BoardState(
        positions=positions,
        turn=turn,
        pieces_on_board={"W": on_w, "B": on_b},
        pieces_placed={"W": placed_w, "B": placed_b},
        pieces_captured={"W": max(0, placed_b - on_b), "B": max(0, placed_w - on_w)},
    )


# ── Malom move selection ───────────────────────────────────────────────────────

def enumerate_malom_winner_moves(board: BoardState, malom_db) -> list[dict]:
    """Return ALL legal moves that lead to a 'L' successor (opponent loses).

    Randomly sampling from this list instead of always picking the highest-DTW
    move trains the net to recognise ANY winning move, not just the optimal path.
    """
    legal = get_all_legal_moves(board)
    winning_moves = []
    for mv in legal:
        try:
            nb = board.apply_move(mv)
        except Exception:
            continue
        result = malom_db.query(nb)
        if result is not None and result["outcome"] == "L":
            winning_moves.append(mv)
    return winning_moves


def find_malom_defense_move(board: BoardState, malom_db) -> Optional[dict]:
    """Best resistance: draw if available, else highest DTW (slowest loss)."""
    legal = get_all_legal_moves(board)
    best_move = None
    best_dtw  = float("-inf")
    for mv in legal:
        try:
            nb = board.apply_move(mv)
        except Exception:
            continue
        result = malom_db.query(nb)
        if result is None:
            if best_move is None:
                best_move = mv
            continue
        if result["outcome"] == "D":
            return mv
        if result["outcome"] == "L":
            dtw = result.get("dtw", -999)
            if dtw > best_dtw:
                best_dtw = dtw
                best_move = mv
    return best_move or (legal[0] if legal else None)


# ── Trajectory runner ──────────────────────────────────────────────────────────

def run_trajectory(
    start_board: BoardState,
    malom_db,
    rng: np.random.Generator,
    sentinel_advisor,
    heuristic_ai_w=None,
    heuristic_ai_b=None,
    max_depth: int = 40,
) -> list[tuple[BoardState, float, str]]:
    """Follow a winning trajectory from start_board.

    At each winner's turn: randomly pick one of ALL winning moves (any 'L' successor).
    At each loser's turn: Malom best-defense (or heuristic AI if provided).

    Label for each position (Option B — Malom-weighted composite):
      malom_sign = +1 (W) / −1 (L) / 0 (D)
      y = malom_sign × best_composite_quality   ∈ [−1, 1]

    Returns list of (board, y, phase) triples.
    """
    trajectory: list[tuple[BoardState, float, str]] = []
    board        = start_board
    winner_color = board.turn

    for _ in range(max_depth):
        term, _ = is_terminal(board)
        if term:
            break

        result = malom_db.query(board)
        if result is None:
            break

        outcome   = result["outcome"]
        malom_sign = _MALOM_SIGN.get(outcome, 0.0)
        legal      = get_all_legal_moves(board)
        composite  = _best_composite(board, legal, sentinel_advisor)
        stm_label  = malom_sign * composite
        phase      = get_game_phase(board, board.turn)
        trajectory.append((board, stm_label, phase))

        if board.turn == winner_color:
            winners = enumerate_malom_winner_moves(board, malom_db)
            if not winners:
                break
            move = winners[int(rng.integers(len(winners)))]
        else:
            ai = heuristic_ai_w if board.turn == "W" else heuristic_ai_b
            move = None
            if ai is not None:
                try:
                    move = ai.choose_move(board)
                except Exception:
                    pass
            if move is None:
                move = find_malom_defense_move(board, malom_db)
            if move is None:
                break

        try:
            board = board.apply_move(move)
        except Exception:
            break

    return trajectory


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_dataset(
    conn: sqlite3.Connection,
    malom_db,
    rng: np.random.Generator,
    sentinel_advisor,
    heuristic_ai_w=None,
    heuristic_ai_b=None,
    n_starts: int = 5000,
    min_placed: int = 7,
    max_traj_depth: int = 40,
    bucket_cap: Optional[int] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build trajectory training data.

    Returns (X, y, phases) where phases is a str array of phase labels per sample.
    """
    rows = conn.execute(
        "SELECT state_key FROM positions WHERE malom_wdl = 'W' ORDER BY RANDOM() LIMIT ?",
        (n_starts * 4,)
    ).fetchall()

    buckets: dict[int, list[str]] = defaultdict(list)
    for (sk,) in rows:
        parts = sk.split("|")
        if len(parts) < 7:
            continue
        try:
            placed = int(parts[3]) + int(parts[4])
        except ValueError:
            continue
        if placed < min_placed:
            continue
        buckets[placed].append(sk)

    starts: list[str] = []
    for stage in sorted(buckets.keys()):
        items = buckets[stage]
        cap = bucket_cap or len(items)
        starts.extend(items[:cap])
        if len(starts) >= n_starts:
            break
    starts = starts[:n_starts]

    if verbose:
        print(f"  Starting positions: {len(starts):,} (target {n_starts})")

    seen_keys: set[str] = set()
    all_samples: list[tuple[np.ndarray, float, str]] = []
    skipped_parse = 0
    total_traj    = 0
    total_steps   = 0

    for i, sk in enumerate(starts):
        if verbose and (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(starts)} trajectories  "
                  f"unique samples: {len(all_samples):,}")

        board = state_key_to_board(sk)
        if board is None:
            skipped_parse += 1
            continue

        traj = run_trajectory(board, malom_db, rng,
                              sentinel_advisor, heuristic_ai_w, heuristic_ai_b,
                              max_depth=max_traj_depth)
        total_traj  += 1
        total_steps += len(traj)

        for b, label, phase in traj:
            canon_key, _ = make_board_state_key(b)
            if canon_key in seen_keys:
                continue
            seen_keys.add(canon_key)
            feats = board_to_features(b, b.turn)
            all_samples.append((feats, label, phase))

    if verbose:
        avg_len = total_steps / max(1, total_traj)
        by_phase: dict[str, int] = defaultdict(int)
        for _, _, ph in all_samples:
            by_phase[ph] += 1
        print(f"\n  Trajectories run: {total_traj:,}  avg length: {avg_len:.1f} steps")
        print(f"  Unique positions: {len(all_samples):,}  "
              + "  ".join(f"{ph}={by_phase[ph]:,}" for ph in ALL_PHASES))
        if skipped_parse:
            print(f"  Skipped (parse error): {skipped_parse}")

    if not all_samples:
        return (np.empty((0, 79), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=object))

    X      = np.stack([s[0] for s in all_samples]).astype(np.float32)
    y      = np.array([s[1] for s in all_samples], dtype=np.float32)
    phases = np.array([s[2] for s in all_samples], dtype=object)
    return X, y, phases


# ── Prediction accuracy benchmark ─────────────────────────────────────────────

def bench_accuracy(
    net: ValueNet,
    conn: sqlite3.Connection,
    n: int = 2000,
    label: str = "",
    min_placed: int = 7,
) -> dict:
    rows = conn.execute(
        "SELECT state_key, malom_wdl FROM positions "
        "WHERE malom_wdl IS NOT NULL ORDER BY RANDOM() LIMIT ?",
        (n * 4,)
    ).fetchall()

    correct = 0
    total   = 0
    for state_key, wdl in rows:
        if total >= n:
            break
        parts = state_key.split("|")
        if len(parts) < 7:
            continue
        try:
            if int(parts[3]) + int(parts[4]) < min_placed:
                continue
        except ValueError:
            continue
        board = state_key_to_board(state_key)
        if board is None:
            continue
        pred = net.predict(board, board.turn)
        if ((pred >  0.1 and wdl == "W") or
                (pred < -0.1 and wdl == "L") or
                (abs(pred) <= 0.1 and wdl == "D")):
            correct += 1
        total += 1

    acc = correct / max(1, total)
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Accuracy: {acc:.1%}  ({correct}/{total})")
    return {"accuracy": acc, "correct": correct, "total": total}


# ── Trajectory-following benchmark ────────────────────────────────────────────

def bench_trajectory_follow(
    net: ValueNet,
    malom_db,
    rng: np.random.Generator,
    n: int = 300,
    min_placed: int = 7,
    label: str = "",
) -> dict:
    """Test how often VN's top move is among the Malom winning moves."""
    results = sqlite3.connect(str(_DB_PATH)).execute(
        "SELECT state_key FROM positions WHERE malom_wdl='W' ORDER BY RANDOM() LIMIT ?",
        (n * 4,)
    ).fetchall()

    matches = 0
    skipped = 0
    total   = 0

    for (sk,) in results:
        if total >= n:
            break
        parts = sk.split("|")
        if len(parts) < 7:
            continue
        try:
            if int(parts[3]) + int(parts[4]) < min_placed:
                continue
        except ValueError:
            continue

        board = state_key_to_board(sk)
        if board is None:
            continue

        winning_moves = enumerate_malom_winner_moves(board, malom_db)
        if not winning_moves:
            skipped += 1
            continue

        legal = get_all_legal_moves(board)
        best_mv    = None
        best_score = float("-inf")
        for mv in legal:
            try:
                nb = board.apply_move(mv)
            except Exception:
                continue
            opp_val = net.predict(nb, nb.turn)
            stm_val = -opp_val
            if stm_val > best_score:
                best_score = stm_val
                best_mv    = mv

        if best_mv is None:
            continue

        if best_mv in winning_moves:
            matches += 1
        total += 1

    rate = matches / max(1, total)
    tag  = f"[{label}] " if label else ""
    print(f"  {tag}Trajectory-follow: {rate:.1%}  "
          f"({matches}/{total}, {skipped} skipped)")
    return {"rate": rate, "matches": matches, "total": total}


# ── Full-game benchmark ────────────────────────────────────────────────────────

def _play_game(ai_w, ai_b, max_plies: int = 400) -> Optional[str]:
    board = BoardState.new_game()
    for _ in range(max_plies):
        terminal, winner = is_terminal(board)
        if terminal:
            return winner
        legal = get_all_legal_moves(board)
        if not legal:
            return "B" if board.turn == "W" else "W"
        ai = ai_w if board.turn == "W" else ai_b
        try:
            move = ai.choose_move(board)
        except Exception:
            return "B" if board.turn == "W" else "W"
        if not move:
            return "B" if board.turn == "W" else "W"
        try:
            board = board.apply_move(move)
        except Exception:
            return None
    return None


def _make_ai(color: str, difficulty: int, value_net=None, vn_blend: int = 80,
             time_budget: float = 0.5):
    from ai.game_ai import GameAI
    from ai.heuristics import HeuristicWeights
    weights = HeuristicWeights(value_net_blend=vn_blend) if (value_net and vn_blend > 0) else None
    return GameAI(
        color=color,
        difficulty=difficulty,
        value_net=value_net,
        weights=weights,
        override_time_budget=time_budget,
    )


def bench_games(
    net,
    n_games: int,
    difficulty: int = 6,
    vn_blend: int = 80,
    label_a: str = "raw",
    label_b: str = "phase-vn",
    time_budget: float = 0.5,
) -> dict:
    wins = draws = losses = 0
    for g in range(n_games):
        vn_color = "W" if g % 2 == 0 else "B"
        ai_w = _make_ai("W", difficulty,
                        value_net=(net if vn_color == "W" else None),
                        vn_blend=vn_blend, time_budget=time_budget)
        ai_b = _make_ai("B", difficulty,
                        value_net=(net if vn_color == "B" else None),
                        vn_blend=vn_blend, time_budget=time_budget)
        winner = _play_game(ai_w, ai_b)
        if winner is None:
            draws += 1
        elif winner == vn_color:
            wins += 1
        else:
            losses += 1
        if (g + 1) % 10 == 0:
            t = g + 1
            print(f"    game {t:3d}/{n_games}  "
                  f"{label_b}: W={wins} D={draws} L={losses}  "
                  f"score={wins/t:.0%}")
    total = max(wins + draws + losses, 1)
    print(f"\n  {label_b} vs {label_a}:  "
          f"W={wins}  D={draws}  L={losses} / {total}  "
          f"score={wins/total:.1%}")
    return {"wins": wins, "draws": draws, "losses": losses, "total": total,
            "score": wins / total}


# ── Per-phase training ─────────────────────────────────────────────────────────

def _train_phase_net(
    phase: str,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch: int,
    continue_from: Optional[Path],
    out_base: Path,
) -> ValueNet:
    """Train and save one phase-specific net.  Returns the trained net."""
    print(f"\n── Phase: {phase}  N={len(X):,} ──")

    if continue_from is not None:
        phase_path = continue_from.parent / f"{continue_from.stem}_{phase}.npz"
        if phase_path.exists():
            print(f"  Fine-tuning from {phase_path}")
            net = ValueNet.load(phase_path)
        else:
            print(f"  Continue-from not found ({phase_path}) — fresh init")
            net = ValueNet()
    else:
        net = ValueNet()

    if len(X) < 10:
        print(f"  WARNING: only {len(X)} samples for {phase} phase — skipping training")
        return net

    if epochs > 0:
        t0 = time.perf_counter()
        losses = net.train(X, y, epochs=epochs, lr=lr, batch_size=batch, verbose=True)
        elapsed = time.perf_counter() - t0
        print(f"  Training done in {elapsed:.1f}s  "
              f"final loss={losses[-1]:.5f}  "
              f"best={min(losses):.5f} (epoch {losses.index(min(losses))+1})")

    out_path = out_base.parent / f"{out_base.stem}_{phase}.npz"
    net.save(out_path)
    print(f"  Saved → {out_path}")
    return net


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train phase-specific ValueNets from Malom winning trajectories."
    )
    ap.add_argument("--db",             default=str(_DB_PATH))
    ap.add_argument("--malom-db",       default=str(_MALOM_PATH))
    ap.add_argument("--sentinel",       default=str(_SENTINEL_PATH),
                    help="Sentinel checkpoint path (default: learned_ai/sentinel/checkpoints/best.pt)")
    ap.add_argument("--out",            default=str(_OUT_BASE),
                    help="Base output path (no extension).  "
                         "Saves {out}_place.npz, {out}_move.npz, {out}_fly.npz")
    ap.add_argument("--phase",          choices=list(ALL_PHASES) + ["all"], default="all",
                    help="Which phase net(s) to train (default: all)")
    ap.add_argument("--n-starts",       type=int, default=5000,
                    help="Starting positions to sample from human_db (default 5000)")
    ap.add_argument("--traj-depth",     type=int, default=40,
                    help="Max plies per trajectory (default 40)")
    ap.add_argument("--min-placed",     type=int, default=7,
                    help="Min total pieces placed for a start position (default 7)")
    ap.add_argument("--bucket-cap",     type=int, default=None)
    ap.add_argument("--use-heuristic",  action="store_true",
                    help="Use heuristic AI for loser's moves (can be slow)")
    ap.add_argument("--heuristic-difficulty", type=int, default=4)
    ap.add_argument("--heuristic-time",       type=float, default=0.05)
    ap.add_argument("--seed",           type=int, default=42,
                    help="RNG seed for winner-move sampling (default 42)")
    ap.add_argument("--epochs",         type=int,   default=40)
    ap.add_argument("--lr",             type=float, default=8e-4)
    ap.add_argument("--batch",          type=int,   default=512)
    ap.add_argument("--continue-from",  default=None,
                    help="Base path of existing phase nets to fine-tune from")
    ap.add_argument("--bench-accuracy", type=int, default=2000)
    ap.add_argument("--bench-traj",     type=int, default=300)
    ap.add_argument("--bench-games",    type=int, default=0)
    ap.add_argument("--bench-gap",      action="store_true")
    ap.add_argument("--difficulty",     type=int, default=6)
    ap.add_argument("--vn-blend",       type=int, default=80)
    ap.add_argument("--time-budget",    type=float, default=0.5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Malom DB ──────────────────────────────────────────────────────────────
    from ai.malom_db import MalomDB
    malom_db = MalomDB(args.malom_db)
    if not malom_db.is_available():
        sys.exit(f"Malom DB not available at {args.malom_db}")
    print(f"Malom DB: {args.malom_db}  ✓")

    # ── Sentinel advisor ───────────────────────────────────────────────────────
    sentinel_advisor = None
    sentinel_path = Path(args.sentinel)
    if sentinel_path.exists():
        try:
            from learned_ai.sentinel.infer import SentinelAdvisor
            sentinel_advisor = SentinelAdvisor(checkpoint_path=str(sentinel_path))
            # Trigger lazy load
            _dummy_board = BoardState.new_game()
            _dummy_moves = [{"from": None, "to": "a1", "capture": None}]
            sentinel_advisor.advise(_dummy_board, _dummy_moves, "W", played_move_idx=0)
            print(f"Sentinel: loaded from {sentinel_path}")
        except Exception as e:
            print(f"Sentinel load failed ({e}) — using heuristics only (weight adjusted)")
            sentinel_advisor = None
    else:
        print(f"Sentinel: not found at {sentinel_path} — using heuristics only")

    if sentinel_advisor is None:
        print("  Note: composite quality will use heuristic score only (sentinel weight ignored)")

    # ── Heuristic AI for loser (opt-in) ───────────────────────────────────────
    if args.use_heuristic:
        print(f"Loser defense: heuristic AI  diff={args.heuristic_difficulty}  "
              f"time={args.heuristic_time}s")
        heuristic_ai_w = _make_ai("W", args.heuristic_difficulty,
                                   time_budget=args.heuristic_time)
        heuristic_ai_b = _make_ai("B", args.heuristic_difficulty,
                                   time_budget=args.heuristic_time)
    else:
        print("Loser defense: Malom best-resistance (no AI search)")
        heuristic_ai_w = None
        heuristic_ai_b = None

    # ── Build trajectory dataset ───────────────────────────────────────────────
    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"DB not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    print(f"Human DB: {db_path}")

    phases_to_train = ALL_PHASES if args.phase == "all" else (args.phase,)

    print(f"\nBuilding trajectory dataset  "
          f"n_starts={args.n_starts}  traj_depth={args.traj_depth}  "
          f"min_placed={args.min_placed}  seed={args.seed}  "
          f"reward=malom_sign×composite...")
    t0 = time.perf_counter()
    X_all, y_all, phases_all = build_dataset(
        conn, malom_db, rng,
        sentinel_advisor,
        heuristic_ai_w, heuristic_ai_b,
        n_starts=args.n_starts,
        min_placed=args.min_placed,
        max_traj_depth=args.traj_depth,
        bucket_cap=args.bucket_cap,
        verbose=True,
    )
    build_time = time.perf_counter() - t0
    print(f"Dataset: {len(X_all):,} positions  (built in {build_time:.1f}s)")
    print(f"y stats: min={y_all.min():.3f}  max={y_all.max():.3f}  mean={y_all.mean():.3f}")
    print("  " + "  ".join(f"{ph}={int(np.sum(phases_all == ph)):,}" for ph in ALL_PHASES))

    if len(X_all) == 0:
        sys.exit("No training data — check Malom DB, human_db, and filters.")

    # ── Phase-specific training ────────────────────────────────────────────────
    out_base      = Path(args.out)
    continue_base = Path(args.continue_from) if args.continue_from else None

    trained_nets: dict[str, ValueNet] = {}

    for phase in phases_to_train:
        mask = phases_all == phase
        Xp, yp = X_all[mask], y_all[mask]

        net = _train_phase_net(
            phase, Xp, yp,
            epochs=args.epochs,
            lr=args.lr,
            batch=args.batch,
            continue_from=continue_base,
            out_base=out_base,
        )
        trained_nets[phase] = net

    # ── Benchmarks ────────────────────────────────────────────────────────────
    bench_net = trained_nets.get("move") or trained_nets.get(list(trained_nets.keys())[0])

    if args.bench_accuracy > 0 and bench_net is not None:
        print(f"\nAccuracy benchmark ({args.bench_accuracy} positions, move-phase net):")
        bench_accuracy(bench_net, conn, args.bench_accuracy,
                       label="phase-move", min_placed=args.min_placed)

    if args.bench_traj > 0 and bench_net is not None:
        print(f"\nTrajectory-follow benchmark ({args.bench_traj} positions):")
        bench_trajectory_follow(bench_net, malom_db, rng,
                                n=args.bench_traj,
                                min_placed=args.min_placed, label="phase-move")

    conn.close()

    # ── Full-game benchmark with PhaseValueNet ────────────────────────────────
    if args.bench_games > 0:
        phase_vn = PhaseValueNet.load_if_exists(out_base)
        game_net = phase_vn if phase_vn is not None else bench_net
        net_label = "PhaseVN" if phase_vn is not None else "phase-move-vn"

        print(f"\n{'='*60}")
        print(f"Full-game benchmark: {args.bench_games} games  "
              f"difficulty={args.difficulty}  vn_blend={args.vn_blend}%  "
              f"net={net_label}")
        bench_games(game_net, n_games=args.bench_games,
                    difficulty=args.difficulty, vn_blend=args.vn_blend,
                    label_a="raw", label_b=net_label,
                    time_budget=args.time_budget)

        if args.bench_gap:
            gap_net_path = ROOT / "data" / "gap_net.npz"
            if gap_net_path.exists():
                gap_net = ValueNet.load(gap_net_path)
                print(f"\nGAP net baseline:")
                bench_games(gap_net, n_games=args.bench_games,
                            difficulty=args.difficulty, vn_blend=args.vn_blend,
                            label_a="raw", label_b="gap-net",
                            time_budget=args.time_budget)


if __name__ == "__main__":
    main()
