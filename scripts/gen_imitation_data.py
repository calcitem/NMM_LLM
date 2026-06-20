"""scripts/gen_imitation_data.py — Generate supervised imitation dataset.

Plays heuristic vs heuristic games and records, for each position:
  * feat_matrix  (k, 62)  — scaffolded features for every legal move
  * value_input  (23,)    — board-level features for value head
  * label_dist   (k,)     — soft probability distribution over legal moves
  * h_eval       float    — evaluate(board, player, strength_mode=True)

Label distribution uses Malom DB when available (DTM-graded per move) and
falls back to sentinel-score-based distribution otherwise.

The resulting .npz can be loaded directly by train_scaffolded_s1.py for
imitation (KL divergence on policy + MSE on value).

Usage
-----
    .venv/bin/python scripts/gen_imitation_data.py [options]

Options
-------
  --games   N       Number of self-play games (default 2000)
  --diff    D       Heuristic difficulty (default 3)
  --out     PATH    Output .npz path (default learned_ai/data/imitation_scaffolded.npz)
  --sentinel PATH   SentinelAdvisor checkpoint (default learned_ai/sentinel/checkpoints/best.pt)
  --malom   PATH    Malom DB directory (default: read from data/settings.json, or skip)
  --max-ply N       Max plies per game before draw (default 300)
  --seed    N       Random seed
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.models.scaffolded_encoder import encode_position, MOVE_FEAT_DIM, VALUE_INPUT_DIM
from learned_ai.sentinel.infer import load_advisor
from learned_ai.sentinel.labels import dtm_quality

# ── Soft-label hyperparameters ─────────────────────────────────────────────────
# Malom: per-category multiplier applied before dtm_quality
_WDL_SCALE = {"win": 1.0, "draw": 0.4, "loss": 0.1}
# Sentinel fallback: softmax temperature (lower = sharper distribution)
_SENTINEL_TEMP = 0.5


def _move_key(mv: dict):
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _compute_soft_label(
    db_moves: list,
    legal_moves: list,
    sentinel_scores: list,
) -> np.ndarray:
    """Compute (k,) soft probability distribution over legal moves.

    Uses Malom DB (DTM-graded) when available; sentinel scores as fallback.
    """
    k = len(legal_moves)

    if db_moves:
        db_lookup = {_move_key(e.get("move", {})): e for e in db_moves}
        weights = np.zeros(k, dtype=np.float64)
        has_db = False
        for i, mv in enumerate(legal_moves):
            entry = db_lookup.get(_move_key(mv))
            if entry is not None:
                wdl = entry.get("wdl", "unknown")
                dtm = entry.get("dtm")
                if wdl in _WDL_SCALE:
                    weights[i] = _WDL_SCALE[wdl] * dtm_quality(wdl, dtm)
                    has_db = True
                else:
                    weights[i] = max(float(sentinel_scores[i]), 1e-6) if sentinel_scores else 0.5
            else:
                weights[i] = max(float(sentinel_scores[i]), 1e-6) if sentinel_scores else 0.5

        if has_db and weights.sum() > 0:
            return (weights / weights.sum()).astype(np.float32)

    # Sentinel fallback: temperature softmax
    s = np.array(sentinel_scores if sentinel_scores else [1.0 / k] * k, dtype=np.float64)
    s = np.exp((s - s.max()) / _SENTINEL_TEMP)
    return (s / s.sum()).astype(np.float32)


def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    # ── load sentinel (optional) ───────────────────────────────────────────────
    sentinel = None
    if args.sentinel and Path(args.sentinel).exists():
        sentinel = load_advisor(args.sentinel)
        if sentinel and sentinel.is_loaded():
            print(f"[gen] Sentinel loaded from {args.sentinel}")
        else:
            sentinel = None
            print("[gen] Sentinel not available — sentinel features will be 0.5")

    # ── load Malom DB (optional) ────────────────────────────────────────────────
    db = None
    malom_path = args.malom or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if db.is_available():
                print(f"[gen] Malom DB loaded from {malom_path}")
            else:
                db = None
                print("[gen] Malom DB path given but not available — sentinel fallback labels")
        except Exception as e:
            print(f"[gen] Malom DB load failed ({e}) — sentinel fallback labels")
            db = None
    else:
        print("[gen] No Malom DB — sentinel fallback labels")

    # ── load Value Net (optional) ──────────────────────────────────────────────
    value_net = None
    vn_path = getattr(args, "value_net", None) or str(_ROOT / "data" / "value_net.npz")
    if vn_path and Path(vn_path).exists():
        try:
            from ai.value_net import ValueNet as _ValueNet
            value_net = _ValueNet.load(vn_path)
            print(f"[gen] Value net loaded from {vn_path}")
        except Exception as e:
            print(f"[gen] Value net load failed ({e}) — VN features will be 0")
    else:
        print("[gen] No value net — VN features will be 0")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── storage ────────────────────────────────────────────────────────────────
    all_feat_matrices: list[np.ndarray] = []
    all_value_inputs:  list[np.ndarray] = []
    all_label_dists:   list[np.ndarray] = []
    all_h_evals:       list[float]      = []
    all_vn_evals:      list[float]      = []

    t_start = time.time()
    wins_w = wins_b = draws = 0
    n_db_labels = n_sentinel_labels = 0

    TIME_BUDGET = 0.05  # fast moves for data generation

    for game_i in range(args.games):
        from learned_ai.agents.heuristic_agent import GameAI as _GA
        ai_w = HeuristicAgent(
            color="W", difficulty=args.diff,
            game_ai=_GA(color="W", difficulty=args.diff,
                        override_time_budget=TIME_BUDGET),
        )
        ai_b = HeuristicAgent(
            color="B", difficulty=args.diff,
            game_ai=_GA(color="B", difficulty=args.diff,
                        override_time_budget=TIME_BUDGET),
        )

        board = BoardState.new_game()
        ply   = 0
        game_positions: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []

        while ply < args.max_ply:
            terminal, winner = is_terminal(board)
            if terminal:
                if winner == "W":
                    wins_w += 1
                elif winner == "B":
                    wins_b += 1
                else:
                    draws += 1
                break

            player = board.turn
            agent  = ai_w if player == "W" else ai_b

            enc = encode_position(board, player, sentinel_advisor=sentinel, db=db, value_net=value_net)
            if enc is None or not enc.legal_moves:
                draws += 1
                break

            # Soft label: Malom DB (DTM-graded) or sentinel fallback
            label_dist = _compute_soft_label(enc.db_moves, enc.legal_moves, enc.sentinel_scores)
            if enc.db_moves and any(e.get("wdl") in _WDL_SCALE for e in enc.db_moves):
                n_db_labels += 1
            else:
                n_sentinel_labels += 1

            chosen_move = agent.choose_move(board)
            if not chosen_move:
                draws += 1
                break

            game_positions.append((
                enc.feat_matrix,
                enc.value_input,
                label_dist,
                enc.h_before,
                enc.vn_before,
            ))

            board = board.apply_move(chosen_move)
            ply += 1
        else:
            draws += 1

        for feat_matrix, value_input, label_dist, h_eval, vn_eval in game_positions:
            all_feat_matrices.append(feat_matrix)
            all_value_inputs.append(value_input)
            all_label_dists.append(label_dist)
            all_h_evals.append(h_eval)
            all_vn_evals.append(vn_eval)

        if (game_i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            n_pos = len(all_label_dists)
            print(
                f"[gen] game {game_i+1}/{args.games} | "
                f"positions {n_pos} | "
                f"W/B/D {wins_w}/{wins_b}/{draws} | "
                f"db_labels={n_db_labels} sent_labels={n_sentinel_labels} | "
                f"{elapsed:.0f}s elapsed"
            )

    # ── save ───────────────────────────────────────────────────────────────────
    n = len(all_label_dists)
    feat_arr  = np.empty(n, dtype=object)
    label_arr = np.empty(n, dtype=object)
    for i, (fm, ld) in enumerate(zip(all_feat_matrices, all_label_dists)):
        feat_arr[i]  = fm
        label_arr[i] = ld

    np.savez(
        out_path,
        feat_matrices=feat_arr,
        value_inputs=np.array(all_value_inputs, dtype=np.float32),
        label_dists=label_arr,
        h_evals=np.array(all_h_evals, dtype=np.float32),
        vn_evals=np.array(all_vn_evals, dtype=np.float32),
    )
    elapsed = time.time() - t_start
    print(f"\n[gen] Saved {n} positions to {out_path}  ({elapsed:.0f}s total)")
    print(f"[gen] Games: W={wins_w}  B={wins_b}  draw={draws}")
    print(f"[gen] Labels: {n_db_labels} Malom-DB  {n_sentinel_labels} sentinel-fallback")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate scaffolded imitation dataset")
    p.add_argument("--games",    type=int,  default=2000)
    p.add_argument("--diff",     type=int,  default=3)
    p.add_argument("--out",      type=str,
                   default=str(_ROOT / "learned_ai" / "data" / "imitation_scaffolded.npz"))
    p.add_argument("--sentinel", type=str,
                   default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"))
    p.add_argument("--malom",     type=str,  default="")
    p.add_argument("--value-net", type=str,  default=str(_ROOT / "data" / "value_net.npz"))
    p.add_argument("--max-ply",  type=int,  default=300)
    p.add_argument("--seed",     type=int,  default=42)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
