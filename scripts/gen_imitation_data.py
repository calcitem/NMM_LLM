"""scripts/gen_imitation_data.py — Generate supervised imitation dataset.

Plays heuristic vs heuristic games and records, for each position:
  * feat_matrix  (k, 62)  — scaffolded features for every legal move
  * value_input  (23,)    — board-level features for value head
  * chosen_idx   int      — index of the move the heuristic actually played
  * h_eval       float    — evaluate(board, player, strength_mode=True)
  * h_eval_after float    — evaluate(board_after_move, player, strength_mode=True)

The resulting .npz can be loaded directly by train_scaffolded_s1.py for
imitation (cross-entropy on policy + MSE on value).

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


def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _move_key(mv: dict):
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


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
                print("[gen] Malom DB path given but not available — DB features will be 0")
        except Exception as e:
            print(f"[gen] Malom DB load failed ({e}) — skipping")
            db = None
    else:
        print("[gen] No Malom DB — DB counterfactual features will be 0")

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
    all_chosen_idxs:   list[int]        = []
    all_h_evals:       list[float]      = []
    all_vn_evals:      list[float]      = []

    t_start = time.time()
    wins_w = wins_b = draws = 0

    TIME_BUDGET = 0.05  # fast moves for data generation

    for game_i in range(args.games):
        # Random colour assignment per game (agent always plays W vs B heuristic)
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
        game_positions: list[tuple[np.ndarray, np.ndarray, int, float, float]] = []

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

            # Encode position BEFORE move
            enc = encode_position(board, player, sentinel_advisor=sentinel, db=db, value_net=value_net)
            if enc is None or not enc.legal_moves:
                draws += 1
                break

            # Heuristic picks the move — we record its index in legal_moves
            chosen_move = agent.choose_move(board)
            if not chosen_move:
                draws += 1
                break

            chosen_key = _move_key(chosen_move)
            chosen_idx = next(
                (i for i, m in enumerate(enc.legal_moves)
                 if _move_key(m) == chosen_key),
                0,
            )

            game_positions.append((
                enc.feat_matrix,
                enc.value_input,
                chosen_idx,
                enc.h_before,
                enc.vn_before,  # NEW
            ))

            board = board.apply_move(chosen_move)
            ply += 1
        else:
            draws += 1

        for feat_matrix, value_input, cidx, h_eval, vn_eval in game_positions:
            all_feat_matrices.append(feat_matrix)
            all_value_inputs.append(value_input)
            all_chosen_idxs.append(cidx)
            all_h_evals.append(h_eval)
            all_vn_evals.append(vn_eval)  # NEW

        if (game_i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            n_pos = len(all_chosen_idxs)
            print(
                f"[gen] game {game_i+1}/{args.games} | "
                f"positions {n_pos} | "
                f"W/B/D {wins_w}/{wins_b}/{draws} | "
                f"{elapsed:.0f}s elapsed"
            )

    # ── save ───────────────────────────────────────────────────────────────────
    # feat_matrices have variable k per position; store as object array
    n = len(all_chosen_idxs)
    feat_arr   = np.empty(n, dtype=object)
    for i, fm in enumerate(all_feat_matrices):
        feat_arr[i] = fm

    np.savez(
        out_path,
        feat_matrices=feat_arr,
        value_inputs=np.array(all_value_inputs, dtype=np.float32),
        chosen_idxs=np.array(all_chosen_idxs, dtype=np.int32),
        h_evals=np.array(all_h_evals, dtype=np.float32),
        vn_evals=np.array(all_vn_evals, dtype=np.float32),
    )
    elapsed = time.time() - t_start
    print(f"\n[gen] Saved {n} positions to {out_path}  ({elapsed:.0f}s total)")
    print(f"[gen] Games: W={wins_w}  B={wins_b}  draw={draws}")


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
