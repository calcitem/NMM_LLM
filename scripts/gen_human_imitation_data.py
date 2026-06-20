"""scripts/gen_human_imitation_data.py — Extract human-game imitation dataset.

Reads all human vs AI game records from data/games/*.jsonl and encodes every
move made by the human player.  Games where the human won are weighted 1.0;
draws are weighted 0.3; lost games are skipped.

Positions where the human deviated from the heuristic's top-1 choice in a won
game are flagged (deviates=True) — these are the highest-signal samples for
Stage 1.5 fine-tuning.

Output .npz arrays
------------------
  feat_matrices : (N,) object array of (k, 62) float32
  value_inputs  : (N, 23) float32
  chosen_idxs   : (N,) int32
  h_evals       : (N,) float32   — h_before for value-head supervision
  h_top1_idxs   : (N,) int32     — heuristic's best move index
  weights       : (N,) float32   — 1.0 (won) or 0.3 (draw)
  deviates      : (N,) bool      — True if human didn't play heuristic top-1

Usage
-----
    .venv/bin/python scripts/gen_human_imitation_data.py [options]

Options
-------
  --games-dir  PATH  Directory of .jsonl game files (default data/games)
  --out        PATH  Output .npz (default learned_ai/data/human_imitation.npz)
  --sentinel   PATH  SentinelAdvisor checkpoint (default sentinel/checkpoints/best.pt)
  --malom      PATH  Malom DB directory (default: read from data/settings.json)
  --won-weight F     Weight for won-game moves (default 1.0)
  --draw-weight F    Weight for drawn-game moves (default 0.3)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from learned_ai.models.scaffolded_encoder import encode_position, MOVE_FEAT_DIM, VALUE_INPUT_DIM
from learned_ai.sentinel.infer import load_advisor


def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _load_game_file(path: Path) -> list[dict]:
    """Load one .jsonl file as a list of game dicts (handles single-JSON and JSONL)."""
    content = path.read_text().strip()
    if not content:
        return []
    try:
        return [json.loads(content)]
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def _move_key(mv: dict) -> tuple:
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def run(args: argparse.Namespace) -> None:
    t_start = time.time()

    # ── load sentinel ──────────────────────────────────────────────────────────
    sentinel = None
    if args.sentinel and Path(args.sentinel).exists():
        sentinel = load_advisor(args.sentinel)
        if sentinel and sentinel.is_loaded():
            print(f"[hgen] Sentinel loaded from {args.sentinel}")
        else:
            sentinel = None
            print("[hgen] Sentinel not available — sentinel features will be 0.5")
    else:
        print("[hgen] No sentinel path given — sentinel features will be 0.5")

    # ── load Malom DB ──────────────────────────────────────────────────────────
    db = None
    malom_path = args.malom or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if db.is_available():
                print(f"[hgen] Malom DB loaded from {malom_path}")
            else:
                db = None
        except Exception as e:
            print(f"[hgen] Malom DB load failed ({e}) — skipping")
    else:
        print("[hgen] No Malom DB — DB features will be 0")

    # ── load Value Net (optional) ──────────────────────────────────────────────
    value_net = None
    vn_path = getattr(args, "value_net", None) or str(_ROOT / "data" / "value_net.npz")
    if vn_path and Path(vn_path).exists():
        try:
            from ai.value_net import ValueNet as _ValueNet
            value_net = _ValueNet.load(vn_path)
            print(f"[hgen] Value net loaded from {vn_path}")
        except Exception as e:
            print(f"[hgen] Value net load failed ({e}) — VN features will be 0")
    else:
        print("[hgen] No value net — VN features will be 0")

    # ── scan game files ────────────────────────────────────────────────────────
    games_dir = Path(args.games_dir)
    game_files = sorted(games_dir.glob("*.jsonl"))
    print(f"[hgen] Found {len(game_files)} game files in {games_dir}")

    all_feat_matrices: list[np.ndarray] = []
    all_value_inputs:  list[np.ndarray] = []
    all_chosen_idxs:   list[int]        = []
    all_h_evals:       list[float]      = []
    all_vn_evals:      list[float]      = []
    all_h_top1_idxs:   list[int]        = []
    all_weights:       list[float]      = []
    all_deviates:      list[bool]       = []

    n_won = n_draw = n_lost = n_selfplay = 0
    n_pos_won = n_pos_draw = 0
    n_deviates = 0
    n_errors = 0
    n_files = len(game_files)

    for file_i, game_file in enumerate(game_files):
        if (file_i + 1) % 100 == 0 or file_i == 0:
            n_pos = len(all_chosen_idxs)
            elapsed = time.time() - t_start
            print(
                f"[hgen] {file_i+1}/{n_files} files | "
                f"positions {n_pos} | won={n_won} draw={n_draw} | {elapsed:.0f}s",
                flush=True,
            )

        for game in _load_game_file(game_file):
            human_color = game.get("human_color")
            winner = game.get("winner")

            # Skip self-play or AI-vs-AI
            if game.get("self_play") or human_color in (None, "self_play"):
                n_selfplay += 1
                continue

            if winner == human_color:
                weight = args.won_weight
                n_won += 1
            elif winner is None:
                weight = args.draw_weight
                n_draw += 1
            else:
                n_lost += 1
                continue   # skip lost games

            moves = game.get("moves", [])
            for mv in moves:
                if mv.get("color") != human_color:
                    continue   # only encode human moves

                fen = mv.get("board_fen_before")
                if not fen:
                    continue

                try:
                    board = BoardState.from_fen_string(fen)
                except Exception:
                    n_errors += 1
                    continue

                enc = encode_position(board, human_color, sentinel_advisor=sentinel, db=db, value_net=value_net)
                if enc is None or not enc.legal_moves:
                    continue

                chosen_key = _move_key(mv)
                chosen_idx = next(
                    (i for i, m in enumerate(enc.legal_moves)
                     if _move_key(m) == chosen_key),
                    None,
                )
                if chosen_idx is None:
                    n_errors += 1
                    continue

                deviates = (chosen_idx != enc.h_top1_idx)
                if deviates:
                    n_deviates += 1

                all_feat_matrices.append(enc.feat_matrix)
                all_value_inputs.append(enc.value_input)
                all_chosen_idxs.append(chosen_idx)
                all_h_evals.append(enc.h_before)
                all_vn_evals.append(enc.vn_before)
                all_h_top1_idxs.append(enc.h_top1_idx)
                all_weights.append(weight)
                all_deviates.append(deviates)

                if winner == human_color:
                    n_pos_won += 1
                else:
                    n_pos_draw += 1

    # ── save ───────────────────────────────────────────────────────────────────
    n = len(all_chosen_idxs)
    if n == 0:
        print("[hgen] No positions extracted — check games directory.")
        return

    feat_arr = np.empty(n, dtype=object)
    for i, fm in enumerate(all_feat_matrices):
        feat_arr[i] = fm

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        feat_matrices=feat_arr,
        value_inputs=np.array(all_value_inputs,  dtype=np.float32),
        chosen_idxs= np.array(all_chosen_idxs,   dtype=np.int32),
        h_evals=     np.array(all_h_evals,        dtype=np.float32),
        vn_evals=    np.array(all_vn_evals,        dtype=np.float32),
        h_top1_idxs= np.array(all_h_top1_idxs,   dtype=np.int32),
        weights=     np.array(all_weights,         dtype=np.float32),
        deviates=    np.array(all_deviates,        dtype=bool),
    )

    elapsed = time.time() - t_start
    print(f"\n[hgen] Saved {n} positions to {out_path}  ({elapsed:.0f}s)")
    print(f"[hgen] Games:  won={n_won}  draw={n_draw}  lost={n_lost}  skipped(self-play)={n_selfplay}")
    print(f"[hgen] Positions: from-won={n_pos_won}  from-draw={n_pos_draw}")
    print(f"[hgen] Human deviated from heuristic top-1: {n_deviates}/{n_pos_won} won-game moves")
    if n_errors:
        print(f"[hgen] Encoding errors skipped: {n_errors}")


def main() -> None:
    p = argparse.ArgumentParser(description="Extract human-game imitation dataset")
    p.add_argument(
        "--games-dir", default=str(_ROOT / "data" / "games"),
    )
    p.add_argument(
        "--out", default=str(_ROOT / "learned_ai" / "data" / "human_imitation.npz"),
    )
    p.add_argument(
        "--sentinel",
        default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"),
    )
    p.add_argument("--malom",       default="")
    p.add_argument("--value-net",   type=str,   default=str(_ROOT / "data" / "value_net.npz"))
    p.add_argument("--won-weight",  type=float, default=1.0)
    p.add_argument("--draw-weight", type=float, default=0.3)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
