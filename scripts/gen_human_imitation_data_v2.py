"""scripts/gen_human_imitation_data_v2.py — Human-game imitation dataset with lookahead.

Identical to gen_human_imitation_data.py but encodes each position with the full
15-ply lookahead block (60 floats) appended to the base 62-float features, producing
(k, 122) feat_matrices compatible with the v2 specialist networks.

Requires: sentinel, value_net, and gap_net (all optional but recommended).

Output .npz arrays
------------------
  feat_matrices : (N,) object array of (k, 122) float32   ← 62 base + 60 lookahead
  value_inputs  : (N, 23) float32
  label_dists   : (N,) object array of (k,) float32
  chosen_idxs   : (N,) int32
  h_evals       : (N,) float32
  h_top1_idxs   : (N,) int32
  weights        : (N,) float32
  deviates       : (N,) bool
  is_winner      : (N,) bool

Usage
-----
    .venv/bin/python scripts/gen_human_imitation_data_v2.py [options]

Options
-------
  --games-dir  PATH  Directory of .jsonl game files (default data/games)
  --out        PATH  Output .npz (default learned_ai/data/human_imitation2.npz)
  --sentinel   PATH  SentinelAdvisor checkpoint
  --malom      PATH  Malom DB directory (read from data/settings.json if omitted)
  --value-net  PATH  ValueNet .npz
  --gap-net    PATH  GapNet .npz (default data/gap_net.npz)
  --won-weight F     Weight for won-game moves (default 1.0)
  --draw-weight F    Weight for drawn-game moves (default 0.3)
  --loser-weight F   Weight for loser-side positions (default 0.5)
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
from learned_ai.models.scaffolded_encoder import (
    encode_position_with_lookahead,
    MOVE_FEAT_DIM_WITH_LOOKAHEAD,
    LOOKAHEAD_FEAT_DIM,
    VALUE_INPUT_DIM,
)
from learned_ai.sentinel.infer import load_advisor
from learned_ai.sentinel.labels import dtm_quality

# ── Soft-label hyperparameters ─────────────────────────────────────────────────
_WDL_SCALE    = {"win": 1.0, "draw": 0.4, "loss": 0.1}
_SENTINEL_TEMP = 0.5
_HUMAN_ALPHA   = 0.4


def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _load_game_file(path: Path) -> list[dict]:
    content = path.read_text().strip()
    if not content:
        return []
    try:
        return [json.loads(content)]
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def _move_key(mv: dict) -> tuple:
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _compute_soft_label(db_moves, legal_moves, sentinel_scores) -> np.ndarray:
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

    s = np.array(sentinel_scores if sentinel_scores else [1.0 / k] * k, dtype=np.float64)
    s = np.exp((s - s.max()) / _SENTINEL_TEMP)
    return (s / s.sum()).astype(np.float32)


def _compute_human_soft_label(db_moves, legal_moves, sentinel_scores, human_idx) -> np.ndarray:
    soft = _compute_soft_label(db_moves, legal_moves, sentinel_scores)
    one_hot = np.zeros(len(legal_moves), dtype=np.float32)
    one_hot[human_idx] = 1.0
    return (1.0 - _HUMAN_ALPHA) * soft + _HUMAN_ALPHA * one_hot


def run(args: argparse.Namespace) -> None:
    t_start = time.time()

    # ── load sentinel ──────────────────────────────────────────────────────────
    sentinel = None
    if args.sentinel and Path(args.sentinel).exists():
        sentinel = load_advisor(args.sentinel)
        if sentinel and sentinel.is_loaded():
            print(f"[hgen2] Sentinel loaded from {args.sentinel}")
        else:
            sentinel = None
            print("[hgen2] Sentinel not available — sentinel features will be 0.5")
    else:
        print("[hgen2] No sentinel path given — sentinel features will be 0.5")

    # ── load Malom DB ──────────────────────────────────────────────────────────
    db = None
    malom_path = args.malom or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if db.is_available():
                print(f"[hgen2] Malom DB loaded from {malom_path}")
            else:
                db = None
        except Exception as e:
            print(f"[hgen2] Malom DB load failed ({e}) — sentinel fallback labels")
    else:
        print("[hgen2] No Malom DB — sentinel fallback labels")

    # ── load Value Net ─────────────────────────────────────────────────────────
    value_net = None
    vn_path = getattr(args, "value_net", None) or str(_ROOT / "data" / "value_net.npz")
    if vn_path and Path(vn_path).exists():
        try:
            from ai.value_net import ValueNet as _ValueNet
            value_net = _ValueNet.load(vn_path)
            print(f"[hgen2] Value net loaded from {vn_path}")
        except Exception as e:
            print(f"[hgen2] Value net load failed ({e}) — VN features will be 0")
    else:
        print("[hgen2] No value net — VN features will be 0")

    # ── load Gap Net ───────────────────────────────────────────────────────────
    gap_net = None
    gn_path = args.gap_net or str(_ROOT / "data" / "gap_net.npz")
    if gn_path and Path(gn_path).exists():
        try:
            from ai.gap_net import GapNet
            gap_net = GapNet.load(gn_path)
            print(f"[hgen2] Gap net loaded from {gn_path}")
        except Exception as e:
            print(f"[hgen2] Gap net load failed ({e}) — gap features will be 0.5")
    else:
        print("[hgen2] No gap net — gap features will be 0.5")

    # ── build LookaheadAdvisor ─────────────────────────────────────────────────
    from learned_ai.models.scaffolded_encoder import _heuristic_eval
    from learned_ai.models.lookahead_advisor import LookaheadAdvisor

    lookahead = LookaheadAdvisor(
        sentinel=sentinel,
        value_net=value_net,
        evaluate_fn=_heuristic_eval,
        gap_net=gap_net,
        use_sentinel=True,
        ply_depth=15,
    )
    print(f"[hgen2] LookaheadAdvisor: ply_depth=15, feat_dim={lookahead.feat_dim}, use_sentinel=True")
    print(f"[hgen2] Expected feat_matrix shape per position: (k, {MOVE_FEAT_DIM_WITH_LOOKAHEAD})")

    # ── scan game files ────────────────────────────────────────────────────────
    games_dir = Path(args.games_dir)
    game_files = sorted(games_dir.glob("*.jsonl"))
    print(f"[hgen2] Found {len(game_files)} game files in {games_dir}")

    all_feat_matrices: list[np.ndarray] = []
    all_value_inputs:  list[np.ndarray] = []
    all_label_dists:   list[np.ndarray] = []
    all_chosen_idxs:   list[int]        = []
    all_h_evals:       list[float]      = []
    all_vn_evals:      list[float]      = []
    all_h_top1_idxs:   list[int]        = []
    all_weights:       list[float]      = []
    all_deviates:      list[bool]       = []
    all_is_winner:     list[bool]       = []

    n_won = n_draw = n_lost = n_selfplay = 0
    n_pos_won = n_pos_draw = n_pos_loser = 0
    n_deviates = 0
    n_db_labels = n_sentinel_labels = 0
    n_errors = 0
    n_files = len(game_files)

    for file_i, game_file in enumerate(game_files):
        if (file_i + 1) % 50 == 0 or file_i == 0:
            n_pos = len(all_chosen_idxs)
            elapsed = time.time() - t_start
            print(
                f"[hgen2] {file_i+1}/{n_files} files | "
                f"positions {n_pos} | won={n_won} draw={n_draw} | {elapsed:.0f}s",
                flush=True,
            )

        for game in _load_game_file(game_file):
            human_color = game.get("human_color")
            winner      = game.get("winner")

            if game.get("self_play") or human_color in (None, "self_play"):
                n_selfplay += 1
                continue

            if winner == human_color:
                n_won += 1
            elif winner is None:
                n_draw += 1
            else:
                n_lost += 1
                continue

            moves  = game.get("moves", [])[:args.max_moves]
            weight = args.won_weight if winner == human_color else args.draw_weight

            # ── Winner/draw side: human player's moves ─────────────────────────
            is_w_pos = winner == human_color
            for mv in moves:
                if mv.get("color") != human_color:
                    continue
                fen = mv.get("board_fen_before")
                if not fen:
                    continue
                try:
                    board = BoardState.from_fen_string(fen)
                except Exception:
                    n_errors += 1
                    continue

                enc = encode_position_with_lookahead(
                    board, human_color,
                    sentinel_advisor=sentinel,
                    db=db,
                    value_net=value_net,
                    lookahead_advisor=lookahead,
                )
                if enc is None or not enc.legal_moves:
                    continue

                chosen_key = _move_key(mv)
                chosen_idx = next(
                    (i for i, m in enumerate(enc.legal_moves) if _move_key(m) == chosen_key),
                    None,
                )
                if chosen_idx is None:
                    n_errors += 1
                    continue

                label_dist = _compute_human_soft_label(
                    enc.db_moves, enc.legal_moves, enc.sentinel_scores, chosen_idx,
                )

                if enc.db_moves and any(e.get("wdl") in _WDL_SCALE for e in enc.db_moves):
                    n_db_labels += 1
                else:
                    n_sentinel_labels += 1

                deviates = (chosen_idx != enc.h_top1_idx)
                if deviates:
                    n_deviates += 1

                all_feat_matrices.append(enc.feat_matrix)
                all_value_inputs.append(enc.value_input)
                all_label_dists.append(label_dist)
                all_chosen_idxs.append(chosen_idx)
                all_h_evals.append(enc.h_before)
                all_vn_evals.append(enc.vn_before)
                all_h_top1_idxs.append(enc.h_top1_idx)
                all_weights.append(weight)
                all_deviates.append(deviates)
                all_is_winner.append(is_w_pos)

                if is_w_pos:
                    n_pos_won += 1
                else:
                    n_pos_draw += 1

            # ── Loser side: AI moves in human-won games ────────────────────────
            if winner == human_color:
                loser_color = "B" if human_color == "W" else "W"
                for mv in moves:
                    if mv.get("color") != loser_color:
                        continue
                    fen = mv.get("board_fen_before")
                    if not fen:
                        continue
                    try:
                        board = BoardState.from_fen_string(fen)
                    except Exception:
                        n_errors += 1
                        continue

                    enc = encode_position_with_lookahead(
                        board, loser_color,
                        sentinel_advisor=sentinel,
                        db=db,
                        value_net=value_net,
                        lookahead_advisor=lookahead,
                    )
                    if enc is None or not enc.legal_moves:
                        continue

                    chosen_key = _move_key(mv)
                    chosen_idx = next(
                        (i for i, m in enumerate(enc.legal_moves) if _move_key(m) == chosen_key),
                        None,
                    )
                    if chosen_idx is None:
                        n_errors += 1
                        continue

                    k = len(enc.legal_moves)
                    h_idx = enc.h_top1_idx
                    loser_label = np.full(k, 0.05 / max(k - 1, 1), dtype=np.float32)
                    if 0 <= h_idx < k:
                        loser_label[h_idx] = 0.95

                    all_feat_matrices.append(enc.feat_matrix)
                    all_value_inputs.append(enc.value_input)
                    all_label_dists.append(loser_label)
                    all_chosen_idxs.append(chosen_idx)
                    all_h_evals.append(enc.h_before)
                    all_vn_evals.append(enc.vn_before)
                    all_h_top1_idxs.append(enc.h_top1_idx)
                    all_weights.append(args.loser_weight)
                    all_deviates.append(False)
                    all_is_winner.append(False)
                    n_pos_loser += 1

    # ── save ───────────────────────────────────────────────────────────────────
    n = len(all_chosen_idxs)
    if n == 0:
        print("[hgen2] No positions extracted — check games directory.")
        return

    # Verify a sample to catch dimension mismatches before saving
    sample = all_feat_matrices[0]
    if sample.shape[1] != MOVE_FEAT_DIM_WITH_LOOKAHEAD:
        print(
            f"[hgen2] WARNING: feat_matrix width={sample.shape[1]}, "
            f"expected {MOVE_FEAT_DIM_WITH_LOOKAHEAD}"
        )

    feat_arr  = np.empty(n, dtype=object)
    label_arr = np.empty(n, dtype=object)
    for i, (fm, ld) in enumerate(zip(all_feat_matrices, all_label_dists)):
        feat_arr[i]  = fm
        label_arr[i] = ld

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        feat_matrices=feat_arr,
        value_inputs= np.array(all_value_inputs,  dtype=np.float32),
        label_dists=  label_arr,
        chosen_idxs=  np.array(all_chosen_idxs,   dtype=np.int32),
        h_evals=      np.array(all_h_evals,        dtype=np.float32),
        vn_evals=     np.array(all_vn_evals,       dtype=np.float32),
        h_top1_idxs=  np.array(all_h_top1_idxs,   dtype=np.int32),
        weights=      np.array(all_weights,         dtype=np.float32),
        deviates=     np.array(all_deviates,        dtype=bool),
        is_winner=    np.array(all_is_winner,       dtype=bool),
    )

    elapsed = time.time() - t_start
    print(f"\n[hgen2] Saved {n} positions to {out_path}  ({elapsed:.0f}s)")
    print(f"[hgen2] Games:  won={n_won}  draw={n_draw}  lost={n_lost}  skipped(self-play)={n_selfplay}")
    print(f"[hgen2] Positions: winner={n_pos_won}  draw={n_pos_draw}  loser={n_pos_loser}")
    print(f"[hgen2] Labels: {n_db_labels} Malom-DB  {n_sentinel_labels} sentinel-fallback")
    print(f"[hgen2] Human deviated from heuristic top-1: {n_deviates}/{n_pos_won} won-game moves")
    print(f"[hgen2] Feature width: {sample.shape[1]} ({MOVE_FEAT_DIM_WITH_LOOKAHEAD} expected)")
    if n_errors:
        print(f"[hgen2] Encoding errors skipped: {n_errors}")


def main() -> None:
    p = argparse.ArgumentParser(description="Human-game imitation dataset (v2, 122-float)")
    p.add_argument("--games-dir",    default=str(_ROOT / "data" / "games"))
    p.add_argument("--out",          default=str(_ROOT / "learned_ai" / "data" / "human_imitation2.npz"))
    p.add_argument("--sentinel",     default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"))
    p.add_argument("--malom",        default="")
    p.add_argument("--value-net",    type=str, default=str(_ROOT / "data" / "value_net.npz"))
    p.add_argument("--gap-net",      type=str, default=str(_ROOT / "data" / "gap_net.npz"))
    p.add_argument("--max-moves",    type=int,   default=120,
                   help="Cap moves per game (skip positions beyond this ply)")
    p.add_argument("--won-weight",   type=float, default=1.0)
    p.add_argument("--draw-weight",  type=float, default=0.3)
    p.add_argument("--loser-weight", type=float, default=0.5)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
