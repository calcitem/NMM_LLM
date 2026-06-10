"""scripts/evaluate_sentinel.py — offline move-level sentinel evaluation.

PRIMARY: Trajectory-level evaluation — every played move in every decisive game
is scored by the sentinel and checked against the Malom solved DB.  Moves are
grouped into winning trajectories (made by the eventual winner) and losing
trajectories (made by the eventual loser).  This is the correct test of whether
the sentinel can distinguish good moves from bad ones in real game play.

SECONDARY: Flat per-move statistics (all legal moves in each position, same as
training) — aggregate accuracy, WDL breakdown, calibration.

Usage:
    python scripts/evaluate_sentinel.py \\
        --checkpoint learned_ai/sentinel/checkpoints/best.pt \\
        [--game-dir data/games] \\
        [--db-path "/mnt/windows/NMM_DB/Entire DB"] \\
        [--dataset processed.npz] \\
        [--device cpu] [--limit N] [--no-flat]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from learned_ai.sentinel.config import load_config
from learned_ai.sentinel.dataset import (
    SentinelDataset,
    _board_from_fen_before,
    _enumerate_legal_moves,
    _heuristic_scores,
    _iter_game_records,
    _normalise_scores,
    _ranks_desc,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.sentinel.feature_builder import build_move_features
from learned_ai.sentinel.infer import SentinelAdvisor
from learned_ai.sentinel.labels import label_move


# ── Trajectory evaluation ─────────────────────────────────────────────────────

def _extract_played_example(board, played_move, player, db):
    """
    Compute the sentinel feature vector and ground-truth quality label for the
    specific move that was played on ``board``.

    Returns (feature_vector, quality, supervision_source) or None if the played
    move cannot be matched against the legal-move list.
    """
    moves = _enumerate_legal_moves(board, player)
    if not moves:
        return None

    raw = _heuristic_scores(board, moves, player)
    norm = _normalise_scores(raw)
    ranks = _ranks_desc(raw)
    n_legal = len(moves)

    all_db_moves: List[Dict] = []
    if db is not None and db.is_available():
        try:
            all_db_moves = db.query_all_moves(board, player)
        except Exception:
            pass

    wdl_by_key: Dict[tuple, str] = {}
    for entry in all_db_moves:
        mv = entry.get("move", {})
        key = (mv.get("from"), mv.get("to"), mv.get("capture"))
        wdl_by_key[key] = entry.get("wdl", "unknown")

    # Match played move against legal-move list (no truncation cap for eval)
    p_from = played_move.get("from")
    p_to = played_move.get("to")
    p_cap = played_move.get("capture")

    match_idx = None
    for i, mv in enumerate(moves):
        if mv.get("from") == p_from and mv.get("to") == p_to and mv.get("capture") == p_cap:
            match_idx = i
            break

    if match_idx is None:
        return None

    mv = moves[match_idx]
    key = (mv.get("from"), mv.get("to"), mv.get("capture"))
    wdl = wdl_by_key.get(key)
    quality, _weight, source = label_move(wdl, heuristic_score_norm=norm[match_idx])

    ctx = {
        "all_moves": all_db_moves,
        "heuristic_rank": ranks[match_idx],
        "n_legal": n_legal,
        "heuristic_score_norm": norm[match_idx],
    }
    try:
        feat = build_move_features(board, mv, player, ctx)
    except Exception:
        return None

    return np.array(feat, dtype=np.float32), float(quality), source


def collect_trajectory_data(game_dir, db, limit=None):
    """
    Walk every decisive game file and extract one example per played move.

    Returns:
        feats         (N, FEATURE_DIM) float32
        qualities     (N,) float32 — ground-truth label [0,1]
        sources       list[str]   — "solved_db" | "heuristic_weak"
        trajs         list[str]   — "winning" | "losing"
        game_indices  list[int]   — game id (0-based, decisive games only)
        n_decisive    int
        n_skipped     int         — draw / unknown games
    """
    all_feats: list = []
    all_qualities: list = []
    all_sources: list = []
    all_trajs: list = []
    game_indices: list = []
    n_decisive = 0
    n_skipped = 0

    paths = sorted(glob.glob(os.path.join(game_dir, "**", "*.jsonl"), recursive=True))
    if limit:
        paths = paths[:limit]

    for path_idx, path in enumerate(paths):
        for record in _iter_game_records(path):
            winner = record.get("winner")
            if winner is None:
                n_skipped += 1
                continue

            game_id = n_decisive
            n_decisive += 1

            for log_move in (record.get("moves") or []):
                fen = log_move.get("board_fen_before")
                if not fen:
                    continue
                board = _board_from_fen_before(fen)
                if board is None:
                    continue
                color = log_move.get("color") or getattr(board, "turn", "W")
                result = _extract_played_example(board, log_move, color, db)
                if result is None:
                    continue
                feat, quality, source = result
                all_feats.append(feat)
                all_qualities.append(quality)
                all_sources.append(source)
                all_trajs.append("winning" if color == winner else "losing")
                game_indices.append(game_id)

        if (path_idx + 1) % 100 == 0:
            print(f"  ... processed {path_idx + 1}/{len(paths)} files", flush=True)

    if not all_feats:
        return None, None, [], [], [], n_decisive, n_skipped

    return (
        np.stack(all_feats).astype(np.float32),
        np.array(all_qualities, dtype=np.float32),
        all_sources,
        all_trajs,
        game_indices,
        n_decisive,
        n_skipped,
    )


def _run_trajectory_eval(advisor, game_dir, db, device, limit=None):
    print("Collecting trajectory data (all played moves in decisive games) ...")
    feats, qualities, sources, trajs, game_indices, n_decisive, n_skipped = \
        collect_trajectory_data(game_dir, db, limit=limit)

    if feats is None:
        print("  No trajectory data found.")
        return

    print(f"  {n_decisive} decisive games | {n_skipped} draw/unknown skipped")
    print(f"  {len(trajs)} played moves collected\n")

    # Batch inference over all played moves
    model = advisor.model
    model.eval()
    x = torch.from_numpy(feats).to(device)
    with torch.no_grad():
        preds = model(x).reshape(-1).cpu().numpy()

    trajs_arr = np.array(trajs, dtype=object)
    sources_arr = np.array(sources, dtype=object)
    game_idx_arr = np.array(game_indices)

    print("=== Trajectory-Level Evaluation ===\n")

    # Per-trajectory breakdown
    for traj in ("winning", "losing"):
        mask = trajs_arr == traj
        n = int(np.sum(mask))
        if n == 0:
            continue

        p = preds[mask]
        g = qualities[mask]
        src = sources_arr[mask]

        # Accuracy: sentinel side-of-0.5 matches ground-truth side-of-0.5
        acc = float(np.mean((p >= 0.5) == (g >= 0.5)))
        mean_pred = float(np.mean(p))
        frac_pred_good = float(np.mean(p >= 0.5))
        frac_gt_good = float(np.mean(g >= 0.5))
        n_db = int(np.sum(src == "solved_db"))

        print(f"{traj.capitalize()} trajectory  (n={n}, "
              f"Malom DB labelled={n_db} / {100*n_db/n:.0f}%)")
        print(f"  Sentinel accuracy vs Malom DB:    {acc:.3f}")
        print(f"  Fraction sentinel scores >=0.5:   {frac_pred_good:.3f}")
        print(f"  Fraction Malom DB quality >=0.5:  {frac_gt_good:.3f}")
        print(f"  Mean sentinel score:              {mean_pred:.3f}")

        # DB-only slice (highest trust)
        db_mask = src == "solved_db"
        if db_mask.any():
            db_acc = float(np.mean((p[db_mask] >= 0.5) == (g[db_mask] >= 0.5)))
            print(f"  Malom DB-only accuracy:           {db_acc:.3f}")
        print()

    # Combined across both trajectory types
    acc_all = float(np.mean((preds >= 0.5) == (qualities >= 0.5)))
    mae_all = float(np.mean(np.abs(preds - qualities)))
    print(f"Combined (winning + losing) accuracy: {acc_all:.3f}  MAE: {mae_all:.3f}\n")

    # Game-level polarity: does winner's mean score beat loser's mean score?
    print("Game-level trajectory polarity  (winner mean score > loser mean score):")
    n_polarity = 0
    n_correct = 0
    win_means: list = []
    loss_means: list = []
    max_gid = int(game_idx_arr.max()) if len(game_idx_arr) else -1
    for gid in range(max_gid + 1):
        gm = game_idx_arr == gid
        gw = gm & (trajs_arr == "winning")
        gl = gm & (trajs_arr == "losing")
        if not np.any(gw) or not np.any(gl):
            continue
        wm = float(np.mean(preds[gw]))
        lm = float(np.mean(preds[gl]))
        win_means.append(wm)
        loss_means.append(lm)
        n_polarity += 1
        if wm > lm:
            n_correct += 1

    if n_polarity:
        pct = 100 * n_correct / n_polarity
        print(f"  {n_correct}/{n_polarity} games ({pct:.1f}%) where mean winner score > mean loser score")
        print(f"  Mean winner score per game:  {np.mean(win_means):.3f} ± {np.std(win_means):.3f}")
        print(f"  Mean loser  score per game:  {np.mean(loss_means):.3f} ± {np.std(loss_means):.3f}")
    print()


# ── Flat evaluation (all legal moves, matches training distribution) ──────────

def _gather(advisor, dataset, device):
    model = advisor.model
    model.eval()
    feats = np.stack([e.features for e in dataset.examples]).astype(np.float32)
    x = torch.from_numpy(feats).to(device)
    with torch.no_grad():
        out = model(x).reshape(-1)
    preds = out.cpu().numpy()
    targets = np.array([e.move_quality for e in dataset.examples], dtype=np.float32)
    sources = np.array([e.supervision_source for e in dataset.examples], dtype=object)
    return preds, targets, sources


def _calibration(pred, gold, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (pred >= lo) & (pred < hi if i < bins - 1 else pred <= hi)
        if not np.any(mask):
            continue
        rows.append((f"[{lo:.1f},{hi:.1f})", int(np.sum(mask)),
                     float(np.mean(pred[mask])), float(np.mean(gold[mask]))))
    return rows


def _run_flat_eval(advisor, game_dir, db, config, device, dataset_path=None, limit=None):
    print("=== Flat Per-Move Statistics (all legal moves per position) ===\n")
    if dataset_path and os.path.exists(dataset_path):
        dataset = SentinelDataset.load_from_disk(dataset_path)
        print(f"Loaded preprocessed dataset from {dataset_path}")
    else:
        dataset = SentinelDataset.load_from_games(
            game_dir, db=db, config=config, limit=limit
        )
    if len(dataset) == 0:
        print("  No flat examples available.")
        return

    print(f"Evaluating on {len(dataset)} examples (all candidates per position).\n")
    preds, targets, sources = _gather(advisor, dataset, device)

    pred_pos = preds >= 0.5
    gold_pos = targets >= 0.5
    acc = float(np.mean(pred_pos == gold_pos))
    mae = float(np.mean(np.abs(preds - targets)))
    print(f"Overall:  accuracy(>=0.5)={acc:.3f}  MAE={mae:.3f}\n")

    print("Win/draw/loss accuracy:")
    for name, sel, ok in (
        ("win",  targets >= 0.99, preds >= 0.5),
        ("draw", np.abs(targets - 0.5) < 1e-3, np.abs(preds - 0.5) < 0.25),
        ("loss", targets <= 0.01, preds < 0.5),
    ):
        n = int(np.sum(sel))
        a = float(np.mean(ok[sel])) if n else float("nan")
        print(f"  {name:5s} n={n:6d} acc={a:.3f}")
    print()

    print("By supervision source:")
    for src in sorted(set(sources.tolist())):
        m = sources == src
        n = int(np.sum(m))
        a = float(np.mean((preds[m] >= 0.5) == (targets[m] >= 0.5))) if n else float("nan")
        print(f"  {src:14s} n={n:6d} accuracy={a:.3f}")
    print()

    print("Move-quality reliability (bin, n, mean_pred, mean_gold):")
    for row in _calibration(preds, targets):
        print(f"  {row[0]} n={row[1]:6d} pred={row[2]:.3f} gold={row[3]:.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate the move-level sentinel")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--game-dir", default="data/games")
    p.add_argument("--dataset", default=None, help="Preprocessed .npz for flat eval")
    p.add_argument("--db-path", default="", help="Path to Malom solved DB for ground truth")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="Max game files to process")
    p.add_argument("--no-flat", action="store_true",
                   help="Skip the flat (all-legal-moves) evaluation section")
    args = p.parse_args()

    config = load_config(args.config)
    advisor = SentinelAdvisor(args.checkpoint, config=config, device=args.device)
    if not advisor.is_loaded():
        print(f"Failed to load checkpoint {args.checkpoint}")
        return 1
    config = advisor.config
    device = torch.device(args.device)

    db = ExternalSolvedDB(
        db_path=args.db_path or config.external_db_path,
        enabled=bool(args.db_path) or config.external_db_enabled,
    )
    print(f"Malom DB available: {db.is_available()}\n")

    _run_trajectory_eval(advisor, args.game_dir, db, device, limit=args.limit)

    if not args.no_flat:
        _run_flat_eval(
            advisor, args.game_dir, db, config, device,
            dataset_path=args.dataset, limit=args.limit,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
