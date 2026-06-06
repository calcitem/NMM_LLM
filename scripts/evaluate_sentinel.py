"""scripts/evaluate_sentinel.py — offline move-level sentinel evaluation.

Reports move-quality accuracy (pred>=0.5 vs label>=0.5), a win/draw/loss
breakdown, and a calibration (reliability bins) of predicted vs. observed
move quality.

Usage:
    python scripts/evaluate_sentinel.py --checkpoint learned_ai/sentinel/checkpoints/best.pt
        [--game-dir data/games] [--dataset processed.npz] [--device cpu]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from learned_ai.sentinel.config import load_config
from learned_ai.sentinel.dataset import SentinelDataset
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.sentinel.infer import SentinelAdvisor


def _gather(advisor: SentinelAdvisor, dataset: SentinelDataset, device):
    """Run the model over every example; return (preds, targets, sources)."""
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


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate the move-level sentinel")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--game-dir", default="data/games")
    p.add_argument("--dataset", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    config = load_config(args.config)
    advisor = SentinelAdvisor(args.checkpoint, config=config, device=args.device)
    if not advisor.is_loaded():
        print(f"Failed to load checkpoint {args.checkpoint}")
        return 1
    config = advisor.config
    device = torch.device(args.device)

    if args.dataset and os.path.exists(args.dataset):
        dataset = SentinelDataset.load_from_disk(args.dataset)
    else:
        db = ExternalSolvedDB(
            db_path=config.external_db_path,
            enabled=config.external_db_enabled,
        )
        print(f"External DB available: {db.is_available()} (ground-truth labels)")
        dataset = SentinelDataset.load_from_games(
            args.game_dir, db=db, config=config, limit=args.limit
        )
    if len(dataset) == 0:
        print("No examples to evaluate.")
        return 1
    print(f"Evaluating on {len(dataset)} examples.\n")

    preds, targets, sources = _gather(advisor, dataset, device)

    # overall accuracy: agreement on the >=0.5 boundary
    pred_pos = preds >= 0.5
    gold_pos = targets >= 0.5
    acc = float(np.mean(pred_pos == gold_pos))
    mae = float(np.mean(np.abs(preds - targets)))
    print(f"Overall: accuracy(>=0.5)={acc:.3f}  MAE={mae:.3f}\n")

    # win/draw/loss breakdown
    print("Win/draw/loss accuracy:")
    for name, sel, ok in (
        ("win", targets >= 0.99, preds >= 0.5),
        ("draw", np.abs(targets - 0.5) < 1e-3, np.abs(preds - 0.5) < 0.25),
        ("loss", targets <= 0.01, preds < 0.5),
    ):
        n = int(np.sum(sel))
        a = float(np.mean(ok[sel])) if n else float("nan")
        print(f"  {name:5s} n={n:6d} acc={a:.3f}")
    print()

    # supervision-source breakdown
    print("By supervision source:")
    for src in sorted(set(sources.tolist())):
        m = sources == src
        n = int(np.sum(m))
        a = float(np.mean((preds[m] >= 0.5) == (targets[m] >= 0.5))) if n else float("nan")
        print(f"  {src:14s} n={n:6d} accuracy={a:.3f}")
    print()

    # calibration of predicted vs. observed quality
    print("Move-quality reliability (bin, n, mean_pred, mean_gold):")
    for row in _calibration(preds, targets):
        print(f"  {row[0]} n={row[1]:6d} pred={row[2]:.3f} gold={row[3]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
