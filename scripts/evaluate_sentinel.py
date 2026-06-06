"""scripts/evaluate_sentinel.py — offline sentinel evaluation metrics.

Reports turning-point precision/recall, mistake-risk calibration (reliability
bins), opportunity-detection quality, and an early-warning rate (how often the
model flags a turning point in the plies leading up to a confirmed one).

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
    """Run the model over every example; return prediction/target arrays."""
    model = advisor.model
    model.eval()
    feats = np.stack([e.state_features for e in dataset.examples]).astype(np.float32)
    x = torch.from_numpy(feats).to(device)
    with torch.no_grad():
        out = model(x)
    preds = {
        "mistake_risk": out.mistake_risk.reshape(-1).cpu().numpy(),
        "opportunity_score": out.opportunity_score.reshape(-1).cpu().numpy(),
        "trajectory_value_delta": out.trajectory_value_delta.reshape(-1).cpu().numpy(),
        "turning_point_confidence": out.turning_point_confidence.reshape(-1).cpu().numpy(),
    }
    targets = {
        "mistake_risk": np.array([e.mistake_risk for e in dataset.examples]),
        "opportunity_score": np.array([e.opportunity_score for e in dataset.examples]),
        "turning_point_confidence": np.array(
            [e.turning_point_confidence for e in dataset.examples]),
        "ply": np.array([e.ply for e in dataset.examples]),
    }
    return preds, targets


def _precision_recall(pred_pos, gold_pos):
    tp = int(np.sum(pred_pos & gold_pos))
    fp = int(np.sum(pred_pos & ~gold_pos))
    fn = int(np.sum(~pred_pos & gold_pos))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _calibration(pred, gold_binary, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (pred >= lo) & (pred < hi if i < bins - 1 else pred <= hi)
        if not np.any(mask):
            continue
        rows.append((f"[{lo:.1f},{hi:.1f})", int(np.sum(mask)),
                     float(np.mean(pred[mask])), float(np.mean(gold_binary[mask]))))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate the sentinel overlay")
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
        dataset = SentinelDataset.load_from_games(
            args.game_dir, db=ExternalSolvedDB(""), config=config, limit=args.limit
        )
    if len(dataset) == 0:
        print("No examples to evaluate.")
        return 1
    print(f"Evaluating on {len(dataset)} examples.\n")

    preds, targets = _gather(advisor, dataset, device)
    thr = config.turning_point_threshold

    # turning-point precision/recall
    tp_pred = preds["turning_point_confidence"] >= thr
    tp_gold = targets["turning_point_confidence"] >= 0.5
    prec, rec, f1 = _precision_recall(tp_pred, tp_gold)
    print(f"Turning-point detection @ threshold {thr}:")
    print(f"  precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    print(f"  positives predicted={int(np.sum(tp_pred))} gold={int(np.sum(tp_gold))}\n")

    # mistake-risk calibration
    print("Mistake-risk reliability (bin, n, mean_pred, mean_gold):")
    for row in _calibration(preds["mistake_risk"], (targets["mistake_risk"] >= 0.5).astype(float)):
        print(f"  {row[0]} n={row[1]:5d} pred={row[2]:.3f} gold={row[3]:.3f}")
    print()

    # opportunity detection
    op_pred = preds["opportunity_score"] >= 0.6
    op_gold = targets["opportunity_score"] >= 0.6
    oprec, orec, of1 = _precision_recall(op_pred, op_gold)
    print(f"Opportunity detection @0.6: precision={oprec:.3f} recall={orec:.3f} f1={of1:.3f}\n")

    # early-warning rate: of confirmed turning points (gold), what fraction have
    # the model already flagging high confidence at ply-1 or ply-2 before them?
    # Approximated here within the flattened dataset by checking the preceding
    # examples' predictions (dataset is ply-ordered per game in load order).
    confirmed_idx = np.where(tp_gold)[0]
    early = 0
    for idx in confirmed_idx:
        lookback = preds["turning_point_confidence"][max(0, idx - 2):idx]
        if lookback.size and np.any(lookback >= thr):
            early += 1
    ew_rate = early / len(confirmed_idx) if len(confirmed_idx) else 0.0
    print(f"Early-warning rate (flag within 2 plies before a turning point): {ew_rate:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
