"""scripts/train_sentinel.py — train the sentinel overlay from watched games.

Trajectory-supervised training. The external solved DB is used as a teacher when
available; otherwise training falls back to game-outcome proxy supervision (the
model still learns, accuracy is just lower).

Usage:
    python scripts/train_sentinel.py [--config configs/sentinel_default.yaml]
                                     [--game-dir data/games]
                                     [--dataset processed.npz]
                                     [--db-path "/mnt/windows/NMM_DB/Entire DB"]
                                     [--resume checkpoint.pt]
                                     [--epochs N] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from learned_ai.sentinel.config import load_config
from learned_ai.sentinel.dataset import SentinelDataset, collate_examples
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.sentinel.model import SentinelNet, sentinel_loss


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _turning_point_pr(model, loader, device, threshold: float):
    """Compute turning-point precision/recall over a loader (target>=0.5 positive)."""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for feats, targets in loader:
            feats = feats.to(device)
            out = model(feats)
            pred = (out.turning_point_confidence.reshape(-1).cpu().numpy() >= threshold)
            gold = (targets["turning_point_confidence"].numpy() >= 0.5)
            tp += int(np.sum(pred & gold))
            fp += int(np.sum(pred & ~gold))
            fn += int(np.sum(~pred & gold))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def main() -> int:
    p = argparse.ArgumentParser(description="Train the sentinel overlay")
    p.add_argument("--config", default=None)
    p.add_argument("--game-dir", default="data/games")
    p.add_argument("--dataset", default=None, help="Preprocessed .npz (skips replay)")
    p.add_argument("--db-path", default="")
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="max game files")
    args = p.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    _set_seed(config.seed)
    device = torch.device(args.device)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────────
    if args.dataset and os.path.exists(args.dataset):
        print(f"Loading preprocessed dataset from {args.dataset}")
        dataset = SentinelDataset.load_from_disk(args.dataset)
    else:
        db = ExternalSolvedDB(
            db_path=args.db_path or config.external_db_path,
            enabled=bool(args.db_path) or config.external_db_enabled,
        )
        print(f"External DB available: {db.is_available()}")
        dataset = SentinelDataset.load_from_games(
            args.game_dir, db=db, config=config, limit=args.limit
        )
    n = len(dataset)
    if n == 0:
        print("No training examples found — nothing to train.")
        return 1
    print(f"Dataset: {n} examples. Classes: {dataset.class_distribution()}")
    print(f"Supervision sources: {dataset.source_distribution()}")

    n_val = max(1, int(n * config.val_fraction)) if n > 1 else 0
    n_train = n - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.seed),
        )
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_examples,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                   collate_fn=collate_examples)
        if val_ds is not None else None
    )

    # ── Model ───────────────────────────────────────────────────────────────────
    model = SentinelNet(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    start_epoch = 0
    best_val = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", best_val)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    def _save(path: str, epoch: int) -> None:
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
            "best_val": best_val,
        }, path)

    # ── Training loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.epochs):
        model.train()
        t0 = time.time()
        running = {"total": 0.0, "mistake_risk": 0.0, "opportunity_score": 0.0,
                   "trajectory_value_delta": 0.0, "turning_point_confidence": 0.0}
        n_batches = 0
        for feats, targets in train_loader:
            feats = feats.to(device)
            tg = {k: targets[k].to(device) for k in targets}
            weight = tg.pop("weight", None)
            out = model(feats)
            losses = sentinel_loss(out, tg, sample_weight=weight,
                                   loss_weights=config.loss_weights)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            running["total"] += float(losses["total"].detach())
            for k in ("mistake_risk", "opportunity_score",
                      "trajectory_value_delta", "turning_point_confidence"):
                running[k] += float(losses[k])
            n_batches += 1

        n_batches = max(1, n_batches)
        train_loss = running["total"] / n_batches

        # validation
        val_loss = float("nan")
        prec = rec = float("nan")
        if val_loader is not None:
            model.eval()
            v_total = 0.0
            v_batches = 0
            with torch.no_grad():
                for feats, targets in val_loader:
                    feats = feats.to(device)
                    tg = {k: targets[k].to(device) for k in targets}
                    weight = tg.pop("weight", None)
                    out = model(feats)
                    losses = sentinel_loss(out, tg, sample_weight=weight,
                                           loss_weights=config.loss_weights)
                    v_total += float(losses["total"])
                    v_batches += 1
            val_loss = v_total / max(1, v_batches)
            prec, rec = _turning_point_pr(
                model, val_loader, device, config.turning_point_threshold
            )

        dt = time.time() - t0
        print(
            f"epoch {epoch + 1}/{config.epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"[mistake={running['mistake_risk'] / n_batches:.3f} "
            f"opp={running['opportunity_score'] / n_batches:.3f} "
            f"delta={running['trajectory_value_delta'] / n_batches:.3f} "
            f"tp={running['turning_point_confidence'] / n_batches:.3f}] "
            f"tp_precision={prec:.3f} tp_recall={rec:.3f} ({dt:.1f}s)"
        )

        _save(os.path.join(config.checkpoint_dir, "latest.pt"), epoch + 1)
        cur_val = val_loss if val_loader is not None else train_loss
        if cur_val < best_val:
            best_val = cur_val
            _save(os.path.join(config.checkpoint_dir, "best.pt"), epoch + 1)

    print(f"Training complete. Best val/train loss: {best_val:.4f}")
    print(f"Checkpoints in {config.checkpoint_dir} (latest.pt, best.pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
