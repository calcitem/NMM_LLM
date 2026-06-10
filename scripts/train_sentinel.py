"""scripts/train_sentinel.py — train the move-level sentinel from watched games.

The sentinel is a per-move quality scorer: each example is one candidate move
in one position, labelled with a single ``move_quality`` in [0, 1] from the
mover's perspective (1.0 = win, 0.5 = draw, 0.0 = loss). Labels come from the
external solved DB when available; otherwise a weak heuristic label is used.

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


def _eval_metrics(model, loader, device):
    """Mean BCE loss plus accuracy and a win/draw/loss breakdown over a loader.

    A prediction is "correct" when ``round`` agreement holds: pred >= 0.5 iff
    target >= 0.5. The WDL breakdown reports per-bucket accuracy using the same
    >=0.5 / <=0.5 / ~0.5 banding the dataset uses for labels.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = total = 0
    buckets = {"win": [0, 0], "draw": [0, 0], "loss": [0, 0]}  # [correct, count]
    with torch.no_grad():
        for feats, quality, weight in loader:
            feats = feats.to(device)
            quality = quality.to(device)
            weight = weight.to(device)
            out = model(feats)
            losses = sentinel_loss(out, quality, sample_weight=weight)
            total_loss += float(losses["total"])
            n_batches += 1

            p = out.reshape(-1).cpu().numpy()
            g = quality.reshape(-1).cpu().numpy()
            pred_pos = p >= 0.5
            gold_pos = g >= 0.5
            correct += int(np.sum(pred_pos == gold_pos))
            total += len(g)
            for i in range(len(g)):
                if g[i] >= 0.99:
                    b = "win"
                    ok = p[i] >= 0.5
                elif g[i] <= 0.01:
                    b = "loss"
                    ok = p[i] < 0.5
                elif abs(g[i] - 0.5) < 1e-3:
                    b = "draw"
                    ok = abs(p[i] - 0.5) < 0.25
                else:
                    continue
                buckets[b][1] += 1
                buckets[b][0] += int(ok)

    val_loss = total_loss / max(1, n_batches)
    acc = correct / max(1, total)
    wdl = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in buckets.items()}
    return val_loss, acc, wdl


def main() -> int:
    p = argparse.ArgumentParser(description="Train the move-level sentinel")
    p.add_argument("--config", default=None)
    p.add_argument("--game-dir", default="data/games")
    p.add_argument("--dataset", default=None, help="Preprocessed .npz (skips replay)")
    p.add_argument("--db-path", default="")
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="max game files")
    p.add_argument("--decisive-only", action="store_true",
                   help="Exclude draw/unknown games; train only on win/loss outcomes")
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
        n = len(dataset)
        if n == 0:
            print("No training examples found — nothing to train.")
            return 1
        print(f"Dataset: {n} examples. Quality: {dataset.quality_distribution()}")
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
    else:
        db = ExternalSolvedDB(
            db_path=args.db_path or config.external_db_path,
            enabled=bool(args.db_path) or config.external_db_enabled,
        )
        print(f"External DB available: {db.is_available()}")
        # Game-level split: whole game files go to either train or val,
        # so no ply from the same game leaks across the split boundary.
        train_ds, val_ds = SentinelDataset.game_level_split(
            args.game_dir,
            val_fraction=config.val_fraction,
            db=db,
            config=config,
            seed=config.seed,
            limit=args.limit,
            decisive_only=args.decisive_only,
        )
        n = len(train_ds)
        if n == 0:
            print("No training examples found — nothing to train.")
            return 1
        print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")
        print(f"Train quality: {train_ds.quality_distribution()}")
        print(f"Train sources: {train_ds.source_distribution()}")

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
        running = 0.0
        n_batches = 0
        for feats, quality, weight in train_loader:
            feats = feats.to(device)
            quality = quality.to(device)
            weight = weight.to(device)
            out = model(feats)
            losses = sentinel_loss(out, quality, sample_weight=weight)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            running += float(losses["total"].detach())
            n_batches += 1

        n_batches = max(1, n_batches)
        train_loss = running / n_batches

        # validation
        val_loss = float("nan")
        acc = float("nan")
        wdl = {"win": float("nan"), "draw": float("nan"), "loss": float("nan")}
        if val_loader is not None:
            val_loss, acc, wdl = _eval_metrics(model, val_loader, device)

        dt = time.time() - t0
        print(
            f"epoch {epoch + 1}/{config.epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f} acc={acc:.3f} "
            f"[win={wdl['win']:.3f} draw={wdl['draw']:.3f} loss={wdl['loss']:.3f}] "
            f"({dt:.1f}s)"
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
