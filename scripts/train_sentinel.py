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
from learned_ai.sentinel.dataset import SentinelDataset, ContrastiveSentinelDataset, collate_examples
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.sentinel.feature_builder import FEATURE_DIM
from learned_ai.sentinel.model import SentinelNet, sentinel_loss, contrastive_ranking_loss

# DB-derived feature slots zeroed when --drop-db-features is active.
# Keeps structural slots [0-40, 46-47] but zeros the DB indicator / DTM slots.
_DB_FEATURE_SLOTS = list(range(41, 46)) + list(range(48, 58))


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _eval_metrics(model, loader, device, lambda_wdl=0.0, db_mask=None):
    """Mean loss plus accuracy and a win/draw/loss breakdown over a loader."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = total = 0
    buckets = {"win": [0, 0], "draw": [0, 0], "loss": [0, 0]}  # [correct, count]
    with torch.no_grad():
        for feats, quality, weight, wdl_cls in loader:
            feats = feats.to(device)
            if db_mask is not None:
                feats = feats * db_mask
            quality = quality.to(device)
            weight = weight.to(device)
            wdl_cls = wdl_cls.to(device)
            use_aux = model.aux_wdl and lambda_wdl > 0
            if use_aux:
                out, wdl_logits = model(feats, return_aux=True)
                losses = sentinel_loss(out, quality, sample_weight=weight,
                                       wdl_logits=wdl_logits, wdl_targets=wdl_cls,
                                       lambda_wdl=lambda_wdl)
            else:
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
                if g[i] > 0.5:
                    b = "win"
                    ok = p[i] >= 0.5
                elif g[i] < 0.5:
                    b = "loss"
                    ok = p[i] < 0.5
                else:
                    b = "draw"
                    ok = abs(p[i] - 0.5) < 0.25
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
    p.add_argument("--human-game-dir", default=None,
                   help="Extra game directory (e.g. data/human_games) merged into the corpus")
    p.add_argument("--dataset", default=None, help="Preprocessed .npz (skips replay)")
    p.add_argument("--db-path", default="")
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=None, help="max game files")
    p.add_argument("--decisive-only", action="store_true",
                   help="Exclude draw/unknown games; train only on win/loss outcomes")
    p.add_argument("--aux-wdl", action="store_true",
                   help="Enable auxiliary WDL classification head (improves calibration)")
    p.add_argument("--lambda-wdl", type=float, default=0.3,
                   help="Weight for auxiliary WDL loss term (default 0.3)")
    p.add_argument("--drop-db-features", action="store_true",
                   help="Zero DB-indicator feature slots during training so the model "
                        "cannot shortcut on oracle features unavailable at inference time")
    p.add_argument("--trajectory-weight", action="store_true",
                   help="Boost training weight of the played move based on game outcome "
                        "(Stage 3 curriculum: win-played moves get 3x, loss-played 2x)")
    p.add_argument("--ai-game-dir", default=None,
                   help="Extra AI-vs-AI game directory (e.g. data/ai_games); merged into corpus")
    p.add_argument("--contrastive", action="store_true",
                   help="Add contrastive ranking loss over same-position (good, bad) move pairs")
    p.add_argument("--lambda-contrastive", type=float, default=0.3,
                   help="Weight for contrastive ranking loss term (default 0.3)")
    p.add_argument("--curriculum", action="store_true",
                   help="Two-phase training: phase 1 freezes trunk (high LR), phase 2 full net (low LR)")
    p.add_argument("--lr-phase1", type=float, default=None,
                   help="Learning rate for curriculum phase 1 (default: config.lr = 1e-3)")
    p.add_argument("--lr-phase2", type=float, default=1e-4,
                   help="Learning rate for curriculum phase 2 (default 1e-4)")
    p.add_argument("--epochs-phase1", type=int, default=None,
                   help="Epochs for curriculum phase 1 (default = total epochs // 3)")
    p.add_argument("--out-dir", default=None,
                   help="Override checkpoint output directory (default: config.checkpoint_dir). "
                        "Use this to save a new version without clobbering the current best.pt, "
                        "e.g. --out-dir learned_ai/sentinel/checkpoints/v2")
    args = p.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.out_dir is not None:
        config.checkpoint_dir = args.out_dir
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
        extra_dirs = [d for d in [args.human_game_dir, args.ai_game_dir] if d]
        if extra_dirs:
            print(f"Extra game dirs: {extra_dirs}")
        if args.drop_db_features:
            print(f"DB feature slots zeroed (--drop-db-features): {_DB_FEATURE_SLOTS}")
        if args.trajectory_weight:
            print("Trajectory weighting active: played moves get win/loss outcome boost")
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
            extra_dirs=extra_dirs,
            trajectory_weight=args.trajectory_weight,
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
        aux_wdl=args.aux_wdl,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    start_epoch = 0
    best_val = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
        if incompatible.missing_keys:
            print(f"  Resume: new keys initialised randomly: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  Resume: unexpected keys ignored: {incompatible.unexpected_keys}")
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                print("  Resume: optimizer state incompatible, starting fresh optimizer")
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
            "aux_wdl": args.aux_wdl,
        }, path)

    use_aux = args.aux_wdl and args.lambda_wdl > 0
    lambda_wdl = args.lambda_wdl if use_aux else 0.0
    lambda_contrastive = args.lambda_contrastive if args.contrastive else 0.0

    # Build DB-feature mask (applied each batch when --drop-db-features is set).
    db_mask = None
    if args.drop_db_features:
        db_mask = torch.ones(FEATURE_DIM, dtype=torch.float32, device=device)
        for s in _DB_FEATURE_SLOTS:
            db_mask[s] = 0.0

    # ── Contrastive loader ───────────────────────────────────────────────────────
    contrastive_loader = None
    contrastive_iter = None
    if args.contrastive:
        # Build over the full training set so pairs see all positions.
        base_examples = list(train_ds.dataset.examples if hasattr(train_ds, "dataset")
                             else train_ds.examples)
        contrastive_ds = ContrastiveSentinelDataset(base_examples, max_pairs=200_000)
        if len(contrastive_ds) > 0:
            contrastive_loader = DataLoader(
                contrastive_ds, batch_size=config.batch_size, shuffle=True,
            )
            print(f"Contrastive pairs: {len(contrastive_ds):,} (lambda={lambda_contrastive})")
        else:
            print("Warning: no contrastive pairs found (need position_key populated)")
            lambda_contrastive = 0.0

    def _get_contrastive_iter():
        """Cycle through contrastive batches; refresh iterator when exhausted."""
        nonlocal contrastive_iter
        if contrastive_loader is None:
            return None
        if contrastive_iter is None:
            contrastive_iter = iter(contrastive_loader)
        try:
            return next(contrastive_iter)
        except StopIteration:
            contrastive_iter = iter(contrastive_loader)
            return next(contrastive_iter)

    # ── Curriculum helpers ───────────────────────────────────────────────────────
    def _freeze_trunk(freeze: bool) -> None:
        for name, p in model.named_parameters():
            if "trunk" in name:
                p.requires_grad_(not freeze)

    def _make_optimizer(lr: float) -> torch.optim.Optimizer:
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=lr)

    total_epochs = config.epochs
    if args.curriculum:
        p1_epochs = args.epochs_phase1 if args.epochs_phase1 is not None else max(1, total_epochs // 3)
        p2_epochs = total_epochs - p1_epochs
        p1_lr = args.lr_phase1 if args.lr_phase1 is not None else config.lr
        phase_schedule = [(1, p1_epochs, True, p1_lr), (2, p2_epochs, False, args.lr_phase2)]
        print(f"Curriculum: phase 1 = {p1_epochs} epochs (frozen trunk, lr={p1_lr})"
              f"  phase 2 = {p2_epochs} epochs (full, lr={args.lr_phase2})")
    else:
        phase_schedule = [(1, total_epochs, False, config.lr)]

    # ── Training loop ────────────────────────────────────────────────────────────
    epoch = start_epoch
    for phase_num, n_epochs, freeze_trunk, lr in phase_schedule:
        if n_epochs <= 0:
            continue
        _freeze_trunk(freeze_trunk)
        optimizer = _make_optimizer(lr)
        if args.curriculum:
            print(f"\n── Phase {phase_num} (freeze_trunk={freeze_trunk}, lr={lr}) ──")

        for _ in range(n_epochs):
            if epoch >= start_epoch + total_epochs and epoch > start_epoch:
                break
            model.train()
            t0 = time.time()
            running = 0.0
            n_batches = 0

            for feats, quality, weight, wdl_cls in train_loader:
                feats = feats.to(device)
                if db_mask is not None:
                    feats = feats * db_mask
                quality = quality.to(device)
                weight = weight.to(device)
                wdl_cls = wdl_cls.to(device)

                if use_aux:
                    out, wdl_logits = model(feats, return_aux=True)
                    losses = sentinel_loss(out, quality, sample_weight=weight,
                                           wdl_logits=wdl_logits, wdl_targets=wdl_cls,
                                           lambda_wdl=lambda_wdl)
                else:
                    out = model(feats)
                    losses = sentinel_loss(out, quality, sample_weight=weight)

                total_loss = losses["total"]

                # Contrastive term: interleave pairs from the contrastive loader.
                if lambda_contrastive > 0 and contrastive_loader is not None:
                    pair_batch = _get_contrastive_iter()
                    if pair_batch is not None:
                        feat_g, feat_b = pair_batch
                        feat_g = feat_g.to(device)
                        feat_b = feat_b.to(device)
                        if db_mask is not None:
                            feat_g = feat_g * db_mask
                            feat_b = feat_b * db_mask
                        scores_g = model(feat_g)
                        scores_b = model(feat_b)
                        c_loss = contrastive_ranking_loss(scores_g, scores_b)
                        total_loss = total_loss + lambda_contrastive * c_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                running += float(total_loss.detach())
                n_batches += 1

            n_batches = max(1, n_batches)
            train_loss = running / n_batches

            val_loss = float("nan")
            acc = float("nan")
            wdl = {"win": float("nan"), "draw": float("nan"), "loss": float("nan")}
            if val_loader is not None:
                val_loss, acc, wdl = _eval_metrics(model, val_loader, device, lambda_wdl, db_mask=db_mask)

            dt = time.time() - t0
            phase_tag = f"[p{phase_num}] " if args.curriculum else ""
            print(
                f"{phase_tag}epoch {epoch + 1}/{start_epoch + total_epochs} "
                f"train={train_loss:.4f} val={val_loss:.4f} acc={acc:.3f} "
                f"[win={wdl['win']:.3f} draw={wdl['draw']:.3f} loss={wdl['loss']:.3f}] "
                f"({dt:.1f}s)"
            )

            _save(os.path.join(config.checkpoint_dir, "latest.pt"), epoch + 1)
            cur_val = val_loss if val_loader is not None else train_loss
            if cur_val < best_val:
                best_val = cur_val
                _save(os.path.join(config.checkpoint_dir, "best.pt"), epoch + 1)

            epoch += 1

    print(f"Training complete. Best val/train loss: {best_val:.4f}")
    print(f"Checkpoints in {config.checkpoint_dir} (latest.pt, best.pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
