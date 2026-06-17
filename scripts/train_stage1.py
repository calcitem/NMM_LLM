"""scripts/train_stage1.py — Stage 1: imitation learning from human games.

Trains the NMMNet policy heads (phase_heads) using cross-entropy loss against human
moves from HumanDB, weighted by each move's win-rate.  Resumes from a Stage 0 checkpoint
to keep the value head grounded.

Usage:
    .venv/bin/python scripts/train_stage1.py [--data PATH] [--resume CKPT] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.models.backbone import NMMNet, PHASE_NAMES


# ── Training helpers ─────────────────────────────────────────────────────────

def _run_epoch(
    model: NMMNet,
    loader: DataLoader,
    device: torch.device,
    optimizer,
    train: bool,
) -> tuple[float, float]:
    """Run one epoch; return (avg_loss, top1_accuracy)."""
    model.train(train)
    total_loss = 0.0
    correct = 0
    total_n = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for states_b, phase_ids_b, actions_b, weights_b in loader:
            states_b  = states_b.to(device)
            actions_b = actions_b.to(device)
            weights_b = weights_b.to(device)
            phase_ids_b = phase_ids_b.to(device)

            # Single backbone pass for the whole batch.
            feats = model.backbone(states_b)  # (B, feat_dim)

            loss = torch.tensor(0.0, device=device)
            for phase_id in range(model.num_phases):
                mask = (phase_ids_b == phase_id)
                if not mask.any():
                    continue
                feats_p   = feats[mask]
                actions_p = actions_b[mask]
                weights_p = weights_b[mask]

                logits_p = model.phase_heads[PHASE_NAMES[phase_id]](feats_p)  # (N, 624)
                ce = F.cross_entropy(logits_p, actions_p, reduction="none")   # (N,)
                loss = loss + (ce * weights_p).mean()

                preds = logits_p.argmax(dim=-1)
                correct += (preds == actions_p).sum().item()
                total_n += mask.sum().item()

            total_loss += loss.item()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    n_batches = max(1, len(loader))
    accuracy = correct / max(1, total_n)
    return total_loss / n_batches, accuracy


def train_phase(
    label: str,
    model: NMMNet,
    train_dl: DataLoader,
    val_dl: DataLoader,
    lr: float,
    max_epochs: int,
    device: torch.device,
    freeze_backbone: bool,
    out_dir: Path,
    patience: int = 5,
) -> float:
    print(f"\n── {label}  (lr={lr}  freeze_backbone={freeze_backbone}) ──")

    for name, param in model.backbone.named_parameters():
        param.requires_grad = not freeze_backbone

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    best_acc = -1.0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        import time; wall0 = time.time()

        train_loss, train_acc = _run_epoch(model, train_dl, device, optimizer, train=True)
        val_loss,   val_acc   = _run_epoch(model, val_dl,   device, optimizer, train=False)

        elapsed = time.time() - wall0
        marker = "✓" if val_acc > best_acc else " "
        print(f"  epoch {epoch:2d}/{max_epochs}  "
              f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"{marker}  ({elapsed:.1f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save({"model": model.state_dict()}, out_dir / "best.pt")
            print(f"  → saved {out_dir}/best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop — no improvement for {patience} epochs.")
                break

    torch.save({"model": model.state_dict()}, out_dir / "latest.pt")
    print(f"  → saved {out_dir}/latest.pt")
    return best_acc


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    pa = argparse.ArgumentParser(description="Stage 1 imitation learning")
    pa.add_argument("--data",    default=str(_ROOT / "learned_ai" / "data" / "stage1_imitation.npz"))
    pa.add_argument("--resume",  default=str(_ROOT / "learned_ai" / "checkpoints" / "stage0" / "best.pt"),
                    help="Stage 0 checkpoint to warm-start from")
    pa.add_argument("--out-dir", default=str(_ROOT / "learned_ai" / "checkpoints" / "stage1"))
    pa.add_argument("--val-frac", type=float, default=0.1)
    pa.add_argument("--batch",   type=int, default=1024)
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    d = np.load(args.data)
    states   = torch.tensor(d["states"],          dtype=torch.float32)
    phase_ids = torch.tensor(d["phase_ids"],       dtype=torch.long)
    primary_actions = torch.tensor(d["primary_actions"], dtype=torch.long)
    weights  = torch.tensor(d["weights"],          dtype=torch.float32)

    N = len(states)
    n_val = max(1, int(N * args.val_frac))
    n_train = N - n_val

    perm = torch.randperm(N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    def _ds(idx):
        return TensorDataset(
            states[idx], phase_ids[idx], primary_actions[idx], weights[idx]
        )

    train_dl = DataLoader(_ds(train_idx), batch_size=args.batch, shuffle=True,  pin_memory=True)
    val_dl   = DataLoader(_ds(val_idx),   batch_size=args.batch, shuffle=False, pin_memory=True)

    print(f"Loaded {N:,} samples from {args.data}")
    phase_dist = {}
    for ph in d["phase_ids"]:
        phase_dist[int(ph)] = phase_dist.get(int(ph), 0) + 1
    print(f"  Phase distribution: {dict(sorted(phase_dist.items()))}")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]  mean={weights.mean():.3f}")
    print(f"Train: {n_train:,}  Val: {n_val:,}  batch={args.batch}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NMMNet()
    resume = Path(args.resume)
    if resume.exists():
        ckpt = torch.load(str(resume), map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Resumed from {resume}")
    else:
        print(f"WARNING: checkpoint not found at {resume} — starting fresh")
    model.to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints → {out_dir}")

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    best1 = train_phase(
        "Phase 1 — frozen backbone",
        model, train_dl, val_dl,
        lr=3e-3, max_epochs=20, device=device,
        freeze_backbone=True, out_dir=out_dir, patience=5,
    )
    ckpt1 = torch.load(str(out_dir / "best.pt"), map_location=device)
    model.load_state_dict(ckpt1["model"] if "model" in ckpt1 else ckpt1)
    print(f"Loaded phase 1 best (val_acc={best1:.4f}) for phase 2")

    # ── Phase 2: full network ─────────────────────────────────────────────────
    best2 = train_phase(
        "Phase 2 — full network",
        model, train_dl, val_dl,
        lr=5e-4, max_epochs=40, device=device,
        freeze_backbone=False, out_dir=out_dir, patience=5,
    )

    print(f"\nStage 1 complete.  Best val accuracy: phase1={best1:.4f}  phase2={best2:.4f}")
    print(f"Checkpoints: {out_dir}/best.pt  (use as --resume for Stage 2)")


if __name__ == "__main__":
    main()
