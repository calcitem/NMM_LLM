"""scripts/train_scaffolded_s1b.py — Stage 1.5: Human-game fine-tuning.

Fine-tunes the policy head of a pre-trained ScaffoldedPolicyNet on human
imitation data (output of gen_human_imitation_data.py).  The value head is
frozen so value estimates from Stage 1 are preserved.

Loss
----
  Weighted soft cross-entropy: -weight * sum(label_dist * log P)
  label_dist is (1-HUMAN_ALPHA)*malom_or_sentinel + HUMAN_ALPHA*one_hot(human_move)
  so the human's move always gets HUMAN_ALPHA probability mass, and Malom-
  winning moves get proportionally more of the remaining mass.

  Positions from won games:  weight=1.0  (default)
  Positions from draw games: weight=0.3  (default)
  Positions where the human deviated from heuristic top-1 receive a
  bonus multiplier (--deviate-bonus, default 1.5) because they carry
  the most information about human-specific tactical patterns.

Usage
-----
    .venv/bin/python scripts/train_scaffolded_s1b.py [options]

Options
-------
  --base-ckpt PATH  s1/best.pt to fine-tune from (required)
  --data      PATH  human_imitation.npz from gen_human_imitation_data.py
  --out-dir   DIR   checkpoint directory (default .../s1b)
  --epochs    N     training epochs (default 5)
  --batch     N     mini-batch size (default 32)
  --lr        F     learning rate (default 3e-5)
  --deviate-bonus F bonus weight multiplier for deviated moves (default 1.5)
  --val-frac  F     validation fraction (default 0.1)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet


def load_dataset(npz_path: str):
    """Load the .npz produced by gen_human_imitation_data.py."""
    data = np.load(npz_path, allow_pickle=True)
    feat_matrices = data["feat_matrices"]    # (N,) object array of (k,62)
    value_inputs  = data["value_inputs"]     # (N, 23)  — not used for training here
    label_dists   = data["label_dists"]      # (N,) object array of (k,) soft labels
    chosen_idxs   = data["chosen_idxs"]      # (N,) int — human move (for deviate flag)
    h_evals       = data["h_evals"]          # (N,) float — unused (value head frozen)
    weights       = data["weights"]          # (N,) float
    deviates      = data["deviates"]         # (N,) bool
    return feat_matrices, value_inputs, label_dists, chosen_idxs, h_evals, weights, deviates


def load_base_model(ckpt_path: str, device: torch.device) -> ScaffoldedPolicyNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        cfg = ckpt.get("model_config", {})
        model = ScaffoldedPolicyNet.from_config(cfg)
        sd_key = "model" if "model" in ckpt else "state_dict"
        model.load_state_dict(ckpt[sd_key])
    else:
        model = ScaffoldedPolicyNet()
        model.load_state_dict(ckpt)
    return model.to(device)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[s1b] Device: {device}")

    # ── Load base checkpoint ───────────────────────────────────────────────────
    print(f"[s1b] Loading base checkpoint: {args.base_ckpt}")
    model = load_base_model(args.base_ckpt, device)

    # Freeze value head
    for param in model.value_mlp.parameters():
        param.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[s1b] Policy head trainable: {n_trainable:,}  |  Value head frozen: {n_frozen:,}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"[s1b] Loading {args.data} ...")
    feat_matrices, value_inputs, label_dists, chosen_idxs, h_evals, weights, deviates = load_dataset(args.data)
    N = len(chosen_idxs)

    # Apply deviate bonus to won-game deviated positions
    effective_weights = weights.copy()
    if args.deviate_bonus != 1.0:
        bonus_mask = (weights >= 1.0) & deviates
        effective_weights[bonus_mask] *= args.deviate_bonus
        n_bonus = int(bonus_mask.sum())
        print(f"[s1b] Deviate bonus {args.deviate_bonus}x applied to {n_bonus} positions")

    n_won_pos  = int((weights >= 1.0).sum())
    n_draw_pos = int((weights < 1.0).sum())
    n_dev      = int(deviates.sum())
    print(f"[s1b] {N} positions: {n_won_pos} from won games, {n_draw_pos} from draws")
    print(f"[s1b] Human deviated from heuristic top-1 in {n_dev} positions")

    # ── Train/val split ────────────────────────────────────────────────────────
    idxs = list(range(N))
    random.shuffle(idxs)
    n_val = max(1, int(N * args.val_frac))
    val_idxs   = idxs[:n_val]
    train_idxs = idxs[n_val:]
    print(f"[s1b] Train: {len(train_idxs)}  Val: {len(val_idxs)}")

    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_idxs)
        model.train()

        ep_loss  = 0.0
        ep_w_sum = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_idxs), args.batch):
            batch = train_idxs[batch_start : batch_start + args.batch]
            if not batch:
                continue

            terms: list[torch.Tensor] = []
            bweights: list[float]     = []

            for i in batch:
                fm         = feat_matrices[i]          # (k, 62)
                label_dist = label_dists[i]            # (k,) soft target
                w          = float(effective_weights[i])
                feat   = torch.tensor(fm, dtype=torch.float32).to(device)
                target = torch.tensor(label_dist, dtype=torch.float32).to(device)
                logits = model.policy_logits(feat)
                log_p  = F.log_softmax(logits, dim=-1)
                # Weighted soft cross-entropy: -sum(label * log_p)
                terms.append(-(target * log_p).sum())
                bweights.append(w)

            w_t    = torch.tensor(bweights, dtype=torch.float32).to(device)
            loss_t = torch.stack(terms)
            loss   = (w_t * loss_t).sum() / w_t.sum().clamp(min=1e-9)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss  += float(loss.item()) * float(w_t.sum())
            ep_w_sum += float(w_t.sum())
            n_batches += 1

        avg_loss = ep_loss / max(ep_w_sum, 1e-9)

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_w    = 0.0
        with torch.no_grad():
            for batch_start in range(0, len(val_idxs), args.batch):
                batch = val_idxs[batch_start : batch_start + args.batch]
                if not batch:
                    continue
                terms2: list[torch.Tensor] = []
                bw2:    list[float]        = []
                for i in batch:
                    fm         = feat_matrices[i]
                    label_dist = label_dists[i]
                    w          = float(weights[i])   # raw weight (no deviate bonus)
                    feat   = torch.tensor(fm, dtype=torch.float32).to(device)
                    target = torch.tensor(label_dist, dtype=torch.float32).to(device)
                    logits = model.policy_logits(feat)
                    log_p  = F.log_softmax(logits, dim=-1)
                    terms2.append(-(target * log_p).sum())
                    bw2.append(w)
                w_t2  = torch.tensor(bw2, dtype=torch.float32).to(device)
                loss2 = (w_t2 * torch.stack(terms2)).sum() / w_t2.sum().clamp(min=1e-9)
                val_loss += float(loss2.item()) * float(w_t2.sum())
                val_w    += float(w_t2.sum())

        avg_val = val_loss / max(val_w, 1e-9)
        elapsed = time.time() - t_start
        print(
            f"[s1b] epoch {epoch:2d}/{args.epochs} | "
            f"train={avg_loss:.4f}  val={avg_val:.4f} | "
            f"{elapsed:.0f}s"
        )

        ckpt = {
            "model":        model.state_dict(),
            "model_config": model.get_config(),
            "stage":        "s1b",
            "epoch":        epoch,
            "val_loss":     avg_val,
            "base_ckpt":    args.base_ckpt,
        }
        torch.save(ckpt, out_dir / "latest.pt")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt, out_dir / "best.pt")
            print(f"[s1b]  → new best val_loss={avg_val:.4f}")

    print(f"\n[s1b] Done. Best val_loss={best_val_loss:.4f}")
    print(f"[s1b] Checkpoint: {out_dir / 'best.pt'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1.5: human-game fine-tuning")
    p.add_argument(
        "--base-ckpt",
        default=str(_ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1" / "best.pt"),
        help="Stage 1 checkpoint to fine-tune from",
    )
    p.add_argument(
        "--data",
        default=str(_ROOT / "learned_ai" / "data" / "human_imitation.npz"),
    )
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1b"),
    )
    p.add_argument("--epochs",        type=int,   default=5)
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-5)
    p.add_argument("--deviate-bonus", type=float, default=1.5)
    p.add_argument("--val-frac",      type=float, default=0.1)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
