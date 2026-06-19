"""learned_ai/training/scaffolded_a2c.py — A2C update for ScaffoldedPolicyNet.

The scaffolded model operates on variable-length move sets, so this update
function differs from a2c.py in two ways:

1. Policy loss: computed per-step (each step has a different k legal moves),
   accumulated into a single loss, then averaged.  No phase routing.

2. Value loss: computed in batch (value_input is always VALUE_INPUT_DIM floats),
   enabling efficient bootstrapping with one batched forward pass.

Both losses flow through a single backward() call so shared module parameters
(none here — policy and value heads are separate) would still receive correct
gradients.

ScaffoldedStep dataclass fields:
  move_features   (k, 62) np.ndarray  — features for current position's legal moves
  value_input     (23,)  np.ndarray   — board-level features for value head
  chosen_idx      int                 — which legal move was selected
  log_prob_old    float               — log P at collection time (for PPO ratio, unused in A2C)
  reward          float               — per-move shaped reward
  next_move_features  (k', 62) np.ndarray
  next_value_input    (23,)  np.ndarray
  done            bool
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ENTROPY_COEF = 0.01
VALUE_COEF   = 0.5
GRAD_CLIP    = 1.0


@dataclass
class ScaffoldedStep:
    """One learner-turn step for the scaffolded A2C update."""

    move_features:      np.ndarray   # (k, 62)
    value_input:        np.ndarray   # (23,)
    chosen_idx:         int
    log_prob_old:       float
    reward:             float
    next_move_features: np.ndarray   # (k', 62) — for optional bootstrapping
    next_value_input:   np.ndarray   # (23,)
    done:               bool


def scaffolded_a2c_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: List[ScaffoldedStep],
    device: torch.device,
    gamma: float = 0.99,
    entropy_coef: float = ENTROPY_COEF,
    value_coef: float = VALUE_COEF,
    grad_clip: float = GRAD_CLIP,
    min_batch: int = 8,
) -> tuple[float, float, float]:
    """One A2C gradient update over a batch of ScaffoldedSteps.

    Returns (policy_loss, value_loss, entropy) as Python floats.
    Returns (0, 0, 0) if batch is too small.
    """
    if len(steps) < min_batch:
        return 0.0, 0.0, 0.0

    B = len(steps)

    # ── Batch value inputs (fixed size — easy to stack) ────────────────────────
    all_vi  = torch.tensor(
        np.stack([s.value_input      for s in steps]), dtype=torch.float32
    ).to(device)  # (B, 23)
    all_nvi = torch.tensor(
        np.stack([s.next_value_input for s in steps]), dtype=torch.float32
    ).to(device)  # (B, 23)
    rewards = torch.tensor(
        [s.reward for s in steps], dtype=torch.float32, device=device
    )                                                    # (B,)
    dones   = torch.tensor(
        [float(s.done) for s in steps], dtype=torch.float32, device=device
    )                                                    # (B,)

    model.train()

    # ── Bootstrap: V(next_state) — no gradient ────────────────────────────────
    with torch.no_grad():
        v_next = model.value(all_nvi)                   # (B,)
        v_next = v_next * (1.0 - dones)

    td_targets = rewards + gamma * v_next               # (B,)

    # ── Current value (with gradient) ─────────────────────────────────────────
    v_curr = model.value(all_vi)                        # (B,)

    # ── Advantage (detached for policy gradient) ───────────────────────────────
    advantages = (td_targets - v_curr).detach()         # (B,)
    if advantages.std() > 1e-3:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── Policy loss + entropy (per-step, variable k) ───────────────────────────
    policy_terms:  list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []

    for i, step in enumerate(steps):
        feat   = torch.tensor(step.move_features, dtype=torch.float32).to(device)
        logits = model.policy_logits(feat)              # (k,)
        log_probs = F.log_softmax(logits, dim=-1)       # (k,)
        policy_terms.append(-log_probs[step.chosen_idx] * advantages[i])
        probs = log_probs.exp()
        entropy_terms.append(-(probs * log_probs).sum())

    policy_loss  = torch.stack(policy_terms).mean()
    entropy_loss = torch.stack(entropy_terms).mean()
    value_loss   = F.mse_loss(v_curr, td_targets.detach())

    total_loss = (
        policy_loss
        - entropy_coef * entropy_loss
        + value_coef * value_loss
    )

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return (
        float(policy_loss.item()),
        float(value_loss.item()),
        float(entropy_loss.item()),
    )


def scaffolded_ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: List[ScaffoldedStep],
    device: torch.device,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    epochs: int = 4,
    entropy_coef: float = ENTROPY_COEF,
    value_coef: float = VALUE_COEF,
    grad_clip: float = GRAD_CLIP,
    min_batch: int = 8,
) -> tuple[float, float, float]:
    """PPO clipped surrogate update over ScaffoldedSteps.

    Returns (policy_loss, value_loss, entropy) averaged over epochs.
    """
    if len(steps) < min_batch:
        return 0.0, 0.0, 0.0

    # Pre-compute TD targets (no grad needed)
    all_vi  = torch.tensor(
        np.stack([s.value_input      for s in steps]), dtype=torch.float32
    ).to(device)
    all_nvi = torch.tensor(
        np.stack([s.next_value_input for s in steps]), dtype=torch.float32
    ).to(device)
    rewards = torch.tensor(
        [s.reward for s in steps], dtype=torch.float32, device=device
    )
    dones   = torch.tensor(
        [float(s.done) for s in steps], dtype=torch.float32, device=device
    )
    log_probs_old = torch.tensor(
        [s.log_prob_old for s in steps], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        v_next     = model.value(all_nvi) * (1.0 - dones)
        td_targets = (rewards + gamma * v_next).detach()
        with torch.no_grad():
            v0    = model.value(all_vi)
        advantages = td_targets - v0
        if advantages.std() > 1e-3:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pl_acc, vl_acc, ent_acc = 0.0, 0.0, 0.0

    model.train()
    for _ in range(epochs):
        v_curr = model.value(all_vi)
        policy_terms:  list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []

        for i, step in enumerate(steps):
            feat      = torch.tensor(step.move_features, dtype=torch.float32).to(device)
            logits    = model.policy_logits(feat)
            log_probs = F.log_softmax(logits, dim=-1)
            lp        = log_probs[step.chosen_idx]
            ratio     = torch.exp(lp - log_probs_old[i])
            adv       = advantages[i]
            surr1     = ratio * adv
            surr2     = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_terms.append(-torch.min(surr1, surr2))
            probs = log_probs.exp()
            entropy_terms.append(-(probs * log_probs).sum())

        policy_loss  = torch.stack(policy_terms).mean()
        entropy_loss = torch.stack(entropy_terms).mean()
        value_loss   = F.mse_loss(v_curr, td_targets)
        total_loss   = policy_loss - entropy_coef * entropy_loss + value_coef * value_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        pl_acc  += float(policy_loss.item())
        vl_acc  += float(value_loss.item())
        ent_acc += float(entropy_loss.item())

    return pl_acc / epochs, vl_acc / epochs, ent_acc / epochs
