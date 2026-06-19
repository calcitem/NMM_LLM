"""learned_ai/models/scaffolded_net.py — ScaffoldedPolicyNet.

A compact policy/value network for the scaffolded meta-policy.

Unlike NMMNet (which takes a fixed 84-float board state and outputs 624 action
logits), ScaffoldedPolicyNet operates on the *move level*:

  Policy head: a shared MLP f(move_feat) → scalar logit.
    Input shape: (k, MOVE_FEAT_DIM) — one row per legal move.
    Output shape: (k,) logits → softmax over the k legal moves.

  Value head: a separate MLP g(board_feat) → scalar in [-1, 1].
    Input shape: (VALUE_INPUT_DIM,) or (B, VALUE_INPUT_DIM).
    Output shape: scalar (or (B,) in batch mode).

Both heads share no weights — the policy head learns "which move to prefer"
while the value head learns "how good is this position".

The variable-k input is handled naturally: just pass the (k, MOVE_FEAT_DIM)
feature matrix for the current position's legal moves.  No action-space padding
or masking is needed because the candidate set IS the action space.

At inference, the agent:
  1. Builds feat_matrix (k, 62) via encode_position()
  2. Calls policy_logits(feat_matrix) → (k,) logits
  3. Applies softmax (+ temperature) → (k,) probs
  4. Samples or takes argmax → chosen move index
  5. Calls value(value_input) for A2C bootstrapping

Checkpoint format saved by training scripts
-------------------------------------------
{
  "model": state_dict,
  "model_config": {
      "policy_hidden": (128, 64),
      "value_hidden": (64, 32),
      "dropout": 0.0,
  },
  "stage": "s1" | "s2" | "s3",
  "game_count": int,
  "best_win_rate": float,
}
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_ai.models.scaffolded_encoder import MOVE_FEAT_DIM, VALUE_INPUT_DIM


def _mlp(sizes: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:   # no activation after last linear
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ScaffoldedPolicyNet(nn.Module):
    """Per-move policy head + global value head for the scaffolded meta-policy."""

    def __init__(
        self,
        move_feat_dim: int = MOVE_FEAT_DIM,
        value_input_dim: int = VALUE_INPUT_DIM,
        policy_hidden: Sequence[int] = (128, 64),
        value_hidden: Sequence[int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.move_feat_dim = move_feat_dim
        self.value_input_dim = value_input_dim

        # Policy: shared MLP applied independently to each move's feature row
        self.policy_mlp = _mlp(
            [move_feat_dim, *policy_hidden, 1], dropout=dropout
        )
        # Value: board-level MLP → scalar estimate of position quality
        self.value_mlp = _mlp(
            [value_input_dim, *value_hidden, 1], dropout=dropout
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    # ── forward helpers ────────────────────────────────────────────────────────

    def policy_logits(self, move_feat: torch.Tensor) -> torch.Tensor:
        """Compute per-move logits.

        ``move_feat``: (k, move_feat_dim) — one row per legal move.
        Returns: (k,) logits.
        """
        if move_feat.dim() == 1:
            move_feat = move_feat.unsqueeze(0)
        out = self.policy_mlp(move_feat)          # (k, 1)
        return out.squeeze(-1)                    # (k,)

    def value(self, value_input: torch.Tensor) -> torch.Tensor:
        """Estimate position value.

        ``value_input``: (value_input_dim,) or (B, value_input_dim).
        Returns: scalar or (B,) tensor.
        """
        if value_input.dim() == 1:
            value_input = value_input.unsqueeze(0)
        out = self.value_mlp(value_input)         # (B, 1)
        out = torch.tanh(out)                     # bound to [-1, 1]
        return out.squeeze(-1)                    # (B,) or scalar

    def forward(
        self,
        move_feat: torch.Tensor,
        value_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Combined forward pass.

        Returns dict with "logits" (k,) and "value" (scalar).
        """
        logits = self.policy_logits(move_feat)
        val = self.value(value_input)
        if val.dim() > 0 and val.shape[0] == 1:
            val = val.squeeze(0)
        return {"logits": logits, "value": val}

    # ── inference helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def policy_probs(
        self,
        move_feat: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return softmax probabilities over legal moves.

        ``move_feat``: (k, move_feat_dim) tensor.
        Returns: (k,) probability tensor.
        """
        logits = self.policy_logits(move_feat)
        scaled = logits / max(float(temperature), 1e-6)
        return F.softmax(scaled, dim=-1)

    # ── checkpoint helpers ─────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return a serialisable config dict matching the constructor kwargs."""
        def _sizes(mlp: nn.Sequential) -> list:
            return [m.in_features for m in mlp if isinstance(m, nn.Linear)] + \
                   [list(mlp.modules())[-1].out_features
                    if isinstance(list(mlp.modules())[-1], nn.Linear) else 1]

        lin_pol = [m for m in self.policy_mlp.modules() if isinstance(m, nn.Linear)]
        lin_val = [m for m in self.value_mlp.modules() if isinstance(m, nn.Linear)]
        p_hidden = tuple(m.out_features for m in lin_pol[:-1])
        v_hidden = tuple(m.out_features for m in lin_val[:-1])
        drop_mods = [m for m in self.policy_mlp.modules() if isinstance(m, nn.Dropout)]
        dropout = drop_mods[0].p if drop_mods else 0.0
        return {
            "move_feat_dim": self.move_feat_dim,
            "value_input_dim": self.value_input_dim,
            "policy_hidden": p_hidden,
            "value_hidden": v_hidden,
            "dropout": dropout,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "ScaffoldedPolicyNet":
        return cls(
            move_feat_dim=cfg.get("move_feat_dim", MOVE_FEAT_DIM),
            value_input_dim=cfg.get("value_input_dim", VALUE_INPUT_DIM),
            policy_hidden=tuple(cfg.get("policy_hidden", (128, 64))),
            value_hidden=tuple(cfg.get("value_hidden", (64, 32))),
            dropout=float(cfg.get("dropout", 0.0)),
        )
