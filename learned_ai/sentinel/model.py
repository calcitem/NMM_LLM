"""learned_ai/sentinel/model.py — SentinelNet, a compact multi-head MLP.

Reuses the small-MLP construction pattern from learned_ai/models/backbone.py
(``_mlp``) rather than inventing a new architecture.

Architecture
------------
  Input: 120-float feature vector (84 base + 36 context)
  Shared trunk: Linear(120, h0) -> ReLU [-> Dropout] -> ... -> Linear(h_{k-1}, h_k) -> ReLU
                (hidden_dims default [256,128,64]; smoke [64,32])
  Heads (each from the trunk output dim T = hidden_dims[-1]):
    mistake_risk_head:           Linear(T,32) -> ReLU -> Linear(32,1) -> Sigmoid
    opportunity_score_head:      Linear(T,32) -> ReLU -> Linear(32,1) -> Sigmoid
    trajectory_value_delta_head: Linear(T,32) -> ReLU -> Linear(32,1) -> Tanh
    turning_point_head:          Linear(T,32) -> ReLU -> Linear(32,1) -> Sigmoid

forward() returns a SentinelOutput dataclass of four tensors. Loss is the
weighted sum of BCE (3 sigmoid heads) + MSE (tanh head), with per-sample weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_ai.sentinel.feature_builder import FEATURE_DIM

_HEAD_HIDDEN = 32


def _mlp(sizes: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    """Linear/ReLU stack (mirrors learned_ai/models/backbone._mlp)."""
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@dataclass
class SentinelOutput:
    """Four head outputs. Tensors are shape (B,) (or scalar when B==1 squeezed)."""

    mistake_risk: torch.Tensor
    opportunity_score: torch.Tensor
    trajectory_value_delta: torch.Tensor
    turning_point_confidence: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "mistake_risk": self.mistake_risk,
            "opportunity_score": self.opportunity_score,
            "trajectory_value_delta": self.trajectory_value_delta,
            "turning_point_confidence": self.turning_point_confidence,
        }


def _head(in_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, _HEAD_HIDDEN),
        nn.ReLU(),
        nn.Linear(_HEAD_HIDDEN, 1),
    )


class SentinelNet(nn.Module):
    """Shared-trunk, four-head sentinel network."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        trunk_sizes = [input_dim, *self.hidden_dims]
        self.trunk = _mlp(trunk_sizes, dropout=dropout)
        feat_dim = self.hidden_dims[-1]

        self.mistake_risk_head = _head(feat_dim)
        self.opportunity_score_head = _head(feat_dim)
        self.trajectory_value_delta_head = _head(feat_dim)
        self.turning_point_head = _head(feat_dim)

    def forward(self, x: torch.Tensor) -> SentinelOutput:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        feats = self.trunk(x)
        mistake = torch.sigmoid(self.mistake_risk_head(feats)).squeeze(-1)
        opp = torch.sigmoid(self.opportunity_score_head(feats)).squeeze(-1)
        delta = torch.tanh(self.trajectory_value_delta_head(feats)).squeeze(-1)
        tp = torch.sigmoid(self.turning_point_head(feats)).squeeze(-1)
        return SentinelOutput(
            mistake_risk=mistake,
            opportunity_score=opp,
            trajectory_value_delta=delta,
            turning_point_confidence=tp,
        )


def sentinel_loss(
    output: SentinelOutput,
    targets: Dict[str, torch.Tensor],
    sample_weight: Optional[torch.Tensor] = None,
    loss_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Weighted multi-task loss.

    targets: dict with keys mistake_risk, opportunity_score,
             trajectory_value_delta, turning_point_confidence (each shape (B,)).
    sample_weight: optional (B,) per-sample weight (e.g. backward-decay weight).
    loss_weights: optional per-head multipliers.

    Returns a dict with 'total' plus the per-head components (all scalar tensors).
    """
    lw = {
        "mistake_risk": 1.0,
        "opportunity_score": 1.0,
        "trajectory_value_delta": 1.0,
        "turning_point_confidence": 1.0,
    }
    if loss_weights:
        lw.update(loss_weights)

    def _reduce(per_elem: torch.Tensor) -> torch.Tensor:
        if sample_weight is not None:
            w = sample_weight.to(per_elem.dtype)
            denom = torch.clamp(w.sum(), min=1e-6)
            return (per_elem * w).sum() / denom
        return per_elem.mean()

    bce = F.binary_cross_entropy
    l_mistake = _reduce(bce(output.mistake_risk, targets["mistake_risk"], reduction="none"))
    l_opp = _reduce(bce(output.opportunity_score, targets["opportunity_score"], reduction="none"))
    l_tp = _reduce(bce(output.turning_point_confidence, targets["turning_point_confidence"], reduction="none"))
    l_delta = _reduce(
        F.mse_loss(output.trajectory_value_delta, targets["trajectory_value_delta"], reduction="none")
    )

    total = (
        lw["mistake_risk"] * l_mistake
        + lw["opportunity_score"] * l_opp
        + lw["trajectory_value_delta"] * l_delta
        + lw["turning_point_confidence"] * l_tp
    )
    return {
        "total": total,
        "mistake_risk": l_mistake.detach(),
        "opportunity_score": l_opp.detach(),
        "trajectory_value_delta": l_delta.detach(),
        "turning_point_confidence": l_tp.detach(),
    }
