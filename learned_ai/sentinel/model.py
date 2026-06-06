"""learned_ai/sentinel/model.py — SentinelNet, a single-output move scorer.

The sentinel was redesigned from a position-level 4-head network into a
*move-level* scorer. Each example is one candidate move in one position, and
the network predicts a single WDL quality score in [0, 1] from the mover's
perspective (1.0 = winning move, 0.5 = draw, 0.0 = losing move).

Architecture
------------
  Input: per-move feature vector (FEATURE_DIM floats; see feature_builder).
  MLP:   Linear(in, h0) -> ReLU -> Dropout -> ... -> Linear(h_{k-1}, 1) -> Sigmoid
         (hidden_dims default [128, 64, 32]).

forward() returns a (B,) float tensor in [0, 1]. Loss is BCELoss with optional
per-sample weights (draw examples are weighted at 0.5 by the dataset).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_ai.sentinel.feature_builder import FEATURE_DIM


class SentinelNet(nn.Module):
    """Single-output move-quality scorer (sigmoid head)."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        layers: List[nn.Module] = []
        prev = input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)  # (B,) in [0, 1]


def sentinel_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Weighted binary cross-entropy on move_quality.

    ``output`` and ``target`` are (B,) tensors in [0, 1]. ``sample_weight`` is an
    optional (B,) per-sample weight (the dataset down-weights draw examples).

    Returns a dict with 'total' (scalar) plus 'bce' (detached) for logging.
    """
    per_elem = F.binary_cross_entropy(output, target, reduction="none")
    if sample_weight is not None:
        w = sample_weight.to(per_elem.dtype)
        denom = torch.clamp(w.sum(), min=1e-6)
        total = (per_elem * w).sum() / denom
    else:
        total = per_elem.mean()
    return {"total": total, "bce": total.detach()}
