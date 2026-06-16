"""learned_ai/sentinel/model.py — SentinelNet, a move-quality scorer.

The sentinel predicts a single WDL quality score in [0, 1] from the mover's
perspective (1.0 = winning move, 0.5 = draw, 0.0 = losing move).

Architecture
------------
  shared trunk: Linear(in, h0) -> ReLU -> Dropout -> ... -> Linear(h_{k-2}, h_{k-1})
  quality head: Linear(h_{k-1}, 1) -> Sigmoid          (primary output, always active)
  wdl head:     Linear(h_{k-1}, 3) [win/draw/loss]     (optional auxiliary head)

forward() returns a (B,) float tensor by default.
forward(x, return_aux=True) returns (quality (B,), wdl_logits (B,3)) when
aux_wdl=True was set at construction time.

Loss is weighted BCE on quality plus optional cross-entropy on WDL class.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_ai.sentinel.feature_builder import FEATURE_DIM


class SentinelNet(nn.Module):
    """Move-quality scorer with optional auxiliary WDL classification head."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.2,
        aux_wdl: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.aux_wdl = aux_wdl

        trunk_layers: List[nn.Module] = []
        prev = input_dim
        for h in self.hidden_dims:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(nn.ReLU())
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)
        self.quality_head = nn.Linear(prev, 1)
        if aux_wdl:
            self.wdl_head = nn.Linear(prev, 3)  # logits: [loss, draw, win]

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.trunk(x)
        quality = torch.sigmoid(self.quality_head(h)).squeeze(-1)  # (B,)
        if return_aux and self.aux_wdl:
            return quality, self.wdl_head(h)  # (B,), (B, 3)
        return quality


def sentinel_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    wdl_logits: Optional[torch.Tensor] = None,
    wdl_targets: Optional[torch.Tensor] = None,
    lambda_wdl: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """Weighted BCE on move_quality plus optional auxiliary WDL cross-entropy.

    ``output`` and ``target`` are (B,) tensors in [0, 1].
    ``sample_weight`` is an optional (B,) weight (draws weighted at 0.5).
    ``wdl_logits`` is (B, 3) from the auxiliary head; ``wdl_targets`` is (B,)
    long tensor with class labels 0=loss / 1=draw / 2=win; -1 = masked out.

    Returns a dict with 'total', 'bce', and optionally 'wdl' for logging.
    """
    per_elem = F.binary_cross_entropy(output, target, reduction="none")
    if sample_weight is not None:
        w = sample_weight.to(per_elem.dtype)
        denom = torch.clamp(w.sum(), min=1e-6)
        bce = (per_elem * w).sum() / denom
    else:
        bce = per_elem.mean()

    total = bce
    result: Dict[str, torch.Tensor] = {"total": total, "bce": bce.detach()}

    if wdl_logits is not None and wdl_targets is not None:
        mask = wdl_targets >= 0
        if mask.any():
            wdl_loss = F.cross_entropy(wdl_logits[mask], wdl_targets[mask])
            result["wdl"] = wdl_loss.detach()
            result["total"] = bce + lambda_wdl * wdl_loss

    return result
