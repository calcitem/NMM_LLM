"""NMMNet: shared MLP backbone + 5 phase-specific policy heads + value head.

The backbone is intentionally small (default 256 -> 256 -> 128). Phase routing
is done at the head level so the network shares board geometry knowledge while
each phase can specialise its action preferences. The 5 heads all output the
same unified 624-dim action logits — masks then prune to the legal slice.

The illegal-action mask is applied by setting logits to a large negative
constant (NEG_INF) before any softmax. This is the *only* place where action
legality enters the model; the legality decision itself comes from the
existing game engine via action_encoder.get_legal_mask.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_ai.models.action_encoder import ACTION_DIM
from learned_ai.models.state_encoder import NUM_PHASES, PHASE_NAMES, STATE_DIM

NEG_INF = -1e9


def _mlp(sizes: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class NMMNet(nn.Module):
    """Shared-backbone, multi-head NMM policy/value network."""

    def __init__(
        self,
        backbone_hidden: Sequence[int] = (256, 256, 128),
        head_hidden: Sequence[int] = (64,),
        dropout: float = 0.0,
        action_dim: int = ACTION_DIM,
        state_dim: int = STATE_DIM,
        num_phases: int = NUM_PHASES,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_phases = num_phases

        backbone_sizes = [state_dim, *backbone_hidden]
        self.backbone = _mlp(backbone_sizes, dropout=dropout)
        feat_dim = backbone_hidden[-1]

        head_input = [feat_dim, *head_hidden]
        self.phase_heads: nn.ModuleDict = nn.ModuleDict(
            {
                PHASE_NAMES[p]: nn.Sequential(
                    _mlp(head_input, dropout=dropout),
                    nn.Linear(head_hidden[-1], action_dim),
                )
                for p in range(num_phases)
            }
        )

        self.value_head: nn.Sequential = nn.Sequential(
            _mlp(head_input, dropout=dropout),
            nn.Linear(head_hidden[-1], 1),
        )

    # ----- helpers -----------------------------------------------------------

    def _featurise(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.backbone(state)

    def _route(self, feats: torch.Tensor, phase_id: int) -> torch.Tensor:
        if not (0 <= phase_id < self.num_phases):
            raise ValueError(f"phase_id {phase_id} out of range")
        head = self.phase_heads[PHASE_NAMES[phase_id]]
        return head(feats)

    # ----- public API --------------------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
        phase_id: int,
        legal_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Return dict with 'logits' (masked) and 'value' for the given phase.

        ``state``     : (state_dim,) or (B, state_dim) tensor.
        ``phase_id``  : int 0..NUM_PHASES-1 (single phase per batch — for
                        mixed-phase batches loop over phases and call once
                        per phase).
        ``legal_mask``: optional bool tensor of shape (action_dim,) or
                        (B, action_dim). Where False, the logits are set to
                        NEG_INF.
        """
        feats = self._featurise(state)
        logits = self._route(feats, phase_id)
        value = self.value_head(feats).squeeze(-1)

        if legal_mask is not None:
            if legal_mask.dim() == 1:
                legal_mask = legal_mask.unsqueeze(0)
            if legal_mask.shape[-1] != self.action_dim:
                raise ValueError(
                    f"legal_mask shape {legal_mask.shape} incompatible with "
                    f"action_dim={self.action_dim}"
                )
            logits = logits.masked_fill(~legal_mask, NEG_INF)

        if logits.shape[0] == 1:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
        return {"logits": logits, "value": value}

    # ----- convenience -------------------------------------------------------

    @torch.no_grad()
    def policy_probs(
        self,
        state: torch.Tensor,
        phase_id: int,
        legal_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Inference helper: returns softmax probabilities over the action space.

        Illegal actions are guaranteed zero probability because their logits
        are pushed to NEG_INF before softmax.
        """
        out = self.forward(state, phase_id, legal_mask)
        logits = out["logits"] / max(temperature, 1e-6)
        return F.softmax(logits, dim=-1)
