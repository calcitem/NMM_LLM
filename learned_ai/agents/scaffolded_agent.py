"""learned_ai/agents/scaffolded_agent.py — ScaffoldedAgent inference wrapper.

Drop-in choose_move() interface for the scaffolded meta-policy.  At inference
the agent needs:
  * a loaded ScaffoldedPolicyNet
  * a loaded SentinelAdvisor (optional but strongly recommended)
  * access to the heuristic evaluate function (via scaffolded_encoder)
  * an optional ExternalSolvedDB for Malom context

Unlike LearnedAgent, there is no fixed action space — the network scores each
legal move directly, so no action masking or phase routing is needed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.models.scaffolded_encoder import encode_position
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet


@dataclass
class ScaffoldedDecision:
    """Trace of the most recent move for trainers / loggers."""

    move_features:  np.ndarray     # (k, 62) for all legal moves at this step
    value_input:    np.ndarray     # (23,)
    chosen_idx:     int            # index into legal_moves
    legal_moves:    list           # full list of legal move dicts
    log_prob:       float          # log P(chosen_idx) at decision time
    value:          float          # estimated V(s)
    # For reward computation:
    sentinel_scores: list[float]
    h_scores_abs:    list[float]
    h_before:        float
    h_top1_idx:      int
    db_moves:        list


class ScaffoldedAgent:
    """Inference wrapper around ScaffoldedPolicyNet for use in gameplay."""

    def __init__(
        self,
        color: str = "B",
        model: Optional[ScaffoldedPolicyNet] = None,
        checkpoint_path: Optional[str] = None,
        sentinel_advisor=None,
        db=None,
        device: str = "auto",
        mode: str = "sample",
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.color = color
        self.sentinel_advisor = sentinel_advisor
        self.db = db
        self.mode = mode
        self.temperature = max(float(temperature), 1e-6)
        self.last_was_blunder = False
        self.last_thinking = "scaffolded"

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model = self._load_checkpoint(checkpoint_path)
        else:
            self.model = ScaffoldedPolicyNet().to(self.device)

        self._gen = torch.Generator(device="cpu")
        if seed is not None:
            self._gen.manual_seed(seed)
        else:
            self._gen.seed()

        self.last_decision: Optional[ScaffoldedDecision] = None

    def _load_checkpoint(self, path: str) -> ScaffoldedPolicyNet:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            cfg = ckpt.get("model_config", {})
            model = ScaffoldedPolicyNet.from_config(cfg)
            sd_key = "model" if "model" in ckpt else "state_dict"
            model.load_state_dict(ckpt[sd_key])
        else:
            model = ScaffoldedPolicyNet()
            model.load_state_dict(ckpt)
        return model.to(self.device)

    def set_mode(self, mode: str) -> None:
        if mode not in {"argmax", "sample"}:
            raise ValueError(f"mode must be 'argmax' or 'sample'; got {mode!r}")
        self.mode = mode

    def set_temperature(self, t: float) -> None:
        self.temperature = max(float(t), 1e-6)

    # ── inference ──────────────────────────────────────────────────────────────

    def choose_move(self, board: BoardState, **_) -> dict:
        player = board.turn

        enc = encode_position(
            board,
            player,
            sentinel_advisor=self.sentinel_advisor,
            db=self.db,
        )
        if enc is None or len(enc.legal_moves) == 0:
            return {}

        feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(self.device)
        vi_t   = torch.tensor(enc.value_input,  dtype=torch.float32).to(self.device)

        with torch.no_grad():
            result = self.model.forward(feat_t, vi_t)
            logits = result["logits"]   # (k,)
            value  = float(result["value"].item())

        chosen_idx, log_prob = self._select(logits)

        self.last_decision = ScaffoldedDecision(
            move_features=enc.feat_matrix,
            value_input=enc.value_input,
            chosen_idx=chosen_idx,
            legal_moves=enc.legal_moves,
            log_prob=float(log_prob),
            value=value,
            sentinel_scores=enc.sentinel_scores,
            h_scores_abs=enc.h_scores_abs,
            h_before=enc.h_before,
            h_top1_idx=enc.h_top1_idx,
            db_moves=enc.db_moves,
        )

        return enc.legal_moves[chosen_idx]

    def _select(self, logits: torch.Tensor) -> tuple[int, float]:
        if self.mode == "argmax" or self.temperature <= 1e-6:
            idx = int(torch.argmax(logits).item())
            log_probs = F.log_softmax(logits, dim=-1)
            return idx, float(log_probs[idx].item())

        scaled    = logits / self.temperature
        log_probs = F.log_softmax(scaled, dim=-1)
        probs     = log_probs.exp()
        if not torch.isfinite(probs).all():
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            probs = probs / probs.sum().clamp(min=1e-9)
        idx = int(torch.multinomial(probs.cpu(), 1, generator=self._gen).item())
        return idx, float(log_probs[idx].item())
