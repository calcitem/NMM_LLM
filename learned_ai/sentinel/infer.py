"""learned_ai/sentinel/infer.py — SentinelAdvisor runtime inference wrapper.

Loads a trained SentinelNet (move-level scorer) and exposes a single fast
``advise()`` call that scores every candidate move in one batched forward pass
and returns a SentinelAdvice dataclass. Designed for the GameAI advisory path:

  * a single CPU forward pass over all candidates, no DB queries at runtime;
  * < 50 ms typical (usually well under 2 ms);
  * never raises on a missing/partial context — returns ``None`` when the model
    is unavailable so the caller degrades gracefully.

The network predicts move quality in [0, 1] from the mover's perspective
(1.0 = winning move, 0.5 = draw, 0.0 = losing move).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from learned_ai.sentinel.config import SentinelConfig
from learned_ai.sentinel.feature_builder import FEATURE_DIM, build_move_features
from learned_ai.sentinel.model import SentinelNet

logger = logging.getLogger(__name__)


@dataclass
class SentinelAdvice:
    """Move-level advisory signal for a single decision.

    ``move_scores[i]`` is the sentinel's quality score in [0, 1] for the i-th
    candidate (same order as the candidate list passed to ``advise``).
    """

    move_scores: List[float]
    best_sentinel_move_idx: int
    played_move_idx: int
    played_move_quality: float
    best_available_quality: float
    opportunity_gap: float          # best_available_quality - played_move_quality
    player: str
    advisory_message: str           # "safe"|"possible_mistake"|"missed_opportunity"|"critical"
    intervention_applied: Optional[str] = None   # "llm_override"|"score_adjust"|"rank1_fallback"|None
    intervention_detail: Optional[str] = None     # human-readable string
    meta: Dict[str, Any] = field(default_factory=dict)


def _advisory_message(played_quality: float, opportunity_gap: float) -> str:
    """Map (played move quality, opportunity gap) to an advisory label."""
    if played_quality < 0.3 and opportunity_gap >= 0.3:
        return "critical"
    if opportunity_gap >= 0.2 and played_quality >= 0.4:
        return "missed_opportunity"
    if opportunity_gap >= 0.1 and played_quality < 0.4:
        return "possible_mistake"
    return "safe"


class SentinelAdvisor:
    """Runtime inference wrapper around a trained move-level SentinelNet."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[SentinelConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or SentinelConfig()
        self.device = torch.device(device)
        self.model: Optional[SentinelNet] = None
        self._loaded = False
        if checkpoint_path:
            self.load(checkpoint_path)

    # ----- loading -------------------------------------------------------------

    def load(self, checkpoint_path: str) -> bool:
        """Load weights from a checkpoint. Returns True on success.

        The checkpoint is the dict written by scripts/train_sentinel.py:
        {"state_dict", "config": {...}, ...}. A raw state_dict is also accepted.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = None
        state_dict = ckpt
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            cfg_dict = ckpt.get("config")
        if cfg_dict:
            self.config = SentinelConfig.from_dict(cfg_dict)
        self.model = SentinelNet(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,  # match training arch; eval() disables dropout
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._loaded = True
        logger.info("[SentinelAdvisor] loaded checkpoint %s", checkpoint_path)
        return True

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    # ----- inference -----------------------------------------------------------

    @torch.no_grad()
    def advise(
        self,
        board_state,
        candidates: List[Dict[str, Any]],
        player: str,
        played_move_idx: int = 0,
        move_ctx_by_idx: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Optional[SentinelAdvice]:
        """Score every candidate move in one batched forward pass.

        ``candidates`` is the ordered apply-move dict list ({from,to,capture})
        from the heuristic search; ``played_move_idx`` is the index the engine
        intends to play (default 0, the heuristic's top choice).

        Returns ``None`` if the model is unavailable or there are no candidates,
        so the caller can skip intervention cleanly.
        """
        if not self.is_loaded() or not candidates:
            return None

        ctx_map = move_ctx_by_idx or {}
        n = len(candidates)
        feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        for i, mv in enumerate(candidates):
            try:
                feats[i] = build_move_features(board_state, mv, player, ctx_map.get(i))
            except Exception:
                feats[i] = 0.0  # neutral row; keeps batch shape stable

        x = torch.from_numpy(feats).to(self.device)
        out = self.model(x).reshape(-1)
        scores = [float(v) for v in out.cpu().numpy()]

        best_idx = int(max(range(n), key=lambda i: scores[i]))
        best_quality = scores[best_idx]
        p_idx = played_move_idx if 0 <= played_move_idx < n else 0
        played_quality = scores[p_idx]
        gap = max(0.0, best_quality - played_quality)
        message = _advisory_message(played_quality, gap)

        return SentinelAdvice(
            move_scores=scores,
            best_sentinel_move_idx=best_idx,
            played_move_idx=p_idx,
            played_move_quality=played_quality,
            best_available_quality=best_quality,
            opportunity_gap=gap,
            player=player,
            advisory_message=message,
        )


def load_advisor(checkpoint_path: str, config: Optional[SentinelConfig] = None,
                 device: str = "cpu") -> Optional[SentinelAdvisor]:
    """Best-effort advisor loader. Returns None (with a logged warning) on any
    failure so callers can wire a sentinel without risking a crash at startup.
    """
    try:
        advisor = SentinelAdvisor(checkpoint_path, config=config, device=device)
        return advisor if advisor.is_loaded() else None
    except Exception as exc:
        logger.warning("[SentinelAdvisor] failed to load %s: %s", checkpoint_path, exc)
        return None
