"""learned_ai/sentinel/infer.py — SentinelAdvisor runtime inference wrapper.

Loads a trained SentinelNet checkpoint and exposes a single fast ``advise()``
call that builds features from a board + decision context and returns a
SentinelAdvice dataclass. Designed for the GameAI advisory path:

  * a single CPU forward pass, no DB queries at runtime;
  * < 50 ms typical (usually well under 2 ms);
  * never raises on a missing/partial context — returns a neutral advice or
    propagates only genuine programming errors (the caller still guards with
    try/except).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from learned_ai.sentinel.config import SentinelConfig
from learned_ai.sentinel.feature_builder import FEATURE_DIM, build_features
from learned_ai.sentinel.model import SentinelNet

logger = logging.getLogger(__name__)

_MESSAGES = {
    "safe": "safe",
    "possible_mistake": "possible_mistake",
    "missed_opportunity": "missed_opportunity",
    "critical": "critical",
}


@dataclass
class SentinelAdvice:
    """Lightweight advisory signal for a single position/decision."""

    mistake_risk: float
    opportunity_score: float
    trajectory_value_delta: float
    turning_point_confidence: float
    is_turning_point: bool
    advisory_message: str   # "safe" | "possible_mistake" | "missed_opportunity" | "critical"

    @classmethod
    def neutral(cls) -> "SentinelAdvice":
        return cls(0.0, 0.0, 0.0, 0.0, False, "safe")


class SentinelAdvisor:
    """Runtime inference wrapper around a trained SentinelNet."""

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
            dropout=0.0,  # eval: no dropout
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
    def advise(self, board_state, move_context: Optional[Dict[str, Any]] = None) -> SentinelAdvice:
        """Run a single forward pass and return a SentinelAdvice.

        Returns a neutral advice if the model is not loaded so callers in
        advisory mode degrade gracefully.
        """
        if not self.is_loaded():
            return SentinelAdvice.neutral()

        feats = build_features(board_state, move_context or {})
        if feats.shape[0] != FEATURE_DIM:  # defensive; build_features guarantees this
            return SentinelAdvice.neutral()
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(self.device)
        out = self.model(x)

        mistake = float(out.mistake_risk.reshape(-1)[0])
        opp = float(out.opportunity_score.reshape(-1)[0])
        delta = float(out.trajectory_value_delta.reshape(-1)[0])
        tp = float(out.turning_point_confidence.reshape(-1)[0])

        is_tp = tp >= self.config.turning_point_threshold
        message = self._message(mistake, opp, tp)
        return SentinelAdvice(
            mistake_risk=mistake,
            opportunity_score=opp,
            trajectory_value_delta=delta,
            turning_point_confidence=tp,
            is_turning_point=is_tp,
            advisory_message=message,
        )

    def _message(self, mistake: float, opp: float, tp: float) -> str:
        if tp >= self.config.reconsider_threshold:
            return _MESSAGES["critical"]
        if mistake >= 0.6:
            return _MESSAGES["possible_mistake"]
        if opp >= 0.6:
            return _MESSAGES["missed_opportunity"]
        return _MESSAGES["safe"]


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
