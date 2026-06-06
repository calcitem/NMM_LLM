"""Strategic Sentinel Overlay — a learned move-level scorer for the NMM engine.

The sentinel scores each candidate move (quality in [0, 1] from the mover's
perspective) and provides lightweight advisory signals to the existing heuristic
GameAI at runtime. It does NOT replace GameAI — advisory by default.

Public surface (imported lazily to keep `import learned_ai.sentinel` cheap and
free of heavy torch import side-effects for callers that only need the config):

    from learned_ai.sentinel.config import SentinelConfig, load_config
    from learned_ai.sentinel.feature_builder import build_move_features, FEATURE_DIM
    from learned_ai.sentinel.db_teacher import ExternalSolvedDB
    from learned_ai.sentinel.labels import label_move, MoveExample
    from learned_ai.sentinel.dataset import SentinelDataset
    from learned_ai.sentinel.model import SentinelNet, sentinel_loss
    from learned_ai.sentinel.infer import SentinelAdvisor, SentinelAdvice
"""

from __future__ import annotations

__all__ = [
    "SentinelConfig",
    "load_config",
]

from learned_ai.sentinel.config import SentinelConfig, load_config
