"""Strategic Sentinel Overlay — a learned advisory model for the NMM heuristic engine.

The sentinel watches game trajectories, learns to flag strategic turning points
across all phases, and provides lightweight advisory signals to the existing
heuristic GameAI at runtime. It does NOT replace GameAI — advisory only.

Public surface (imported lazily to keep `import learned_ai.sentinel` cheap and
free of heavy torch import side-effects for callers that only need the config):

    from learned_ai.sentinel.config import SentinelConfig, load_config
    from learned_ai.sentinel.feature_builder import build_features
    from learned_ai.sentinel.db_teacher import ExternalSolvedDB
    from learned_ai.sentinel.labels import backward_label_trajectory, LabelledExample
    from learned_ai.sentinel.dataset import SentinelDataset
    from learned_ai.sentinel.model import SentinelNet, SentinelOutput
    from learned_ai.sentinel.infer import SentinelAdvisor, SentinelAdvice
"""

from __future__ import annotations

__all__ = [
    "SentinelConfig",
    "load_config",
]

from learned_ai.sentinel.config import SentinelConfig, load_config
