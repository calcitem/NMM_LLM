"""learned_ai/sentinel/config.py — SentinelConfig dataclass + load_config().

A single source of truth for sentinel hyper-parameters, paths, and runtime
behaviour. Loaded from YAML (configs/sentinel_*.yaml) or constructed with
defaults. Unknown keys in the YAML are ignored so older/newer config files do
not crash a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a declared dependency
    yaml = None  # type: ignore


@dataclass
class SentinelConfig:
    # ── Model ────────────────────────────────────────────────────────────────
    # Move-level scorer: input is the per-move feature vector (FEATURE_DIM=58).
    input_dim: int = 58
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2

    # ── Training ─────────────────────────────────────────────────────────────
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    val_fraction: float = 0.15
    seed: int = 42
    checkpoint_dir: str = "learned_ai/sentinel/checkpoints"
    log_dir: str = "learned_ai/sentinel/logs"

    # ── External DB (training-time teacher only) ─────────────────────────────
    external_db_path: str = ""                 # e.g. /mnt/windows/NMM_DB/Entire DB
    external_db_enabled: bool = False          # False = gracefully skip DB supervision

    # ── Runtime ──────────────────────────────────────────────────────────────
    sentinel_mode: str = "advisory"            # "advisory" | "score_adjust" | "reconsider"
    score_adjust_scale: float = 0.05           # reserved tunable for score_adjust mode
    reconsider_threshold: float = 0.3          # opportunity_gap to trigger reconsider

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SentinelConfig":
        """Build a config from a dict, ignoring unknown keys."""
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in (d or {}).items() if k in valid}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_config(path: Optional[str] = None) -> SentinelConfig:
    """Load a SentinelConfig from a YAML file, or return defaults when path is None."""
    if path is None:
        return SentinelConfig()
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load sentinel config files")
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    return SentinelConfig.from_dict(d)
