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
    input_dim: int = 120                       # 84 base + 36 context features
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1

    # ── Training ─────────────────────────────────────────────────────────────
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    val_fraction: float = 0.15
    class_weights: Dict[str, float] = field(default_factory=lambda: {
        "safe_continuation": 0.5,
        "mistake_start": 2.0,
        "missed_opportunity": 2.0,
        "critical_turning_point": 3.0,
        "neutral_state": 0.3,
    })
    seed: int = 42
    checkpoint_dir: str = "learned_ai/sentinel/checkpoints"
    log_dir: str = "learned_ai/sentinel/logs"

    # ── Backward labelling ───────────────────────────────────────────────────
    # backward_decay[i] = weight for the state i plies before a confirmed
    # turning point. Distances beyond the list reuse the final entry.
    backward_decay: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2])

    # ── External DB (training-time teacher only) ─────────────────────────────
    external_db_path: str = ""                 # e.g. /mnt/windows/NMM_DB/Entire DB
    external_db_enabled: bool = False          # False = gracefully skip DB supervision

    # ── Runtime ──────────────────────────────────────────────────────────────
    sentinel_mode: str = "advisory"            # "advisory" | "score_adjust" | "reconsider"
    score_adjust_scale: float = 0.05           # max heuristic score delta in score_adjust mode
    reconsider_threshold: float = 0.8          # turning_point_confidence to trigger reconsider
    turning_point_threshold: float = 0.5       # is_turning_point cutoff for advisory flags

    # ── Loss weighting (per-head multipliers) ────────────────────────────────
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "mistake_risk": 1.0,
        "opportunity_score": 1.0,
        "trajectory_value_delta": 1.0,
        "turning_point_confidence": 1.0,
    })

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
