"""learned_ai/sentinel/labels.py — move-quality labelling for the sentinel.

The sentinel is a move-level scorer: each example is one candidate move in one
position, labelled with a single ``move_quality`` float in [0, 1] from the
mover's perspective:

    1.0 = the solved DB says this move wins for the mover
    0.5 = the DB says draw
    0.0 = the DB says loss
    weak label = heuristic score normalised to [0, 1] when the DB is unavailable

Draw examples carry a reduced training weight (default 0.5) so the BCE target of
0.5 does not dominate the win/loss signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

# Map a per-move WDL string (mover's perspective) to a quality target.
WDL_QUALITY = {"win": 1.0, "draw": 0.5, "loss": 0.0}

# Down-weight draw examples (target 0.5 is otherwise ambiguous under BCE).
DRAW_WEIGHT = 0.5
# Weak (heuristic-only) labels are trusted less than solved-DB labels.
WEAK_LABEL_WEIGHT = 0.4


@dataclass
class MoveExample:
    """One supervised move-level training example."""

    features: np.ndarray            # (FEATURE_DIM,) float32
    move_quality: float             # [0, 1] label, mover's perspective
    training_weight: float          # per-sample BCE weight
    supervision_source: str         # "solved_db" | "heuristic_weak"
    ply: int = 0                    # ply within the game (audit)
    move_notation: str = ""         # optional, for diagnostics
    meta: Dict[str, Any] = field(default_factory=dict)

    def target(self) -> float:
        return float(self.move_quality)


def quality_from_wdl(wdl: Optional[str]) -> Optional[float]:
    """Quality target from a DB WDL string, or None if unknown/unavailable."""
    if wdl is None:
        return None
    return WDL_QUALITY.get(wdl)


def label_move(
    wdl: Optional[str],
    heuristic_score_norm: float = 0.5,
) -> "tuple[float, float, str]":
    """Return (move_quality, training_weight, supervision_source) for one move.

    Prefers the solved-DB WDL. Falls back to the normalised heuristic score as a
    weak label when the DB has no entry for this move.
    """
    q = quality_from_wdl(wdl)
    if q is not None:
        weight = DRAW_WEIGHT if wdl == "draw" else 1.0
        return q, weight, "solved_db"
    q = float(min(1.0, max(0.0, heuristic_score_norm)))
    return q, WEAK_LABEL_WEIGHT, "heuristic_weak"
