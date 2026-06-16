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

# DTM normalisation constants.  win-in-N → quality = 1.0 - N/scale (clamped to
# [0.55, 1.0]).  loss-in-N → quality = N/scale (clamped to [0.0, 0.45]).
DTM_WIN_SCALE = 100
DTM_LOSS_SCALE = 100

# Moves where the opponent can force a win in ≤ this many plies get extra weight
# so the sentinel prioritises catching catastrophic blunders during training.
BAD_MOVE_DTM_THRESHOLD = 10
BAD_MOVE_WEIGHT = 2.0

# Trajectory stage: extra training weight for the move actually played in a game,
# conditioned on the game outcome.  Win-trajectory moves get a larger boost
# because correct winning choices are the signal we most want to reinforce.
TRAJECTORY_WIN_BOOST = 3.0
TRAJECTORY_LOSS_BOOST = 2.0


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


def dtm_quality(wdl: str, dtm: Optional[int]) -> float:
    """DTM-graded quality in [0,1].

    Wins are graded [0.55, 1.0]: win-in-1 ≈ 0.99, win-in-100 ≈ 0.55.
    Losses are graded [0.0, 0.45]: loss-in-1 ≈ 0.01, loss-in-100 ≈ 0.45.
    Draws or unknown DTM fall back to binary WDL_QUALITY.
    """
    if dtm is None:
        return WDL_QUALITY.get(wdl, 0.5)
    if wdl == "win":
        return 1.0 - min(abs(dtm) / DTM_WIN_SCALE, 0.45)
    if wdl == "loss":
        return min(abs(dtm) / DTM_LOSS_SCALE, 0.45)
    return 0.5


def label_move(
    wdl: Optional[str],
    heuristic_score_norm: float = 0.5,
    dtm: Optional[int] = None,
) -> "tuple[float, float, str]":
    """Return (move_quality, training_weight, supervision_source) for one move.

    Prefers the solved-DB WDL+DTM. Falls back to normalised heuristic score as a
    weak label when the DB has no entry for this move.
    """
    q = quality_from_wdl(wdl)
    if q is not None:
        if wdl == "draw":
            return q, DRAW_WEIGHT, "solved_db"
        if dtm is not None and wdl in ("win", "loss"):
            dtm_q = dtm_quality(wdl, dtm)
            weight = BAD_MOVE_WEIGHT if (wdl == "loss" and abs(dtm) <= BAD_MOVE_DTM_THRESHOLD) else 1.0
            return dtm_q, weight, "solved_db_dtm"
        return q, 1.0, "solved_db"
    q = float(min(1.0, max(0.0, heuristic_score_norm)))
    return q, WEAK_LABEL_WEIGHT, "heuristic_weak"
