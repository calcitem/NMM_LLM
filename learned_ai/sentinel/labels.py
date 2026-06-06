"""learned_ai/sentinel/labels.py — backward label propagation for the sentinel.

Training examples come from *played games*. The external solved DB is a
training-time teacher: where it can classify a state we use direct supervision;
elsewhere we propagate the nearest later judgement backward along the
trajectory, and fall back to the recorded game winner as a weak proxy.

Label types
-----------
  safe_continuation       played move preserved a strong trajectory
  mistake_start           played move began deterioration vs better alternatives
  missed_opportunity      a stronger move existed but was not chosen
  critical_turning_point  small choice caused a large downstream outcome change
  neutral_state           no strong evidence of strategic decisiveness

Each LabelledExample carries the four regression/classification targets the
SentinelNet predicts, plus a per-sample training weight and the supervision
source for auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

LABEL_TYPES = (
    "safe_continuation",
    "mistake_start",
    "missed_opportunity",
    "critical_turning_point",
    "neutral_state",
)

# Default backward-decay weights by distance (plies) from a confirmed turning
# point. Distances beyond the list reuse the final entry.
DEFAULT_BACKWARD_DECAY: List[float] = [1.0, 0.8, 0.6, 0.4, 0.2]

# Map a WDL string (side-to-move perspective) to a scalar value.
_WDL_VALUE = {"W": 1.0, "D": 0.0, "L": -1.0}


@dataclass
class LabelledExample:
    """One supervised sentinel training example."""

    state_features: np.ndarray              # (120,) float32
    label: str                              # one of LABEL_TYPES
    turning_point_confidence: float         # [0,1]
    value_delta: float                      # estimated strategic shift [-1,1]
    mistake_risk: float                     # [0,1]
    opportunity_score: float                # [0,1]
    training_weight: float                  # per-sample loss weight
    supervision_source: str                 # see SUPERVISION_SOURCES
    ply: int = 0                            # index within the game (for audit)
    meta: Dict[str, Any] = field(default_factory=dict)

    def target_dict(self) -> Dict[str, float]:
        """The four head targets the model is trained against."""
        return {
            "mistake_risk": float(self.mistake_risk),
            "opportunity_score": float(self.opportunity_score),
            "trajectory_value_delta": float(self.value_delta),
            "turning_point_confidence": float(self.turning_point_confidence),
            "weight": float(self.training_weight),
        }


SUPERVISION_SOURCES = (
    "direct_solved",
    "backward_propagated",
    "trajectory_outcome",
    "weak_proxy",
)


def _decay_weight(distance: int, decay: Sequence[float]) -> float:
    if not decay:
        return 0.2
    if distance < len(decay):
        return float(decay[distance])
    return float(decay[-1])


def _winner_value_for_color(winner: Optional[str], color: str) -> Optional[float]:
    """Game-outcome value from ``color``'s perspective: +1 win, -1 loss, 0 draw."""
    if winner is None:
        return None
    w = str(winner).upper()
    if w in ("D", "DRAW", "NONE", ""):
        return 0.0
    if w in ("W", "B"):
        return 1.0 if w == color else -1.0
    return None


def _classify(value_delta: float, mistake_risk: float, opportunity: float,
              tp_conf: float) -> str:
    """Derive a categorical label from the continuous targets.

    Thresholds are deliberately simple; the categorical label is mostly for
    diagnostics / class-distribution checks, while the heads train on the
    continuous targets.
    """
    if tp_conf >= 0.6:
        return "critical_turning_point"
    if mistake_risk >= 0.6 and value_delta < 0:
        return "mistake_start"
    if opportunity >= 0.6:
        return "missed_opportunity"
    if value_delta >= 0.15 and mistake_risk < 0.4:
        return "safe_continuation"
    return "neutral_state"


def backward_label_trajectory(
    game_record: Dict[str, Any],
    states: Sequence[Any],
    features: Sequence[np.ndarray],
    move_contexts: Sequence[Dict[str, Any]],
    db=None,
    backward_decay: Optional[Sequence[float]] = None,
) -> List[LabelledExample]:
    """Label a single game's trajectory.

    Parameters
    ----------
    game_record   : parsed game dict (uses ``winner`` for proxy supervision).
    states        : BoardState before each labelled ply (len N).
    features      : precomputed 120-float feature vectors, aligned to ``states``.
    move_contexts : per-ply context dicts (provide ``color``, ``was_blunder``,
                    ``chosen_rank``, ``game_ai_score`` when available).
    db            : ExternalSolvedDB-like teacher (or None / unavailable).
    backward_decay: decay weights by distance; defaults to DEFAULT_BACKWARD_DECAY.

    Algorithm
    ---------
    1. Walk forward; query the DB for each state (None when unavailable).
    2. Where the DB gives a direct WDL, that state gets ``direct_solved``
       supervision.
    3. For states without a direct result, find the nearest *later* DB-resolved
       state and propagate its value backward with the decay weight; mark
       ``backward_propagated``.
    4. States still unresolved use the game winner as a weak proxy
       (``trajectory_outcome`` near the end of the game, ``weak_proxy`` earlier).
    5. A turning point is flagged where the per-mover value swings sharply
       between consecutive resolved states, or where a logged blunder occurs.
    """
    decay = list(backward_decay) if backward_decay else DEFAULT_BACKWARD_DECAY
    n = len(states)
    if not (n == len(features) == len(move_contexts)):
        raise ValueError("states, features, move_contexts must be equal length")

    winner = game_record.get("winner")

    # ── Step 1: direct DB values per state (None where unavailable). ──────────
    db_wdl: List[Optional[str]] = [None] * n
    if db is not None and getattr(db, "is_available", lambda: False)():
        try:
            db_wdl = list(db.query_trajectory(list(states)))
            if len(db_wdl) != n:
                db_wdl = [None] * n
        except Exception:
            db_wdl = [None] * n

    # Per-state value from the *mover at that ply* perspective.
    # encode each resolved WDL (side-to-move) as a scalar.
    resolved_value: List[Optional[float]] = [
        (_WDL_VALUE.get(w) if w is not None else None) for w in db_wdl
    ]
    resolved_source: List[Optional[str]] = [
        ("direct_solved" if v is not None else None) for v in resolved_value
    ]

    # ── Step 2/3: backward-propagate nearest later resolved value. ────────────
    # next_resolved[i] = (value, distance) of the nearest resolved state at j>=i.
    next_resolved: List[Optional[tuple]] = [None] * n
    carry: Optional[tuple] = None  # (value, index)
    for i in range(n - 1, -1, -1):
        if resolved_value[i] is not None:
            carry = (resolved_value[i], i)
            next_resolved[i] = (resolved_value[i], 0)
        elif carry is not None:
            val, idx = carry
            next_resolved[i] = (val, idx - i)

    # ── Step 4: outcome proxy for anything still unresolved. ──────────────────
    examples: List[LabelledExample] = []
    last_value: Optional[float] = None
    for i in range(n):
        ctx = move_contexts[i] or {}
        color = ctx.get("color") or getattr(states[i], "turn", "W")
        weight = 1.0
        if resolved_value[i] is not None:
            value = resolved_value[i]
            source = "direct_solved"
        elif next_resolved[i] is not None:
            val, dist = next_resolved[i]
            value = val
            weight = _decay_weight(dist, decay)
            source = "backward_propagated"
        else:
            proxy = _winner_value_for_color(winner, color)
            if proxy is None:
                value = 0.0
                source = "weak_proxy"
                weight = _decay_weight(99, decay)  # smallest weight
            else:
                value = proxy
                # nearer the end of the game the outcome is a stronger signal.
                frac = (i + 1) / float(n)
                if frac >= 0.66:
                    source = "trajectory_outcome"
                    weight = 0.6
                else:
                    source = "weak_proxy"
                    weight = _decay_weight(99, decay)

        # ── Turning-point / mistake / opportunity signals. ────────────────────
        was_blunder = bool(ctx.get("was_blunder"))
        chosen_rank = ctx.get("chosen_rank", 0) or 0
        n_cand = len(ctx.get("candidates") or [])

        # value swing vs previous resolved/proxy value (mover perspective flips
        # each ply, so compare magnitudes of change in mover-relative value).
        swing = 0.0 if last_value is None else abs(value - (-last_value))
        last_value = value

        tp_conf = 0.0
        if was_blunder:
            tp_conf = max(tp_conf, 0.85)
        tp_conf = max(tp_conf, min(1.0, swing))  # large swing => turning point

        # mistake_risk: high when the trajectory value is bad for the mover or a
        # blunder was logged.
        mistake_risk = 0.0
        if value < 0:
            mistake_risk = min(1.0, 0.5 + 0.5 * abs(value))
        if was_blunder:
            mistake_risk = max(mistake_risk, 0.9)

        # opportunity_score: high when a clearly better move was available but not
        # taken. Two evidence sources:
        #   1. an explicit better-ranked candidate (enriched self-play logs), or
        #   2. a logged blunder in a non-losing position (the engine itself flags
        #      that a better move existed), since basic game logs carry only the
        #      single played move.
        opportunity = 0.0
        if n_cand > 1 and chosen_rank > 0:
            opportunity = min(1.0, chosen_rank / float(n_cand - 1))
            if value < -0.5:
                opportunity *= 0.5  # already losing — less of a "missed" chance
        elif was_blunder and value > -0.5:
            opportunity = 0.7

        label = _classify(value, mistake_risk, opportunity, tp_conf)
        # weak proxies should not over-train the categorical extremes.
        examples.append(
            LabelledExample(
                state_features=np.asarray(features[i], dtype=np.float32),
                label=label,
                turning_point_confidence=float(tp_conf),
                value_delta=float(max(-1.0, min(1.0, value))),
                mistake_risk=float(mistake_risk),
                opportunity_score=float(opportunity),
                training_weight=float(weight),
                supervision_source=source,
                ply=i,
                meta={"color": color, "winner": winner},
            )
        )
    return examples
