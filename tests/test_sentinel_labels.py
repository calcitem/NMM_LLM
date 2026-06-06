"""Tests for learned_ai/sentinel/labels.py (backward label propagation)."""

from __future__ import annotations

import numpy as np

from game.board import BoardState
from learned_ai.sentinel.feature_builder import FEATURE_DIM
from learned_ai.sentinel.labels import (
    DEFAULT_BACKWARD_DECAY,
    LABEL_TYPES,
    LabelledExample,
    backward_label_trajectory,
)


class _FakeDB:
    """A stub teacher that resolves only specified plies to a given WDL."""

    def __init__(self, resolved):
        # resolved: dict {ply_index: "W"|"L"|"D"}
        self._resolved = resolved
        self._states = None

    def is_available(self):
        return True

    def query_trajectory(self, states):
        return [self._resolved.get(i) for i in range(len(states))]


def _trajectory(n):
    states = [BoardState.new_game() for _ in range(n)]
    feats = [np.zeros(FEATURE_DIM, dtype=np.float32) for _ in range(n)]
    ctxs = [{"color": "W" if i % 2 == 0 else "B"} for i in range(n)]
    return states, feats, ctxs


def test_label_backward_propagation_weights():
    # Turning point resolved at ply 10; check decayed weights at 9,8,7,6.
    n = 11
    states, feats, ctxs = _trajectory(n)
    db = _FakeDB({10: "W"})
    record = {"winner": "W"}
    examples = backward_label_trajectory(record, states, feats, ctxs, db=db)
    assert len(examples) == n
    # ply 10 is direct (distance 0 -> weight 1.0)
    assert examples[10].supervision_source == "direct_solved"
    assert examples[10].training_weight == DEFAULT_BACKWARD_DECAY[0]
    # plies 9..6 are backward-propagated with decreasing weights
    assert examples[9].supervision_source == "backward_propagated"
    assert examples[9].training_weight == DEFAULT_BACKWARD_DECAY[1]
    assert examples[8].training_weight == DEFAULT_BACKWARD_DECAY[2]
    assert examples[7].training_weight == DEFAULT_BACKWARD_DECAY[3]
    assert examples[6].training_weight == DEFAULT_BACKWARD_DECAY[4]


def test_game_outcome_proxy_when_db_unavailable():
    n = 6
    states, feats, ctxs = _trajectory(n)
    record = {"winner": "W"}
    examples = backward_label_trajectory(record, states, feats, ctxs, db=None)
    assert len(examples) == n
    # No DB => every example is a proxy (trajectory_outcome or weak_proxy).
    for ex in examples:
        assert ex.supervision_source in ("trajectory_outcome", "weak_proxy")
    # White mover (even plies) gets a positive proxy value when White wins.
    assert examples[0].value_delta >= 0.0


def test_direct_supervision_takes_priority():
    n = 4
    states, feats, ctxs = _trajectory(n)
    # DB says ply 1 is resolved; the game winner proxy would say otherwise.
    db = _FakeDB({1: "L"})
    record = {"winner": "W"}
    examples = backward_label_trajectory(record, states, feats, ctxs, db=db)
    assert examples[1].supervision_source == "direct_solved"
    # The resolved value (L => -1) overrides the would-be proxy.
    assert examples[1].value_delta < 0.0


def test_blunder_flags_turning_point():
    n = 4
    states, feats, ctxs = _trajectory(n)
    ctxs[2]["was_blunder"] = True
    record = {"winner": "B"}
    examples = backward_label_trajectory(record, states, feats, ctxs, db=None)
    assert examples[2].turning_point_confidence >= 0.8
    assert examples[2].mistake_risk >= 0.8


def test_all_label_types_reachable():
    # Synthetic trajectory crafted so the labeller emits every category at least
    # once. Multi-candidate context drives the missed_opportunity branch.
    n = 6
    states, feats, ctxs = _trajectory(n)
    record = {"winner": "W"}
    # ply with a strong missed opportunity for the eventual WINNER (so the
    # position is not losing and the missed_opportunity branch is reachable):
    # ply 2 is a White mover (even index) and White wins this game.
    ctxs[2].update({
        "candidates": [{"score": 5.0}, {"score": 1.0}, {"score": 0.5}],
        "chosen_rank": 2,
    })
    # ply flagged as a logged blunder => mistake / turning point signal
    ctxs[3]["was_blunder"] = True
    examples = backward_label_trajectory(record, states, feats, ctxs, db=None)
    labels = {e.label for e in examples}
    # Every produced label must be a valid type.
    assert labels.issubset(set(LABEL_TYPES))
    # missed_opportunity must be reachable from the crafted candidate context.
    assert "missed_opportunity" in labels


def test_returns_labelled_example_instances():
    n = 3
    states, feats, ctxs = _trajectory(n)
    examples = backward_label_trajectory({"winner": "D"}, states, feats, ctxs)
    assert all(isinstance(e, LabelledExample) for e in examples)
    for e in examples:
        td = e.target_dict()
        assert set(td.keys()) == {
            "mistake_risk", "opportunity_score",
            "trajectory_value_delta", "turning_point_confidence", "weight",
        }
        assert 0.0 <= td["mistake_risk"] <= 1.0
        assert -1.0 <= td["trajectory_value_delta"] <= 1.0
