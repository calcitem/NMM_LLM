"""tests/test_phase_detection.py — verify 5-way phase classification."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game.board import BoardState, POSITIONS
from learned_ai.models.state_encoder import (
    PHASE_ENDGAME,
    PHASE_FLYING,
    PHASE_FULL_PLACEMENT,
    PHASE_MIDGAME,
    PHASE_OPENING_PLACEMENT,
    detect_phase,
)


def _make_state(positions, turn, *, placed=None, on=None):
    pos_map = {p: "" for p in POSITIONS}
    pos_map.update(positions)
    w_on = sum(1 for v in pos_map.values() if v == "W")
    b_on = sum(1 for v in pos_map.values() if v == "B")
    pieces_placed = placed if placed is not None else {"W": w_on, "B": b_on}
    pieces_on = on if on is not None else {"W": w_on, "B": b_on}
    return BoardState(
        positions=pos_map,
        turn=turn,
        pieces_on_board=pieces_on,
        pieces_placed=pieces_placed,
        pieces_captured={"W": 0, "B": 0},
    )


class TestPhaseDetection(unittest.TestCase):
    def test_opening_placement_new_game(self):
        board = BoardState.new_game()
        self.assertEqual(detect_phase(board), PHASE_OPENING_PLACEMENT)

    def test_full_placement_threshold(self):
        # 4 placements down for white -> full_placement.
        board = _make_state(
            positions={"a7": "W", "d7": "W", "g7": "W", "a4": "W"},
            turn="W",
            placed={"W": 4, "B": 3},
            on={"W": 4, "B": 3},
        )
        self.assertEqual(detect_phase(board), PHASE_FULL_PLACEMENT)

    def test_midgame_after_placement(self):
        board = _make_state(
            positions={
                "a7": "W", "d7": "W", "g7": "W", "a4": "W", "g4": "W",
                "a1": "W", "d1": "W", "g1": "W", "b4": "W",
                "c5": "B", "d5": "B", "e5": "B", "c4": "B", "e4": "B",
                "c3": "B", "d3": "B", "e3": "B", "b6": "B",
            },
            turn="W",
            placed={"W": 9, "B": 9},
            on={"W": 9, "B": 9},
        )
        self.assertEqual(detect_phase(board), PHASE_MIDGAME)

    def test_flying_phase(self):
        # White placed 9, currently has 3 pieces -> flying for white.
        board = _make_state(
            positions={"a7": "W", "d7": "W", "g7": "W", "c5": "B", "d5": "B", "e5": "B", "c4": "B"},
            turn="W",
            placed={"W": 9, "B": 9},
            on={"W": 3, "B": 4},
        )
        self.assertEqual(detect_phase(board), PHASE_FLYING)

    def test_endgame_when_opponent_low(self):
        # STM has 7, opponent has 3 -> opponent is in flying, STM is in endgame from spec.
        board = _make_state(
            positions={
                "a7": "W", "d7": "W", "g7": "W", "a4": "W", "g4": "W", "a1": "W", "d1": "W",
                "c5": "B", "d5": "B", "e5": "B",
            },
            turn="W",
            placed={"W": 9, "B": 9},
            on={"W": 7, "B": 3},
        )
        self.assertEqual(detect_phase(board), PHASE_ENDGAME)

    def test_endgame_when_stm_low_but_not_flying(self):
        # STM has 4 pieces, not flying yet (fly triggers at 3).
        board = _make_state(
            positions={"a7": "W", "d7": "W", "g7": "W", "a4": "W", "c5": "B", "d5": "B", "e5": "B", "c4": "B", "d3": "B"},
            turn="W",
            placed={"W": 9, "B": 9},
            on={"W": 4, "B": 5},
        )
        self.assertEqual(detect_phase(board), PHASE_ENDGAME)


if __name__ == "__main__":
    unittest.main()
