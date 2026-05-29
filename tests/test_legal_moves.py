"""tests/test_legal_moves.py — integration with the existing move generator.

The learned-AI subsystem must never enumerate legal moves itself; the only
authority is game.rules.get_all_legal_moves. These checks make sure the
encoder's masks line up with the engine's enumeration across phases.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves
from learned_ai.models.action_encoder import (
    CAPTURE_OFFSET,
    encode_action,
    get_legal_mask,
)


def primary_indices_from_legal(board):
    return {encode_action(mv)[0] for mv in get_all_legal_moves(board)}


class TestLegalMoves(unittest.TestCase):
    def test_opening_legal_set_matches_mask(self):
        board = BoardState.new_game()
        legal_primaries = primary_indices_from_legal(board)
        mask = get_legal_mask(board)
        mask_primaries = {
            i for i in range(CAPTURE_OFFSET) if bool(mask[i].item())
        }
        self.assertEqual(legal_primaries, mask_primaries)

    def test_movement_phase_mask_matches(self):
        board = BoardState.from_setup(
            positions={
                "a7": "W", "d7": "W", "g7": "B", "g4": "B",
                "a1": "W", "g1": "B", "d1": "W",
            },
            turn="W",
            phase="move",
        )
        legal_primaries = primary_indices_from_legal(board)
        mask = get_legal_mask(board)
        mask_primaries = {
            i for i in range(CAPTURE_OFFSET) if bool(mask[i].item())
        }
        self.assertEqual(legal_primaries, mask_primaries)

    def test_fly_phase_allows_non_adjacent(self):
        # White flying: 3 pieces only, all placed.
        positions = {"a7": "W", "d7": "W", "g7": "W"}
        for sq in ("c5", "d5", "e5", "c4"):
            positions[sq] = "B"
        board = BoardState(
            positions={p: positions.get(p, "") for p in POSITIONS},
            turn="W",
            pieces_on_board={"W": 3, "B": 4},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )
        legal = get_all_legal_moves(board)
        # Flying white can move any of its 3 pieces to any empty square.
        empties = sum(1 for v in board.positions.values() if v == "")
        self.assertEqual(len(legal), 3 * empties + 0)


if __name__ == "__main__":
    unittest.main()
