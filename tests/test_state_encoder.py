"""tests/test_state_encoder.py — verify the 84-float state encoder."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from game.board import BoardState, POSITIONS
from learned_ai.models.state_encoder import (
    STATE_DIM,
    NUM_PHASES,
    PHASE_OPENING_PLACEMENT,
    PHASE_FULL_PLACEMENT,
    PHASE_MIDGAME,
    PHASE_ENDGAME,
    PHASE_FLYING,
    detect_phase,
    encode_state,
    encode_state_with_phase,
)


class TestStateEncoder(unittest.TestCase):
    def test_shape_and_dtype(self):
        board = BoardState.new_game()
        vec = encode_state(board)
        self.assertEqual(vec.shape, (STATE_DIM,))
        self.assertEqual(vec.dtype, torch.float32)

    def test_new_game_one_hot_layout(self):
        board = BoardState.new_game()
        vec = encode_state(board)
        # All 24 positions empty: every triplet should be (1, 0, 0).
        for i in range(24):
            base = i * 3
            self.assertAlmostEqual(vec[base + 0].item(), 1.0)
            self.assertAlmostEqual(vec[base + 1].item(), 0.0)
            self.assertAlmostEqual(vec[base + 2].item(), 0.0)
        # White to move.
        self.assertAlmostEqual(vec[72].item(), 0.0)
        # Phase one-hot: exactly one bit set, equal to opening_placement.
        phase_block = vec[73:73 + NUM_PHASES]
        self.assertEqual(phase_block.sum().item(), 1.0)
        self.assertEqual(int(torch.argmax(phase_block).item()), PHASE_OPENING_PLACEMENT)
        # Counts all zero.
        for k in (78, 79, 80, 81, 82, 83):
            self.assertAlmostEqual(vec[k].item(), 0.0)

    def test_piece_placement_sets_correct_channel(self):
        board = BoardState.new_game()
        board = board.apply_move({"from": None, "to": "d7", "capture": None})
        vec = encode_state(board)
        # d7 occupies POSITIONS index 1 — channel 1 (white) should be set.
        idx = POSITIONS.index("d7")
        base = idx * 3
        self.assertAlmostEqual(vec[base + 0].item(), 0.0)
        self.assertAlmostEqual(vec[base + 1].item(), 1.0)
        self.assertAlmostEqual(vec[base + 2].item(), 0.0)
        # Side to move flipped to black.
        self.assertAlmostEqual(vec[72].item(), 1.0)

    def test_phase_detection_opening_placement(self):
        board = BoardState.new_game()
        self.assertEqual(detect_phase(board), PHASE_OPENING_PLACEMENT)

    def test_phase_detection_full_placement(self):
        # White placed 4 pieces -> full_placement phase for white.
        board = BoardState.new_game()
        for sq in ["a7", "g7", "a1", "g1"]:
            # Alternate fake placements to advance counters; pieces dont matter
            # for the phase check.
            board = board.apply_move({"from": None, "to": sq, "capture": None})
            # Place a black piece between each white placement to keep turns even.
            empties = [p for p in POSITIONS if board.positions[p] == ""]
            board = board.apply_move({"from": None, "to": empties[0], "capture": None})
        # Now it should be white's turn with 4 placements down.
        if board.turn != "W":
            # Skip one black placement to align turns.
            empties = [p for p in POSITIONS if board.positions[p] == ""]
            board = board.apply_move({"from": None, "to": empties[0], "capture": None})
        self.assertGreaterEqual(board.pieces_placed["W"], 4)
        self.assertEqual(detect_phase(board), PHASE_FULL_PLACEMENT)

    def test_encode_state_with_phase_consistency(self):
        board = BoardState.new_game()
        vec, phase_id = encode_state_with_phase(board)
        self.assertEqual(phase_id, detect_phase(board))
        # The one-hot bit position must agree with phase_id.
        self.assertAlmostEqual(vec[73 + phase_id].item(), 1.0)

    def test_round_trip_consistency_same_board_same_vector(self):
        board = BoardState.new_game()
        v1 = encode_state(board)
        v2 = encode_state(board)
        self.assertTrue(torch.equal(v1, v2))

    def test_invalid_piece_value_raises(self):
        board = BoardState.new_game()
        bad = BoardState(
            positions=dict(board.positions, a1="X"),
            turn=board.turn,
            pieces_on_board=dict(board.pieces_on_board),
            pieces_placed=dict(board.pieces_placed),
            pieces_captured=dict(board.pieces_captured),
        )
        with self.assertRaises(ValueError):
            encode_state(bad)


if __name__ == "__main__":
    unittest.main()
