"""tests/test_action_encoder.py — verify move <-> index encoding."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves
from learned_ai.models.action_encoder import (
    ACTION_DIM,
    CAPTURE_OFFSET,
    MOVE_OFFSET,
    PLACE_OFFSET,
    decode_action,
    encode_action,
    get_legal_mask,
    get_legal_moves,
    move_requires_capture,
)


class TestActionEncoder(unittest.TestCase):
    def test_action_dim(self):
        self.assertEqual(ACTION_DIM, 624)

    def test_placement_round_trip(self):
        board = BoardState.new_game()
        for mv in get_all_legal_moves(board):
            primary, cap = encode_action(mv)
            self.assertEqual(cap, None)
            self.assertTrue(PLACE_OFFSET <= primary < MOVE_OFFSET)
            decoded = decode_action(primary, board)
            self.assertEqual(decoded["from"], mv["from"])
            self.assertEqual(decoded["to"], mv["to"])
            self.assertEqual(decoded.get("capture"), mv.get("capture"))

    def test_legal_mask_shape_and_values(self):
        board = BoardState.new_game()
        mask = get_legal_mask(board)
        self.assertEqual(mask.shape, (ACTION_DIM,))
        # Exactly the 24 placement bits should be set in opening placement.
        self.assertEqual(int(mask[PLACE_OFFSET:MOVE_OFFSET].sum().item()), 24)
        # No movement / capture bits set.
        self.assertEqual(int(mask[MOVE_OFFSET:CAPTURE_OFFSET].sum().item()), 0)
        self.assertEqual(int(mask[CAPTURE_OFFSET:].sum().item()), 0)

    def test_no_illegal_moves_in_mask(self):
        board = BoardState.new_game()
        # Place a piece, then verify mask matches the new legal set.
        board = board.apply_move({"from": None, "to": "d7", "capture": None})
        mask = get_legal_mask(board)
        legal_indices = {encode_action(m)[0] for m in get_all_legal_moves(board)}
        # Every primary index that's True in the mask must be in legal_indices.
        for idx in range(MOVE_OFFSET):
            if mask[idx]:
                self.assertIn(idx, legal_indices)
        for idx, sq in enumerate(POSITIONS):
            # Empty squares cant be captured; only black pieces could have been
            # placed but board has only one white piece -> no captures possible.
            self.assertFalse(mask[CAPTURE_OFFSET + idx].item())

    def test_move_round_trip_in_movement_phase(self):
        # Build a movement-phase position.
        board = BoardState.from_setup(
            positions={"a7": "W", "d7": "B", "g7": "B", "a1": "W"},
            turn="W",
            phase="move",
        )
        legal = get_all_legal_moves(board)
        self.assertTrue(legal, "expected at least one legal movement")
        for mv in legal:
            primary, cap = encode_action(mv)
            self.assertGreaterEqual(primary, MOVE_OFFSET)
            decoded = decode_action(primary, board, capture_index=cap)
            self.assertEqual(decoded["from"], mv["from"])
            self.assertEqual(decoded["to"], mv["to"])
            self.assertEqual(decoded.get("capture"), mv.get("capture"))

    def test_capture_required_detection(self):
        # White already has a7 and d7; placing g7 would close the top mill.
        board = BoardState.from_setup(
            positions={"a7": "W", "d7": "W"},
            turn="W",
            phase="place",
        )
        primary, _ = encode_action({"from": None, "to": "g7", "capture": None})
        self.assertTrue(move_requires_capture(board, primary))
        # An unrelated placement does not require a capture.
        primary2, _ = encode_action({"from": None, "to": "a4", "capture": None})
        self.assertFalse(move_requires_capture(board, primary2))

    def test_get_legal_moves_proxy(self):
        board = BoardState.new_game()
        engine_moves = get_all_legal_moves(board)
        proxied = get_legal_moves(board)
        self.assertEqual(len(engine_moves), len(proxied))


if __name__ == "__main__":
    unittest.main()
