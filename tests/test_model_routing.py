"""tests/test_model_routing.py — verify backbone + per-phase head routing."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from game.board import BoardState
from learned_ai.models.action_encoder import ACTION_DIM, get_legal_mask
from learned_ai.models.backbone import NEG_INF, NMMNet
from learned_ai.models.state_encoder import (
    NUM_PHASES,
    PHASE_NAMES,
    STATE_DIM,
    encode_state,
)


class TestModelRouting(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = NMMNet(backbone_hidden=(32, 32, 16), head_hidden=(16,))

    def test_phase_head_count(self):
        self.assertEqual(len(self.net.phase_heads), NUM_PHASES)
        for name in PHASE_NAMES:
            self.assertIn(name, self.net.phase_heads)

    def test_output_shapes_unbatched(self):
        board = BoardState.new_game()
        state = encode_state(board)
        mask = get_legal_mask(board)
        out = self.net.forward(state, phase_id=0, legal_mask=mask)
        self.assertEqual(out["logits"].shape, (ACTION_DIM,))
        self.assertEqual(out["value"].shape, ())

    def test_output_shapes_batched(self):
        board = BoardState.new_game()
        state = encode_state(board).unsqueeze(0).repeat(4, 1)
        mask = get_legal_mask(board).unsqueeze(0).repeat(4, 1)
        out = self.net.forward(state, phase_id=0, legal_mask=mask)
        self.assertEqual(out["logits"].shape, (4, ACTION_DIM))
        self.assertEqual(out["value"].shape, (4,))

    def test_legal_mask_pushes_illegal_to_neg_inf(self):
        board = BoardState.new_game()
        state = encode_state(board)
        mask = get_legal_mask(board)
        out = self.net.forward(state, phase_id=0, legal_mask=mask)
        logits = out["logits"]
        # Illegal positions must be exactly NEG_INF.
        self.assertTrue(torch.all(logits[~mask] == NEG_INF))
        # Legal positions must be finite.
        self.assertTrue(torch.all(torch.isfinite(logits[mask])))

    def test_softmax_over_legal_actions_sums_to_one(self):
        board = BoardState.new_game()
        state = encode_state(board)
        mask = get_legal_mask(board)
        probs = self.net.policy_probs(state, phase_id=0, legal_mask=mask)
        # Illegal -> zero probability.
        self.assertAlmostEqual(probs[~mask].sum().item(), 0.0, places=5)
        # Legal slice sums to ~1.
        self.assertAlmostEqual(probs[mask].sum().item(), 1.0, places=5)

    def test_different_phases_use_different_heads(self):
        board = BoardState.new_game()
        state = encode_state(board)
        out0 = self.net.forward(state, phase_id=0, legal_mask=None)
        out1 = self.net.forward(state, phase_id=1, legal_mask=None)
        # Different heads should produce different logits.
        self.assertFalse(torch.allclose(out0["logits"], out1["logits"]))
        # But the value head is shared (same input -> same value).
        self.assertTrue(torch.allclose(out0["value"], out1["value"]))

    def test_invalid_phase_raises(self):
        board = BoardState.new_game()
        state = encode_state(board)
        with self.assertRaises(ValueError):
            self.net.forward(state, phase_id=NUM_PHASES, legal_mask=None)

    def test_state_dim_consistency(self):
        self.assertEqual(self.net.state_dim, STATE_DIM)


if __name__ == "__main__":
    unittest.main()
