"""tests/test_checkpoint_save_load.py — verify checkpoint round-trip integrity."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from game.board import BoardState
from learned_ai.agents.learned_agent import LearnedAgent
from learned_ai.models.action_encoder import get_legal_mask
from learned_ai.models.backbone import NMMNet
from learned_ai.models.state_encoder import encode_state


class TestCheckpointSaveLoad(unittest.TestCase):
    def test_state_dict_round_trip(self):
        torch.manual_seed(0)
        net = NMMNet(backbone_hidden=(32, 32, 16), head_hidden=(16,))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            path = tmp.name
        try:
            torch.save(net.state_dict(), path)
            net2 = NMMNet(backbone_hidden=(32, 32, 16), head_hidden=(16,))
            net2.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
            for (k1, v1), (k2, v2) in zip(net.state_dict().items(), net2.state_dict().items()):
                self.assertEqual(k1, k2)
                self.assertTrue(torch.equal(v1, v2))
        finally:
            os.unlink(path)

    def test_inference_unchanged_after_reload(self):
        torch.manual_seed(0)
        agent = LearnedAgent(
            color="W", backbone_hidden=(32, 32, 16), head_hidden=(16,), mode="argmax", seed=0,
            device="cpu",
        )
        board = BoardState.new_game()
        state = encode_state(board)
        mask = get_legal_mask(board)
        before = agent.model.forward(state, phase_id=0, legal_mask=mask)
        before_logits = before["logits"].detach().clone()
        before_value = before["value"].detach().clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            path = tmp.name
        try:
            torch.save(agent.model.state_dict(), path)
            agent2 = LearnedAgent(
                color="W",
                backbone_hidden=(32, 32, 16),
                head_hidden=(16,),
                mode="argmax",
                checkpoint_path=path,
                seed=0,
                device="cpu",
            )
            after = agent2.model.forward(state, phase_id=0, legal_mask=mask)
            self.assertTrue(torch.equal(before_logits, after["logits"]))
            self.assertTrue(torch.equal(before_value, after["value"]))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
