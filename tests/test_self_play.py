"""tests/test_self_play.py — 3-game self-play smoke test."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from learned_ai.agents.learned_agent import LearnedAgent
from learned_ai.training.self_play import assign_rewards, play_game


class TestSelfPlay(unittest.TestCase):
    def test_three_games_complete(self):
        torch.manual_seed(0)
        white = LearnedAgent(
            color="W", backbone_hidden=(32, 32, 16), head_hidden=(16,), seed=0
        )
        black = LearnedAgent(
            color="B", backbone_hidden=(32, 32, 16), head_hidden=(16,), seed=1
        )
        for _ in range(3):
            result = play_game(white, black, max_plies=200)
            self.assertGreater(result.plies, 0)
            self.assertTrue(result.trajectory, "trajectory should be non-empty")
            self.assertEqual(len(result.move_log), result.plies)
            # Winner is W / B / None (draw); never an unexpected value.
            self.assertIn(result.winner, ("W", "B", None))

    def test_reward_assignment(self):
        torch.manual_seed(0)
        white = LearnedAgent(
            color="W", backbone_hidden=(16, 16, 8), head_hidden=(8,), seed=0
        )
        black = LearnedAgent(
            color="B", backbone_hidden=(16, 16, 8), head_hidden=(8,), seed=1
        )
        result = play_game(white, black, max_plies=200)
        transitions = assign_rewards(result, win_reward=1.0, loss_reward=-1.0, draw_reward=0.0, gamma=1.0)
        self.assertEqual(len(transitions), len(result.trajectory))
        if result.winner is not None:
            for tr in transitions:
                if tr.side_to_move == result.winner:
                    self.assertAlmostEqual(tr.reward, 1.0)
                else:
                    self.assertAlmostEqual(tr.reward, -1.0)
        # Final transition flagged as done.
        if transitions:
            self.assertTrue(transitions[-1].done)


if __name__ == "__main__":
    unittest.main()
