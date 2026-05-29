"""tests/test_heuristic_vs_learned.py — 2 games of heuristic vs learned, no crash."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.agents.learned_agent import LearnedAgent
from learned_ai.training.self_play import play_game


class TestHeuristicVsLearned(unittest.TestCase):
    def test_two_games_no_crash(self):
        torch.manual_seed(0)
        for run in range(2):
            learned = LearnedAgent(
                color="W" if run == 0 else "B",
                backbone_hidden=(16, 16, 8),
                head_hidden=(8,),
                seed=run,
            )
            heuristic = HeuristicAgent(
                color="B" if run == 0 else "W", difficulty=1
            )
            if run == 0:
                result = play_game(learned, heuristic, max_plies=200)
            else:
                result = play_game(heuristic, learned, max_plies=200)
            self.assertGreater(result.plies, 0)
            self.assertIn(result.winner, ("W", "B", None))


if __name__ == "__main__":
    unittest.main()
