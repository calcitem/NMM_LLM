"""scripts/benchmark_vs_heuristic.py — benchmark the learned AI vs the heuristic AI.

Usage:
    python scripts/benchmark_vs_heuristic.py [--checkpoint path] --games N
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.agents.learned_agent import LearnedAgent
from learned_ai.evaluation.evaluator import evaluate_match


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="Path to learned-AI checkpoint")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    def learned_factory(color: str) -> LearnedAgent:
        return LearnedAgent(color=color, checkpoint_path=args.checkpoint, mode="argmax")

    def heuristic_factory(color: str) -> HeuristicAgent:
        return HeuristicAgent(color=color, difficulty=args.difficulty)

    result = evaluate_match(
        agent1_factory=learned_factory,
        agent2_factory=heuristic_factory,
        games=args.games,
        agent1_name="learned",
        agent2_name=f"heuristic-d{args.difficulty}",
        output_json_path=args.output,
    )
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
