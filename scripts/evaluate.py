"""scripts/evaluate.py — head-to-head match between any two agents.

Usage:
    python scripts/evaluate.py \
        --agent1 [heuristic|learned|random] \
        --agent2 [heuristic|learned|random] \
        --games 50 \
        [--agent1-checkpoint path] [--agent2-checkpoint path]
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.agents.learned_agent import LearnedAgent
from learned_ai.agents.random_agent import RandomAgent
from learned_ai.evaluation.evaluator import evaluate_match


def make_factory(kind: str, checkpoint: str | None, difficulty: int):
    def factory(color: str):
        if kind == "heuristic":
            return HeuristicAgent(color=color, difficulty=difficulty)
        if kind == "random":
            return RandomAgent(color=color, seed=random.randint(0, 1 << 31))
        if kind == "learned":
            return LearnedAgent(
                color=color, checkpoint_path=checkpoint, mode="argmax"
            )
        raise ValueError(f"unknown agent kind {kind}")

    return factory


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent1", required=True, choices=["heuristic", "learned", "random"])
    p.add_argument("--agent2", required=True, choices=["heuristic", "learned", "random"])
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--agent1-checkpoint", default=None)
    p.add_argument("--agent2-checkpoint", default=None)
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--output", default=None, help="optional JSON output path")
    args = p.parse_args()

    result = evaluate_match(
        agent1_factory=make_factory(args.agent1, args.agent1_checkpoint, args.difficulty),
        agent2_factory=make_factory(args.agent2, args.agent2_checkpoint, args.difficulty),
        games=args.games,
        agent1_name=args.agent1,
        agent2_name=args.agent2,
        output_json_path=args.output,
    )
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
