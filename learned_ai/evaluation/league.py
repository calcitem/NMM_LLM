"""Checkpoint-vs-checkpoint round-robin matches."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from learned_ai.agents.learned_agent import LearnedAgent
from learned_ai.evaluation.evaluator import EvalResult, evaluate_match


def _agent_factory(checkpoint_path: str):
    def factory(color: str) -> LearnedAgent:
        return LearnedAgent(
            color=color, checkpoint_path=checkpoint_path, mode="argmax"
        )

    return factory


def run_league(
    checkpoint_paths: List[str],
    games_per_pair: int = 20,
) -> Dict[Tuple[str, str], EvalResult]:
    """Pairwise matches between every (ckpt_a, ckpt_b) with a != b.

    Returns a dict keyed by (path_a, path_b) -> EvalResult.
    """
    results: Dict[Tuple[str, str], EvalResult] = {}
    for i, a in enumerate(checkpoint_paths):
        for b in checkpoint_paths[i + 1 :]:
            name_a = Path(a).stem
            name_b = Path(b).stem
            res = evaluate_match(
                agent1_factory=_agent_factory(a),
                agent2_factory=_agent_factory(b),
                games=games_per_pair,
                agent1_name=name_a,
                agent2_name=name_b,
            )
            results[(a, b)] = res
    return results
