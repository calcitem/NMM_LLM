"""scripts/run_self_play.py — generate and save self-play games.

Usage:
    python scripts/run_self_play.py --episodes 100 \
        [--checkpoint path] [--save-dir learned_ai/self_play_games]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learned_ai.agents.learned_agent import LearnedAgent
from learned_ai.data.game_logger import GameLogger
from learned_ai.training.self_play import play_game


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--save-dir", default="learned_ai/self_play_games")
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    logger = GameLogger(args.save_dir)
    print(f"Logging self-play to {logger.path}")

    for ep in range(args.episodes):
        white = LearnedAgent(
            color="W",
            checkpoint_path=args.checkpoint,
            mode="sample",
            temperature=args.temperature,
        )
        black = LearnedAgent(
            color="B",
            checkpoint_path=args.checkpoint,
            mode="sample",
            temperature=args.temperature,
        )
        result = play_game(white, black, max_plies=400)
        logger.log_game(
            winner=result.winner,
            moves=result.move_log,
            meta={
                "episode": ep,
                "plies": result.plies,
                "draw_reason": result.draw_reason,
            },
        )
        print(
            f"  ep {ep}: winner={result.winner} plies={result.plies}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
