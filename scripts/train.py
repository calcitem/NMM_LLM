"""scripts/train.py — main training entry point.

Usage:
    python scripts/train.py [--config path] [--resume checkpoint] [--stage N]
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learned_ai.training.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="learned_ai/config/default_config.yaml",
        help="Path to YAML config",
    )
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument(
        "--stage",
        type=int,
        default=None,
        help="If set, override curriculum start_stage (1..5)",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Override training.max_episodes from the config",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(cfg, resume_path=args.resume)

    if args.stage is not None:
        from learned_ai.training.curriculum import Curriculum

        trainer.curriculum = Curriculum.from_config(cfg.get("curriculum", {}), start_stage=args.stage)

    print(f"Loaded config {args.config}")
    print(f"Stage budgets: {trainer.curriculum.state.stage_budgets}")
    print(f"Starting at stage {trainer.curriculum.state.current_stage}")

    trainer.train(max_episodes=args.max_episodes)
    print("Training finished.")
    print(f"Final stats: {trainer.stats.__dict__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
