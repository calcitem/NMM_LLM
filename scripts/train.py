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
    p.add_argument("--resume", nargs="?", const="learned_ai/checkpoints/latest.pt",
                   default=None, help="Resume from checkpoint (default: latest.pt)")
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
    p.add_argument(
        "--level",
        default=None,
        help="Stage 3 sub-level to start at, e.g. b80 b60 b40 b20 d1 d3 d10 (implies --stage 3)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(cfg, resume_path=args.resume)

    # Resolve effective stage: --level implies stage 3.
    effective_stage = args.stage
    if args.level is not None and effective_stage is None:
        effective_stage = 3

    if effective_stage is not None:
        from learned_ai.training.curriculum import Curriculum

        trainer.curriculum = Curriculum.from_config(cfg.get("curriculum", {}), start_stage=effective_stage)

    if args.level is not None:
        cur = trainer.curriculum
        valid = []
        matched = False
        for idx, (diff, blunder) in enumerate(cur._levels):
            label = f"b{int(round(blunder * 100))}" if blunder > 0 else f"d{diff}"
            valid.append(label)
            if label == args.level:
                cur.state.heuristic_level_idx = idx
                matched = True
                break
        if not matched:
            print(f"Unknown level '{args.level}'. Valid stage 3 levels: {valid}", file=sys.stderr)
            return 1

    print(f"Config       : {args.config}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    if effective_stage is not None:
        print(f"Start stage  : {effective_stage} (overridden by --stage/--level)")
    if args.level is not None:
        print(f"Start level  : {args.level} (idx {trainer.curriculum.state.heuristic_level_idx})")
    print(f"Stage budgets: {trainer.curriculum.state.stage_budgets}")

    trainer.train(max_episodes=args.max_episodes, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
