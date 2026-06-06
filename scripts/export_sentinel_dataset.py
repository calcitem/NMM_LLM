"""scripts/export_sentinel_dataset.py — persist a processed sentinel dataset.

Replays all games once and writes a compact ``.npz`` (default) or ``.jsonl``
file so training runs can reuse the processed set without re-replaying.

Usage:
    python scripts/export_sentinel_dataset.py --game-dir data/games \
        --output learned_ai/sentinel/processed.npz [--format npz|jsonl] \
        [--db-path ...] [--config configs/sentinel_default.yaml]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learned_ai.sentinel.config import load_config
from learned_ai.sentinel.dataset import SentinelDataset
from learned_ai.sentinel.db_teacher import ExternalSolvedDB


def main() -> int:
    p = argparse.ArgumentParser(description="Export processed sentinel dataset")
    p.add_argument("--game-dir", default="data/games")
    p.add_argument("--output", default="learned_ai/sentinel/processed.npz")
    p.add_argument("--format", choices=["npz", "jsonl"], default="npz")
    p.add_argument("--db-path", default="")
    p.add_argument("--config", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    config = load_config(args.config)
    db = ExternalSolvedDB(db_path=args.db_path or config.external_db_path,
                          enabled=bool(args.db_path) or config.external_db_enabled)

    ds = SentinelDataset.load_from_games(
        args.game_dir, db=db, config=config, limit=args.limit
    )
    print(f"Loaded {len(ds)} examples. Quality distribution: {ds.quality_distribution()}")
    print(f"Supervision sources: {ds.source_distribution()}")

    if args.format == "npz":
        ds.save_to_disk(args.output)
    else:
        with open(args.output, "w") as out:
            for ex in ds.examples:
                out.write(json.dumps({
                    "move_quality": ex.move_quality,
                    "training_weight": ex.training_weight,
                    "supervision_source": ex.supervision_source,
                    "ply": ex.ply,
                    "move_notation": ex.move_notation,
                    "features": ex.features.tolist(),
                }) + "\n")
    print(f"Wrote dataset to {args.output} ({args.format})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
