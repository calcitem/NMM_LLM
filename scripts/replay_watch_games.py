"""scripts/replay_watch_games.py — replay game logs + attach sentinel supervision.

Replays every game in a directory, builds per-ply decision context, attaches
solved-DB (or game-outcome proxy) supervision via the sentinel labelling layer,
and writes one labelled example per line to a JSONL file.

Usage:
    python scripts/replay_watch_games.py --game-dir data/games \
        [--db-path "/mnt/windows/NMM_DB/Entire DB"] \
        --output supervised_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learned_ai.sentinel.config import load_config
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.sentinel.dataset import _iter_game_records, examples_from_game


def main() -> int:
    p = argparse.ArgumentParser(description="Replay games + attach sentinel supervision")
    p.add_argument("--game-dir", default="data/games")
    p.add_argument("--db-path", default="")
    p.add_argument("--config", default=None)
    p.add_argument("--output", default="supervised_dataset.jsonl")
    p.add_argument("--limit", type=int, default=None, help="max number of game files")
    args = p.parse_args()

    config = load_config(args.config)
    db = ExternalSolvedDB(db_path=args.db_path or config.external_db_path,
                          enabled=bool(args.db_path) or config.external_db_enabled)
    print(f"External DB available: {db.is_available()}  ({db!r})")

    paths = sorted(
        os.path.join(args.game_dir, f)
        for f in os.listdir(args.game_dir)
        if f.endswith(".jsonl")
    )
    if args.limit is not None:
        paths = paths[: args.limit]

    n_examples = 0
    n_games = 0
    with open(args.output, "w") as out:
        for path in paths:
            for record in _iter_game_records(path):
                n_games += 1
                for ex in examples_from_game(
                    record, db=db, backward_decay=config.backward_decay
                ):
                    row = {
                        "label": ex.label,
                        "turning_point_confidence": ex.turning_point_confidence,
                        "value_delta": ex.value_delta,
                        "mistake_risk": ex.mistake_risk,
                        "opportunity_score": ex.opportunity_score,
                        "training_weight": ex.training_weight,
                        "supervision_source": ex.supervision_source,
                        "ply": ex.ply,
                        "features": ex.state_features.tolist(),
                    }
                    out.write(json.dumps(row) + "\n")
                    n_examples += 1

    print(f"Wrote {n_examples} labelled examples from {n_games} games to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
