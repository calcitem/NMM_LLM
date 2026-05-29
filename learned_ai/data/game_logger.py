"""Append-only JSONL logger for self-play game trajectories."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class GameLogger:
    def __init__(self, log_dir: str, run_name: Optional[str] = None) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if run_name is None:
            run_name = time.strftime("selfplay-%Y%m%d-%H%M%S")
        self.run_name = run_name
        self.path = self.log_dir / f"{run_name}.jsonl"

    def log_game(
        self,
        winner: Optional[str],
        moves: List[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "winner": winner,
            "moves": moves,
            "meta": meta or {},
            "logged_at": time.time(),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def iter_games(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
