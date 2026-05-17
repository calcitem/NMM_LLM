"""ai/trajectory_db.py — Game trajectory memory for move guidance.

Indexes all saved game JSONL files by move-sequence prefix so the AI can
ask: "given the moves played so far, which next moves historically correlated
with a win for my colour?"

Covers the full game (placement + movement phases) using checkpoint depths
that grow from 4 to 48 half-moves.  Longer matches are preferred; the query
falls back to shorter prefixes when no deep match is found.

After each game the caller should invoke add_game() to keep the index
current without a full reload.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Checkpoint depths (half-moves from game start) used when building the index.
_DEPTHS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48)

_UNICODE_X = "×"   # ×


def _norm(notation: str) -> str:
    """Normalise notation: replace Unicode × with ASCII x."""
    return notation.replace(_UNICODE_X, "x")


class TrajectoryDB:
    """
    In-memory index of historical game trajectories.

    The index maps  prefix_string → {next_notation → outcome_counts}
    where prefix_string is the pipe-joined list of the first D normalised
    move notations in a game, and outcome_counts is
        {"W": int, "B": int, "D": int, "total": int}.

    query() returns a per-move score delta (positive = historically good for
    the colour about to move, negative = historically bad) so the engine can
    boost moves that have won before and avoid those that have lost.
    """

    def __init__(self, games_dir: Path | str) -> None:
        self._games_dir = Path(games_dir)
        self._index: dict[str, dict[str, dict]] = {}
        self._game_count = 0

    # ── Build / update ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Index every *.jsonl file in the games directory from scratch."""
        self._index.clear()
        self._game_count = 0
        if not self._games_dir.exists():
            logger.warning("TrajectoryDB: games directory not found: %s", self._games_dir)
            return
        for path in sorted(self._games_dir.glob("*.jsonl")):
            try:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    self._index_game(json.loads(text))
            except Exception as exc:
                logger.debug("TrajectoryDB: skipping %s — %s", path.name, exc)
        logger.info(
            "TrajectoryDB: indexed %d games → %d prefix entries.",
            self._game_count, len(self._index),
        )

    def _index_game(self, record: dict) -> None:
        winner = record.get("winner")       # "W", "B", or None/missing
        moves  = record.get("moves", [])
        if not moves:
            return

        notations = [_norm(m.get("notation", "")) for m in moves if m.get("notation")]
        if not notations:
            return

        self._game_count += 1

        for depth in _DEPTHS:
            if len(notations) <= depth:
                break
            prefix   = "|".join(notations[:depth])
            next_mv  = notations[depth]

            bucket = self._index.setdefault(prefix, {})
            entry  = bucket.setdefault(next_mv, {"W": 0, "B": 0, "D": 0, "total": 0})
            entry["total"] += 1
            if winner in ("W", "B"):
                entry[winner] += 1
            else:
                entry["D"] += 1

    def add_game(self, record: dict) -> None:
        """Incrementally add one completed game without a full reload."""
        self._index_game(record)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        move_notations: list[str],
        current_color: str,
        min_samples: int = 2,
    ) -> dict[str, float]:
        """
        Return a score-delta dict for candidate next-move notations.

        Positive delta  → this move historically correlates with `current_color`
                          winning (max +0.5 when 100 % win rate).
        Negative delta  → correlates with a loss (min -0.5).
        Returns {}      when no trajectory data are found for the current depth.

        Tries the longest matching prefix first and falls back to shorter ones.
        Normalises notation before lookup so ×/x variants both match.
        """
        normed = [_norm(n) for n in move_notations]

        for depth in reversed(_DEPTHS):
            if len(normed) < depth:
                continue
            prefix     = "|".join(normed[:depth])
            candidates = self._index.get(prefix)
            if not candidates:
                continue

            total_samples = sum(c["total"] for c in candidates.values())
            if total_samples < min_samples:
                continue

            result: dict[str, float] = {}
            for notation, stats in candidates.items():
                total = stats["total"]
                if total == 0:
                    continue
                wins  = stats.get(current_color, 0)
                draws = stats.get("D", 0)
                # Win rate (draws worth 0.4), centred on 0.0
                score = (wins + 0.4 * draws) / total
                result[notation] = score - 0.5
            return result

        return {}

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def game_count(self) -> int:
        return self._game_count

    @property
    def entry_count(self) -> int:
        return len(self._index)
