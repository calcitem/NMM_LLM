"""ai/trajectory_db.py — Game trajectory memory for move guidance.

Indexes all saved game JSONL files by move-sequence prefix so the AI can
ask: "given the moves played so far, which next moves historically correlated
with a win for my colour?"

Covers the full game (placement + movement phases) using checkpoint depths
that grow from 4 to 48 half-moves.  Longer matches are preferred; the query
falls back to shorter prefixes when no deep match is found.

D4 board symmetry is applied at indexing time: every prefix is stored in its
canonical (lex-min under D4) form so rotations and reflections of the same
game share statistics.

After each game the caller should invoke add_game() to keep the index
current without a full reload.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ai.board_symmetry import (
    canonical_sequence as _canonical_sequence,
    prefix_query_canonicals as _prefix_query_canonicals,
    transform_notation as _transform_notation,
    SYM_INVERSE as _SYM_INVERSE,
)

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

    The index maps  canon_prefix → {canon_next_notation → outcome_counts}
    where canon_prefix is the pipe-joined D4-canonical form of the first D
    normalised move notations, and outcome_counts is
        {"W": int, "B": int, "D": int, "total": int}.

    All 8 D4 symmetric equivalents of a game share statistics, multiplying
    effective sample size by up to 8×.

    query() returns a per-move score delta (positive = historically good for
    the colour about to move, negative = historically bad) so the engine can
    boost moves that have won before and avoid those that have lost.
    """

    def __init__(self, games_dir: Path | str) -> None:
        self._games_dir = Path(games_dir)
        self._index: dict[str, dict[str, dict]] = {}
        self._bans:  dict[str, set[str]] = {}   # actual prefix → set of banned notations
        self._game_count = 0

    # ── Build / update ────────────────────────────────────────────────────────

    def load(self, bad_moves_path: Path | str | None = None) -> None:
        """
        Index every *.jsonl file in the games directory from scratch.
        If bad_moves_path is provided, load persistent ban list from it.
        """
        self._index.clear()
        self._bans.clear()
        self._game_count = 0
        if not self._games_dir.exists():
            logger.warning("TrajectoryDB: games directory not found: %s", self._games_dir)
            return
        for path in sorted(self._games_dir.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._index_game(json.loads(line))
                except Exception as exc:
                    logger.debug("TrajectoryDB: skipping line in %s — %s", path.name, exc)
        logger.info(
            "TrajectoryDB: indexed %d games → %d prefix entries.",
            self._game_count, len(self._index),
        )
        if bad_moves_path is not None:
            self.load_bad_moves(bad_moves_path)

    def _index_game(self, record: dict) -> None:
        # Skip adaptive-softened games — blunder-inflated play pollutes the library.
        if record.get("adaptive_softened"):
            return
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

            prefix_notations = notations[:depth]
            next_mv_raw      = notations[depth]

            # Canonicalise the prefix; transform the next move by the same symmetry.
            canon_prefix_list, sym_idx = _canonical_sequence(prefix_notations)
            canon_next_mv = _transform_notation(next_mv_raw, sym_idx)
            if canon_next_mv is None:
                continue

            canon_prefix_key = "|".join(canon_prefix_list)
            bucket = self._index.setdefault(canon_prefix_key, {})
            entry  = bucket.setdefault(canon_next_mv, {"W": 0, "B": 0, "D": 0, "total": 0})
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
        All 8 D4 symmetric equivalents of each prefix are queried; results are
        inverse-transformed back to the actual game notation.
        Normalises notation before lookup so ×/x variants both match.
        """
        normed = [_norm(n) for n in move_notations]

        for depth in reversed(_DEPTHS):
            if len(normed) < depth:
                continue

            # Collect stats across all D4 equivalents of this query prefix.
            merged: dict[str, dict] = {}
            found_any = False

            for canon_prefix_key, sym_idx in _prefix_query_canonicals(normed, depth):
                candidates = self._index.get(canon_prefix_key)
                if not candidates:
                    continue
                found_any = True
                inv = _SYM_INVERSE[sym_idx]
                for canon_notation, stats in candidates.items():
                    actual_notation = _transform_notation(canon_notation, inv)
                    if actual_notation is None:
                        continue
                    if actual_notation not in merged:
                        merged[actual_notation] = {"W": 0, "B": 0, "D": 0, "total": 0}
                    entry = merged[actual_notation]
                    entry["total"] += stats["total"]
                    entry["W"]     += stats["W"]
                    entry["B"]     += stats["B"]
                    entry["D"]     += stats["D"]

            if not found_any:
                continue

            total_samples = sum(c["total"] for c in merged.values())
            if total_samples < min_samples:
                continue

            # Bans are stored in actual (non-canonical) notation so they apply
            # precisely to the specific sequence the user flagged.
            banned = self._bans.get("|".join(normed[:depth]), set())
            result: dict[str, float] = {}
            for notation, stats in merged.items():
                if _norm(notation) in banned:
                    result[notation] = -1.0   # hard-ban sentinel (outside statistical range)
                    continue
                total = stats["total"]
                if total == 0:
                    continue
                wins  = stats.get(current_color, 0)
                draws = stats.get("D", 0)
                score = (wins + 0.4 * draws) / total
                result[notation] = score - 0.5
            for bad_n in banned:
                if bad_n not in result:
                    result[bad_n] = -1.0
            return result

        # Even with no statistical match, surface bans at the shortest applicable depth.
        for depth in _DEPTHS:
            if len(normed) < depth:
                break
            prefix = "|".join(normed[:depth])
            banned = self._bans.get(prefix, set())
            if banned:
                return {n: -1.0 for n in banned}

        return {}

    def query_opponent_loss(
        self,
        move_notations: list[str],
        opponent_color: str,
        min_samples: int = 2,
    ) -> dict[str, float]:
        """Score candidate moves by how often the opponent loses from this position.

        Positive delta  → opponent historically loses frequently after this next move
                          (max +0.5 when opponent always loses).
        Negative delta  → opponent wins frequently; avoid this line.
        Returns {}      when no trajectory data match.

        Complements query(): where query() rewards moves that correlate with
        our wins, this method rewards moves that correlate with their losses —
        a subtly different signal when the database has many drawn games.
        """
        normed = [_norm(n) for n in move_notations]

        for depth in reversed(_DEPTHS):
            if len(normed) < depth:
                continue

            merged: dict[str, dict] = {}
            found_any = False

            for canon_prefix_key, sym_idx in _prefix_query_canonicals(normed, depth):
                candidates = self._index.get(canon_prefix_key)
                if not candidates:
                    continue
                found_any = True
                inv = _SYM_INVERSE[sym_idx]
                for canon_notation, stats in candidates.items():
                    actual_notation = _transform_notation(canon_notation, inv)
                    if actual_notation is None:
                        continue
                    if actual_notation not in merged:
                        merged[actual_notation] = {"W": 0, "B": 0, "D": 0, "total": 0}
                    entry = merged[actual_notation]
                    entry["total"] += stats["total"]
                    entry["W"]     += stats["W"]
                    entry["B"]     += stats["B"]
                    entry["D"]     += stats["D"]

            if not found_any:
                continue

            total_samples = sum(c["total"] for c in merged.values())
            if total_samples < min_samples:
                continue

            result: dict[str, float] = {}
            for notation, stats in merged.items():
                total = stats["total"]
                if total == 0:
                    continue
                opp_losses = stats.get(
                    "W" if opponent_color == "B" else "B", 0
                )
                loss_rate = opp_losses / total
                result[notation] = loss_rate - 0.5
            return result

        return {}

    # ── Bad move bans ─────────────────────────────────────────────────────────

    def mark_bad_move(self, prior_notations: list[str], bad_notation: str) -> None:
        """
        Permanently penalise `bad_notation` from the position described by
        `prior_notations`.  query() will return -1.0 (hard-ban) for that
        move at any prefix depth that matches.

        Bans are stored in actual notation (not canonical) so they apply
        precisely to the specific game situation the user flagged.
        """
        normed_prior = [_norm(n) for n in prior_notations]
        normed_bad   = _norm(bad_notation)
        for depth in _DEPTHS:
            if len(normed_prior) < depth:
                break
            prefix = "|".join(normed_prior[:depth])
            self._bans.setdefault(prefix, set()).add(normed_bad)

    def load_bad_moves(self, path: Path | str) -> None:
        """Load a persistent bad-moves JSON file and apply all bans."""
        path = Path(path)
        if not path.exists():
            return
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
            for e in entries:
                self.mark_bad_move(e.get("prior_notations", []), e.get("bad_notation", ""))
            logger.info("TrajectoryDB: loaded %d bad-move bans from %s", len(entries), path.name)
        except Exception as exc:
            logger.warning("TrajectoryDB: could not load bad moves from %s — %s", path, exc)

    def save_bad_move(
        self, path: Path | str, prior_notations: list[str], bad_notation: str
    ) -> None:
        """Append one bad-move entry to the persistent JSON file and apply the ban."""
        path = Path(path)
        existing: list = []
        try:
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        existing.append({"prior_notations": list(prior_notations), "bad_notation": bad_notation})
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        self.mark_bad_move(prior_notations, bad_notation)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def game_count(self) -> int:
        return self._game_count

    @property
    def entry_count(self) -> int:
        return len(self._index)
