"""
ai/opening_book.py — Opening book management for Nine Men's Morris.

Manages three JSON files:
  data/openings/book_openings.json    — read-only canonical book (shipped with project)
  data/openings/openings.json         — mutable copy of book openings with updated stats
  data/openings/learned_openings.json — novel/learned openings discovered at runtime

book_openings.json is NEVER written.  openings.json is seeded from the book on
first use and tracks per-opening outcome stats for the 11 canonical lines.
Learned openings (novel game sequences, self-play discoveries) live exclusively
in learned_openings.json and can be pruned without touching book data.

On first run after the split is introduced, any learned entries found inside
openings.json are migrated to learned_openings.json automatically (a backup of
openings.json is created first).
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_AUTO_NAME_PREFIXES = (
    "Novel Opening novel-",
    "Novel Opening (",
    "Novel Opening",
    "Self-Play Line",
)


def is_auto_named(name: str) -> bool:
    """Return True if `name` is a machine-generated placeholder (not an LLM name)."""
    if not name or not name.strip():
        return True
    for prefix in _AUTO_NAME_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BranchMove:
    branch_id: str
    deviation_ply: int
    deviation_move: str
    name: str
    line_continuation: list[str]
    strategic_notes: str
    seed_source: str        # "book" | "human" | "learned"
    outcome_stats: dict     # {"W": int, "B": int, "D": int}


@dataclass
class Opening:
    opening_id: str
    name: str
    aliases: list[str]
    family: str
    side: str               # "W", "B", or "both"
    seed_source: str        # "book" | "human" | "learned"
    line_moves: list[str]   # alternating W/B placement notation, e.g. ["d2","d6","f4","b4"]
    branch_moves: list[BranchMove]
    opening_fen_signatures: list[dict]  # [{"ply": int, "fen": str}]
    strategic_notes: str
    common_blunders: list[str]
    recommended_responses: dict         # {"W": [...], "B": [...]}
    outcome_stats: dict                 # {"W":int,"B":int,"D":int, plus human_*/ai_* breakdown}
    confidence: float
    tags: list[str]
    source_reference: str = ""
    needs_llm_name: bool = False        # True when auto-named without LLM; candidate for naming

    def opening_score(self, ai_color: str = "W", penalty: float = 0.0) -> float:
        """
        Rate this opening from 0.0 (bad for ai_color) to 1.0 (excellent).
        Draws count 0.4.  Unexplored openings return 0.55 so the AI is
        mildly curious about trying them before penalising or boosting them.
        An optional penalty (0.0–0.3) reduces the score for recent poor performance.
        """
        stats = self.outcome_stats
        w = stats.get("W", 0)
        b = stats.get("B", 0)
        d = stats.get("D", 0)
        total = w + b + d
        if total == 0:
            return max(0.0, 0.55 - penalty)
        wins = w if ai_color == "W" else b
        raw = (wins + 0.4 * d) / total
        return max(0.0, raw - penalty)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dict_to_branch(d: dict) -> BranchMove:
    """Deserialise a dict into a BranchMove, tolerating missing fields."""
    return BranchMove(
        branch_id=d.get("branch_id", ""),
        deviation_ply=d.get("deviation_ply", 0),
        deviation_move=d.get("deviation_move", ""),
        name=d.get("name", ""),
        line_continuation=d.get("line_continuation", []),
        strategic_notes=d.get("strategic_notes", ""),
        seed_source=d.get("seed_source", "book"),
        outcome_stats=d.get("outcome_stats", {"W": 0, "B": 0, "D": 0}),
    )


def _dict_to_opening(d: dict) -> Opening:
    """Deserialise a dict (from JSON) into an Opening, including nested BranchMoves."""
    branch_moves = [
        _dict_to_branch(b) if isinstance(b, dict) else b
        for b in d.get("branch_moves", [])
    ]
    return Opening(
        opening_id=d.get("opening_id", ""),
        name=d.get("name", ""),
        aliases=d.get("aliases", []),
        family=d.get("family", ""),
        side=d.get("side", "both"),
        seed_source=d.get("seed_source", "book"),
        line_moves=d.get("line_moves", []),
        branch_moves=branch_moves,
        opening_fen_signatures=d.get("opening_fen_signatures", []),
        strategic_notes=d.get("strategic_notes", ""),
        common_blunders=d.get("common_blunders", []),
        recommended_responses=d.get("recommended_responses", {"W": [], "B": []}),
        outcome_stats=d.get("outcome_stats", {"W": 0, "B": 0, "D": 0}),
        confidence=d.get("confidence", 1.0),
        tags=d.get("tags", []),
        source_reference=d.get("source_reference", ""),
        needs_llm_name=d.get("needs_llm_name", False),
    )


def _opening_to_dict(o: Opening) -> dict:
    """Serialise an Opening (and its BranchMoves) to a JSON-serialisable dict."""
    return asdict(o)


def _load_json_list(path: Path, label: str) -> list[dict]:
    """Load a JSON array from path; return [] on any error."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            logger.warning("%s is not a JSON array; treating as empty.", label)
            return []
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", label, exc)
        return []


# ── OpeningBook ───────────────────────────────────────────────────────────────

class OpeningBook:
    """
    Load, query, and persist the Nine Men's Morris opening book.

    book_openings.json is NEVER written; it is the canonical source.
    openings.json tracks outcome stats for the 11 book lines.
    learned_openings.json stores all novel/self-play discoveries; these
    can be pruned individually without affecting the book.
    """

    def __init__(
        self,
        book_path: str = "data/openings/book_openings.json",
        openings_path: str = "data/openings/openings.json",
        learned_path: str = "data/openings/learned_openings.json",
        penalties_path: str = "data/openings/penalties.json",
    ) -> None:
        self._book_path = Path(book_path)
        self._openings_path = Path(openings_path)
        self._learned_path = Path(learned_path)
        self._penalties_path = Path(penalties_path)
        self._index: dict[str, Opening] = {}   # opening_id -> Opening
        self._book_ids: set[str] = set()       # IDs from the canonical book file
        self._learned_ids: set[str] = set()    # IDs in learned_openings.json
        # opening_id -> {"penalty": float, "last_updated": str}
        self._penalties: dict[str, dict] = {}
        self.load()

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        1. Read book_openings.json (read-only).
        2. Migrate: if openings.json contains learned entries and
           learned_openings.json doesn't exist yet, split them out.
        3. Seed openings.json from the book if it doesn't exist.
        4. Read openings.json (book entries with updated stats).
        5. Seed learned_openings.json as [] if it doesn't exist.
        6. Read learned_openings.json into the index.
        """
        # Step 1: canonical book
        for raw in _load_json_list(self._book_path, "book_openings.json"):
            try:
                opening = _dict_to_opening(raw)
                self._index[opening.opening_id] = opening
                self._book_ids.add(opening.opening_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed book entry: %s", exc)

        if not self._book_ids:
            logger.warning(
                "book_openings.json not found or empty at %s.", self._book_path
            )

        # Step 2: migration — split learned out of openings.json if needed
        if self._openings_path.exists() and not self._learned_path.exists():
            self._migrate_split_learned()

        # Step 3: seed openings.json from book if missing
        if not self._openings_path.exists():
            self._openings_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_openings_json()
            logger.info(
                "openings.json did not exist; seeded from book (%d entries).",
                len(self._index),
            )

        # Step 4: read openings.json (book entries + their updated stats)
        for raw in _load_json_list(self._openings_path, "openings.json"):
            try:
                opening = _dict_to_opening(raw)
                self._index[opening.opening_id] = opening
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed opening entry: %s", exc)

        # Step 5: seed learned_openings.json as empty if missing
        if not self._learned_path.exists():
            self._learned_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_learned_json()
            logger.info("learned_openings.json created (empty).")

        # Step 6: read learned_openings.json
        for raw in _load_json_list(self._learned_path, "learned_openings.json"):
            try:
                opening = _dict_to_opening(raw)
                self._index[opening.opening_id] = opening
                self._learned_ids.add(opening.opening_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed learned entry: %s", exc)

        # Step 7: load penalties
        self._penalties = self._load_penalties()

        logger.info(
            "OpeningBook loaded: %d opening(s) (%d book, %d learned).",
            len(self._index),
            len(self._book_ids),
            len(self._learned_ids),
        )

    # ── Migration ─────────────────────────────────────────────────────────────

    def _migrate_split_learned(self) -> None:
        """
        One-time migration: split learned entries out of openings.json into
        learned_openings.json.  A backup of openings.json is written first.
        """
        data = _load_json_list(self._openings_path, "openings.json (migration)")
        if not data:
            return

        book_entries = [e for e in data if e.get("seed_source") == "book"]
        learned_entries = [e for e in data if e.get("seed_source") != "book"]

        if not learned_entries:
            return  # nothing to split

        # Back up openings.json
        backup = self._openings_path.with_suffix(".json.pre-split-backup")
        try:
            shutil.copy2(self._openings_path, backup)
            logger.info("Migration: backed up openings.json → %s", backup.name)
        except OSError as exc:
            logger.warning("Migration: could not create backup: %s", exc)

        # Write learned_openings.json
        try:
            self._learned_path.parent.mkdir(parents=True, exist_ok=True)
            with self._learned_path.open("w", encoding="utf-8") as fh:
                json.dump(learned_entries, fh, indent=2, ensure_ascii=False)
            logger.info(
                "Migration: wrote %d learned opening(s) → learned_openings.json",
                len(learned_entries),
            )
        except OSError as exc:
            logger.error("Migration: failed to write learned_openings.json: %s", exc)
            return  # abort; openings.json untouched

        # Rewrite openings.json with only book entries
        try:
            with self._openings_path.open("w", encoding="utf-8") as fh:
                json.dump(book_entries, fh, indent=2, ensure_ascii=False)
            logger.info(
                "Migration: rewrote openings.json with %d book entry/entries.",
                len(book_entries),
            )
        except OSError as exc:
            logger.error("Migration: failed to rewrite openings.json: %s", exc)

    # ── Penalty helpers ───────────────────────────────────────────────────────

    def _load_penalties(self) -> dict[str, dict]:
        if not self._penalties_path.exists():
            return {}
        try:
            with self._penalties_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read penalties.json: %s", exc)
            return {}

    def _write_penalties(self) -> None:
        self._penalties_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._penalties_path.open("w", encoding="utf-8") as fh:
                json.dump(self._penalties, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("Failed to write penalties.json: %s", exc)

    def _apply_outcome_penalty(
        self, opening_id: str, winner: str, human_color: Optional[str]
    ) -> None:
        """Update the decay penalty for an opening after a game outcome.

        Only applies when human_color is known (so we know which side the AI was).
        AI loss → +0.05 penalty (capped at 0.3).  AI win → −0.02 (floor 0.0).
        """
        if human_color not in ("W", "B"):
            return
        from datetime import datetime, timezone
        ai_color = "B" if human_color == "W" else "W"
        entry = self._penalties.setdefault(
            opening_id, {"penalty": 0.0, "last_updated": ""}
        )
        if winner == ai_color:
            entry["penalty"] = max(0.0, entry["penalty"] - 0.02)
        elif winner in ("W", "B"):
            entry["penalty"] = min(0.3, entry["penalty"] + 0.05)
        entry["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._write_penalties()

    def get_penalty(self, opening_id: str) -> float:
        """Return the current decay penalty for an opening (0.0 if none recorded)."""
        return self._penalties.get(opening_id, {}).get("penalty", 0.0)

    def get_adjusted_score(self, opening_id: str, ai_color: str = "W") -> float:
        """Return opening_score with penalty applied — use this for display."""
        opening = self._index.get(opening_id)
        if opening is None:
            return 0.0
        return opening.opening_score(ai_color, penalty=self.get_penalty(opening_id))

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_by_id(self, opening_id: str) -> Optional[Opening]:
        """Return the Opening for the given ID, or None if not found."""
        return self._index.get(opening_id)

    def get_by_name(self, name: str) -> list[Opening]:
        """Case-insensitive partial match on the name field."""
        needle = name.casefold()
        return [o for o in self._index.values() if needle in o.name.casefold()]

    def get_by_family(self, family: str) -> list[Opening]:
        """Exact match on the family field."""
        return [o for o in self._index.values() if o.family == family]

    def get_by_tag(self, tag: str) -> list[Opening]:
        """Return openings whose tags list contains the given tag."""
        return [o for o in self._index.values() if tag in o.tags]

    def get_by_seed_source(self, source: str) -> list[Opening]:
        """Exact match on seed_source."""
        return [o for o in self._index.values() if o.seed_source == source]

    def get_unnamed_openings(self) -> list[Opening]:
        """Return all openings flagged needs_llm_name=True (queued for LLM naming)."""
        return [o for o in self._index.values() if o.needs_llm_name]

    def is_prunable(self, opening_id: str) -> bool:
        """Return True if this opening can be pruned (i.e. it lives in learned_openings.json)."""
        return opening_id in self._learned_ids

    def values(self):
        """Iterate over all Opening objects in the index."""
        return self._index.values()

    # ── Mutation ──────────────────────────────────────────────────────────────

    def save_opening(self, opening: Opening) -> None:
        """
        Write/update an opening in the appropriate file.

        Learned openings (seed_source != "book" or already in _learned_ids) go
        to learned_openings.json.  Book openings go to openings.json.

        Raises ValueError if seed_source == "book" and the ID is brand-new
        (prevents accidentally minting new canonical book entries at runtime).
        """
        if (
            opening.seed_source == "book"
            and opening.opening_id not in self._index
        ):
            raise ValueError(
                f"Cannot add new opening with seed_source='book' at runtime "
                f"(id={opening.opening_id!r}).  Only 'human' or 'learned' "
                f"openings may be created dynamically."
            )

        is_learned = (
            opening.seed_source != "book"
            or opening.opening_id in self._learned_ids
        )
        if is_learned:
            self._learned_ids.add(opening.opening_id)

        self._index[opening.opening_id] = opening

        if is_learned:
            self._write_learned_json()
        else:
            self._write_openings_json()

    def update_outcome_stats(
        self,
        opening_id: str,
        winner: str,
        human_color: Optional[str] = None,
    ) -> None:
        """
        Record the outcome of a game that used this opening.

        winner must be "W", "B", or "D".
        human_color, if supplied, splits the tally into human_wins/losses/draws
        and ai_wins/losses/draws so the AI can distinguish its own performance.
        """
        opening = self._index.get(opening_id)
        if opening is None:
            logger.warning(
                "update_outcome_stats: opening_id %r not found.", opening_id
            )
            return

        if winner not in ("W", "B", "D"):
            logger.warning(
                "update_outcome_stats: invalid winner %r (must be W/B/D).", winner
            )
            return

        stats = opening.outcome_stats
        stats[winner] = stats.get(winner, 0) + 1

        if human_color in ("W", "B"):
            if winner == "D":
                stats["human_draws"] = stats.get("human_draws", 0) + 1
                stats["ai_draws"] = stats.get("ai_draws", 0) + 1
            elif winner == human_color:
                stats["human_wins"] = stats.get("human_wins", 0) + 1
                stats["ai_losses"] = stats.get("ai_losses", 0) + 1
            else:
                stats["ai_wins"] = stats.get("ai_wins", 0) + 1
                stats["human_losses"] = stats.get("human_losses", 0) + 1

        if opening_id in self._learned_ids:
            self._write_learned_json()
        else:
            self._write_openings_json()

        self._apply_outcome_penalty(opening_id, winner, human_color)

    def prune_opening(self, opening_id: str) -> bool:
        """
        Remove a learned opening permanently.

        Returns True if the opening was removed, False if not found or if
        the opening is a protected book entry.
        """
        if opening_id not in self._learned_ids:
            logger.warning(
                "prune_opening: %r is not a prunable learned opening.", opening_id
            )
            return False
        if opening_id not in self._index:
            return False
        del self._index[opening_id]
        self._learned_ids.discard(opening_id)
        if opening_id in self._penalties:
            del self._penalties[opening_id]
            self._write_penalties()
        self._write_learned_json()
        return True

    def set_name(
        self,
        opening_id: str,
        name: str,
        needs_llm_name: bool = False,
    ) -> bool:
        """
        Set the display name on any opening (book or learned).

        Returns True if the opening was found and updated, False otherwise.
        """
        opening = self._index.get(opening_id)
        if opening is None:
            return False
        opening.name = name
        opening.needs_llm_name = needs_llm_name
        if opening_id in self._learned_ids:
            self._write_learned_json()
        else:
            self._write_openings_json()
        return True

    def select_opening(
        self,
        ai_color: str = "W",
        exploration_rate: float = 0.25,
        temperature: float = 0.18,
    ) -> Optional["Opening"]:
        """
        Pick an opening for the AI to target at game start using UCB1 with
        temperature-weighted random sampling so different openings are tried
        each game rather than the same UCB-max winner every time.

        temperature controls variety: lower → more deterministic (best always
        picked); higher → more random.  0.18 gives good first-move variety
        while still strongly preferring well-scored openings.
        """
        import math
        import random

        openings = [
            o for o in self._index.values()
            if o.side in (ai_color, "both")
        ]
        if not openings:
            return None

        global_games = sum(
            o.outcome_stats.get("W", 0)
            + o.outcome_stats.get("B", 0)
            + o.outcome_stats.get("D", 0)
            for o in openings
        )
        log_n = math.log(max(1, global_games) + 1)

        def _ucb(op: "Opening") -> float:
            penalty = self._penalties.get(op.opening_id, {}).get("penalty", 0.0)
            base = op.opening_score(ai_color, penalty=penalty)
            local = (
                op.outcome_stats.get("W", 0)
                + op.outcome_stats.get("B", 0)
                + op.outcome_stats.get("D", 0)
            )
            return base + exploration_rate * math.sqrt(log_n / (local + 1))

        scores = [_ucb(op) for op in openings]
        max_score = max(scores)
        weights = [math.exp((s - max_score) / temperature) for s in scores]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for op, w in zip(openings, weights):
            cumulative += w
            if r <= cumulative:
                return op
        return openings[-1]

    def record_deviation(
        self, opening_id: str, ply: int, move_played: str, board_fen: str
    ) -> Optional[BranchMove]:
        """
        Record that the game deviated from opening `opening_id` at `ply`
        by playing `move_played`.

        - Returns the existing BranchMove if one already covers this deviation.
        - Creates, saves, and returns a new BranchMove with seed_source="learned"
          if no matching branch exists.
        - Returns None if the opening_id is not in the index.
        """
        opening = self._index.get(opening_id)
        if opening is None:
            return None

        for branch in opening.branch_moves:
            if (
                branch.deviation_ply == ply
                and branch.deviation_move == move_played
            ):
                return branch

        branch_id = f"{opening_id}-dev-{ply}-{move_played}"
        new_branch = BranchMove(
            branch_id=branch_id,
            deviation_ply=ply,
            deviation_move=move_played,
            name=f"{opening.name} — Deviation at ply {ply}",
            line_continuation=[],
            strategic_notes="",
            seed_source="learned",
            outcome_stats={"W": 0, "B": 0, "D": 0},
        )
        opening.branch_moves.append(new_branch)
        self.save_opening(opening)
        return new_branch

    def save_novel_opening(
        self,
        move_sequence: list[str],
        board_fen_signatures: list[dict],
        outcome: Optional[str] = None,
        needs_llm_name: bool = False,
    ) -> Opening:
        """
        Create, persist, and return a new 'learned' Opening from an observed
        move sequence that didn't match any known opening.

        outcome, if provided, must be "W", "B", or "D".
        needs_llm_name=True marks this opening as pending LLM naming.
        """
        outcome_stats: dict[str, int] = {"W": 0, "B": 0, "D": 0}
        if outcome in outcome_stats:
            outcome_stats[outcome] = 1

        opening_id = f"novel-{uuid.uuid4().hex[:8]}"
        opening = Opening(
            opening_id=opening_id,
            name=f"Novel Opening {opening_id}",
            aliases=[],
            family="novel",
            side="both",
            seed_source="learned",
            line_moves=list(move_sequence),
            branch_moves=[],
            opening_fen_signatures=list(board_fen_signatures),
            strategic_notes="",
            common_blunders=[],
            recommended_responses={"W": [], "B": []},
            outcome_stats=outcome_stats,
            confidence=0.3,
            tags=["novel", "learned"],
            source_reference="",
            needs_llm_name=needs_llm_name,
        )
        self.save_opening(opening)
        return opening

    # ── Deduplication ─────────────────────────────────────────────────────────

    def find_similar(
        self,
        move_sequence: list[str],
        min_common: int = 4,
    ) -> list["Opening"]:
        """Return openings whose first `min_common` moves match `move_sequence`."""
        if len(move_sequence) < min_common:
            return []
        prefix = tuple(move_sequence[:min_common])
        return [
            o for o in self._index.values()
            if len(o.line_moves) >= min_common
            and tuple(o.line_moves[:min_common]) == prefix
        ]

    def merge_duplicates(self, min_common: int = 4) -> int:
        """Merge auto-named openings that share the same first `min_common` moves.

        Rules:
        - Named openings (LLM or book names) are NEVER merged with each other.
        - Auto-named openings sharing a prefix with a named opening are merged INTO
          the named opening that has the most recorded games.
        - Auto-named openings sharing a prefix with only other auto-named openings
          are merged into the one with the most recorded games; the result keeps
          needs_llm_name=True so it gets named in the next naming pass.

        Returns the number of entries removed.
        """
        from collections import defaultdict

        _stat_keys = ("W", "B", "D",
                      "human_wins", "human_losses", "human_draws",
                      "ai_wins", "ai_losses", "ai_draws")

        groups: dict[tuple, list[Opening]] = defaultdict(list)
        for o in self._index.values():
            if len(o.line_moves) >= min_common:
                key = tuple(o.line_moves[:min_common])
                groups[key].append(o)

        removed = 0
        for group in groups.values():
            if len(group) <= 1:
                continue

            named   = [o for o in group if not is_auto_named(o.name)]
            unnamed = [o for o in group if is_auto_named(o.name)]

            if not unnamed:
                continue

            def _by_games(o: Opening) -> int:
                return sum(o.outcome_stats.get(k, 0) for k in ("W", "B", "D"))

            if named:
                canonical = max(named, key=_by_games)
                to_merge  = unnamed
            else:
                unnamed.sort(key=_by_games, reverse=True)
                canonical = unnamed[0]
                canonical.needs_llm_name = True
                to_merge  = unnamed[1:]

            for dup in to_merge:
                for k in _stat_keys:
                    canonical.outcome_stats[k] = (
                        canonical.outcome_stats.get(k, 0)
                        + dup.outcome_stats.get(k, 0)
                    )
                del self._index[dup.opening_id]
                self._learned_ids.discard(dup.opening_id)
                self._penalties.pop(dup.opening_id, None)
                removed += 1

            self._index[canonical.opening_id] = canonical

        if removed:
            self._write_openings_json()
            self._write_learned_json()
            self._write_penalties()
        return removed

    # ── Persistence ───────────────────────────────────────────────────────────

    def _write_openings_json(self) -> None:
        """Serialise non-learned openings (book entries with updated stats) to openings.json."""
        self._openings_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            _opening_to_dict(o) for o in self._index.values()
            if o.opening_id not in self._learned_ids
        ]
        try:
            with self._openings_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("Failed to write openings.json: %s", exc)
            raise

    def _write_learned_json(self) -> None:
        """Serialise learned openings to learned_openings.json."""
        self._learned_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            _opening_to_dict(o) for o in self._index.values()
            if o.opening_id in self._learned_ids
        ]
        try:
            with self._learned_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("Failed to write learned_openings.json: %s", exc)
            raise
