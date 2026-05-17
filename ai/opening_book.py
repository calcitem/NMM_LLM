"""
ai/opening_book.py — Opening book management for Nine Men's Morris.

Manages two JSON files:
  data/openings/book_openings.json  — read-only canonical book (shipped with project)
  data/openings/openings.json       — mutable working copy (seeded from book on first run)

The _index is keyed by opening_id and always reflects the merged state, with
openings.json taking precedence over book_openings.json for duplicate IDs.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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

    def opening_score(self, ai_color: str = "W") -> float:
        """
        Rate this opening from 0.0 (bad for ai_color) to 1.0 (excellent).
        Draws count 0.4.  Unexplored openings return 0.55 so the AI is
        mildly curious about trying them before penalising or boosting them.
        """
        stats = self.outcome_stats
        w = stats.get("W", 0)
        b = stats.get("B", 0)
        d = stats.get("D", 0)
        total = w + b + d
        if total == 0:
            return 0.55  # unexplored — slight curiosity bonus
        wins = w if ai_color == "W" else b
        return (wins + 0.4 * d) / total


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
    d = asdict(o)
    return d


# ── OpeningBook ───────────────────────────────────────────────────────────────

class OpeningBook:
    """
    Load, query, and persist the Nine Men's Morris opening book.

    book_openings.json is NEVER written; it is the canonical source.
    openings.json is the mutable working copy seeded from book_openings.json
    on first use.  All runtime mutations are applied to openings.json only.
    """

    def __init__(
        self,
        book_path: str = "data/openings/book_openings.json",
        openings_path: str = "data/openings/openings.json",
    ) -> None:
        self._book_path = Path(book_path)
        self._openings_path = Path(openings_path)
        self._index: dict[str, Opening] = {}   # opening_id -> Opening
        # Track which IDs originated exclusively from the read-only book file
        # (before openings.json may have overridden them).
        self._book_ids: set[str] = set()
        self.load()

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        1. Read book_openings.json (read-only).  Warn if missing.
        2. Seed openings.json from the book if it doesn't exist yet.
        3. Read openings.json and merge into _index (takes precedence).
        """
        book_data: list[dict] = []

        # Step 1: read book
        if self._book_path.exists():
            try:
                with self._book_path.open("r", encoding="utf-8") as fh:
                    book_data = json.load(fh)
                if not isinstance(book_data, list):
                    logger.warning(
                        "book_openings.json is not a JSON array; skipping book load."
                    )
                    book_data = []
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read book_openings.json: %s", exc)
                book_data = []
        else:
            logger.warning(
                "book_openings.json not found at %s; continuing without book data.",
                self._book_path,
            )

        # Populate _index with book entries first
        for raw in book_data:
            try:
                opening = _dict_to_opening(raw)
                self._index[opening.opening_id] = opening
                self._book_ids.add(opening.opening_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed book entry: %s", exc)

        # Step 2: seed openings.json if it doesn't exist
        if not self._openings_path.exists():
            self._openings_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_openings_json()
            logger.info(
                "openings.json did not exist; seeded from book (%d entries).",
                len(self._index),
            )

        # Step 3: read openings.json and merge (overrides book entries for same ID)
        try:
            with self._openings_path.open("r", encoding="utf-8") as fh:
                openings_data: list[dict] = json.load(fh)
            if not isinstance(openings_data, list):
                logger.warning(
                    "openings.json is not a JSON array; skipping openings load."
                )
                openings_data = []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read openings.json: %s", exc)
            openings_data = []

        for raw in openings_data:
            try:
                opening = _dict_to_opening(raw)
                self._index[opening.opening_id] = opening
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed opening entry: %s", exc)

        logger.info("OpeningBook loaded: %d opening(s) in index.", len(self._index))

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

    def values(self):
        """Iterate over all Opening objects in the index."""
        return self._index.values()

    # ── Mutation ──────────────────────────────────────────────────────────────

    def save_opening(self, opening: Opening) -> None:
        """
        Write/update an opening in openings.json (never touches book_openings.json).

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

        self._index[opening.opening_id] = opening
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
            ai_color = "B" if human_color == "W" else "W"
            if winner == "D":
                stats["human_draws"] = stats.get("human_draws", 0) + 1
                stats["ai_draws"] = stats.get("ai_draws", 0) + 1
            elif winner == human_color:
                stats["human_wins"] = stats.get("human_wins", 0) + 1
                stats["ai_losses"] = stats.get("ai_losses", 0) + 1
            else:
                stats["ai_wins"] = stats.get("ai_wins", 0) + 1
                stats["human_losses"] = stats.get("human_losses", 0) + 1

        self._write_openings_json()

    def select_opening(
        self,
        ai_color: str = "W",
        exploration_rate: float = 0.25,
    ) -> Optional["Opening"]:
        """
        Pick an opening for the AI to target at game start using UCB1.

        UCB1 naturally balances exploitation of known-good openings with
        exploration of under-tried ones.  exploration_rate scales the
        exploration term; higher values favour variety over winning percentage.
        """
        import math

        # Only consider openings where this AI colour plays the winning side.
        # side='both' means the line is colour-neutral (e.g. draws, or unknown outcome).
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
            base = op.opening_score(ai_color)
            local = (
                op.outcome_stats.get("W", 0)
                + op.outcome_stats.get("B", 0)
                + op.outcome_stats.get("D", 0)
            )
            # UCB exploration term — large for untried openings
            return base + exploration_rate * math.sqrt(log_n / (local + 1))

        return max(openings, key=_ucb)

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

        # Look for an existing branch that covers this exact deviation
        for branch in opening.branch_moves:
            if (
                branch.deviation_ply == ply
                and branch.deviation_move == move_played
            ):
                return branch

        # Create a new learned branch
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
        needs_llm_name=True marks this opening as pending LLM naming (use when
        no LLM is available at game time — run tools/name_openings.py later).
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

    # ── Persistence ───────────────────────────────────────────────────────────

    def _write_openings_json(self) -> None:
        """
        Serialise the entire _index to openings.json.

        All openings (including those originally seeded from book_openings.json)
        are written, making openings.json a self-contained mutable copy.
        """
        self._openings_path.parent.mkdir(parents=True, exist_ok=True)
        data = [_opening_to_dict(o) for o in self._index.values()]
        try:
            with self._openings_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error("Failed to write openings.json: %s", exc)
            raise
