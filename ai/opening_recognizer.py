"""
ai/opening_recognizer.py — Real-time opening recognition for Nine Men's Morris.

Recognition is ONLY active during the placement phase (both sides placing;
ply <= 18).  Once placement ends (both sides have placed 9 pieces each) the
current result is frozen and returned unchanged for subsequent calls.

Recognition pipeline per ply
-----------------------------
1. Append the move to move_sequence; compute ply = len(move_sequence).
2. Exact-prefix match against all openings in the book.
3. Deviation detection — if previous ply had candidates but current ply has
   none, look up branch moves on the previous candidates.
4. FEN transposition — compare board.to_fen_string() against all
   opening_fen_signatures at this ply across the whole book.
5. Novel — ply >= 4 and nothing matched.
6. Set book_move from the matched opening's line_moves[ply] if available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState

from ai.opening_book import Opening, OpeningBook

logger = logging.getLogger(__name__)


# ── RecognitionResult ─────────────────────────────────────────────────────────

@dataclass
class RecognitionResult:
    opening_id: Optional[str]
    name: Optional[str]
    family: Optional[str]
    confidence: float           # 0.0 – 1.0
    status: str                 # "exact" | "probable" | "transposition" | "novel" | "inactive"
    matched_ply: int            # how many plies matched
    deviation_ply: Optional[int]
    deviation_move: Optional[str]   # what was played at the deviation point
    book_move: Optional[str]        # book's next recommended move (line_moves[ply])
    branch_name: Optional[str]
    strategic_notes: str
    common_blunders: list[str]
    tags: list[str]


#: Sentinel returned before any moves have been observed.
INACTIVE_RESULT: RecognitionResult = RecognitionResult(
    opening_id=None,
    name=None,
    family=None,
    confidence=0.0,
    status="inactive",
    matched_ply=0,
    deviation_ply=None,
    deviation_move=None,
    book_move=None,
    branch_name=None,
    strategic_notes="",
    common_blunders=[],
    tags=[],
)


# ── OpeningRecognizer ─────────────────────────────────────────────────────────

class OpeningRecognizer:
    """
    Tracks the move sequence and classifies it against the opening book in
    real-time.  Call update() after each placement-phase move.
    """

    def __init__(self, book: OpeningBook) -> None:
        self.book = book
        self.move_sequence: list[str] = []
        self.current_result: RecognitionResult = INACTIVE_RESULT
        self._active_candidates: list[Opening] = []
        self._prev_candidates: list[Opening] = []   # candidates from previous ply
        self._last_matched_opening: Optional[Opening] = None
        self._placement_phase_ended: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state; ready for a new game."""
        self.move_sequence = []
        self.current_result = INACTIVE_RESULT
        self._active_candidates = []
        self._prev_candidates = []
        self._last_matched_opening = None
        self._placement_phase_ended = False

    def update(self, move_notation: str, board: "BoardState") -> RecognitionResult:
        """
        Advance recognition by one move.

        Parameters
        ----------
        move_notation:
            The move just played in placement notation (e.g. "d2").
        board:
            The BoardState *after* the move has been applied (used for FEN
            transposition checks and placement-phase detection).

        Returns
        -------
        RecognitionResult
            The updated recognition state.
        """
        # Once the placement phase ends, freeze and return unchanged.
        if self._placement_phase_ended:
            return self.current_result

        # Check whether placement just ended.
        if (
            board.pieces_placed.get("W", 0) >= 9
            and board.pieces_placed.get("B", 0) >= 9
        ):
            self._placement_phase_ended = True
            return self.current_result

        # ── Step 1: append move, compute ply ─────────────────────────────────
        self.move_sequence.append(move_notation)
        ply = len(self.move_sequence)

        # Carry forward previous candidates before overwriting.
        self._prev_candidates = list(self._active_candidates)

        # ── Step 2: exact prefix match ────────────────────────────────────────
        candidates: list[Opening] = [
            o for o in self.book.values()
            if len(o.line_moves) >= ply
            and o.line_moves[:ply] == self.move_sequence
        ]
        self._active_candidates = candidates

        if candidates:
            # Decide status based on number of matches.
            if len(candidates) == 1 and ply >= 2:
                status = "exact"
                confidence = 1.0
                matched_opening = candidates[0]
            else:
                status = "probable"
                confidence = 1.0 / len(candidates)
                # Pick the longest (most specific) candidate as representative.
                matched_opening = max(candidates, key=lambda o: len(o.line_moves))

            self._last_matched_opening = matched_opening

            # Book move: the next expected move in the line.
            book_move: Optional[str] = None
            if len(matched_opening.line_moves) > ply:
                book_move = matched_opening.line_moves[ply]

            result = RecognitionResult(
                opening_id=matched_opening.opening_id,
                name=matched_opening.name,
                family=matched_opening.family,
                confidence=confidence,
                status=status,
                matched_ply=ply,
                deviation_ply=None,
                deviation_move=None,
                book_move=book_move,
                branch_name=None,
                strategic_notes=matched_opening.strategic_notes,
                common_blunders=list(matched_opening.common_blunders),
                tags=list(matched_opening.tags),
            )
            self.current_result = result
            return result

        # ── Step 3: deviation detection ───────────────────────────────────────
        # Previous ply had candidates but this ply has none — we deviated.
        if self._prev_candidates:
            deviation_ply = ply
            deviation_move = move_notation

            # Search for a pre-registered branch on any previous candidate.
            for prev_opening in self._prev_candidates:
                for branch in prev_opening.branch_moves:
                    if (
                        branch.deviation_ply == deviation_ply
                        and branch.deviation_move == deviation_move
                    ):
                        # Found a known branch.
                        result = RecognitionResult(
                            opening_id=prev_opening.opening_id,
                            name=prev_opening.name,
                            family=prev_opening.family,
                            confidence=0.5,
                            status="probable",
                            matched_ply=ply - 1,   # last ply that fully matched
                            deviation_ply=deviation_ply,
                            deviation_move=deviation_move,
                            book_move=None,
                            branch_name=branch.name,
                            strategic_notes=branch.strategic_notes,
                            common_blunders=list(prev_opening.common_blunders),
                            tags=list(prev_opening.tags),
                        )
                        self.current_result = result
                        self._active_candidates = []
                        return result

            # No matching branch — fall through to transposition / novel.

        # ── Step 4: transposition detection ──────────────────────────────────
        board_fen = board.to_fen_string()
        for opening in self.book.values():
            for sig in opening.opening_fen_signatures:
                if sig.get("ply") == ply and sig.get("fen") == board_fen:
                    self._last_matched_opening = opening

                    # Book move from line if available.
                    book_move = None
                    if len(opening.line_moves) > ply:
                        book_move = opening.line_moves[ply]

                    result = RecognitionResult(
                        opening_id=opening.opening_id,
                        name=opening.name,
                        family=opening.family,
                        confidence=0.7,
                        status="transposition",
                        matched_ply=ply,
                        deviation_ply=None,
                        deviation_move=None,
                        book_move=book_move,
                        branch_name=None,
                        strategic_notes=opening.strategic_notes,
                        common_blunders=list(opening.common_blunders),
                        tags=list(opening.tags),
                    )
                    self.current_result = result
                    self._active_candidates = []
                    return result

        # ── Step 5: novel ─────────────────────────────────────────────────────
        if ply >= 4:
            result = RecognitionResult(
                opening_id=None,
                name=None,
                family=None,
                confidence=0.0,
                status="novel",
                matched_ply=ply - 1 if self._prev_candidates else 0,
                deviation_ply=ply if self._prev_candidates else None,
                deviation_move=move_notation if self._prev_candidates else None,
                book_move=None,
                branch_name=None,
                strategic_notes="",
                common_blunders=[],
                tags=[],
            )
            self.current_result = result
            self._active_candidates = []
            return result

        # ply < 4 and no match yet — stay inactive (too early to classify).
        self.current_result = INACTIVE_RESULT
        self._active_candidates = []
        return self.current_result

    # ── Convenience accessors ─────────────────────────────────────────────────

    def get_next_book_move(self) -> Optional[str]:
        """Return the book's recommended next move, or None."""
        return self.current_result.book_move

    def get_current_result(self) -> RecognitionResult:
        """Return the latest RecognitionResult."""
        return self.current_result
