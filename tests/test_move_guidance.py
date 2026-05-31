"""Tests for ai/move_guidance.py (B-65)."""

import unittest

from game.board import BoardState
from ai.game_ai import GameAI
from ai.opening_book import Opening, OpeningBook
from ai.opening_recognizer import OpeningRecognizer, INACTIVE_RESULT
from ai.move_guidance import (
    build_choose_move_kwargs,
    compute_force_book_early,
    format_trajectory_context,
    pick_target_opening,
    synthesize_opening_recognition,
)


class _StubBook(OpeningBook):
    def __init__(self, opening: Opening):
        super().__init__()
        self._index = {opening.opening_id: opening}

    def select_opening(self, ai_color: str = "W", **kwargs):
        for op in self._index.values():
            if op.side in (ai_color, "both"):
                return op
        return None


class TestMoveGuidance(unittest.TestCase):
    def _opening(self) -> Opening:
        return Opening(
            opening_id="test-1",
            name="Test Line",
            aliases=[],
            family="Test",
            side="B",
            seed_source="book",
            line_moves=["d6", "d2", "f4"],
            branch_moves=[],
            opening_fen_signatures=[],
            strategic_notes="note",
            common_blunders=[],
            recommended_responses={"W": [], "B": []},
            outcome_stats={"W": 0, "B": 0, "D": 0},
            confidence=0.9,
            tags=[],
        )

    def test_synthesize_requires_legal_book_move(self):
        op = self._opening()
        board = BoardState.new_game()
        rec = synthesize_opening_recognition(
            INACTIVE_RESULT, op, board, [], game_sym_idx=0,
        )
        self.assertEqual(rec.status, "probable")
        self.assertEqual(rec.book_move, "d6")

        # Occupied square — no synthesis
        blocked = BoardState.from_setup({"d6": "W"}, "B", "place")
        rec2 = synthesize_opening_recognition(
            INACTIVE_RESULT, op, blocked, [], game_sym_idx=0,
        )
        self.assertEqual(rec2.status, "inactive")

    def test_force_book_early_first_two_ai_placements(self):
        board = BoardState.new_game()
        self.assertTrue(compute_force_book_early(board, [], "B"))
        one = [{"color": "B", "type": "place", "notation": "d2"}]
        self.assertTrue(compute_force_book_early(board, one, "B"))
        two = one + [{"color": "W", "type": "place", "notation": "d6"},
                     {"color": "B", "type": "place", "notation": "f4"}]
        self.assertFalse(compute_force_book_early(board, two, "B"))

    def test_format_trajectory_context(self):
        text = format_trajectory_context({"c4": 0.5, "d7": -0.1})
        self.assertIn("c4 +0.50", text)
        self.assertIn("d7 -0.10", text)

    def test_pick_target_opening_validates_side(self):
        op = self._opening()
        op.side = "W"
        book = _StubBook(op)
        target, sym = pick_target_opening(book, "B")
        self.assertIsNone(target)
        self.assertEqual(sym, 0)

    def test_build_choose_move_kwargs_includes_trajectory_context(self):
        board = BoardState.new_game()
        ai = GameAI(color="B", difficulty=1)
        rec = OpeningRecognizer(_StubBook(self._opening()))
        kwargs = build_choose_move_kwargs(
            board, ai, [],
            opening_recognizer=rec,
            target_opening=self._opening(),
            trajectory_db=None,
        )
        self.assertIn("trajectory_context", kwargs)
        self.assertIn("force_book_early", kwargs)
        self.assertEqual(kwargs["recognition"].book_move, "d6")


if __name__ == "__main__":
    unittest.main()
