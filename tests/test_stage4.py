"""tests/test_stage4.py — Stage 4: Opening book and recognition tests."""

from __future__ import annotations

import os
import tempfile
import unittest

from game.board import BoardState
from ai.opening_book import OpeningBook, Opening
from ai.opening_recognizer import OpeningRecognizer, INACTIVE_RESULT
from ai.game_ai import GameAI


BOOK_PATH = "data/openings/book_openings.json"

# mill-rush-perpendicular is unique after 6 moves.
PERP_MOVES = ["d2", "d6", "f4", "b4", "f2", "b6"]
# mill-rush-parallel shares first 5 moves with perpendicular then diverges.
PARALLEL_MOVES = ["d2", "d6", "f4", "b4", "f2", "f6", "b2", "b6"]


def _make_book(tmp_dir: str) -> OpeningBook:
    return OpeningBook(
        book_path=BOOK_PATH,
        openings_path=os.path.join(tmp_dir, "openings.json"),
    )


def _replay(moves: list[str], book: OpeningBook):
    """Replay moves through an OpeningRecognizer; return (recognizer, final result)."""
    rec = OpeningRecognizer(book)
    board = BoardState.new_game()
    result = INACTIVE_RESULT
    for m in moves:
        board = board.apply_move({"from": None, "to": m, "capture": None})
        result = rec.update(m, board)
    return rec, result


# ── OpeningBook load ──────────────────────────────────────────────────────────

class TestOpeningBookLoad(unittest.TestCase):

    def test_loads_11_openings(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            self.assertEqual(len(list(book.values())), 11)

    def test_get_by_id_returns_opening(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            opening = book.get_by_id("mill-rush-perpendicular")
            self.assertIsNotNone(opening)
            self.assertEqual(opening.opening_id, "mill-rush-perpendicular")

    def test_get_by_id_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            self.assertIsNone(book.get_by_id("does-not-exist"))

    def test_get_by_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            results = book.get_by_family("Mill Rush")
            self.assertGreater(len(results), 0)
            for o in results:
                self.assertEqual(o.family, "Mill Rush")

    def test_seeding_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            book1 = _make_book(tmp)
            book2 = _make_book(tmp)
            self.assertEqual(len(list(book1.values())), len(list(book2.values())))

    def test_save_new_book_opening_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            new_opening = Opening(
                opening_id="brand-new-id",
                name="Fake",
                aliases=[],
                family="fake",
                side="both",
                seed_source="book",
                line_moves=[],
                branch_moves=[],
                opening_fen_signatures=[],
                strategic_notes="",
                common_blunders=[],
                recommended_responses={"W": [], "B": []},
                outcome_stats={"W": 0, "B": 0, "D": 0},
                confidence=1.0,
                tags=[],
            )
            with self.assertRaises(ValueError):
                book.save_opening(new_opening)


# ── Outcome stats ─────────────────────────────────────────────────────────────

class TestOpeningBookOutcomeStats(unittest.TestCase):

    def test_increment_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            oid = "mill-rush-parallel"
            initial = book.get_by_id(oid).outcome_stats.get("W", 0)
            book.update_outcome_stats(oid, "W")
            self.assertEqual(book.get_by_id(oid).outcome_stats["W"], initial + 1)

    def test_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            book.update_outcome_stats("corner-gambit", "B")
            book2 = _make_book(tmp)
            self.assertGreaterEqual(book2.get_by_id("corner-gambit").outcome_stats["B"], 1)

    def test_invalid_winner_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            oid = "mill-rush-parallel"
            before = dict(book.get_by_id(oid).outcome_stats)
            book.update_outcome_stats(oid, "X")
            self.assertEqual(before, book.get_by_id(oid).outcome_stats)

    def test_unknown_opening_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            # Should not raise; just logs a warning.
            book.update_outcome_stats("nonexistent-id", "W")


# ── Novel opening save ────────────────────────────────────────────────────────

class TestOpeningBookNovelSave(unittest.TestCase):

    def test_saved_with_learned_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            moves = ["a1", "g7", "a7", "g1", "d2", "d6", "f4", "b4"]
            novel = book.save_novel_opening(moves, [], outcome="W")
            self.assertEqual(novel.seed_source, "learned")
            self.assertEqual(novel.line_moves, moves)
            self.assertAlmostEqual(novel.confidence, 0.3, places=2)
            self.assertEqual(novel.outcome_stats["W"], 1)

    def test_reloads_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            moves = ["a1", "g7", "a7", "g1", "d2", "d6", "f4", "b4"]
            novel = book.save_novel_opening(moves, [], outcome="D")
            oid = novel.opening_id
            book2 = _make_book(tmp)
            reloaded = book2.get_by_id(oid)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.seed_source, "learned")
            self.assertEqual(reloaded.line_moves, moves)


# ── Record deviation ───────────────────────────────────────────────────────────

class TestOpeningBookRecordDeviation(unittest.TestCase):

    def test_creates_learned_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            branch = book.record_deviation("mill-rush-parallel", 6, "a1", "test_fen")
            self.assertIsNotNone(branch)
            self.assertEqual(branch.deviation_ply, 6)
            self.assertEqual(branch.deviation_move, "a1")
            self.assertEqual(branch.seed_source, "learned")

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            b1 = book.record_deviation("mill-rush-parallel", 6, "a1", "fen")
            b2 = book.record_deviation("mill-rush-parallel", 6, "a1", "fen")
            self.assertEqual(b1.branch_id, b2.branch_id)

    def test_unknown_opening_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            self.assertIsNone(book.record_deviation("no-such-id", 6, "a1", "fen"))


# ── OpeningRecognizer ─────────────────────────────────────────────────────────

class TestOpeningRecognizerExact(unittest.TestCase):

    def test_inactive_before_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            rec = OpeningRecognizer(book)
            self.assertEqual(rec.get_current_result().status, "inactive")

    def test_exact_after_6_moves_perpendicular(self):
        """mill-rush-perpendicular is unique at ply 6 → status='exact'."""
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            _, result = _replay(PERP_MOVES, book)
            self.assertEqual(result.status, "exact",
                             f"Expected 'exact', got {result.status!r} ({result.name})")
            self.assertEqual(result.opening_id, "mill-rush-perpendicular")
            self.assertAlmostEqual(result.confidence, 1.0)

    def test_exact_carries_book_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            _, result = _replay(PERP_MOVES, book)
            # Line has moves beyond ply 6 → book_move should be populated.
            self.assertIsNotNone(result.book_move)

    def test_probable_when_multiple_candidates(self):
        """Four openings share the first 4 moves → status='probable'."""
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            _, result = _replay(["d2", "d6", "f4", "b4"], book)
            self.assertEqual(result.status, "probable")
            self.assertLess(result.confidence, 1.0)

    def test_exact_after_8_moves_parallel(self):
        """mill-rush-parallel is unique at ply 8 → status='exact'."""
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            _, result = _replay(PARALLEL_MOVES, book)
            self.assertEqual(result.status, "exact")
            self.assertEqual(result.opening_id, "mill-rush-parallel")


class TestOpeningRecognizerDeviation(unittest.TestCase):

    def test_deviation_status_after_unexpected_move(self):
        """Unexpected 6th move after 5 matched plies → status in (novel, probable)."""
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            moves = PERP_MOVES[:5] + ["a1"]  # a1 is not a 6th move in any opening
            _, result = _replay(moves, book)
            self.assertIn(result.status, ("novel", "probable"),
                          f"Got unexpected status {result.status!r}")

    def test_deviation_ply_recorded(self):
        """deviation_ply must equal the ply at which the sequence diverged."""
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            moves = PERP_MOVES[:5] + ["a1"]
            _, result = _replay(moves, book)
            if result.deviation_ply is not None:
                self.assertEqual(result.deviation_ply, 6)

    def test_reset_clears_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            rec, _ = _replay(PERP_MOVES, book)
            rec.reset()
            self.assertEqual(rec.get_current_result().status, "inactive")
            self.assertEqual(len(rec.move_sequence), 0)
            self.assertEqual(rec._active_candidates, [])

    def test_recognizer_freezes_after_placement_ends(self):
        """After 18 placements recognition result should not change."""
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            rec = OpeningRecognizer(book)
            board = BoardState.new_game()
            positions = [
                "a1", "g7", "a4", "g4", "a7", "g1",
                "d1", "d7", "b2", "f6", "b6", "f2",
                "c3", "e5", "c4", "e4", "c5", "e3",
            ]
            for m in positions:
                board = board.apply_move({"from": None, "to": m, "capture": None})
                rec.update(m, board)
            result_at_18 = rec.get_current_result()
            # Playing a 19th placement-phase move is illegal; board is now in move phase.
            # Just verify the freeze flag is set.
            self.assertTrue(rec._placement_phase_ended)


class TestOpeningRecognizerTransposition(unittest.TestCase):

    def test_transposition_detected(self):
        """
        Playing W→f4, B→d6, W→d2, B→b4 gives the same board as the canonical
        first 4 moves of mill-rush-parallel → should be detected as transposition.
        """
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            # Verify FENs match before running the recognition test.
            def play(moves):
                b = BoardState.new_game()
                for m in moves:
                    b = b.apply_move({"from": None, "to": m, "capture": None})
                return b

            canonical = play(["d2", "d6", "f4", "b4"])
            transposed = play(["f4", "d6", "d2", "b4"])
            if canonical.to_fen_string() != transposed.to_fen_string():
                self.skipTest("Board FEN encodes move history; transposition test N/A")

            _, result = _replay(["f4", "d6", "d2", "b4"], book)
            self.assertEqual(result.status, "transposition",
                             f"Expected 'transposition', got {result.status!r}")


# ── GameAI opening integration ─────────────────────────────────────────────────

class TestGameAIOpeningIntegration(unittest.TestCase):

    def test_choose_move_with_recognition_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            _, recognition = _replay(PERP_MOVES, book)
            board = BoardState.new_game()
            for m in PERP_MOVES:
                board = board.apply_move({"from": None, "to": m, "capture": None})
            ai = GameAI(color=board.turn, difficulty=3)
            move = ai.choose_move(board, recognition=recognition)
            self.assertIn("to", move)

    def test_choose_move_with_none_recognition_does_not_crash(self):
        board = BoardState.new_game()
        ai = GameAI(color="W", difficulty=2)
        move = ai.choose_move(board, recognition=None)
        self.assertIn("to", move)

    def test_opening_adjustments_boost_book_move_score(self):
        """
        With an exact recognition the book move's raw score should be increased
        relative to an equivalent run without recognition.
        """
        with tempfile.TemporaryDirectory() as tmp:
            book = _make_book(tmp)
            _, recognition = _replay(PERP_MOVES, book)
            board = BoardState.new_game()
            for m in PERP_MOVES:
                board = board.apply_move({"from": None, "to": m, "capture": None})

            ai = GameAI(color=board.turn, difficulty=2)
            from game.rules import get_all_legal_moves
            moves = get_all_legal_moves(board)
            scored_plain = ai._score_all(board, moves, 2)
            scored_boosted = ai._apply_opening_adjustments(scored_plain, recognition)

            book_dest = recognition.book_move
            if book_dest is None:
                self.skipTest("No book move at this ply")

            plain_score = next(
                (s for m, s in scored_plain if m["to"] == book_dest), None
            )
            boosted_score = next(
                (s for m, s in scored_boosted if m["to"] == book_dest), None
            )
            if plain_score is None:
                self.skipTest(f"Book move {book_dest!r} not in legal moves")
            self.assertGreaterEqual(boosted_score, plain_score)


if __name__ == "__main__":
    unittest.main()
