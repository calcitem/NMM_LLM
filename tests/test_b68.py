"""
tests/test_b68.py — Regression for B-68: opening book bonus must not override
the B-64 dead/near-dead placement penalty.

Two scenarios:
  1. Book suppression: when the book recommends a dead square, _apply_opening_adjustments
     must NOT apply the book_bonus_abs delta so the tactical score wins.
  2. Forced-block label: when the only legal block is a dead square, last_thinking
     should read "Forced block (dead square — unavoidable)", not "Dead/near-dead
     placement (B-64)".
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from game.board import BoardState
from ai.game_ai import GameAI, _immediate_mill_threats
from ai.heuristics import HeuristicWeights


def _make_board(white_squares, black_squares, turn="W") -> BoardState:
    """Quick builder: place White on white_squares, Black on black_squares."""
    pos = {}
    for sq in white_squares:
        pos[sq] = "W"
    for sq in black_squares:
        pos[sq] = "B"
    return BoardState.from_setup(pos, turn=turn, phase="place")


def _make_recognition(book_move: str, status: str = "matched") -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        book_move=book_move,
        common_blunders=[],
    )


class TestB68BookBonusSuppression(unittest.TestCase):
    """B-68 fix 1: book bonus is suppressed for dead-square placements."""

    def setUp(self):
        # Board at turn 8 of placement phase.
        # g7 neighbours: d7=Black, g4=Black → 0 free after placing.
        # White has tactically better alternatives (e.g. c4 or b4).
        self.board = _make_board(
            white_squares=["c1", "e1", "d5", "b6"],
            black_squares=["d7", "g4", "a4", "e7"],
            turn="W",
        )
        weights = HeuristicWeights(opening_adherence=75)
        self.ai = GameAI(color="W", difficulty=3, weights=weights)

    def test_dead_book_move_gets_no_bonus(self):
        """_apply_opening_adjustments must not boost g7 when g7 is dead."""
        recognition = _make_recognition(book_move="g7")
        # Two candidate moves: g7 (dead) and c4 (has free neighbours)
        scored = [
            ({"to": "g7"}, -1500),   # base score already penalised by B-64
            ({"to": "c4"}, 200),     # a live, tactically sound square
        ]
        adjusted = self.ai._apply_opening_adjustments(scored, recognition, self.board)
        score_map = {m["to"]: s for m, s in adjusted}
        # Book bonus (2250 at 75%) must NOT be added to g7
        self.assertEqual(score_map["g7"], -1500,
                         "Book bonus should be suppressed for dead square g7")
        # c4 should be untouched (it is not the book_dest)
        self.assertEqual(score_map["c4"], 200)

    def test_live_book_move_still_gets_bonus(self):
        """_apply_opening_adjustments still boosts live squares normally."""
        recognition = _make_recognition(book_move="c4")
        scored = [
            ({"to": "c4"}, 200),
            ({"to": "g7"}, -1500),
        ]
        adjusted = self.ai._apply_opening_adjustments(scored, recognition, self.board)
        score_map = {m["to"]: s for m, s in adjusted}
        expected_bonus = int(3000 * 75 / 100)  # 2250
        self.assertEqual(score_map["c4"], 200 + expected_bonus,
                         "Book bonus must still apply to live square c4")

    def test_dead_book_move_mill_closing_keeps_bonus(self):
        """A dead-square book move that closes a mill keeps its bonus (B-64 exemption)."""
        # Set up White pieces so that placing at g7 closes a mill.
        # g1-g4-g7 is a valid mill; White has g1 and g4 would close it.
        # But g4 is Black in setUp... use a different mill: a7-d7-g7.
        # Place White at a7 and White at d7 so g7 closes the mill.
        board = _make_board(
            white_squares=["a7", "d7", "c1"],
            black_squares=["g4", "a4"],
            turn="W",
        )
        recognition = _make_recognition(book_move="g7")
        scored = [({"to": "g7"}, 0)]
        adjusted = self.ai._apply_opening_adjustments(scored, recognition, board)
        score_map = {m["to"]: s for m, s in adjusted}
        expected_bonus = int(3000 * 75 / 100)
        self.assertEqual(score_map["g7"], 0 + expected_bonus,
                         "Mill-closing placement should still get the book bonus")

    def test_movement_phase_book_bonus_unchanged(self):
        """Movement-phase book moves (have 'from') are never suppressed."""
        recognition = _make_recognition(book_move="g7")
        # Movement move: from=g4 to=g7 (dead square, but movement phase)
        scored = [({"from": "g4", "to": "g7"}, -200)]
        adjusted = self.ai._apply_opening_adjustments(scored, recognition, self.board)
        score_map = {(m.get("from"), m["to"]): s for m, s in adjusted}
        expected_bonus = int(3000 * 75 / 100)
        self.assertEqual(score_map[("g4", "g7")], -200 + expected_bonus,
                         "Movement-phase book bonus must not be suppressed")


class TestB68ForcedBlockLabel(unittest.TestCase):
    """B-68 fix 2: forced block at dead square shows correct label."""

    def test_forced_block_dead_square_label(self):
        """When the mandatory block is a dead square, last_thinking says 'Forced block'."""
        # Black threatens a7-d7-g7 (has a7 and d7).
        # g7 neighbours: d7=Black, g4=Black → 0 free → dead.
        # White must block at g7 (only blocking move).
        board = _make_board(
            white_squares=["c1", "e1", "d5"],
            black_squares=["a7", "d7", "g4", "b4"],
            turn="W",
        )
        threats = _immediate_mill_threats(board)
        self.assertIn("g7", threats, "g7 must be a mandatory blocking square")

        ai = GameAI(color="W", difficulty=2)
        ai.choose_move(board)
        self.assertEqual(
            ai.last_thinking,
            "Forced block (dead square — unavoidable)",
            f"Expected forced-block label, got: {ai.last_thinking!r}",
        )

    def test_non_dead_forced_block_keeps_normal_label(self):
        """Mandatory block on a live square keeps its original tactical label."""
        # Black threatens b2-b4-b6 (has b2 and b6), block at b4.
        # b4 neighbours: a4, c4, b2, b6 — after blocking, a4 and c4 are free → live.
        board = _make_board(
            white_squares=["d2", "d6", "f4"],
            black_squares=["b2", "b6", "g4", "a1"],
            turn="W",
        )
        threats = _immediate_mill_threats(board)
        self.assertIn("b4", threats, "b4 must be a mandatory blocking square")

        ai = GameAI(color="W", difficulty=2)
        ai.choose_move(board)
        # b4 has free neighbours (a4, c4) so it is live — label must NOT be forced-block
        self.assertNotEqual(
            ai.last_thinking,
            "Forced block (dead square — unavoidable)",
            "Live forced-block square must not get the dead-square label",
        )


if __name__ == "__main__":
    unittest.main()
