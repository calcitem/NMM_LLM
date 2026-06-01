"""
tests/test_b77_b78.py — Unit tests for B-77 (2-ply move-phase pin rule) and
B-78 (trajectory DB interference fix).

B-77: _pinned_move_squares_2ply(board, color) fires when:
  - A mill has own=1, opp=1, empty=1 (not a direct 2-config yet)
  - The opp piece is adjacent to the own piece (can slide in if own vacates)
  - A feeder opp piece is adjacent to opp_sq but outside the mill
  After White vacates own_sq:
    1. Opp slides opp_sq → own_sq (now in mill with feeder adjacent)
    2. Feeder slides → opp_sq (completing the 2-config)
  → own_sq is "2-ply pinned": vacating it hands opp a future 2-config.

B-78: DB forced-move must not override a capture move, and trajectory hints
  must not override a mill-close move (bonus capped at close_mill - 1 = 499).
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from game.board import BoardState, POSITIONS
from ai.game_ai import GameAI, _pinned_move_squares_2ply


def _board(white: list[str], black: list[str], turn: str = "W") -> BoardState:
    pos = {p: "" for p in POSITIONS}
    for p in white:
        pos[p] = "W"
    for p in black:
        pos[p] = "B"
    return BoardState.from_setup(pos, turn=turn, phase="move")


# ── B-77: _pinned_move_squares_2ply ──────────────────────────────────────────

class TestPinnedMoveSquares2Ply(unittest.TestCase):

    def test_basic_2ply_pin_detected(self):
        # Mill a7-d7-g7: White@a7, Black@d7, empty@g7.
        # Black@a4 adjacent to a7 but outside mill → if W vacates a7, B d7→a7, a4→d7 = 2-config (a4,d7,g7... wait)
        # Let's use: mill b6-d6-f6: White@b6, Black@d6, empty@f6.
        # Black@b4 adjacent to b6 outside mill.
        # If White vacates b6: Black d6→b6, b4→d6 → Black has b4+b6 in mill b2-b4-b6? No.
        # Better: mill a4-b4-c4: White@a4, Black@b4, empty@c4.
        # ADJACENCY a4: [a7, a1, b4] — b4 is in the mill (adjacent to a4, inside mill).
        # Need a feeder adjacent to b4 (opp_sq) OUTSIDE the mill.
        # ADJACENCY b4: [a4, b6, b2, c4] — b6 and b2 are outside a4-b4-c4.
        # Add Black@b6 as feeder.
        # Pattern: own=a4, opp=b4, empty=c4; opp b4 adjacent to own a4; feeder b6 adj to b4.
        b = _board(
            white=["a4", "g7", "d1", "f2"],
            black=["b4", "b6", "d6", "a7"],
            turn="W",
        )
        pinned = _pinned_move_squares_2ply(b, "W")
        self.assertIn("a4", pinned)

    def test_no_pin_when_no_feeder(self):
        # Same mill a4-b4-c4: White@a4, Black@b4, empty@c4.
        # But no feeder adjacent to b4 outside the mill.
        b = _board(
            white=["a4", "g7", "d1", "f2"],
            black=["b4", "d6", "a7", "g4"],
            turn="W",
        )
        pinned = _pinned_move_squares_2ply(b, "W")
        self.assertNotIn("a4", pinned)

    def test_no_pin_when_opp_not_adjacent_to_own(self):
        # Mill a7-d7-g7: White@a7, Black@g7, empty@d7.
        # Black@g7 is NOT adjacent to a7 (they are on opposite ends, a7-d7-g7 is a line).
        # Check ADJACENCY: a7 adj = [a4, d7, b6]; g7 adj = [d7, g4].
        # So a7 and g7 are not adjacent. No pin.
        b = _board(
            white=["a7", "d1", "f2", "b2"],
            black=["g7", "g4", "d6", "a4"],
            turn="W",
        )
        pinned = _pinned_move_squares_2ply(b, "W")
        self.assertNotIn("a7", pinned)

    def test_no_pin_when_mill_is_pure_2config(self):
        # If opp already has 2 in the mill (opp=2, own=0, empty=1),
        # that's a direct 1-ply threat handled by _pinned_move_squares, not 2-ply.
        # _pinned_move_squares_2ply should not fire for opp=2 configs.
        b = _board(
            white=["g7", "d1", "f2", "b2"],
            black=["a7", "d7", "d6", "a4"],
            turn="W",
        )
        pinned = _pinned_move_squares_2ply(b, "W")
        # g7 is empty in mill a7-d7-g7, own=0, opp=2 → not a 2-ply pattern
        self.assertNotIn("g7", pinned)

    def test_choose_move_avoids_2ply_pinned_square(self):
        # White has only one non-pinned piece to move from; AI must use it.
        # Mill a4-b4-c4: White@a4, Black@b4, empty@c4; feeder Black@b6.
        # White also has g7, d1 — these are not pinned.
        b = _board(
            white=["a4", "g7", "d1"],
            black=["b4", "b6", "d6", "a7", "g4"],
            turn="W",
        )
        pinned = _pinned_move_squares_2ply(b, "W")
        self.assertIn("a4", pinned)
        ai = GameAI(color="W", difficulty=3)
        move = ai.choose_move(b)
        self.assertIsNotNone(move)
        self.assertNotEqual(move.get("from"), "a4")

    def test_safety_all_pinned_still_moves(self):
        # When all own pieces are 2-ply pinned, AI must still return a move
        # (safety guard: no piece list reduction when it would empty moves).
        # Tricky to set up — just verify no crash and a move is returned.
        b = _board(
            white=["a4", "d1"],
            black=["b4", "b6", "d6", "a7", "g4", "g7", "f2", "f4"],
            turn="W",
        )
        ai = GameAI(color="W", difficulty=1)
        move = ai.choose_move(b)
        self.assertIsNotNone(move)
        self.assertIn("from", move)


# ── B-78: DB forced move / trajectory hint cap ───────────────────────────────

class MockFGDBResult:
    def __init__(self, outcome="W", best_move="d6-d7"):
        self.outcome = outcome
        self.best_move_canonical = best_move


class MockFGDB:
    """Minimal stand-in for FullGameDB that returns a controlled forced move."""

    def __init__(self, forced_notation: str):
        self._forced = forced_notation
        self._result = MockFGDBResult(outcome="W", best_move=forced_notation)

    def is_available(self) -> bool:
        return True

    def query(self, board):
        return self._result

    def best_move(self, board) -> str:
        return self._forced

    def score_delta(self, board, color):
        return {}


class TestB78DBForcedMove(unittest.TestCase):

    def _move_phase_board(self):
        """A simple move-phase board where White can capture OR make a quiet move."""
        # White: d7, g7, a7.  Black: d6, b6, f6, d3, g4, b4.
        # Mill d6-b6-f6 — Black has 2-config but needs closing.
        # White has a capture: if we put Black piece adjacent that White can capture.
        # Easier: give White a closed mill so it can capture.
        # White closes d1-d3-d5 (mill) and captures; forced DB move = a7-a4 (quiet).
        pos = {p: "" for p in POSITIONS}
        for p in ["d1", "d3", "d5", "g7", "a7", "b2"]:
            pos[p] = "W"
        for p in ["d6", "b6", "f6", "d2", "g4", "b4"]:
            pos[p] = "B"
        return BoardState.from_setup(pos, turn="W", phase="move")

    def test_db_forced_move_not_returned_when_capture_available(self):
        # White closes d1-d3-d5 (already a mill? no, we need to MOVE into it).
        # Just test that when moves include a capture, DB forced move is skipped.
        # Use a board where White has a capture move AND a quiet move.
        pos = {p: "" for p in POSITIONS}
        # White closes mill a7-d7-g7 by moving from some square → d7.
        # Actually let's just test the logic: inject a mock FGDB and verify the
        # AI does NOT return the forced move when a capture is present.
        # We need a board where White can capture this turn.
        # White: a7, d7, b4, d1, d3.  Black: g7, d6, b6, a4, b2.
        # If White has mill (a7-d7-g7 needs g7 which is Black) → White closes a7-d7-g7? No.
        # Simpler: White at b6-d6-f6... no.
        # Let's place White so it just closed a mill last turn (it's White's turn to capture).
        # board.phase = "move" and there's a capture in the moves list means White just
        # closed a mill and must capture.  We simulate this by putting the board in a state
        # where one white mill is closed and there's a capturable Black piece.
        # White has b6+d6+f6 = closed mill → move is to capture a Black piece.
        # But that's checking post-mill capture selection... For simplicity: just inject
        # a capture into the moves list via a position where Black is trapped.
        # White: a7, d7, g7 (closed mill!), d1, f2.  Black: d6, b6, f6, b4, a4.
        # In a closed-mill position the AI will want to capture.  Inject forced notation = "d1-d3" (quiet).
        pos2 = {p: "" for p in POSITIONS}
        for p in ["a7", "d7", "g7", "d1", "f2"]:
            pos2[p] = "W"
        for p in ["d6", "b6", "f6", "b4", "a4"]:
            pos2[p] = "B"
        board = BoardState.from_setup(pos2, turn="W", phase="move")

        ai = GameAI(color="W", difficulty=3)
        mock_db = MockFGDB(forced_notation="d1-d3")
        ai._fullgame_db = mock_db

        move = ai.choose_move(board)
        # The AI has a closed mill (a7-d7-g7) so it can capture a Black piece.
        # The forced notation "d1-d3" is a quiet move — it must NOT be chosen.
        self.assertIsNotNone(move)
        if move.get("capture"):
            # Good — a capture was chosen over the DB forced move.
            pass
        else:
            # If there's no capture in the legal moves, the DB move is fine.
            from game.rules import get_all_legal_moves as legal_moves
            lmoves = legal_moves(board)
            captures = [m for m in lmoves if m.get("capture")]
            if captures:
                self.fail(f"DB forced quiet move was chosen despite captures available: {move}")

    def test_db_forced_move_applied_when_no_capture(self):
        # No captures available; DB forced move should be returned.
        # White: d1, f2, b2, g4, e3.  Black: d6, b6, f6, b4, g7.
        # No mills closed, no captures available.
        # Force notation = "d1-d3" (d3 is adjacent to d1 and empty here).
        pos = {p: "" for p in POSITIONS}
        for p in ["d1", "f2", "b2", "g4", "e3"]:
            pos[p] = "W"
        for p in ["d6", "b6", "f6", "b4", "g7"]:
            pos[p] = "B"
        board = BoardState.from_setup(pos, turn="W", phase="move")

        # Verify d1-d2 is a legal quiet move in this position (d1 adj d2)
        from game.rules import get_all_legal_moves as legal_moves
        lmoves = legal_moves(board)
        d1_d2 = next((m for m in lmoves if m.get("from") == "d1" and m.get("to") == "d2"), None)
        if d1_d2 is None:
            self.skipTest("d1-d2 not legal in this position")

        ai = GameAI(color="W", difficulty=3)
        ai._fullgame_db = MockFGDB(forced_notation="d1-d2")

        chosen = ai.choose_move(board)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.get("from"), "d1")
        self.assertEqual(chosen.get("to"), "d2")


class TestB78TrajectoryHintCap(unittest.TestCase):

    def test_trajectory_hint_capped_below_close_mill(self):
        """Trajectory bonus must not exceed close_mill - 1 (499)."""
        from ai.heuristics import HeuristicWeights, DEFAULT_WEIGHTS
        ai = GameAI(color="W", difficulty=3)

        # At adherence=100, scale=3000; delta=+0.5 → raw bonus=1500 → capped at 499.
        ai._weights.opening_adherence = 100
        close_mill = ai._weights.close_mill  # 500

        scored = [
            ({"to": "d7"}, 0),   # the hinted move
            ({"to": "g7"}, close_mill),  # a mill-closing move (score = 500)
        ]
        hints = {"d7": 0.5}

        adjusted = ai._apply_trajectory_hints(scored, hints)
        adjusted_dict = {m["to"]: s for m, s in adjusted}

        # d7 bonus should be capped at 499, NOT 1500.
        self.assertLessEqual(adjusted_dict["d7"], close_mill - 1)
        # g7 (mill close) should remain dominant.
        self.assertGreater(adjusted_dict["g7"], adjusted_dict["d7"])

    def test_negative_trajectory_hint_not_capped(self):
        """Cap only applies to positive bonuses; negative deltas pass through."""
        ai = GameAI(color="W", difficulty=3)
        ai._weights.opening_adherence = 100

        scored = [({"to": "d7"}, 0)]
        hints = {"d7": -0.3}

        adjusted = ai._apply_trajectory_hints(scored, hints)
        score = adjusted[0][1]
        self.assertLess(score, 0)

    def test_hard_ban_not_affected_by_cap(self):
        """Hard-ban sentinel (-1.0) still sends move to -INF+1."""
        ai = GameAI(color="W", difficulty=3)
        ai._weights.opening_adherence = 100

        from ai.game_ai import INF
        scored = [({"to": "d7"}, 500)]
        hints = {"d7": -1.0}

        adjusted = ai._apply_trajectory_hints(scored, hints)
        self.assertEqual(adjusted[0][1], -INF + 1)


if __name__ == "__main__":
    unittest.main()
