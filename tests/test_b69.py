"""
tests/test_b69.py — Regression for B-69: hard-filter dead placements from the
move list before any search so that iterative deepening cannot override B-64.

A "dead placement" is placing on a square with 0 free (empty) adjacent squares
that does not close a mill.  Such a piece is permanently immobile: it can never
be part of a mill or slide away.

Regression positions taken from live game failures (2026-05-31):
  • a4-game: after W:a7,b4,g7,d6,d1 / B:d7,a1,g4,g1 — Black must NOT place on
    a4 or b6 (both dead: 0 free neighbours in that position).
  • b6-last-piece: board where b6 has d6=B, b4=B → dead for any side.
"""
from __future__ import annotations

import unittest

from game.board import ADJACENCY, BoardState, POSITIONS
from ai.game_ai import GameAI, _is_dead_placement, _dead_has_mill_potential


def _free_neighbors(board: BoardState, sq: str) -> int:
    return sum(1 for nb in ADJACENCY.get(sq, []) if board.positions.get(nb) == "")


class TestIsDeadPlacement(unittest.TestCase):
    """Unit tests for the _is_dead_placement helper."""

    def _board(self, white, black, turn="W"):
        pos = {sq: "W" for sq in white}
        pos.update({sq: "B" for sq in black})
        return BoardState.from_setup(pos, turn=turn, phase="place")

    def test_dead_square_is_detected(self):
        """a4 with a1=B, a7=W, b4=W is dead."""
        board = self._board(
            white=["a7", "b4", "g7", "d6", "d1"],
            black=["d7", "a1", "g4", "g1"],
            turn="B",
        )
        self.assertTrue(_is_dead_placement(board, {"to": "a4"}))

    def test_live_square_is_not_dead(self):
        """d2 with two free neighbours is not dead."""
        board = self._board(
            white=["a7", "b4", "g7", "d6", "d1"],
            black=["d7", "a1", "g4", "g1"],
            turn="B",
        )
        self.assertFalse(_is_dead_placement(board, {"to": "d2"}))

    def test_movement_move_never_dead(self):
        """A move with 'from' key is never a dead placement."""
        board = self._board(
            white=["a7", "b4", "g7", "d6", "d1"],
            black=["d7", "a1", "g4", "g1"],
            turn="B",
        )
        self.assertFalse(_is_dead_placement(board, {"from": "d7", "to": "a4"}))

    def test_mill_closing_dead_square_exempted(self):
        """Placing on a dead square that closes a mill is NOT filtered."""
        # a7-d7-g7 is a mill; Black has a7 and d7; placing at g7 closes it.
        # g7 neighbours: d7(B) and g4 — if g4 is occupied too, g7 is dead.
        board = self._board(
            white=["c1"],
            black=["a7", "d7", "g4"],
            turn="B",
        )
        # g7 neighbours: d7=B, g4=B → 0 free → would be dead without mill check
        self.assertEqual(_free_neighbors(board, "g7"), 0)
        # but closing a7-d7-g7 → should be exempted
        self.assertFalse(_is_dead_placement(board, {"to": "g7"}))


class TestDeadPlacementHardFilter(unittest.TestCase):
    """Integration: AI must never choose a dead placement at any difficulty."""

    def _dead_squares(self, board: BoardState) -> set:
        return {
            sq for sq in ADJACENCY
            if board.positions.get(sq, "") == ""
            and _free_neighbors(board, sq) == 0
        }

    def test_a4_game_position(self):
        """
        Regression: after W:a7,b4,g7,d6,d1 / B:d7,a1,g4,g1 (Black's 5th move)
        the AI must not place on a4 or b6 (both dead in this position).
        """
        board = BoardState.from_setup(
            {"a7": "W", "b4": "W", "g7": "W", "d6": "W", "d1": "W",
             "d7": "B", "a1": "B", "g4": "B", "g1": "B"},
            turn="B",
            phase="place",
        )
        dead = self._dead_squares(board)
        self.assertIn("a4", dead, "Test setup: a4 should be dead here")
        self.assertIn("b6", dead, "Test setup: b6 should be dead here")

        for diff in (1, 2, 3):
            ai = GameAI(color="B", difficulty=diff)
            move = ai.choose_move(board)
            self.assertIsNotNone(move)
            to = move["to"]
            self.assertNotIn(
                to, dead,
                f"difficulty={diff}: AI chose dead square {to}; dead={dead}",
            )

    def test_b6_last_piece_position(self):
        """
        Regression: when b6 has b4=B, d6=B (0 free neighbours) it must not
        be chosen even as the final placement.
        """
        # Position where b4 and d6 are Black (so b6 is dead for either side),
        # with White needing to place last piece and live alternatives exist.
        board = BoardState.from_setup(
            {"a7": "W", "g7": "W", "d1": "W", "g1": "W",
             "d7": "B", "b4": "B", "d6": "B", "g4": "B"},
            turn="W",
            phase="place",
        )
        self.assertEqual(_free_neighbors(board, "b6"), 0, "b6 must be dead here")

        for diff in (1, 2):
            ai = GameAI(color="W", difficulty=diff)
            move = ai.choose_move(board)
            self.assertIsNotNone(move)
            self.assertNotEqual(
                move["to"], "b6",
                f"difficulty={diff}: AI chose dead square b6",
            )

    def test_forced_block_at_dead_square_still_played(self):
        """
        Safety: if the only mandatory block is a dead square the filter must
        allow it through (never leave a mill threat unblocked).
        Black threatens a7-d7-g7 (has a7 and d7); g7 neighbours g4=B → dead.
        White must still block at g7.
        """
        board = BoardState.from_setup(
            {"c1": "W", "e1": "W", "d5": "W"},
            turn="W",
            phase="place",
        )
        # Manually inject Black pieces so the threat exists
        pos = dict(board.positions)
        pos["a7"] = "B"
        pos["d7"] = "B"
        pos["g4"] = "B"
        pos["b4"] = "B"
        from game.board import hash_board
        board2 = BoardState(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 3, "B": 4},
            pieces_placed={"W": 3, "B": 4},
            pieces_captured={"W": 0, "B": 0},
            hash_key=0,
        )
        board2 = board2.__class__(
            positions=pos,
            turn="W",
            pieces_on_board={"W": 3, "B": 4},
            pieces_placed={"W": 3, "B": 4},
            pieces_captured={"W": 0, "B": 0},
            hash_key=hash_board(board2),
        )
        # g7 is dead here (neighbours d7=B, g4=B) but is the only block
        self.assertEqual(_free_neighbors(board2, "g7"), 0)

        ai = GameAI(color="W", difficulty=2)
        move = ai.choose_move(board2)
        self.assertIsNotNone(move)
        self.assertEqual(
            move["to"], "g7",
            f"Forced block at dead square should still be played; got {move}",
        )


def _b79_board() -> BoardState:
    """Board from the B-79 regression game (AI vs AI level 3, 2026-06-01).

    After: 1.d6 f4, 2.g4 d2, 3.d7 d5, 4.d3 c4, 5.b4 a4, 6.g1 g7, 7.a1 d1, 8.e5 e4
    White to place 9th piece; all 8 remaining empty squares are dead.
    """
    pos = {p: "" for p in POSITIONS}
    for p in ["d6", "g4", "d7", "d3", "b4", "g1", "a1", "e5"]:
        pos[p] = "W"
    for p in ["f4", "d2", "d5", "c4", "a4", "g7", "d1", "e4"]:
        pos[p] = "B"
    from game.board import hash_board
    b = BoardState(
        positions=pos,
        turn="W",
        pieces_on_board={"W": 8, "B": 8},
        pieces_placed={"W": 8, "B": 8},
        pieces_captured={"W": 0, "B": 0},
        hash_key=0,
    )
    return b.__class__(
        positions=pos,
        turn="W",
        pieces_on_board={"W": 8, "B": 8},
        pieces_placed={"W": 8, "B": 8},
        pieces_captured={"W": 0, "B": 0},
        hash_key=hash_board(b),
    )


class TestDeadMillPotential(unittest.TestCase):
    """Unit tests for _dead_has_mill_potential (B-79 secondary filter)."""

    def test_a7_has_no_mill_potential(self):
        """a7: (a7,d7,g7) blocked by g7=B; (a1,a4,a7) blocked by a4=B → no potential."""
        board = _b79_board()
        self.assertFalse(_dead_has_mill_potential(board, "a7"))

    def test_f2_has_no_mill_potential(self):
        """f2: (f6,f4,f2) blocked by f4=B; (f2,d2,b2) blocked by d2=B → no potential."""
        board = _b79_board()
        self.assertFalse(_dead_has_mill_potential(board, "f2"))

    def test_c5_has_no_mill_potential(self):
        """c5: (c5,d5,e5) blocked by d5=B; (c3,c4,c5) blocked by c4=B → no potential."""
        board = _b79_board()
        self.assertFalse(_dead_has_mill_potential(board, "c5"))

    def test_b6_has_mill_potential(self):
        """b6: (b6,d6,f6) has d6=W, f6=empty (no opponent) → has potential."""
        board = _b79_board()
        self.assertTrue(_dead_has_mill_potential(board, "b6"))

    def test_e3_has_mill_potential(self):
        """e3: (e3,d3,c3) has d3=W, c3=empty (no opponent) → has potential."""
        board = _b79_board()
        self.assertTrue(_dead_has_mill_potential(board, "e3"))


class TestAllDeadFallback(unittest.TestCase):
    """Integration: when all placements are dead the AI must not choose a square
    with zero surviving mill potential (B-79 regression — 2026-06-01 AI vs AI game)."""

    def test_ai_avoids_fully_blocked_dead_squares(self):
        """AI must not place on a7, f2, or c5 — all opponent-blocked on every mill line."""
        board = _b79_board()
        zero_potential = {"a7", "f2", "c5"}

        for diff in (1, 2, 3):
            ai = GameAI(color="W", difficulty=diff)
            move = ai.choose_move(board)
            self.assertIsNotNone(move)
            self.assertNotIn(
                move["to"],
                zero_potential,
                f"difficulty={diff}: AI chose fully-blocked dead square {move['to']}",
            )


if __name__ == "__main__":
    unittest.main()
