"""tests/test_ai.py — Stage 2: heuristics and GameAI acceptance tests."""

import time
import unittest

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves
from ai.heuristics import evaluate, INF, _closed_mills, _blocked_count, _two_configs, _double_mills
from ai.game_ai import GameAI


# ── Helper ────────────────────────────────────────────────────────────────────

def _board_from_pos(white: list[str], black: list[str], turn: str = "W",
                    w_placed: int = 9, b_placed: int = 9) -> BoardState:
    pos = {p: "" for p in POSITIONS}
    for p in white:
        pos[p] = "W"
    for p in black:
        pos[p] = "B"
    return BoardState(
        positions=pos,
        turn=turn,
        pieces_on_board={"W": len(white), "B": len(black)},
        pieces_placed={"W": w_placed, "B": b_placed},
        pieces_captured={"W": 0, "B": 0},
    )


# ── Heuristic unit tests ──────────────────────────────────────────────────────

class TestClosedMills(unittest.TestCase):
    def test_outer_top_mill(self):
        b = _board_from_pos(["a7", "d7", "g7"], [])
        self.assertEqual(_closed_mills(b, "W"), 1)

    def test_no_mills(self):
        b = _board_from_pos(["a7", "d7"], ["g7"])
        self.assertEqual(_closed_mills(b, "W"), 0)

    def test_two_mills(self):
        # Outer top + outer bottom
        b = _board_from_pos(["a7", "d7", "g7", "g1", "d1", "a1"], [])
        self.assertEqual(_closed_mills(b, "W"), 2)


class TestTwoConfigs(unittest.TestCase):
    def test_one_open_mill(self):
        # Two W at a7, d7; g7 empty → one potential mill
        b = _board_from_pos(["a7", "d7"], [])
        self.assertEqual(_two_configs(b, "W"), 1)

    def test_blocked_by_opponent(self):
        # W at a7, d7; B at g7 → not an open mill for W
        b = _board_from_pos(["a7", "d7"], ["g7"])
        self.assertEqual(_two_configs(b, "W"), 0)


class TestDoubleMillPivot(unittest.TestCase):
    def test_no_double_mills(self):
        b = _board_from_pos(["a7", "d7", "g7"], [])
        self.assertEqual(_double_mills(b, "W"), 0)

    def test_pivot_piece(self):
        # d7 is in (a7,d7,g7) and (d7,d6,d5) — if all 3 are filled
        b = _board_from_pos(["a7", "d7", "g7", "d6", "d5"], [])
        self.assertEqual(_double_mills(b, "W"), 1)  # d7 is the pivot


class TestBlockedCount(unittest.TestCase):
    def test_surrounded_piece(self):
        # a7's neighbours are d7 and a4. Both occupied by Black → a7 blocked.
        # g7, g1, d1 each have at least one empty neighbour → not blocked.
        # 4 W pieces → move phase (pieces_placed=9 but pieces_on_board=4 > 3).
        b = _board_from_pos(
            ["a7", "g7", "g1", "d1"],
            ["d7", "a4"],
            turn="W", w_placed=9, b_placed=9,
        )
        self.assertEqual(_blocked_count(b, "W"), 1)

    def test_fly_phase_never_blocked(self):
        # In fly phase (≤3 pieces after placing 9) pieces are never blocked.
        fly_board = BoardState(
            positions={p: ("W" if p == "d5" else ("B" if p in ("c5", "e5", "d6") else ""))
                       for p in POSITIONS},
            turn="W",
            pieces_on_board={"W": 1, "B": 3},
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 0, "B": 0},
        )
        self.assertEqual(_blocked_count(fly_board, "W"), 0)


class TestEvaluateTerminal(unittest.TestCase):
    def test_win_returns_positive_inf(self):
        # B has only 2 pieces after placing 9 → W wins
        b = _board_from_pos(["a7", "d7", "g7"], ["b6", "b4"],
                             turn="W", w_placed=9, b_placed=9)
        score = evaluate(b, "W")
        self.assertEqual(score, INF)

    def test_loss_returns_negative_inf(self):
        b = _board_from_pos(["a7", "d7", "g7"], ["b6", "b4"],
                             turn="W", w_placed=9, b_placed=9)
        score = evaluate(b, "B")
        self.assertEqual(score, -INF)


# ── GameAI integration tests ──────────────────────────────────────────────────

class TestGameAIChooseMove(unittest.TestCase):
    def test_picks_legal_move(self):
        board = BoardState.new_game()
        ai = GameAI(color="W", difficulty=2)
        move = ai.choose_move(board)
        legal = get_all_legal_moves(board)
        self.assertIn(move, legal)

    def test_difficulty_3_within_time(self):
        board = BoardState.new_game()
        ai = GameAI(color="W", difficulty=3)
        start = time.time()
        move = ai.choose_move(board)
        elapsed = time.time() - start
        self.assertIn(move, get_all_legal_moves(board))
        self.assertLess(elapsed, 5.0, "Difficulty 3 must respond in under 5 seconds")

    def test_completes_obvious_mill(self):
        # W at a7, d7; B has two isolated pieces (b6, g1) that cannot reform a mill.
        # Placing at g7 completes mill (a7,d7,g7) and captures one Black piece,
        # reducing B to 1 piece — a large advantage vs any other placement.
        b = _board_from_pos(
            ["a7", "d7"],
            ["b6", "g1"],       # B pieces not connected by any mill together
            turn="W",
            w_placed=2,
            b_placed=2,
        )
        ai = GameAI(color="W", difficulty=1)   # depth=2: shallow enough to prefer immediate gains
        move = ai.choose_move(b)
        self.assertEqual(move["to"], "g7", "AI should complete the mill at g7")

    def test_score_move_blunder_vs_optimal(self):
        # Same position. Completing mill at g7 + capture must score higher than
        # a random corner placement with no mill and no capture.
        b = _board_from_pos(
            ["a7", "d7"],
            ["b6", "g1"],
            turn="W",
            w_placed=2,
            b_placed=2,
        )
        ai = GameAI(color="W", difficulty=2)
        # g7 forms a mill → must include a capture; g1 is already Black's → use b6
        optimal = {"from": None, "to": "g7", "capture": "b6"}
        blunder  = {"from": None, "to": "c5", "capture": None}   # inner corner, no mill
        score_opt = ai.score_move(b, optimal)
        score_bad = ai.score_move(b, blunder)
        self.assertGreater(score_opt, score_bad,
                           "Completing the mill must outscore a random placement")


class TestBlunderMode(unittest.TestCase):
    def test_blunder_always_plays_bad_with_prob_1(self):
        board = BoardState.new_game()
        ai = GameAI(color="W", difficulty=2, blunder_probability=1.0)
        move = ai.choose_move(board)
        self.assertTrue(ai.last_was_blunder)
        self.assertIn(move, get_all_legal_moves(board))

    def test_no_blunder_with_prob_0(self):
        board = BoardState.new_game()
        ai = GameAI(color="W", difficulty=2, blunder_probability=0.0)
        ai.choose_move(board)
        self.assertFalse(ai.last_was_blunder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
