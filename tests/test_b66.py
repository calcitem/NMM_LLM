"""
tests/test_b66.py — Regression for B-66: prefer own mill close over passive block.

Game trace (aggressive Black, after 9... d2-b2):
  White threatens b6 (b2-b4-b6).  Black can close c3-c4-c5 via d3-c3 (+ capture).
  d3-d2 does NOT form a mill; the bug report misidentified the target square.

Before fix, mandatory block restricted candidates to {to: b6}, so choose_move
played d6-b6.  After fix, Black closes the c-line mill.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from game.board import BoardState
from game.notation import parse_move_string
from game.rules import get_all_legal_moves
from ai.game_ai import GameAI, _immediate_mill_threats
from ai.heuristics import HeuristicWeights

_ROOT = Path(__file__).resolve().parent.parent
_B66_MOVES = [
    "d2", "d6", "d7", "g4", "d1", "d3", "b4", "a1", "a4", "c4",
    "f6", "g1", "g7", "a7", "f4", "f2", "d5", "c5", "d2-b2",
]


def _replay(moves: list[str]) -> BoardState:
    board = BoardState.new_game()
    for tok in moves:
        spec = parse_move_string(tok)
        legal = get_all_legal_moves(board)
        move = next(
            (
                m for m in legal
                if m.get("from") == spec.get("from")
                and m["to"] == spec["to"]
                and m.get("capture") == spec.get("capture")
            ),
            None,
        )
        if move is None:
            raise ValueError(f"illegal move in replay: {tok}")
        board = board.apply_move(move)
    return board


def _aggressive_weights() -> HeuristicWeights:
    with open(_ROOT / "data/personalities/aggressive.json", encoding="utf-8") as fh:
        wdict = json.load(fh)
    hw = HeuristicWeights()
    for key, val in wdict.items():
        if hasattr(hw, key):
            setattr(hw, key, val)
    return hw


class TestB66MillCloseVsBlock(unittest.TestCase):
    def test_position_after_white_d2_b2(self):
        board = _replay(_B66_MOVES)
        self.assertEqual(board.turn, "B")
        # White threatens b6; Black has c4-c5 with empty c3.
        self.assertEqual(_immediate_mill_threats(board), set())

    def test_aggressive_black_closes_c_line_mill(self):
        board = _replay(_B66_MOVES)
        ai = GameAI(color="B", difficulty=5, weights=_aggressive_weights())
        move = ai.choose_move(board)
        self.assertEqual(move.get("from"), "d3")
        self.assertEqual(move["to"], "c3")
        self.assertIsNotNone(move.get("capture"))

    def test_d3_d2_is_not_the_mill_close(self):
        """d3-d2 was cited in the report but does not close a mill on this board."""
        from ai.heuristics import _closed_mills

        board = _replay(_B66_MOVES)
        after = board.apply_move({"from": "d3", "to": "d2", "capture": None})
        self.assertEqual(_closed_mills(after, "B"), 0)


_TACTICAL_MOVES = [
    "d6", "d2", "f4", "b4", "c4", "e4", "d3", "d5", "a4", "d7",
    "d1", "e5", "e3", "c3", "c5", "a7", "g7", "b6",
    "d1-g1",  # White move 10 — now Black to move
]


class TestMillCloseVsPassiveSlide(unittest.TestCase):
    """Regression: Black should close b2-b4-b6 via d2-b2, not slide b4-b2.

    After White plays d1-g1 (threatening g1-g4-g7 next turn), Black has
    b4+b6 in place with b2 empty and d2 adjacent to b2.  The B-66 move-phase
    carveout must clear the single-g4 threat so choose_move can close the mill.
    """

    def test_no_mandatory_block_when_own_mill_available(self):
        board = _replay(_TACTICAL_MOVES)
        self.assertEqual(board.turn, "B")
        # B-66 carveout should have cleared the g4 threat.
        self.assertEqual(_immediate_mill_threats(board), set())

    def test_choose_move_closes_b_line_mill(self):
        board = _replay(_TACTICAL_MOVES)
        ai = GameAI(color="B", difficulty=4)
        move = ai.choose_move(board)
        self.assertEqual(move.get("from"), "d2")
        self.assertEqual(move["to"], "b2")
        self.assertIsNotNone(move.get("capture"))


if __name__ == "__main__":
    unittest.main()
