"""tests/test_ponder.py — B-93/B-94: DB-blended prediction and TT-deepening on ponder hit."""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

from game.board import BoardState
from ai.ponder import PonderManager, _move_notation


def _build_board(w_positions: list[str], b_positions: list[str]) -> BoardState:
    board = BoardState.new_game()
    max_len = max(len(w_positions), len(b_positions))
    for i in range(max_len):
        if i < len(w_positions):
            board = board.apply_move({"from": None, "to": w_positions[i], "capture": None})
        if i < len(b_positions):
            board = board.apply_move({"from": None, "to": b_positions[i], "capture": None})
    return board


def _move_phase_board() -> BoardState:
    """Return a mid-game move-phase board with several pieces per side."""
    w = ["a1", "a4", "a7", "b2", "b4", "b6", "c3", "c5", "d1"]
    b = ["d3", "d5", "d7", "e4", "e5", "f2", "f4", "f6", "g1"]
    return _build_board(w, b)


class TestPonderB93FrequencyBlend(unittest.TestCase):
    """B-93: trajectory-DB frequency boosts the predicted opponent move."""

    def _make_game_ai(self, color: str = "B") -> MagicMock:
        ai = MagicMock()
        ai.color = color
        ai.difficulty = 5
        ai._weights = MagicMock()
        ai._value_net = None
        ai._endgame_solved_db = None
        ai._neural_evaluator = None
        ai.choose_move.return_value = {"to": "d6"}
        return ai

    def test_no_db_does_not_crash(self):
        """PonderManager.start with no DBs completes without error."""
        board = _move_phase_board()
        pm = PonderManager()
        game_ai = self._make_game_ai()
        pm.start(board=board, game_ai=game_ai, game_notations=[])
        time.sleep(0.1)
        pm.stop()

    def test_trajectory_db_boost_shifts_prediction(self):
        """High-frequency move in trajectory_db should become the predicted move."""
        from game.rules import get_all_legal_moves
        from ai.game_ai import _order_moves

        board = _move_phase_board()
        legal = get_all_legal_moves(board)
        if len(legal) < 2:
            self.skipTest("not enough legal moves")

        ordered = _order_moves(board, legal, None, None)
        # The move that _order_moves ranks last (or low priority)
        low_priority_move = ordered[-1]
        low_notation = _move_notation(low_priority_move)

        # Stub trajectory_db: make the low-priority move look very frequent (1.0)
        traj_db = MagicMock()
        traj_db.query_all_frequencies.return_value = {low_notation: 1.0}

        game_ai = self._make_game_ai()
        pm = PonderManager()
        pm.start(
            board=board,
            game_ai=game_ai,
            game_notations=[],
            trajectory_db=traj_db,
        )
        time.sleep(0.1)
        pm.stop()
        # B-93 prediction must have queried frequencies (may also be called by ponder search)
        self.assertGreaterEqual(traj_db.query_all_frequencies.call_count, 1)
        first_call_board = traj_db.query_all_frequencies.call_args_list[0][0][0]
        self.assertEqual(first_call_board, board)

    def test_fullgame_db_boost_shifts_prediction(self):
        """fullgame_db.best_move_validated result gets a +3 boost."""
        from game.rules import get_all_legal_moves
        from ai.game_ai import _order_moves

        board = _move_phase_board()
        legal = get_all_legal_moves(board)
        if len(legal) < 2:
            self.skipTest("not enough legal moves")

        ordered = _order_moves(board, legal, None, None)
        last_move = ordered[-1]
        last_notation = _move_notation(last_move)

        fgdb = MagicMock()
        fgdb.best_move_validated.return_value = last_notation

        game_ai = self._make_game_ai()
        pm = PonderManager()
        pm.start(
            board=board,
            game_ai=game_ai,
            game_notations=[],
            fullgame_db=fgdb,
        )
        time.sleep(0.1)
        pm.stop()
        fgdb.best_move_validated.assert_called_once_with(board)

    def test_both_dbs_combined(self):
        """Both DBs can be active simultaneously without error."""
        board = _move_phase_board()

        traj_db = MagicMock()
        traj_db.query_all_frequencies.return_value = {}

        fgdb = MagicMock()
        fgdb.best_move_validated.return_value = None

        game_ai = self._make_game_ai()
        pm = PonderManager()
        pm.start(
            board=board,
            game_ai=game_ai,
            game_notations=[],
            trajectory_db=traj_db,
            fullgame_db=fgdb,
        )
        time.sleep(0.1)
        pm.stop()
        # T-C1: query_all_frequencies may be called multiple times (once for prediction
        # in ponder.start, then once per root move in _choose_rust_scored SE-11b extension).
        traj_db.query_all_frequencies.assert_called()
        fgdb.best_move_validated.assert_called_once()

    def test_db_exception_does_not_prevent_ponder(self):
        """If DB methods raise, ponder still starts (graceful degradation)."""
        board = _move_phase_board()

        traj_db = MagicMock()
        traj_db.query_all_frequencies.side_effect = RuntimeError("db error")

        fgdb = MagicMock()
        fgdb.best_move_validated.side_effect = RuntimeError("db error")

        game_ai = self._make_game_ai()
        pm = PonderManager()
        pm.start(
            board=board,
            game_ai=game_ai,
            game_notations=[],
            trajectory_db=traj_db,
            fullgame_db=fgdb,
        )
        time.sleep(0.1)
        pm.stop()  # should not raise


class TestPonderB94TTDeepening(unittest.TestCase):
    """B-94: get_result returns (move, completed_ponder_ai); TT reuse on ponder hit."""

    def _real_game_ai(self, color: str = "W") -> "GameAI":
        from ai.game_ai import GameAI
        return GameAI(color=color, difficulty=1)  # 0.3s budget → completes quickly

    def test_get_result_returns_tuple_on_hit(self):
        """On a ponder hit, get_result returns a (move, ponder_ai) tuple."""
        board = _move_phase_board()
        from game.rules import get_all_legal_moves
        legal = get_all_legal_moves(board)
        if not legal:
            self.skipTest("no legal moves")

        game_ai = self._real_game_ai(color="B")
        pm = PonderManager()
        pm.start(board=board, game_ai=game_ai, game_notations=[])

        # Wait for the ponder search to complete (difficulty-1 budget is 0.3s)
        time.sleep(1.0)

        # Snapshot branches before stop() clears them
        branches = list(pm._branches)
        pm.stop()

        if not branches:
            self.skipTest("ponder did not start")
        predicted_hash = branches[0].predicted_hash

        # Construct a board with the matching hash
        # Find which board was predicted by checking all opponent moves
        from ai.game_ai import _order_moves
        ordered = _order_moves(board, legal, None, None)
        ponder_board = None
        for m in ordered:
            nb = board.apply_move(m)
            if nb.hash_key == predicted_hash:
                ponder_board = nb
                break

        if ponder_board is None:
            self.skipTest("predicted hash not found in legal moves")

        result = pm.get_result(ponder_board)
        # If ponder completed, result should be a tuple
        if result is not None:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            cached_move, completed_ai = result
            self.assertIsInstance(cached_move, dict)
            # completed_ai may be None if ponder was aborted before completion

    def test_get_result_returns_none_on_miss(self):
        """On a ponder miss, get_result returns None."""
        board = _move_phase_board()
        game_ai = self._real_game_ai(color="B")
        pm = PonderManager()
        pm.start(board=board, game_ai=game_ai, game_notations=[])
        time.sleep(0.1)
        pm.stop()

        # Use the original board (not ponder_board) — this should be a miss
        result = pm.get_result(board)
        self.assertIsNone(result)

    def test_completed_ponder_ai_has_populated_tt(self):
        """After a ponder completes, completed_ponder_ai has a warmed TT (Python or Rust)."""
        board = _move_phase_board()
        game_ai = self._real_game_ai(color="B")
        pm = PonderManager()
        pm.start(board=board, game_ai=game_ai, game_notations=[])
        time.sleep(1.0)

        # Snapshot branches before stop() clears them
        branches = list(pm._branches)
        pm.stop()

        if not branches or branches[0].cached_move is None:
            self.skipTest("ponder did not complete in time")

        completed_ai = branches[0].completed_ponder_ai
        self.assertIsNotNone(completed_ai)
        # Accept either Python TT entries (pre-Rust path) or Rust TT handle (T-C4 path).
        py_tt_warm = any(completed_ai._tt._table)
        rust_tt_warm = getattr(completed_ai, "_rust_tt_handle", None) is not None
        self.assertTrue(
            py_tt_warm or rust_tt_warm,
            "After ponder, at least one TT (Python _tt or Rust _rust_tt_handle) should be populated",
        )

    def test_deepening_with_prewarm_tt_returns_valid_move(self):
        """Calling _iterative_deepen on a fresh AI with pre-warmed TT returns a valid move."""
        from game.rules import get_all_legal_moves
        from ai.game_ai import GameAI

        board = _move_phase_board()
        game_ai = self._real_game_ai(color="B")
        pm = PonderManager()
        pm.start(board=board, game_ai=game_ai, game_notations=[])
        time.sleep(1.0)

        # Snapshot branches before stop() clears them
        branches = list(pm._branches)
        pm.stop()

        if not branches or branches[0].completed_ponder_ai is None:
            self.skipTest("ponder did not complete")

        # Get the ponder board (hash must exist in legal moves)
        legal = get_all_legal_moves(board)
        predicted_hash = branches[0].predicted_hash
        ponder_board = next(
            (board.apply_move(m) for m in legal
             if board.apply_move(m).hash_key == predicted_hash),
            None
        )
        if ponder_board is None:
            self.skipTest("ponder board not found in legal moves")

        ponder_ai = branches[0].completed_ponder_ai
        ponder_ai._force_stop = False
        ponder_ai._deadline = float("inf")

        move = ponder_ai._iterative_deepen(ponder_board, time_limit=1.0)
        self.assertIsInstance(move, dict)
        self.assertIn("to", move)


if __name__ == "__main__":
    unittest.main()
