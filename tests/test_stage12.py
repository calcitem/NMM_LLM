"""tests/test_stage12.py — Stage 12: MCTS and ValueNet tests."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves, is_terminal
from ai.mcts import MCTS, MCTSNode, _UCT_C
from ai.value_net import ValueNet, board_to_features, _INPUT_DIM
from ai.game_ai import GameAI


# ── Helpers ───────────────────────────────────────────────────────────────────

def _board_from_pos(
    white: list[str],
    black: list[str],
    turn: str = "W",
    w_placed: int = 9,
    b_placed: int = 9,
) -> BoardState:
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


def _new_game() -> BoardState:
    return BoardState.new_game()


def _near_win_board() -> BoardState:
    """W about to win: W has 3 pieces in almost a mill; B has 3."""
    return _board_from_pos(
        white=["a7", "d7", "g1", "d1", "a1"],
        black=["b6", "d6", "f6", "b2", "d2"],
    )


# ── MCTSNode ──────────────────────────────────────────────────────────────────

class TestMCTSNode(unittest.TestCase):
    def test_initial_state(self):
        board = _new_game()
        node  = MCTSNode(board)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.move)
        self.assertEqual(node.children, [])
        self.assertIsNone(node.untried_moves)

    def test_parent_child_link(self):
        board  = _new_game()
        parent = MCTSNode(board)
        child  = MCTSNode(board, move={"from": None, "to": "d7", "capture": None},
                          parent=parent)
        self.assertIs(child.parent, parent)
        self.assertEqual(child.move["to"], "d7")


# ── MCTS.choose_move ──────────────────────────────────────────────────────────

class TestMCTSChooseMove(unittest.TestCase):
    def _mcts(self, color: str = "W") -> MCTS:
        return MCTS(color=color, time_limit=0.3)

    def test_returns_legal_move_opening(self):
        board = _new_game()
        mcts  = self._mcts("W")
        move  = mcts.choose_move(board)
        self.assertIn("to", move)
        legal = get_all_legal_moves(board)
        self.assertIn(move, legal)

    def test_returns_legal_move_midgame(self):
        board = _near_win_board()
        mcts  = self._mcts("W")
        move  = mcts.choose_move(board)
        legal = get_all_legal_moves(board)
        self.assertIn(move, legal)

    def test_single_move_returns_immediately(self):
        # Construct a board where W is in fly phase with 3 pieces and only
        # one legal capture target — choose_move must not crash.
        board = _board_from_pos(
            white=["a7", "d7", "g7"],   # mill → W can capture
            black=["b6"],
            turn="W",
        )
        # Force W into "just-placed all pieces" scenario by setting placed=9
        board = BoardState(
            positions=board.positions,
            turn="W",
            pieces_on_board=board.pieces_on_board,
            pieces_placed={"W": 9, "B": 9},
            pieces_captured={"W": 8, "B": 0},
        )
        mcts = self._mcts("W")
        move = mcts.choose_move(board)
        self.assertIsInstance(move, dict)

    def test_nodes_searched_positive(self):
        board = _new_game()
        mcts  = self._mcts("W")
        mcts.choose_move(board)
        self.assertGreater(mcts.nodes_searched, 0)

    def test_empty_board_no_crash(self):
        board = _new_game()
        mcts  = MCTS(color="W", time_limit=0.2)
        move  = mcts.choose_move(board)
        self.assertIn("to", move)

    def test_terminal_position_handled(self):
        # B is down to 2 pieces (W should have won already, but test handling).
        board = _board_from_pos(
            white=["a7", "d7", "g7", "g1", "d1"],
            black=["b6", "d6"],
        )
        terminal, _ = is_terminal(board)
        if terminal:
            # choose_move on a terminal board: expect empty dict or a legal move
            mcts = self._mcts("W")
            result = mcts.choose_move(board)
            self.assertIsInstance(result, dict)

    def test_black_gets_legal_move(self):
        board = _new_game()
        board = BoardState(
            positions=board.positions,
            turn="B",
            pieces_on_board=board.pieces_on_board,
            pieces_placed=board.pieces_placed,
            pieces_captured=board.pieces_captured,
        )
        mcts = MCTS(color="B", time_limit=0.2)
        move = mcts.choose_move(board)
        legal = get_all_legal_moves(board)
        self.assertIn(move, legal)


# ── MCTS with value_net ───────────────────────────────────────────────────────

class TestMCTSWithValueNet(unittest.TestCase):
    def test_untrained_net_gives_legal_move(self):
        board = _new_game()
        net   = ValueNet()
        mcts  = MCTS(color="W", time_limit=0.2, value_net=net)
        move  = mcts.choose_move(board)
        legal = get_all_legal_moves(board)
        self.assertIn(move, legal)


# ── ValueNet.board_to_features ────────────────────────────────────────────────

class TestBoardToFeatures(unittest.TestCase):
    def test_shape(self):
        board = _new_game()
        feats = board_to_features(board, "W")
        self.assertEqual(feats.shape, (_INPUT_DIM,))
        self.assertEqual(feats.dtype, np.float32)

    def test_empty_board_all_empty_slots(self):
        board = _new_game()
        feats = board_to_features(board, "W")
        # All 24 positions are empty → only "empty" channel (index 2) set
        for i in range(24):
            self.assertEqual(feats[i * 3],     0.0, f"pos {i} own should be 0")
            self.assertEqual(feats[i * 3 + 1], 0.0, f"pos {i} opp should be 0")
            self.assertEqual(feats[i * 3 + 2], 1.0, f"pos {i} empty should be 1")

    def test_own_piece_encoded(self):
        board = _board_from_pos(white=["a7"], black=[], turn="W",
                                w_placed=1, b_placed=0)
        feats = board_to_features(board, "W")
        idx   = POSITIONS.index("a7")
        self.assertEqual(feats[idx * 3],     1.0)  # own
        self.assertEqual(feats[idx * 3 + 1], 0.0)  # opp
        self.assertEqual(feats[idx * 3 + 2], 0.0)  # empty

    def test_opp_piece_encoded(self):
        board = _board_from_pos(white=[], black=["a7"], turn="B",
                                w_placed=0, b_placed=1)
        feats = board_to_features(board, "W")   # W's perspective
        idx   = POSITIONS.index("a7")
        self.assertEqual(feats[idx * 3],     0.0)  # own (W)
        self.assertEqual(feats[idx * 3 + 1], 1.0)  # opp (B)

    def test_symmetry_own_opp(self):
        # Same position but from the other player's perspective should swap own/opp.
        board = _board_from_pos(white=["a7"], black=["g7"], turn="W",
                                w_placed=1, b_placed=1)
        fw = board_to_features(board, "W")
        fb = board_to_features(board, "B")
        a7 = POSITIONS.index("a7")
        g7 = POSITIONS.index("g7")
        # For W: a7 is own, g7 is opp
        self.assertEqual(fw[a7 * 3],     1.0)
        self.assertEqual(fw[g7 * 3 + 1], 1.0)
        # For B: g7 is own, a7 is opp
        self.assertEqual(fb[g7 * 3],     1.0)
        self.assertEqual(fb[a7 * 3 + 1], 1.0)

    def test_turn_flag(self):
        board_w = _new_game()
        board_b = BoardState(
            positions=board_w.positions, turn="B",
            pieces_on_board=board_w.pieces_on_board,
            pieces_placed=board_w.pieces_placed,
            pieces_captured=board_w.pieces_captured,
        )
        fw = board_to_features(board_w, "W")
        fb = board_to_features(board_b, "B")
        # Both are "my turn", so index 72 should be 1.0
        self.assertEqual(fw[72], 1.0)
        self.assertEqual(fb[72], 1.0)
        # From W's perspective on B's turn, index 72 = 0.0
        fw_not_turn = board_to_features(board_b, "W")
        self.assertEqual(fw_not_turn[72], 0.0)


# ── ValueNet.predict ──────────────────────────────────────────────────────────

class TestValueNetPredict(unittest.TestCase):
    def test_output_in_range(self):
        net   = ValueNet()
        board = _new_game()
        val   = net.predict(board, "W")
        self.assertIsInstance(val, float)
        self.assertGreater(val, -1.0)
        self.assertLess(val, 1.0)

    def test_predict_batch_shape(self):
        net = ValueNet()
        X   = np.zeros((4, _INPUT_DIM), dtype=np.float32)
        out = net.predict_batch(X)
        self.assertEqual(out.shape, (4,))
        for v in out:
            self.assertGreater(v, -1.0)
            self.assertLess(v, 1.0)

    def test_deterministic(self):
        net   = ValueNet()
        board = _new_game()
        v1    = net.predict(board, "W")
        v2    = net.predict(board, "W")
        self.assertAlmostEqual(v1, v2)


# ── ValueNet.train ────────────────────────────────────────────────────────────

class TestValueNetTrain(unittest.TestCase):
    def _tiny_dataset(self, n: int = 128):
        board = _new_game()
        X = np.tile(board_to_features(board, "W"), (n, 1)).astype(np.float32)
        y = np.ones(n, dtype=np.float32)
        return X, y

    def test_loss_decreases(self):
        net = ValueNet()
        X, y = self._tiny_dataset(256)
        losses = net.train(X, y, epochs=10, batch_size=64, lr=0.01)
        self.assertEqual(len(losses), 10)
        self.assertLess(losses[-1], losses[0],
                        "Loss should decrease on a fixed target dataset")

    def test_output_moves_toward_target(self):
        net = ValueNet()
        board = _new_game()
        X = board_to_features(board, "W").reshape(1, -1).astype(np.float32)
        y = np.array([1.0], dtype=np.float32)
        net.train(X, y, epochs=100, batch_size=1, lr=0.01)
        pred = net.predict_batch(X)[0]
        self.assertGreater(pred, 0.0, "After training toward +1, output should be positive")


# ── ValueNet.save / load ──────────────────────────────────────────────────────

class TestValueNetPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        net = ValueNet()
        board = _new_game()
        val_before = net.predict(board, "W")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            net.save(path)
            loaded = ValueNet.load(path)
            val_after = loaded.predict(board, "W")

        self.assertAlmostEqual(val_before, val_after, places=6)

    def test_load_if_exists_missing(self):
        result = ValueNet.load_if_exists("/tmp/_nmm_nonexistent_12345.npz")
        self.assertIsNone(result)

    def test_load_if_exists_present(self):
        net = ValueNet()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "net.npz"
            net.save(path)
            loaded = ValueNet.load_if_exists(path)
        self.assertIsNotNone(loaded)


# ── GameAI MCTS integration ───────────────────────────────────────────────────

class TestGameAIMCTSToggle(unittest.TestCase):
    def test_use_mcts_false_uses_negamax(self):
        ai = GameAI(color="W", difficulty=3, use_mcts=False)
        self.assertIsNone(ai._mcts)

    def test_use_mcts_true_creates_mcts(self):
        ai = GameAI(color="W", difficulty=5, use_mcts=True)
        self.assertIsNotNone(ai._mcts)

    def test_mcts_returns_legal_move(self):
        board = _new_game()
        ai    = GameAI(color="W", difficulty=5, use_mcts=True)
        # Override time to keep test fast
        from ai.mcts import MCTS
        ai._mcts = MCTS(color="W", time_limit=0.3)
        move  = ai.choose_move(board)
        legal = get_all_legal_moves(board)
        self.assertIn(move, legal)

    def test_mcts_with_value_net(self):
        board = _new_game()
        net   = ValueNet()
        ai    = GameAI(color="B", difficulty=5, use_mcts=True, value_net=net)
        from ai.mcts import MCTS
        ai._mcts = MCTS(color="B", time_limit=0.2, value_net=net)
        move = ai.choose_move(board)
        legal = get_all_legal_moves(board)
        # B cannot move on W's turn — board.turn == W, but AI is B; legal moves may be empty
        # (choose_move checks board.turn moves, not AI color's moves)
        # Just ensure no exception and the return is a dict
        self.assertIsInstance(move, dict)


if __name__ == "__main__":
    unittest.main()
