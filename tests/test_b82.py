"""
B-82: Mill-closing suppressed by multi-threat filter.

Two bugs:
  1. _pinned_move_squares fires a false positive when the adjacent opp piece is
     already inside the mill (can't slide to the vacated square and complete it).
  2. B-66 carveout (allow mill-close even with opp threats) was restricted to
     len(threats)==1; with 3 threats it was skipped and g4→g7 was hard-filtered.

Reference game:
  1.b2 f4 / 2.b6 b4 / 3.d6 f6 / 4.f2 d2 / 5.d7 d5 / 6.d3 e4
  7.g4 c5  / 8.e5 c4 / 9.a4 c3xd3 / 10.a4-a7 c3-d3 / (White to move)

At move 11 White has a7-d7-g7 as a 2-config (close at g7 via g4→g7).
Black has 3 threats: {a4, c3, d1}.  AI must be allowed to close the mill.
"""
import pytest
from game.board import BoardState
from ai.game_ai import (
    GameAI, _immediate_mill_threats, _pinned_move_squares, _stm_can_close_mill,
)


# Move-11 board: White to move; owns 2-config (a7-d7-g7), close at g7 via g4→g7.
MOVE11_POS = {
    "b2":"W","b6":"W","d6":"W","f2":"W","d7":"W","g4":"W","e5":"W","a7":"W",
    "f4":"B","b4":"B","f6":"B","d2":"B","d5":"B","e4":"B","c5":"B","c4":"B","d3":"B",
}


@pytest.fixture
def move11_board():
    return BoardState.from_setup(MOVE11_POS, "W", "move")


class TestPinnedMoveSquaresFix:
    def test_g4_not_falsely_pinned(self, move11_board):
        """g4 blocked (g4-f4-e4) but f4 is internal — no external Black piece
        adjacent to g4 can slide in, so g4 must NOT be marked as pinned."""
        pinned = _pinned_move_squares(move11_board, "W")
        assert "g4" not in pinned

    def test_f2_still_pinned(self, move11_board):
        """f2 is legitimately pinned: mill (f6-f4-f2) has 2 Black + f2=White,
        and d2=B (external) is adjacent to f2 — can slide in after f2 moves."""
        pinned = _pinned_move_squares(move11_board, "W")
        assert "f2" in pinned

    def test_e5_still_pinned(self, move11_board):
        """e5 is legitimately pinned: mill (c5-d5-e5) has 2 Black + e5=White,
        and e4=B (external) is adjacent to e5."""
        pinned = _pinned_move_squares(move11_board, "W")
        assert "e5" in pinned


class TestMultiThreatMillCloseCarveout:
    def test_threats_cleared_by_b66(self, move11_board):
        """After the in-mill adjacency fix (_immediate_mill_threats now excludes
        in-mill pieces when checking adjacency), d1 and a4 are no longer false
        positives (their only adjacent Black pieces are inside the respective mills).
        The one real threat — c3, closeable by d3→c3 — is then cleared by the
        B-66 single-threat carveout (White can close a7-d7-g7 this turn).
        Result: threats = {}, White is free to close the mill."""
        threats = _immediate_mill_threats(move11_board)
        assert len(threats) == 0, f"expected 0 after B-66 clears sole real threat, got {threats}"

    def test_stm_can_close_mill(self, move11_board):
        """White can close (a7-d7-g7) this turn — prerequisite for the B-66 carveout."""
        assert _stm_can_close_mill(move11_board, "W")


class TestAIChoosesMillClose:
    @pytest.mark.parametrize("difficulty", [5, 7, 10])
    def test_ai_closes_mill_not_blocks(self, move11_board, difficulty):
        """AI must play a mill-closing move (to g7) rather than a pure blocking move."""
        ai = GameAI(difficulty=difficulty)
        ai.color = "W"
        move = ai.choose_move(move11_board)
        assert move["to"] == "g7", (
            f"difficulty={difficulty}: expected move to g7 (close mill a7-d7-g7), "
            f"got {move.get('from','?')}->{move['to']}"
        )
