"""
Cycling-mill exception: when STM has a closed mill and the opponent has exactly
one immediate threat, the AI must be allowed to cycle (open) the closed mill
rather than being forced to block.

Reference game (move 14, Black to move):
  1.d6 d2 / 2.f4 b4 / 3.f6 b6 / 4.f2xb6 b6 / 5.d3 a7
  6.d7 g7 / 7.d5xg7 g7 / 8.d1 g4 / 9.a1 g1xd1
  10.a1-d1 d2-b2xd1 / 11.f2-d2 g1-d1 / 12.d2-f2xg4 a7-a4
  13.f2-d2 d1-g1 / 14.d5-c5 [Black to move]

Board after White 14.d5-c5:
  White: d6, f4, f6, d2, d3, d7, c5
  Black: b2, b4, b6, a4, g7, g1

White's only threat: d2→f2 closes f4-f6-f2 mill.
Black has closed mill b2-b4-b6.  AI was incorrectly forced to block f2;
it should instead be free to cycle the b-mill (e.g. b4→c4) and re-close
next turn for a capture — especially since the resulting diamond formation
(b2, a4, c4, b6) retains mill-closing ability even after White's capture.
"""
import pytest
from game.board import BoardState
from ai.game_ai import GameAI, _immediate_mill_threats, _stm_can_close_mill


TURN14_POS = {
    "d6": "W", "f4": "W", "f6": "W", "d2": "W", "d3": "W", "d7": "W", "c5": "W",
    "b2": "B", "b4": "B", "b6": "B", "a4": "B", "g7": "B", "g1": "B",
}


@pytest.fixture
def turn14_board():
    return BoardState.from_setup(TURN14_POS, "B", "move")


class TestCyclingMillDetection:
    def test_threats_include_f2(self, turn14_board):
        """White's f4-f6-f2 (closeable by d2→f2) must be a detected threat."""
        threats = _immediate_mill_threats(turn14_board)
        assert "f2" in threats

    def test_no_threat_reachable_by_black(self, turn14_board):
        """After the in-mill adjacency fix, White's threats are {f2, d5}.
        Neither is adjacent to any Black piece (f2 neighbors are f4/d2 both
        White; d5 neighbors are c5/d6 both White plus e5 empty).
        Black cannot block either threat, so the blocking filter produces no
        base-blocking moves — the filter is effectively driven entirely by
        the B-66 mill-close carveout (g4 and c4)."""
        from game.rules import get_all_legal_moves
        threats = _immediate_mill_threats(turn14_board)
        legal = get_all_legal_moves(turn14_board)
        reachable = {m["to"] for m in legal if m["to"] in threats}
        assert len(reachable) == 0, f"expected no reachable threats, got {reachable}"
        assert "f2" in threats
        assert "d5" in threats

    def test_b_mill_is_closed(self, turn14_board):
        """Black's b2-b4-b6 is a complete closed mill (no 2-config in that mill).
        Black does have a separate g-mill 2-config (g1-[g4]-g7), so B-66 fires
        and allows g7→g4 — but that is the inferior move.  The cycling exception
        must also allow moves from the closed b-mill (b4→c4 etc.)."""
        from game.board import MILLS
        b_mill_is_full = all(
            turn14_board.positions[p] == "B"
            for p in next(m for m in MILLS if set(m) == {"b2", "b4", "b6"})
        )
        assert b_mill_is_full, "b2-b4-b6 should be fully occupied by Black"


class TestCyclingExceptionAllowedMoves:
    def test_cycling_moves_in_candidates(self, turn14_board):
        """After the cycling exception, choose_move candidates must include
        at least one move FROM a b-mill piece (b2, b4, b6)."""
        ai = GameAI(difficulty=5)
        ai.color = "B"
        # Inject a hook to observe the candidate list after filtering.
        # We can indirectly verify by asserting the chosen move is from b-mill OR to f2.
        move = ai.choose_move(turn14_board)
        frm = move.get("from", "")
        to  = move.get("to", "")
        assert frm in {"b2", "b4", "b6"} or to == "f2", (
            f"Expected cycling (from b2/b4/b6) or blocking (to f2), got {frm}->{to}"
        )

    def test_ai_prefers_cycling_over_block(self, turn14_board):
        """At level 10, AI should prefer cycling the b-mill over passively
        blocking f2, because the diamond formation survives any single capture."""
        ai = GameAI(difficulty=10)
        ai.color = "B"
        move = ai.choose_move(turn14_board)
        frm = move.get("from", "")
        assert frm in {"b2", "b4", "b6"}, (
            f"Expected cycling move from closed b-mill (b2/b4/b6), got {frm}->{move.get('to')}"
        )
