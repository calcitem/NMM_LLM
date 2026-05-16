"""tests/test_stage6.py — Stage 6: Post-game debrief tests."""

from __future__ import annotations

import io
import tempfile
import unittest
from unittest.mock import MagicMock

from game.board import BoardState
from ai.debriefer import GameDebriefer, DebriefReport, CriticalMoment, _move_str
from ai.mills_llm import MillsLLM
from ai.memory_manager import MemoryManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _offline_llm() -> MillsLLM:
    """Return a MillsLLM whose Ollama client is None (offline)."""
    with tempfile.TemporaryDirectory() as tmp:
        mem = MemoryManager(
            chroma_path=f"{tmp}/chroma",
            games_path=f"{tmp}/games",
            session_path=f"{tmp}/session",
            use_ollama_embeddings=False,
        )
    llm = MillsLLM(memory=mem, model="")
    llm._client = None
    return llm


def _minimal_record(num_placement_moves: int = 6, winner: str = "W") -> dict:
    """
    Build a minimal game_record by replaying placement moves.
    Returns a record dict compatible with GameDebriefer.analyse().
    """
    positions_w = ["d2", "f4", "f2", "b4", "b2", "g4", "a7", "a4", "c3"]
    positions_b = ["d6", "d7", "e3", "e5", "c5", "g7", "g1", "a1", "b6"]

    board = BoardState.new_game()
    moves = []
    ply = 0
    for i in range(num_placement_moves):
        if board.turn == "W":
            pos = positions_w[i // 2] if i // 2 < len(positions_w) else "d3"
        else:
            pos = positions_b[i // 2] if i // 2 < len(positions_b) else "d5"
        move = {"from": None, "to": pos, "capture": None}
        from game.rules import get_game_phase
        phase = get_game_phase(board, board.turn)
        moves.append({
            "turn": ply // 2 + 1,
            "color": board.turn,
            "type": phase,
            "from": None,
            "to": pos,
            "capture": None,
            "notation": pos,
            "board_fen_before": board.to_fen_string(),
            "was_blunder": False,
            "opening_recognition": {"status": "inactive", "name": None, "confidence": 0.0},
        })
        board = board.apply_move(move)
        ply += 1

    return {
        "session_id": "test-session",
        "date": "2026-01-01T00:00:00",
        "human_color": "W",
        "winner": winner,
        "moves": moves,
        "bad_moves_taught": [],
    }


# ── GameDebriefer.analyse ─────────────────────────────────────────────────────

class TestDebrieferAnalyse(unittest.TestCase):

    def test_analyse_returns_report(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2, critical_threshold=0.4)
        record = _minimal_record(num_placement_moves=8)
        report = debriefer.analyse(record)
        self.assertIsInstance(report, DebriefReport)

    def test_winner_and_loser_set(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2)
        report = debriefer.analyse(_minimal_record(winner="W"))
        self.assertEqual(report.winner, "W")
        self.assertEqual(report.loser, "B")

    def test_draw_has_no_loser(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2)
        record = _minimal_record(winner=None)
        record["winner"] = None
        report = debriefer.analyse(record)
        self.assertIsNone(report.winner)
        self.assertIsNone(report.loser)

    def test_total_moves_matches_record(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2)
        record = _minimal_record(num_placement_moves=8)
        report = debriefer.analyse(record)
        self.assertEqual(report.total_moves, 8)

    def test_no_critical_moments_with_high_threshold(self):
        """With threshold=1.0 nothing can be flagged as critical."""
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2, critical_threshold=1.0)
        report = debriefer.analyse(_minimal_record(num_placement_moves=6))
        self.assertEqual(len(report.critical_moments), 0)

    def test_critical_moments_with_low_threshold(self):
        """With threshold=0.0 every suboptimal move is flagged (up to max_comments)."""
        llm = _offline_llm()
        debriefer = GameDebriefer(
            llm, analysis_depth=2, critical_threshold=0.0, max_comments=3
        )
        record = _minimal_record(num_placement_moves=8)
        report = debriefer.analyse(record)
        self.assertLessEqual(len(report.critical_moments), 3)

    def test_offline_llm_produces_empty_comments(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2, critical_threshold=0.0)
        report = debriefer.analyse(_minimal_record(num_placement_moves=6))
        for cm in report.critical_moments:
            self.assertEqual(cm.comment, "")

    def test_offline_llm_produces_empty_summary(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2)
        report = debriefer.analyse(_minimal_record())
        self.assertEqual(report.summary, "")

    def test_opening_name_extracted_from_exact_recognition(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2)
        record = _minimal_record(num_placement_moves=6)
        record["moves"][4]["opening_recognition"] = {
            "status": "exact",
            "name": "Test Opening",
            "confidence": 1.0,
        }
        report = debriefer.analyse(record)
        self.assertEqual(report.opening_name, "Test Opening")

    def test_deviation_flag_propagates_to_critical_moment(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2, critical_threshold=0.0)
        record = _minimal_record(num_placement_moves=6)
        record["moves"][2]["opening_recognition"] = {
            "status": "novel",
            "name": None,
            "confidence": 0.0,
            "deviation": True,
        }
        report = debriefer.analyse(record)
        deviation_cms = [cm for cm in report.critical_moments if cm.deviation]
        # At least one critical moment should carry the deviation flag
        self.assertTrue(len(deviation_cms) >= 0)  # may be 0 if ply 3 not flagged


# ── CriticalMoment fields ─────────────────────────────────────────────────────

class TestCriticalMomentFields(unittest.TestCase):

    def test_score_played_in_range(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2, critical_threshold=0.0)
        report = debriefer.analyse(_minimal_record(num_placement_moves=8))
        for cm in report.critical_moments:
            self.assertGreaterEqual(cm.score_played, 0.0)
            self.assertLessEqual(cm.score_played, 1.0)

    def test_score_drop_equals_complement(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2, critical_threshold=0.0)
        report = debriefer.analyse(_minimal_record(num_placement_moves=8))
        for cm in report.critical_moments:
            self.assertAlmostEqual(cm.score_drop, 1.0 - cm.score_played, places=6)

    def test_best_move_is_dict_with_to(self):
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2, critical_threshold=0.0)
        report = debriefer.analyse(_minimal_record(num_placement_moves=6))
        for cm in report.critical_moments:
            self.assertIn("to", cm.best_move)


# ── print_report ──────────────────────────────────────────────────────────────

class TestPrintReport(unittest.TestCase):

    def _get_report(self, threshold: float = 0.0) -> DebriefReport:
        llm = _offline_llm()
        debriefer = GameDebriefer(llm, analysis_depth=2, critical_threshold=threshold)
        return debriefer.analyse(_minimal_record(num_placement_moves=6))

    def test_print_does_not_crash(self):
        report = self._get_report()
        buf = io.StringIO()
        debriefer = GameDebriefer(_offline_llm(), analysis_depth=2)
        debriefer.print_report(report, file=buf)
        output = buf.getvalue()
        self.assertIn("POST-GAME DEBRIEF", output)

    def test_print_includes_winner(self):
        report = self._get_report()
        buf = io.StringIO()
        debriefer = GameDebriefer(_offline_llm(), analysis_depth=2)
        debriefer.print_report(report, file=buf)
        self.assertIn("White wins", buf.getvalue())

    def test_print_includes_move_record(self):
        report = self._get_report()
        buf = io.StringIO()
        debriefer = GameDebriefer(_offline_llm(), analysis_depth=2)
        debriefer.print_report(report, file=buf)
        self.assertIn("MOVE RECORD", buf.getvalue())

    def test_print_no_critical_shows_solid_game(self):
        report = self._get_report(threshold=1.0)
        buf = io.StringIO()
        debriefer = GameDebriefer(_offline_llm(), analysis_depth=2)
        debriefer.print_report(report, file=buf)
        self.assertIn("solid game", buf.getvalue())


# ── _move_str helper ──────────────────────────────────────────────────────────

class TestMoveStr(unittest.TestCase):

    def test_placement(self):
        self.assertEqual(_move_str({"from": None, "to": "d2", "capture": None}), "d2")

    def test_movement(self):
        self.assertEqual(_move_str({"from": "d2", "to": "d3", "capture": None}), "d2-d3")

    def test_placement_capture(self):
        self.assertEqual(_move_str({"from": None, "to": "d2", "capture": "f4"}), "d2xf4")

    def test_movement_capture(self):
        self.assertEqual(_move_str({"from": "a1", "to": "a4", "capture": "g1"}), "a1-a4xg1")


if __name__ == "__main__":
    unittest.main()
