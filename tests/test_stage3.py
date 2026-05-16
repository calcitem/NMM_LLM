"""tests/test_stage3.py — Stage 3: MemoryManager, MillsLLM, Coordinator tests."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from game.board import BoardState
from ai.memory_manager import MemoryManager
from ai.mills_llm import MillsLLM
from ai.coordinator import Coordinator
from ai.game_ai import GameAI


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_memory(tmp_dir: str) -> MemoryManager:
    return MemoryManager(
        chroma_path=f"{tmp_dir}/chroma",
        games_path=f"{tmp_dir}/games",
        session_path=f"{tmp_dir}/session",
        use_ollama_embeddings=False,
    )


# ── MemoryManager tests ───────────────────────────────────────────────────────

class TestStrategySeeding(unittest.TestCase):
    def test_strategy_collection_has_10_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = _make_memory(tmp)
            count = mem._strategy.count()
            self.assertEqual(count, 10, "Strategy collection must have exactly 10 seeded entries")

    def test_seeding_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_memory(tmp)
            mem2 = _make_memory(tmp)
            self.assertEqual(mem2._strategy.count(), 10)

    def test_retrieve_mill_abandonment(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = _make_memory(tmp)
            results = mem.retrieve_strategy("I abandoned my mill to gain mobility", n=1)
            self.assertTrue(len(results) > 0)
            self.assertIn("abandonment", results[0].lower())


class TestBadMoveMemory(unittest.TestCase):
    def test_store_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = _make_memory(tmp)
            board = BoardState.new_game()
            fen = board.to_fen_string()
            move = {"from": None, "to": "d2", "capture": None}
            mem.store_bad_move(fen, move, "Left centre uncontested")
            results = mem.retrieve_similar_positions(fen, n_results=1)
            self.assertEqual(len(results), 1)
            self.assertIn("d2", results[0]["document"])

    def test_retrieve_empty_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = _make_memory(tmp)
            board = BoardState.new_game()
            results = mem.retrieve_similar_positions(board.to_fen_string())
            self.assertEqual(results, [])


class TestGameRecords(unittest.TestCase):
    def test_save_and_load_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = _make_memory(tmp)
            record = {
                "session_id": "test-123",
                "date": "2026-05-16T10:00:00",
                "human_color": "W",
                "winner": "B",
                "moves": [],
            }
            mem.save_game_record(record)
            loaded = mem.load_recent_games(n=10)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["session_id"], "test-123")


class TestSessionNarratives(unittest.TestCase):
    def test_save_and_retrieve_narrative(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = _make_memory(tmp)
            mem.save_session_narrative("Human played aggressively and won with a double mill.")
            results = mem.retrieve_relevant_narratives("double mill", n=1)
            self.assertEqual(len(results), 1)
            self.assertIn("double mill", results[0])


# ── MillsLLM tests ────────────────────────────────────────────────────────────

class TestMillsLLMNoOllama(unittest.TestCase):
    """Tests for MillsLLM when Ollama is unavailable (graceful degradation)."""

    def _make_llm(self, tmp: str) -> MillsLLM:
        mem = _make_memory(tmp)
        llm = MillsLLM(memory=mem, ollama_url="http://localhost:9999", model="llama3.2")
        llm._client = None  # force offline
        return llm

    def test_evaluate_human_move_returns_none_for_small_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = self._make_llm(tmp)
            board = BoardState.new_game()
            result = llm.evaluate_human_move(
                board_before=board,
                human_move={"from": None, "to": "d2", "capture": None},
                score_before=0.5,
                score_after=0.45,
                score_drop_threshold=0.3,
            )
            self.assertIsNone(result)

    def test_evaluate_human_move_returns_none_when_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = self._make_llm(tmp)
            board = BoardState.new_game()
            result = llm.evaluate_human_move(
                board_before=board,
                human_move={"from": None, "to": "d2", "capture": None},
                score_before=0.8,
                score_after=0.3,
                score_drop_threshold=0.3,
            )
            # Offline → _chat returns "" → None
            self.assertIsNone(result)

    def test_ask_for_move_opinion_returns_empty_when_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = self._make_llm(tmp)
            board = BoardState.new_game()
            text, notation = llm.ask_for_move_opinion(board, [], {"to": "d2"})
            self.assertEqual(text, "")
            self.assertIsNone(notation)


class TestMillsLLMWithMock(unittest.TestCase):
    """Tests for MillsLLM using a mocked Ollama client."""

    def _make_llm_mocked(self, tmp: str, reply: str = "test reply") -> MillsLLM:
        mem = _make_memory(tmp)
        llm = MillsLLM(memory=mem)
        mock_response = MagicMock()
        mock_response.message.content = reply
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_response
        llm._client = mock_client
        return llm

    def test_evaluate_human_move_large_delta_returns_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = self._make_llm_mocked(tmp, reply="That move weakened your position.")
            board = BoardState.new_game()
            result = llm.evaluate_human_move(
                board_before=board,
                human_move={"from": None, "to": "d2", "capture": None},
                score_before=0.8,
                score_after=0.2,
                score_drop_threshold=0.3,
            )
            self.assertIsNotNone(result)
            self.assertIn("weakened", result)

    def test_ask_for_move_opinion_returns_tuple(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = self._make_llm_mocked(tmp, reply="MOVE: d2\nREASON: Central control.")
            board = BoardState.new_game()
            from game.rules import get_all_legal_moves
            legal = get_all_legal_moves(board)
            text, notation = llm.ask_for_move_opinion(board, legal, legal[0])
            self.assertIsInstance(text, str)
            # notation should be a valid legal move string or None
            notations = [
                (m["from"] + "-" if m.get("from") else "") + m["to"] +
                ("x" + m["capture"] if m.get("capture") else "")
                for m in legal
            ]
            if notation is not None:
                self.assertIn(notation, notations)

    def test_record_human_feedback_stores_bad_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = self._make_llm_mocked(tmp)
            board = BoardState.new_game()
            move = {"from": None, "to": "d2", "capture": None}
            llm.record_human_feedback(board, move, "Left centre open")
            results = llm._memory.retrieve_similar_positions(board.to_fen_string(), n_results=1)
            self.assertEqual(len(results), 1)


# ── Coordinator tests ─────────────────────────────────────────────────────────

class TestCoordinator(unittest.TestCase):
    def _make_coordinator(self, tmp: str) -> tuple[Coordinator, MillsLLM]:
        mem = _make_memory(tmp)
        game_ai = GameAI(color="W", difficulty=1)
        llm = MillsLLM(memory=mem)
        llm._client = None  # offline
        coord = Coordinator(game_ai=game_ai, mills_llm=llm, memory=mem)
        return coord, llm

    def test_deliberate_returns_legal_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            coord, _ = self._make_coordinator(tmp)
            coord.on_game_start()
            board = BoardState.new_game()
            from game.rules import get_all_legal_moves
            move = coord.deliberate(board)
            legal = get_all_legal_moves(board)
            self.assertIn(move, legal)

    def test_deliberate_emits_gameai_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            coord, _ = self._make_coordinator(tmp)
            coord.on_game_start()
            board = BoardState.new_game()
            coord.deliberate(board)
            lines = coord.flush_dialogue()
            self.assertTrue(any("[GameAI]" in l for l in lines))

    def test_react_to_human_move_no_comment_small_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            coord, _ = self._make_coordinator(tmp)
            coord.on_game_start()
            board = BoardState.new_game()
            move = {"from": None, "to": "d2", "capture": None}
            board_after = board.apply_move(move)
            coord.react_to_human_move(board, board_after, move)
            lines = coord.flush_dialogue()
            self.assertFalse(any("[MillsLLM]" in l for l in lines))

    def test_on_game_start_resets_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            coord, _ = self._make_coordinator(tmp)
            coord._poor_move_count = 3
            coord._turn_num = 10
            coord.on_game_start()
            self.assertEqual(coord._poor_move_count, 0)
            self.assertEqual(coord._turn_num, 0)

    def test_build_game_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            coord, _ = self._make_coordinator(tmp)
            coord.on_game_start()
            board = BoardState.new_game()
            coord.deliberate(board)
            record = coord.build_game_record(winner="W", human_color="B")
            self.assertIn("session_id", record)
            self.assertEqual(record["winner"], "W")
            self.assertEqual(len(record["moves"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
