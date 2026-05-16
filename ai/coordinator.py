"""ai/coordinator.py — AI dialogue coordinator (GameAI ↔ MillsLLM)."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState

from ai.game_ai import GameAI
from ai.mills_llm import MillsLLM
from ai.memory_manager import MemoryManager
from ai.opening_recognizer import OpeningRecognizer, INACTIVE_RESULT
from ai.endgame_recognizer import EndgameRecognizer, INACTIVE_ENDGAME
from game.rules import get_all_legal_moves, get_game_phase


class Coordinator:
    def __init__(
        self,
        game_ai: GameAI,
        mills_llm: MillsLLM,
        memory: MemoryManager,
        think_time: float = 3.0,
        poor_move_threshold: float = 0.3,
        max_poor_move_comments: int = 5,
        opening_recognizer: OpeningRecognizer | None = None,
        endgame_recognizer: EndgameRecognizer | None = None,
    ) -> None:
        self.game_ai = game_ai
        self.mills_llm = mills_llm
        self.memory = memory
        self.think_time = think_time
        self.poor_move_threshold = poor_move_threshold
        self.max_poor_move_comments = max_poor_move_comments
        self.opening_recognizer = opening_recognizer
        self.endgame_recognizer = endgame_recognizer

        self.dialogue_log: list[str] = []
        self._poor_move_count = 0
        self._last_comment_turn = -2
        self._turn_num = 0
        self._session_id = str(uuid.uuid4())
        self._game_moves: list[dict] = []
        self._endgame_state = INACTIVE_ENDGAME

    # ── Internal helpers ──────────────────────────────────────────────────────

    def emit(self, speaker: str, text: str, tag: str = "normal") -> None:
        if text:
            self.dialogue_log.append(f"[{speaker}] {text}")

    def flush_dialogue(self) -> list[str]:
        lines = self.dialogue_log[:]
        self.dialogue_log.clear()
        return lines

    def _can_comment(self) -> bool:
        if self._poor_move_count >= self.max_poor_move_comments:
            return False
        if self._turn_num - self._last_comment_turn < 2:
            return False
        return True

    # ── Game lifecycle ────────────────────────────────────────────────────────

    def on_game_start(self) -> None:
        self._poor_move_count = 0
        self._last_comment_turn = -2
        self._turn_num = 0
        self._game_moves = []
        self._session_id = str(uuid.uuid4())
        self.dialogue_log.clear()
        self.mills_llm.conversation_history.clear()
        self.mills_llm.narrative_memory = ""
        if self.opening_recognizer:
            self.opening_recognizer.reset()
        if self.endgame_recognizer:
            self.endgame_recognizer.reset()
        self._endgame_state = INACTIVE_ENDGAME

        recent = self.memory.load_recent_games(n=5)
        if recent:
            patterns = self.memory.analyse_patterns(recent)
            self.mills_llm.narrative_memory = (
                f"Recent pattern analysis: {json.dumps(patterns, indent=2)}"
            )

    def on_game_end(self, game_record: dict) -> None:
        self.memory.save_game_record(game_record)

        # Auto-save novel opening sequences discovered during this game.
        if self.opening_recognizer:
            final = self.opening_recognizer.get_current_result()
            if final.status == "novel":
                self._save_novel_opening(game_record)

        summary = self.mills_llm.summarise_session([game_record])
        if summary:
            self.memory.save_session_narrative(summary)
            self.emit("MillsLLM", summary)

    @staticmethod
    def _compute_fen_signatures(placement_moves: list[str]) -> list[dict]:
        from game.board import BoardState
        board = BoardState.new_game()
        sigs = []
        for i, pos in enumerate(placement_moves):
            board = board.apply_move({"from": None, "to": pos, "capture": None})
            ply = i + 1
            if ply in (4, 6, 8, 10):
                sigs.append({"ply": ply, "fen": board.to_fen_string()})
        return sigs

    def _save_novel_opening(self, game_record: dict) -> None:
        placement_moves = [
            m["to"] for m in game_record.get("moves", [])
            if m.get("type") == "place"
        ]
        if len(placement_moves) < 6:
            return

        book = self.opening_recognizer.book  # type: ignore[union-attr]
        name = self.mills_llm.name_novel_opening(placement_moves)
        sigs = self._compute_fen_signatures(placement_moves)
        novel = book.save_novel_opening(
            placement_moves, sigs, outcome=game_record.get("winner")
        )
        novel.name = name
        book.save_opening(novel)
        self.emit("MillsLLM", f"I've recorded this opening as \"{name}\"")

    # ── AI deliberation ───────────────────────────────────────────────────────

    # How much the LLM recommendation must outScore GameAI's choice (after bonus)
    # before GameAI defers to it.
    LLM_BONUS = 0.15

    def deliberate(self, board: "BoardState") -> dict:
        self._turn_num += 1
        legal = get_all_legal_moves(board)
        if not legal:
            raise RuntimeError("No legal moves available")

        # 1. Get current opening recognition and endgame state
        recognition = (
            self.opening_recognizer.get_current_result()
            if self.opening_recognizer else INACTIVE_RESULT
        )
        if self.endgame_recognizer:
            self._endgame_state = self.endgame_recognizer.update(board)
            for msg in self.endgame_recognizer.transition_announcements():
                self.emit("MillsLLM", msg)
        endgame_state = self._endgame_state

        # 2. GameAI picks its best move (with opening bonus and endgame depth boost)
        ai_move = self.game_ai.choose_move(
            board, recognition=recognition, endgame_state=endgame_state
        )
        ai_score = self.game_ai.score_move(board, ai_move)

        # 3. Expose score hint to MillsLLM for its prompt
        self.mills_llm._last_ai_score = ai_score

        # 4. Ask MillsLLM for a recommendation (with opening + endgame context)
        opinion, llm_notation = self.mills_llm.ask_for_move_opinion(
            board, legal, ai_move, recognition=recognition, endgame_state=endgame_state
        )

        # 4. Try to adopt the LLM's recommendation if it scores well enough
        move = ai_move
        if llm_notation:
            from ai.mills_llm import _notation_to_move
            llm_move = _notation_to_move(llm_notation, legal)
            if llm_move and llm_move != ai_move:
                llm_score = self.game_ai.score_move(board, llm_move)
                if llm_score + self.LLM_BONUS > ai_score:
                    move = llm_move
                    self.emit(
                        "GameAI",
                        f"MillsLLM recommends {llm_notation} "
                        f"(score {llm_score:.2f} vs engine {ai_score:.2f}) — adopting",
                    )
                else:
                    self.emit(
                        "GameAI",
                        f"MillsLLM suggests {llm_notation} "
                        f"(score {llm_score:.2f}), engine stays with {_move_str(ai_move)} "
                        f"(score {ai_score:.2f})",
                    )

        # 5. Log LLM's reasoning (strip the MOVE: line — already acted on)
        if opinion:
            reason = _extract_reason(opinion)
            if reason:
                self.emit("MillsLLM", reason)

        move_str = _move_str(move)
        self.emit("GameAI", f"Playing {move_str}")

        if self.game_ai.last_was_blunder:
            blunder_msg = self.mills_llm.announce_blunder(board, move)
            self.emit("MillsLLM", blunder_msg if blunder_msg else
                      "I just made a mistake there — can you spot what I should have done instead?")

        # 8. Update recognizer with AI's move notation
        if self.opening_recognizer:
            self.opening_recognizer.update(move.get("to", ""), board)

        rec = recognition
        self._game_moves.append({
            "turn": self._turn_num,
            "color": board.turn,
            "type": get_game_phase(board, board.turn),
            "from": move.get("from"),
            "to": move.get("to"),
            "capture": move.get("capture"),
            "notation": move_str,
            "board_fen_before": board.to_fen_string(),
            "was_blunder": self.game_ai.last_was_blunder,
            "opening_recognition": {
                "status": rec.status,
                "name": rec.name,
                "confidence": rec.confidence,
            },
        })

        return move

    # ── Human move reaction ───────────────────────────────────────────────────

    def react_to_human_move(
        self,
        board_before: "BoardState",
        board_after: "BoardState",
        human_move: dict,
    ) -> None:
        self._turn_num += 1

        # Update endgame state
        if self.endgame_recognizer:
            self._endgame_state = self.endgame_recognizer.update(board_after)
            for msg in self.endgame_recognizer.transition_announcements():
                self.emit("MillsLLM", msg)

        # Update recognizer with human's move
        recognition = INACTIVE_RESULT
        if self.opening_recognizer:
            recognition = self.opening_recognizer.update(
                human_move.get("to", ""), board_after
            )
            # Emit opening name when first recognised
            if recognition.status == "exact" and recognition.name:
                self.emit("MillsLLM", f"Opening recognised: {recognition.name}")
            elif recognition.status == "transposition" and recognition.name:
                self.emit("MillsLLM", f"Transposition to: {recognition.name}")

        score_before = self.game_ai.score_move(board_before, human_move)

        self._game_moves.append({
            "turn": self._turn_num,
            "color": board_before.turn,
            "type": get_game_phase(board_before, board_before.turn),
            "from": human_move.get("from"),
            "to": human_move.get("to"),
            "capture": human_move.get("capture"),
            "notation": _move_str(human_move),
            "board_fen_before": board_before.to_fen_string(),
            "game_ai_score": score_before,
            "opening_recognition": {
                "status": recognition.status,
                "name": recognition.name,
                "confidence": recognition.confidence,
                "deviation": recognition.deviation_ply is not None,
            },
        })

        if not self._can_comment():
            return

        score_after = 1.0 - score_before
        comment = self.mills_llm.evaluate_human_move(
            board_before=board_before,
            human_move=human_move,
            score_before=score_before,
            score_after=score_after,
            score_drop_threshold=self.poor_move_threshold,
            recognition=recognition,
        )
        if comment:
            self.emit("MillsLLM", comment, tag="warning")
            self._poor_move_count += 1
            self._last_comment_turn = self._turn_num

    # ── Export ────────────────────────────────────────────────────────────────

    def build_game_record(self, winner: str | None, human_color: str) -> dict:
        return {
            "session_id": self._session_id,
            "date": datetime.now().isoformat(),
            "human_color": human_color,
            "winner": winner,
            "moves": self._game_moves,
            "bad_moves_taught": [],
        }


def _move_str(move: dict) -> str:
    if move.get("from"):
        s = f"{move['from']}-{move['to']}"
    else:
        s = move["to"]
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


def _extract_reason(response: str) -> str:
    """Return the REASON line from a structured LLM response, stripping the MOVE line."""
    lines = []
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("MOVE:"):
            continue
        if stripped.upper().startswith("REASON:"):
            lines.append(stripped[7:].strip())
        elif lines:
            lines.append(stripped)
    return " ".join(lines).strip()
