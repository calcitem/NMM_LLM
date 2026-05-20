"""ai/coordinator.py — AI dialogue coordinator (GameAI ↔ MillsLLM)."""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState

from ai.game_ai import GameAI
from ai.mills_llm import MillsLLM
from ai.memory_manager import MemoryManager
from ai.opening_book import Opening
from ai.opening_recognizer import OpeningRecognizer, RecognitionResult, INACTIVE_RESULT
from ai.endgame_recognizer import EndgameRecognizer, INACTIVE_ENDGAME
from ai.trajectory_db import TrajectoryDB
from ai.endgame_db import EndgameDB
from ai.board_symmetry import transform_notation as _transform_book_notation
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
        trajectory_db: TrajectoryDB | None = None,
        endgame_db: EndgameDB | None = None,
        vs_human: bool = True,
        human_color: str = "W",
    ) -> None:
        self.game_ai = game_ai
        self.mills_llm = mills_llm
        self.memory = memory
        self.think_time = think_time
        self.poor_move_threshold = poor_move_threshold
        self.max_poor_move_comments = max_poor_move_comments
        self.opening_recognizer = opening_recognizer
        self.endgame_recognizer = endgame_recognizer
        self.trajectory_db = trajectory_db
        self.endgame_db = endgame_db
        self.vs_human = vs_human
        self.human_color = human_color

        self.dialogue_log: list[str] = []
        self._poor_move_count = 0
        self._general_comment_count = 0
        self._last_comment_turn = -2
        self._human_turn_num = 0
        self._turn_num = 0
        self._session_id = str(uuid.uuid4())
        self._game_moves: list[dict] = []
        self._endgame_state = INACTIVE_ENDGAME
        self._target_opening: Opening | None = None
        self._game_sym_idx: int = 0   # D4 symmetry applied to book moves this game
        self._last_novel_id: str | None = None   # set when an unnamed opening is saved
        self._dominant_turn_streak: int = 0
        self.resignation_offered: bool = False

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

    def _can_comment_general(self) -> bool:
        return self._turn_num - self._last_comment_turn >= 2

    # ── Game lifecycle ────────────────────────────────────────────────────────

    def on_game_start(self) -> None:
        self._poor_move_count = 0
        self._general_comment_count = 0
        self._last_comment_turn = -2
        self._human_turn_num = 0
        self._turn_num = 0
        self._game_moves = []
        self._session_id = str(uuid.uuid4())
        self._dominant_turn_streak = 0
        self.resignation_offered = False
        self.dialogue_log.clear()
        self.mills_llm.conversation_history.clear()
        self.mills_llm.narrative_memory = ""
        if self.opening_recognizer:
            self.opening_recognizer.reset()
        if self.endgame_recognizer:
            self.endgame_recognizer.reset()
        self._endgame_state = INACTIVE_ENDGAME

        # Pick an opening to target this game using UCB-scored selection.
        # select_opening() already filters by side, but double-check so a stale
        # 'both' entry from an unknown-outcome game is accepted for either colour.
        self._target_opening = None
        self._game_sym_idx = 0
        self._last_novel_id = None
        if self.opening_recognizer:
            candidate = self.opening_recognizer.book.select_opening(
                ai_color=self.game_ai.color,
            )
            if candidate and candidate.side in (self.game_ai.color, "both"):
                self._target_opening = candidate
                # When playing White, pick a random D4 symmetry so the opening
                # plays out as a different rotation or reflection each game.
                # The trajectory and endgame DBs use canonical D4 forms, so the
                # AI's mid-game guidance is unaffected by the chosen orientation.
                if self.game_ai.color == "W" and self._target_opening.line_moves:
                    self._game_sym_idx = random.randint(0, 7)
                first_mv = ""
                if self._target_opening.line_moves:
                    raw = self._target_opening.line_moves[0]
                    first_mv = _transform_book_notation(raw, self._game_sym_idx) or raw
                    first_mv = f" → {first_mv}"
                self.emit(
                    "GameAI",
                    f"Targeting opening: {self._target_opening.name} "
                    f"(score {self._target_opening.opening_score(self.game_ai.color):.2f})"
                    f"{first_mv}",
                )

        recent = self.memory.load_recent_games(n=10)
        if recent:
            patterns = self.memory.analyse_patterns(recent)
            game_summaries = []
            for g in recent[:10]:
                notations = [m.get("notation", "") for m in g.get("moves", []) if m.get("notation")]
                move_str = " ".join(notations) if notations else "no moves recorded"
                game_summaries.append(
                    f"winner={g.get('winner','?')} opening={g.get('recognised_opening_name','unknown')} "
                    f"moves=({move_str})"
                )
            self.mills_llm.narrative_memory = (
                f"Recent pattern analysis: {json.dumps(patterns, indent=2)}\n\n"
                f"Last {len(recent)} games:\n" + "\n".join(game_summaries)
            )

    def on_game_end(self, game_record: dict) -> None:
        self.memory.save_game_record(game_record)
        if self.trajectory_db is not None:
            self.trajectory_db.add_game(game_record)
        if self.endgame_db is not None:
            self.endgame_db.add_game(game_record)

        winner = game_record.get("winner")
        human_color = game_record.get("human_color", "W")

        if self.opening_recognizer:
            final = self.opening_recognizer.get_current_result()
            if final.status in ("novel", "inactive"):
                self._save_novel_opening(game_record)
            elif final.opening_id and final.status in ("exact", "probable", "transposition"):
                # Record this game's outcome against the recognised opening so
                # future UCB selection can learn which openings perform well.
                self.opening_recognizer.book.update_outcome_stats(
                    final.opening_id,
                    winner=winner or "D",
                    human_color=human_color,
                )

        summary = self.mills_llm.summarise_session([game_record])
        if summary:
            self.memory.save_session_narrative(summary)
            self.emit("MillsAI", summary)

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
        from ai.opening_book import is_auto_named

        placement_moves = [
            m["to"] for m in game_record.get("moves", [])
            if m.get("type") == "place"
        ]
        if len(placement_moves) < 6:
            return

        book = self.opening_recognizer.book  # type: ignore[union-attr]
        winner = game_record.get("winner")

        # Check if an existing opening shares the same first 4+ moves.
        # If so, merge this game's outcome into it rather than creating a duplicate.
        similar = book.find_similar(placement_moves, min_common=4)
        if similar:
            canonical = max(
                similar,
                key=lambda o: sum(o.outcome_stats.get(k, 0) for k in ("W", "B", "D")),
            )
            if winner in ("W", "B", "D"):
                canonical.outcome_stats[winner] = canonical.outcome_stats.get(winner, 0) + 1
            # Name it now if it's still carrying an auto-generated placeholder
            if is_auto_named(canonical.name) or canonical.needs_llm_name:
                name = self.mills_llm.name_novel_opening(canonical.line_moves)
                if name and not is_auto_named(name):
                    canonical.name = name
                    canonical.needs_llm_name = False
                    self.emit("MillsAI", f"Named this opening family \"{name}\"")
            book.save_opening(canonical)
            if canonical.needs_llm_name or is_auto_named(canonical.name):
                self._last_novel_id = canonical.opening_id
            return

        # No similar opening — create a new one.
        llm_available = self.mills_llm._client is not None
        name = self.mills_llm.name_novel_opening(placement_moves)
        sigs = self._compute_fen_signatures(placement_moves)
        needs_name = is_auto_named(name) or not llm_available
        novel = book.save_novel_opening(
            placement_moves, sigs,
            outcome=winner,
            needs_llm_name=needs_name,
        )
        novel.name = name
        book.save_opening(novel)
        if needs_name:
            self._last_novel_id = novel.opening_id
        else:
            self.emit("MillsAI", f"I've recorded this opening as \"{name}\"")

    # ── Tactical pre-screen ───────────────────────────────────────────────────

    def _tactical_situation(self, board: "BoardState") -> dict:
        """Classify the tactical urgency before move selection.

        Returns a dict with flags used both for logging and to inform the LLM
        about the immediate tactical context.
        """
        from ai.heuristics import (
            detect_double_mills, detect_feeder_mills,
            detect_diamonds, opponent_mills_in_n_moves,
        )
        from ai.heuristics import _closeable_mills
        ai_color  = board.turn
        opp_color = "B" if ai_color == "W" else "W"

        can_close      = _closeable_mills(board, ai_color) > 0
        opp_can_close  = _closeable_mills(board, opp_color) > 0
        opp_doubles    = detect_double_mills(board, opp_color)
        ai_doubles     = detect_double_mills(board, ai_color)
        opp_diamonds   = detect_diamonds(board, opp_color)
        opp_threats_2  = opponent_mills_in_n_moves(board, opp_color, n=2)

        return {
            "urgent":             can_close or opp_can_close or bool(opp_doubles),
            "can_close_mill":     can_close,
            "must_block_opponent": opp_can_close,
            "opp_double_mills":   opp_doubles,
            "ai_double_mills":    ai_doubles,
            "opp_diamonds":       opp_diamonds,
            "opp_threats_in_2":  opp_threats_2,
        }

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

        # If recognition hasn't found an opening yet but we have a target,
        # synthesise a recognition hint from the target so the AI's opening
        # bonus steers it along the preferred line from the very first move.
        phase = get_game_phase(board, board.turn)
        if (
            phase == "place"
            and self._target_opening is not None
            and recognition.status in ("inactive", "novel")
        ):
            ply = len(self._game_moves)
            line = self._target_opening.line_moves
            # Apply the per-game D4 symmetry so the entire opening plays out
            # in the chosen rotation/reflection.  Falls back to the raw move
            # if the transform maps to an off-board square (shouldn't happen
            # on a valid NMM position but keeps things safe).
            _raw_mv = line[ply] if ply < len(line) else None
            _book_mv = (
                (_transform_book_notation(_raw_mv, self._game_sym_idx) or _raw_mv)
                if _raw_mv else None
            )
            legal_dests = {m["to"] for m in legal}
            if _book_mv is not None and _book_mv in legal_dests:
                recognition = RecognitionResult(
                    opening_id=self._target_opening.opening_id,
                    name=self._target_opening.name,
                    family=self._target_opening.family,
                    confidence=self._target_opening.confidence,
                    status="probable",
                    matched_ply=ply,
                    deviation_ply=None,
                    deviation_move=None,
                    book_move=_book_mv,
                    branch_name=None,
                    strategic_notes=self._target_opening.strategic_notes,
                    common_blunders=list(self._target_opening.common_blunders),
                    tags=list(self._target_opening.tags),
                )
        if self.endgame_recognizer:
            self._endgame_state = self.endgame_recognizer.update(board)
            for msg in self.endgame_recognizer.transition_announcements():
                self.emit("MillsAI", msg)
        endgame_state = self._endgame_state

        # 2. Query trajectory DB for historical move-outcome hints
        trajectory_hints: dict | None = None
        if self.trajectory_db is not None and self._game_moves:
            notations = [m.get("notation", "") for m in self._game_moves if m.get("notation")]
            if notations:
                trajectory_hints = self.trajectory_db.query(notations, board.turn) or None

        # 2b. Query endgame DB for position-based hints (merged on top of trajectory hints)
        if self.endgame_db is not None and endgame_state.active:
            eg_hints = self.endgame_db.query(board, board.turn)
            if eg_hints:
                if trajectory_hints:
                    for notation, delta in eg_hints.items():
                        trajectory_hints[notation] = (
                            trajectory_hints.get(notation, 0.0) + delta
                        ) / 2.0
                else:
                    trajectory_hints = eg_hints

        # 3. Tactical pre-screen: log urgency level so the AI and LLM know the context
        tac = self._tactical_situation(board)
        if tac["can_close_mill"]:
            self.emit("GameAI", "Mill closure available — prioritising tactical completion")
        elif tac["must_block_opponent"]:
            self.emit("GameAI", "Opponent threatens a mill — defensive priority")
        elif tac["opp_double_mills"]:
            pivots = ", ".join(tac["opp_double_mills"][:2])
            self.emit("GameAI", f"Disrupting opponent cycling mill pivot at {pivots}")

        # 4. GameAI picks its best move (with opening bonus, trajectory hints, endgame depth)
        # Force the book move for the AI's first 2 placements so opening variety is
        # visible regardless of adherence slider.  At 100% adherence the book move is
        # forced for any ply where recognition is active.
        ai_placements_so_far = sum(
            1 for m in self._game_moves
            if m.get("color") == self.game_ai.color and m.get("type") == "place"
        )
        force_book_early = (phase == "place" and ai_placements_so_far < 2)
        ai_move = self.game_ai.choose_move(
            board,
            recognition=recognition,
            endgame_state=endgame_state,
            trajectory_hints=trajectory_hints,
            force_book_early=force_book_early,
        )
        ai_score = self.game_ai.score_move(board, ai_move)

        # 5. Expose score hint to MillsLLM for its prompt
        self.mills_llm._last_ai_score = ai_score

        # 5. Ask MillsLLM for a recommendation (with opening + endgame context)
        _notations_so_far = [m.get("notation", "") for m in self._game_moves if m.get("notation")]
        opinion, llm_notation = self.mills_llm.ask_for_move_opinion(
            board, legal, ai_move, recognition=recognition, endgame_state=endgame_state,
            audience="human" if self.vs_human else "ai",
            move_history=_notations_so_far,
        )

        # 6. Try to adopt the LLM's recommendation if it scores well enough
        move = ai_move
        if llm_notation:
            from ai.mills_llm import _notation_to_move
            llm_move = _notation_to_move(llm_notation, legal)
            if llm_move and llm_move != ai_move:
                # Don't adopt if the LLM recommendation is a banned move
                _fen = board.to_fen_string()
                _banned = self.game_ai._pos_bans.get(_fen, set())
                if self.game_ai._move_notation(llm_move) in _banned:
                    llm_move = None
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

        # 5. Log LLM's reasoning only when it successfully recommended a move
        if opinion and llm_notation:
            reason = _extract_reason(opinion)
            if reason:
                self.emit("MillsLLM", reason)

        move_str = _move_str(move)
        self.emit("GameAI", f"Playing {move_str}")

        # Resignation check: if human has dominated for 3 consecutive AI turns
        if not self.resignation_offered:
            try:
                from ai.heuristics import evaluate, TANH_SCALE
                human_color = "B" if self.game_ai.color == "W" else "W"
                post = board.apply_move(move)
                raw  = evaluate(post, human_color)
                norm = math.tanh(raw / TANH_SCALE.get(get_game_phase(post, human_color), 180))
                if norm > 0.95:
                    self._dominant_turn_streak += 1
                else:
                    self._dominant_turn_streak = 0
                if self._dominant_turn_streak >= 3:
                    self.resignation_offered = True
                    farewell = random.choice([
                        "Your position is overwhelming — I concede. Well played.",
                        "I see no path forward. A masterful performance.",
                        "You've outplayed me completely. I yield.",
                        "My position is beyond recovery. Congratulations.",
                    ])
                    self.emit("MillsAI", farewell)
            except Exception:
                pass

        if self.game_ai.last_was_blunder:
            blunder_msg = self.mills_llm.announce_blunder(board, move, move_history=_notations_so_far)
            self.emit("MillsAI", blunder_msg if blunder_msg else
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
        self._human_turn_num += 1

        # Update endgame state
        if self.endgame_recognizer:
            self._endgame_state = self.endgame_recognizer.update(board_after)
            for msg in self.endgame_recognizer.transition_announcements():
                self.emit("MillsAI", msg)

        # Update recognizer with human's move
        recognition = INACTIVE_RESULT
        if self.opening_recognizer:
            recognition = self.opening_recognizer.update(
                human_move.get("to", ""), board_after
            )
            if recognition.status == "exact" and recognition.name:
                self.emit("MillsAI", f"Opening recognised: {recognition.name}")
            elif recognition.status == "transposition" and recognition.name:
                self.emit("MillsAI", f"Transposition to: {recognition.name}")

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

        if not self._can_comment_general():
            return

        has_capture = bool(human_move.get("capture"))
        score_after = 1.0 - score_before
        _notations = [m.get("notation", "") for m in self._game_moves if m.get("notation")]

        # 1. Mill/capture commentary — always comment when human forms a mill
        if has_capture:
            comment = self.mills_llm.comment_on_mill(
                board_after, human_move,
                human_color=self.human_color, move_history=_notations,
            )
            if comment:
                self.emit("MillsAI", comment)
                self._general_comment_count += 1
                self._last_comment_turn = self._turn_num
                return

        # 2. Poor-move warning (capped by max_poor_move_comments)
        if self._can_comment():
            comment = self.mills_llm.evaluate_human_move(
                board_before=board_before,
                human_move=human_move,
                score_before=score_before,
                score_after=score_after,
                score_drop_threshold=self.poor_move_threshold,
                recognition=recognition,
                human_color=self.human_color,
                move_history=_notations,
            )
            if comment:
                self.emit("MillsAI", comment, tag="warning")
                self._poor_move_count += 1
                self._last_comment_turn = self._turn_num
                return

        # 3. Positive commentary on strong moves
        if score_before >= 0.75:
            comment = self.mills_llm.comment_on_good_move(
                board_after, human_move, score_before,
                human_color=self.human_color, move_history=_notations,
            )
            if comment:
                self.emit("MillsAI", comment)
                self._general_comment_count += 1
                self._last_comment_turn = self._turn_num
                return

        # 4. Periodic strategic question every 8 human moves
        if self._human_turn_num % 8 == 0:
            question = self.mills_llm.ask_strategic_question(
                board_after, human_color=self.human_color, move_history=_notations,
            )
            if question:
                self.emit("MillsAI", question)
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
