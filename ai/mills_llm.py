"""ai/mills_llm.py — Ollama interface for LLM-assisted Nine Men's Morris."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import BoardState
    from ai.memory_manager import MemoryManager

_MAX_HISTORY = 16

_BOARD_RULES = """\
You are MillsAI, an assistant for Nine Men's Morris.
Also called: Mills, Mühle, Merels.

This is NOT chess.
Never use chess terms, chess ideas, or chess piece names.

GAME FACTS:
- Board has exactly 24 valid nodes:
  Outer:  a7 d7 g7 g4 g1 d1 a1 a4
  Middle: b6 d6 f6 f4 f2 d2 b2 b4
  Inner:  c5 d5 e5 e4 e3 d3 c3 c4
- Players: White (W) and Black (B), 9 pieces each.
- Phases:
  1) place  -> place on any empty node
  2) move   -> slide along a connected line to an adjacent empty node
  3) fly    -> if a side has exactly 3 pieces, it may move to any empty node
- Mill: 3 own pieces in a row on a legal board line
- After forming a mill, remove one opponent piece
- Win by reducing the opponent to 2 pieces, or leaving them with no legal move

MOVE NOTATION:
- Place: d2
- Move: a4-a7
- Place + capture: d2xb6
- Move + capture: a4-a7xb6

STRICT VALIDITY:
- Only use node names from the 24-node list above
- Only choose from the provided LEGAL MOVES
- Never invent notation
"""

_DECISION_POLICY = """\
DECISION PRIORITIES FOR NINE MEN'S MORRIS:

Opening / placement priorities:
1. Prefer control of central/cardinal points and flexible positions
2. Avoid self-crowding and dead placements
3. Avoid making flashy early mills if they reduce future mobility
4. Block dangerous opponent mills, especially strong cardinal mills
5. Preserve the ability to create two future threats instead of one short-lived threat

Midgame priorities:
1. Keep or regain initiative
2. Preserve mobility and restrict opponent mobility
3. Prefer moves that maintain multiple future mill threats
4. Break or punish unstable opponent structures
5. Avoid moves that trap your own pieces

Endgame priorities:
1. Prevent immediate loss
2. Create or stop forced mills
3. Maximize mobility
4. In flying positions, value forcing threats and dual threats highly

GENERAL STYLE:
- Be concrete, not poetic
- Prefer safe, strong, legal moves over speculative ones
- If opening context is given, use it
- If endgame context is given, use it
- If strategic memory is given, treat it as a hint, not a rule
"""

_MOVE_SYSTEM = _BOARD_RULES + "\n" + _DECISION_POLICY + """

TASK:
Choose the single best move for the side to move from LEGAL MOVES.

YOUR RESPONSE MUST BEGIN WITH EXACTLY THIS FORMAT — NO EXCEPTIONS:
MOVE: <exact string from LEGAL MOVES>
REASON: <one sentence, max 18 words>

CRITICAL RULES:
- Line 1 MUST be "MOVE: " followed by one exact entry from LEGAL MOVES
- Line 2 MUST be "REASON: " followed by one sentence
- Do NOT write anything before the MOVE: line
- Do NOT add markdown, headers, or extra explanation
- If you are unsure, still pick the safest move from LEGAL MOVES
"""

_COMMENT_SYSTEM = _BOARD_RULES + """

TASK:
Comment briefly on the human's last move.

OUTPUT RULES:
- Write exactly one short sentence, max 18 words
- Focus on mobility, initiative, mill threat, blunder risk, or positional consequence
- Do not suggest a move
- Do not mention chess
- If the move is acceptable and not worth commenting on, reply exactly:
NO_COMMENT
"""

_QUESTION_SYSTEM = _BOARD_RULES + """

TASK:
Ask the human one brief useful question to learn their plan.

OUTPUT RULES:
- One sentence only
- Max 14 words
- No lecture
- No move suggestion
"""

_BLUNDER_SYSTEM = _BOARD_RULES + """

TASK:
You intentionally made a bad move for teaching.
Tell the human this was deliberate and invite them to find the better idea.

OUTPUT RULES:
- 1 or 2 short sentences
- Do not reveal the correct move
- Do not mention chess
"""

_SESSION_SYSTEM = _BOARD_RULES + """

TASK:
Write a compact markdown session summary.

FORMAT:
## Session
- Winner and result pattern
- Opening or early-game pattern if known
- One key turning point
- One lesson about mobility, initiative, or mill timing
- One note about the human's habits or improvements

Keep it concise and specific.
"""

_DEBRIEF_GAME_SYSTEM = _BOARD_RULES + """

TASK:
Write a clear game debrief.

FORMAT:
## Result
## Opening
## Turning Point
## Mistakes
## Lessons

STYLE:
- concise
- concrete
- coaching tone
- no move invention
"""

_DEBRIEF_POSITION_SYSTEM = _BOARD_RULES + """

TASK:
Explain one replay position.

OUTPUT RULES:
- 2 short paragraphs max
- Explain why the played move mattered
- If a better move existed, explain the difference in plain language
- If this was a turning point, say why
"""

_OPENING_NAME_SYSTEM = _BOARD_RULES + """

TASK:
Invent a short traditional-sounding name for a novel opening sequence.

OUTPUT RULES:
- Reply with only the name
- 2 to 4 words
- No punctuation at the end
- Tone: memorable, serious, game-opening style
"""

_PLAYER_CHAT_SYSTEM = _BOARD_RULES + """

TASK:
Respond to the human player's message during a live game.

OUTPUT RULES:
- 1 to 3 sentences maximum
- Stay focused on Nine Men's Morris strategy or their question
- You may comment on the current position if relevant
- Do NOT suggest a specific move to play next
- Do NOT use chess terminology
- Be concise and coaching in tone
"""

_POSITIVE_COMMENT_SYSTEM = _BOARD_RULES + """

TASK:
Comment briefly on the human's strong move.

OUTPUT RULES:
- Write exactly one short sentence, max 18 words
- Focus on what makes this move strong: mobility gain, mill threat, positional control
- Be encouraging but concise
- Do not suggest another move
- Do not mention chess
- If the move is not worth commenting on, reply exactly: NO_COMMENT
"""

_MILL_COMMENT_SYSTEM = _BOARD_RULES + """

TASK:
Comment briefly on the human forming a mill and capturing a piece.

OUTPUT RULES:
- Write exactly one short sentence, max 18 words
- Note the tactical achievement and what it means for the position going forward
- Do not suggest a move
- Do not mention chess
"""

_POSITION_QUESTION_SYSTEM = _BOARD_RULES + """

TASK:
Ask the human one brief useful question about their strategic plan.

OUTPUT RULES:
- One sentence only
- Max 14 words
- No lecture, no move suggestion
- Ask something that invites them to think about mobility, threats, or mill formation
"""


def _move_history_block(notations: list[str], limit: int = 40) -> str:
    """Format recent move notations as a compact numbered move list."""
    if not notations:
        return ""
    recent = notations[-limit:]
    offset = len(notations) - len(recent)
    lines = []
    for i, n in enumerate(recent, start=offset + 1):
        lines.append(f"{i}. {n}")
    return "\n--- GAME MOVES SO FAR ---\n" + "  ".join(lines) + "\n---"


def _endgame_context_block(endgame_state) -> str:
    lines = [
        "",
        "--- ENDGAME CONTEXT ---",
        f"Phase: {endgame_state.phase}",
        f"Pieces: W={endgame_state.pieces_white} B={endgame_state.pieces_black} total={endgame_state.total_pieces}",
        f"Mobility: W={endgame_state.mobility_white} B={endgame_state.mobility_black}",
        f"Zugzwang risk: {'yes' if endgame_state.zugzwang_risk else 'no'}",
        f"Pattern: {endgame_state.pattern or 'none'}",
    ]
    if getattr(endgame_state, "pattern_notes", None):
        lines.append(f"Pattern notes: {endgame_state.pattern_notes}")
    lines.append("---")
    return "\n".join(lines)


def _opening_context_block(recognition) -> str:
    lines = [
        "",
        "--- OPENING CONTEXT ---",
        f"Name: {recognition.name or 'Unknown / Novel'}",
        f"Family: {recognition.family or '—'}",
        f"Status: {recognition.status}",
        f"Confidence: {recognition.confidence:.0%}",
        f"Book move now: {recognition.book_move or 'none / exhausted'}",
        f"Strategic idea: {recognition.strategic_notes or '—'}",
        f"Common blunders: {', '.join(recognition.common_blunders) if recognition.common_blunders else 'none recorded'}",
        "---",
    ]
    return "\n".join(lines)


def _move_to_notation(move: dict) -> str:
    if move.get("from"):
        s = f"{move['from']}-{move['to']}"
    else:
        s = move["to"]
    if move.get("capture"):
        s += f"x{move['capture']}"
    return s


def _notation_to_move(notation: str, legal: list[dict]) -> dict | None:
    notation = notation.strip().lower()
    for m in legal:
        if _move_to_notation(m) == notation:
            return m
    for m in legal:
        if m["to"] == notation and not m.get("capture"):
            return m
    return None


class MillsLLM:
    def __init__(
        self,
        memory: "MemoryManager",
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
    ) -> None:
        self.model = model
        self._url = ollama_url
        self._memory = memory
        self.conversation_history: list[dict] = []
        self.narrative_memory: str = ""
        self.bad_moves_context: list[dict] = []
        self._client = self._make_client()

    def _make_client(self):
        try:
            import httpx
            import ollama
            # 5 s connect timeout, 30 s read timeout — prevents blocking forever
            # when Ollama is cold-loading a model or swapping between models.
            return ollama.Client(
                host=self._url,
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
        except Exception:
            return None

    def _chat(self, system: str, user: str, keep_history: bool = False) -> str:
        if self._client is None:
            return ""
        messages = [{"role": "system", "content": system}]
        if keep_history:
            messages.extend(self.conversation_history[-_MAX_HISTORY:])
        messages.append({"role": "user", "content": user})
        try:
            response = self._client.chat(model=self.model, messages=messages)
            reply = response.message.content or ""
            if keep_history:
                self.conversation_history.append({"role": "user", "content": user})
                self.conversation_history.append({"role": "assistant", "content": reply})
                self.conversation_history = self.conversation_history[-_MAX_HISTORY:]
            return reply.strip()
        except Exception:
            return ""

    def _strategy_context(self, board_fen: str) -> str:
        snippets = self._memory.retrieve_strategy(board_fen, n=2)
        if not snippets:
            return ""
        return "\n".join(f"- {s[:140]}" for s in snippets)

    def ask_for_move_opinion(
        self,
        board: "BoardState",
        legal_moves: list[dict],
        game_ai_suggestion: dict,
        recognition=None,
        endgame_state=None,
        audience: str = "human",  # "human" or "ai"
        move_history: list[str] | None = None,
    ) -> tuple[str, str | None]:
        notations = [_move_to_notation(m) for m in legal_moves]
        ai_notation = _move_to_notation(game_ai_suggestion)
        ai_score = getattr(self, "_last_ai_score", None)
        score_hint = f"{ai_score:+.2f}" if ai_score is not None else "n/a"
        strategy = self._strategy_context(board.to_fen_string())

        user_parts = [
            "LEGAL MOVES:",
            "\n".join(notations),
            "",
            f"SIDE TO MOVE: {board.turn}",
            f"ENGINE TOP CHOICE: {ai_notation}",
            f"ENGINE SCORE: {score_hint}",
            "",
            "BOARD:",
            board.to_display_grid(),
        ]

        if move_history:
            user_parts.append(_move_history_block(move_history))

        if recognition and recognition.status not in ("inactive", "novel"):
            user_parts.append(_opening_context_block(recognition))

        if endgame_state and endgame_state.active:
            user_parts.append(_endgame_context_block(endgame_state))

        if strategy:
            user_parts.extend(["", "STRATEGIC MEMORY:", strategy])

        user_parts.extend([
            "",
            "Choose the best move from LEGAL MOVES.",
            "Return exactly:",
            "MOVE: <exact legal move>",
            "REASON: <one short sentence>",
        ])

        audience_note = (
            "\nAUDIENCE: You are playing against a human player. Keep commentary engaging and accessible."
            if audience == "human"
            else "\nAUDIENCE: You are playing against another AI engine. Use precise technical language."
        )
        system = _MOVE_SYSTEM + audience_note
        user = "\n".join(user_parts)
        reply = self._chat(system, user, keep_history=False)
        notation = self._parse_move(reply, notations)

        if notation is None and self._client is not None:
            retry_system = _BOARD_RULES + """
Reply with exactly one line:
MOVE: <exact legal move>
No other text.
"""
            retry_user = "LEGAL MOVES:\n" + "\n".join(notations)
            retry_reply = self._chat(retry_system, retry_user, keep_history=False)
            notation = self._parse_move(retry_reply, notations)
            if notation:
                reply = retry_reply

        return reply, notation

    def _parse_move(self, response: str, legal_notations: list[str]) -> str | None:
        for line in response.splitlines():
            clean = re.sub(r"\*+", "", line).strip()
            if re.match(r"(?i)^move\s*:", clean):
                candidate = re.sub(r"(?i)^move\s*:", "", clean).strip()
                match = self._match_notation(candidate, legal_notations)
                if match:
                    return match

        for token in re.findall(r"[a-g][1-7](?:-[a-g][1-7])?(?:x[a-g][1-7])?", response.lower()):
            match = self._match_notation(token, legal_notations)
            if match:
                return match
        return None

    @staticmethod
    def _match_notation(candidate: str, legal_notations: list[str]) -> str | None:
        candidate = re.sub(r"[.\s)\]]+$", "", candidate.strip().lower())
        if candidate in legal_notations:
            return candidate
        base = re.split(r"x", candidate)[0]
        for legal in legal_notations:
            if legal == base or legal.startswith(base + "x"):
                return legal
        return None

    def evaluate_human_move(
        self,
        board_before: "BoardState",
        human_move: dict,
        score_before: float,
        score_after: float,
        score_drop_threshold: float = 0.3,
        recognition=None,
        human_color: str = "",
        move_history: list[str] | None = None,
    ) -> str | None:
        delta = score_after - score_before
        if delta > -score_drop_threshold:
            return None

        move_notation = _move_to_notation(human_move)
        color_ctx = f"HUMAN PLAYS AS: {'White' if human_color == 'W' else 'Black'}\n" if human_color else ""
        user_parts = [
            f"{color_ctx}HUMAN MOVE: {move_notation}",
            f"SCORE CHANGE: {delta:+.2f}",
            "",
            "BOARD AFTER MOVE:",
            board_before.to_display_grid(),
        ]
        if move_history:
            user_parts.append(_move_history_block(move_history))
        if recognition and recognition.status not in ("inactive", "novel"):
            user_parts.append(_opening_context_block(recognition))
        reply = self._chat(_COMMENT_SYSTEM, "\n".join(user_parts), keep_history=False)
        if not reply or reply == "NO_COMMENT":
            return None
        return reply.strip()

    def announce_blunder(
        self, board: "BoardState", move: dict, move_history: list[str] | None = None
    ) -> str:
        move_notation = _move_to_notation(move)
        history_block = _move_history_block(move_history) if move_history else ""
        user = f"Deliberate bad move played: {move_notation}{history_block}\n\nBOARD:\n{board.to_display_grid()}"
        return self._chat(_BLUNDER_SYSTEM, user, keep_history=False)

    def record_human_feedback(self, board: "BoardState", move: dict, reason: str) -> None:
        self._memory.store_bad_move(
            board_fen=board.to_fen_string(),
            move=move,
            reason=reason,
            full_board_ascii=board.to_display_grid(),
        )
        self.bad_moves_context = self._memory.retrieve_similar_positions(
            board.to_fen_string(), n_results=5
        )

    def comment_on_good_move(
        self, board: "BoardState", move: dict, score: float,
        human_color: str = "", move_history: list[str] | None = None,
    ) -> str | None:
        move_notation = _move_to_notation(move)
        color_ctx = f"HUMAN PLAYS AS: {'White' if human_color == 'W' else 'Black'}\n" if human_color else ""
        history_block = _move_history_block(move_history) if move_history else ""
        user = (
            f"{color_ctx}HUMAN MOVE: {move_notation}\n"
            f"MOVE QUALITY (0=worst, 1=best): {score:.2f}\n"
            f"{history_block}\n"
            f"BOARD:\n{board.to_display_grid()}"
        )
        reply = self._chat(_POSITIVE_COMMENT_SYSTEM, user, keep_history=False)
        if not reply or reply.strip() == "NO_COMMENT":
            return None
        return reply.strip()

    def comment_on_mill(
        self, board: "BoardState", move: dict,
        human_color: str = "", move_history: list[str] | None = None,
    ) -> str | None:
        move_notation = _move_to_notation(move)
        color_ctx = f"HUMAN PLAYS AS: {'White' if human_color == 'W' else 'Black'}\n" if human_color else ""
        history_block = _move_history_block(move_history) if move_history else ""
        user = f"{color_ctx}HUMAN MILL + CAPTURE: {move_notation}{history_block}\n\nBOARD:\n{board.to_display_grid()}"
        reply = self._chat(_MILL_COMMENT_SYSTEM, user, keep_history=False)
        return reply.strip() if reply else None

    def ask_strategic_question(
        self, board: "BoardState", human_color: str = "", move_history: list[str] | None = None,
    ) -> str | None:
        color_ctx = f"HUMAN PLAYS AS: {'White' if human_color == 'W' else 'Black'}\n" if human_color else ""
        history_block = _move_history_block(move_history) if move_history else ""
        user = f"{color_ctx}{history_block}\nBOARD:\n{board.to_display_grid()}"
        reply = self._chat(_POSITION_QUESTION_SYSTEM, user, keep_history=False)
        return reply.strip() if reply.strip() else None

    def generate_question_for_human(self, board: "BoardState") -> str | None:
        user = f"BOARD:\n{board.to_display_grid()}"
        reply = self._chat(_QUESTION_SYSTEM, user, keep_history=False)
        return reply.strip() if reply.strip() else None

    def player_chat(
        self, message: str, board: "BoardState", move_history: list[str] | None = None,
    ) -> str:
        """Respond to an in-game message from the human player."""
        history_block = _move_history_block(move_history) if move_history else ""
        user = f"Player: {message}{history_block}\n\nCURRENT BOARD:\n{board.to_display_grid()}"
        reply = self._chat(_PLAYER_CHAT_SYSTEM, user, keep_history=True)
        return reply.strip() if reply else ""

    def summarise_session(self, game_records: list[dict]) -> str:
        if not game_records:
            return ""
        lines = []
        for rec in game_records:
            moves = rec.get("moves", [])
            notations = [m.get("notation", "") for m in moves if m.get("notation")]
            move_seq = " ".join(notations) if notations else "none"
            lines.append(
                f"- winner={rec.get('winner', '?')} "
                f"opening={rec.get('recognised_opening_name') or rec.get('opening_name', 'unknown')} "
                f"total_moves={len(moves)} "
                f"move_sequence: {move_seq}"
            )
        return self._chat(_SESSION_SYSTEM, "\n".join(lines), keep_history=False)

    def name_novel_opening(self, move_sequence: list[str]) -> str:
        move_str = ", ".join(move_sequence)
        reply = self._chat(
            _OPENING_NAME_SYSTEM,
            f"Opening sequence: {move_str}",
            keep_history=False,
        )
        name = reply.strip().strip("\"'")
        if not name:
            first = move_sequence[0] if move_sequence else "?"
            name = f"Novel Opening ({first}...)"
        return name

    def debrief_game(self, report) -> str:
        user = (
            f"Winner: {report.winner}\n"
            f"Loser: {report.loser}\n"
            f"Opening: {report.opening_name or 'unknown'}\n"
            f"Moves: {len(report.game_record.get('moves', []))}"
        )
        return self._chat(_DEBRIEF_GAME_SYSTEM, user, keep_history=False)

    def debrief_position(
        self,
        board: "BoardState",
        ply: int,
        move_played: dict,
        best_move: dict,
        score_played: float,
        score_best: float,
        is_critical: bool,
        opening_name: str | None,
        context: str,
    ) -> str:
        user = (
            f"Ply: {ply}\n"
            f"Played: {_move_to_notation(move_played)} score={score_played:+.2f}\n"
            f"Best: {_move_to_notation(best_move)} score={score_best:+.2f}\n"
            f"Critical: {'yes' if is_critical else 'no'}\n"
            f"Opening: {opening_name or 'unknown'}\n"
            f"Context: {context}\n\n"
            f"BOARD:\n{board.to_display_grid()}"
        )
        return self._chat(_DEBRIEF_POSITION_SYSTEM, user, keep_history=False)
