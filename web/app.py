"""web/app.py — FastAPI server for Nine Men's Morris."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_ROOT = Path(__file__).parent.parent
_WEB  = Path(__file__).parent

import sys
sys.path.insert(0, str(_ROOT))

from game.game_engine import GameEngine
from game.rules import get_all_legal_moves, get_game_phase
from ai.game_ai import GameAI
from ai.memory_manager import MemoryManager
from ai.mills_llm import MillsLLM
from ai.coordinator import Coordinator
from ai.opening_book import OpeningBook
from ai.opening_recognizer import OpeningRecognizer
from ai.endgame_recognizer import EndgameRecognizer


def _load_settings() -> dict:
    try:
        return json.loads((_ROOT / "data" / "settings.json").read_text())
    except Exception:
        return {}


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    def __init__(
        self,
        engine: GameEngine,
        game_ai: Optional[GameAI],
        coordinator: Optional[Coordinator],
        human_color: str,
        vs_human: bool,
    ) -> None:
        self.engine      = engine
        self.game_ai     = game_ai
        self.coordinator = coordinator
        self.human_color = human_color
        self.vs_human    = vs_human
        self._pending: Optional[dict] = None   # move awaiting a capture choice
        self._board_before = None              # board snapshot before human move
        self._board_history: list = []         # list of BoardState for undo


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Nine Men's Morris")
app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")
templates = Jinja2Templates(directory=str(_WEB / "templates"))


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state(session: Session) -> dict:
    engine = session.engine
    board  = engine.board
    color  = board.turn
    phase  = get_game_phase(board, color)

    legal  = get_all_legal_moves(board)
    legal_dests   = list({m["to"]   for m in legal})
    legal_sources = list({m["from"] for m in legal if m.get("from")})
    move_pairs    = [[m["from"], m["to"]] for m in legal if m.get("from")]

    eval_score = 0.0
    if session.game_ai:
        try:
            eval_score = session.game_ai.position_eval(board)
        except Exception:
            pass

    return {
        "type":             "state",
        "board":            dict(board.positions),
        "turn":             color,
        "phase":            phase,
        "is_human_turn":    session.vs_human or color == session.human_color,
        "human_color":      session.human_color,
        "pieces_placed":    dict(board.pieces_placed),
        "pieces_captured":  dict(board.pieces_captured),
        "pieces_w":         sum(1 for v in board.positions.values() if v == "W"),
        "pieces_b":         sum(1 for v in board.positions.values() if v == "B"),
        "legal_dests":      legal_dests,
        "legal_sources":    legal_sources,
        "move_pairs":       move_pairs,
        "eval_score":       eval_score,
        "finished":         engine.finished,
        "winner":           engine.winner,
        "moves":            [
            {"color": m["color"], "notation": m["notation"]}
            for m in engine.game_record.get("moves", [])
        ],
    }


async def _send(ws: WebSocket, obj: dict) -> None:
    await ws.send_text(json.dumps(obj))


async def _commentary(ws: WebSocket, session: Session) -> None:
    if session.coordinator:
        for line in session.coordinator.flush_dialogue():
            await _send(ws, {"type": "commentary", "text": line})


async def _game_over(ws: WebSocket, session: Session) -> None:
    winner = session.engine.winner
    msg = f"{'White' if winner == 'W' else 'Black'} wins!" if winner else "Draw!"
    await _send(ws, {"type": "game_over", "winner": winner, "message": msg})

    if session.coordinator:
        record = session.coordinator.build_game_record(
            winner=winner, human_color=session.human_color
        )
        await asyncio.to_thread(session.coordinator.on_game_end, record)
        await _commentary(ws, session)


async def _ai_turn(ws: WebSocket, session: Session) -> None:
    board = session.engine.board
    await _send(ws, {"type": "thinking", "color": board.turn})

    if session.coordinator:
        move = await asyncio.to_thread(session.coordinator.deliberate, board)
    else:
        move = await asyncio.to_thread(session.game_ai.choose_move, board)

    session.engine.apply_move(move)

    await _send(ws, {
        "type":        "ai_move",
        "from":        move.get("from"),
        "to":          move.get("to"),
        "capture":     move.get("capture"),
        "was_blunder": bool(session.game_ai and session.game_ai.last_was_blunder),
    })
    await _commentary(ws, session)
    await _send(ws, _state(session))

    if session.engine.finished:
        await _game_over(ws, session)


def _is_ai_turn(session: Session) -> bool:
    if session.vs_human or session.engine.finished:
        return False
    return session.engine.board.turn != session.human_color


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session: Optional[Session] = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            kind = msg.get("type")

            # ── new_game ──────────────────────────────────────────────────────
            if kind == "new_game":
                import random as _random
                hc_raw    = msg.get("human_color", "W")
                hc        = _random.choice(["W", "B"]) if hc_raw == "R" else hc_raw
                diff      = max(1, min(10, int(msg.get("difficulty", 3))))
                vs_human  = bool(msg.get("vs_human", False))
                use_llm   = bool(msg.get("use_llm", True))
                settings  = _load_settings()

                engine   = GameEngine(human_color=hc)
                game_ai  = None
                coord    = None

                if not vs_human:
                    ai_color = "B" if hc == "W" else "W"
                    game_ai  = GameAI(color=ai_color, difficulty=diff)

                    if use_llm:
                        url   = settings.get("ollama_url",   "http://localhost:11434")
                        model = settings.get("ollama_model", "llama3.1:8b")
                        mem   = MemoryManager(ollama_url=url, ollama_model=model)
                        llm   = MillsLLM(memory=mem, ollama_url=url, model=model)
                        book  = OpeningBook()
                        rec   = OpeningRecognizer(book)
                        egr   = EndgameRecognizer(
                            active_threshold=settings.get("endgame_active_threshold", 11),
                            deep_threshold=settings.get("endgame_deep_threshold", 8),
                            zugzwang_threshold=settings.get("endgame_zugzwang_threshold", 0.4),
                        )
                        coord = Coordinator(
                            game_ai=game_ai, mills_llm=llm, memory=mem,
                            poor_move_threshold=settings.get("poor_move_threshold", 0.3),
                            max_poor_move_comments=settings.get("max_poor_move_comments_per_game", 5),
                            opening_recognizer=rec, endgame_recognizer=egr,
                        )
                        await asyncio.to_thread(coord.on_game_start)

                session = Session(engine, game_ai, coord, hc, vs_human)
                await _send(websocket, _state(session))
                await _commentary(websocket, session)

                if _is_ai_turn(session):
                    await _ai_turn(websocket, session)

            # ── move ──────────────────────────────────────────────────────────
            elif kind == "move" and session:
                frm = msg.get("from")  # None for placement
                to  = msg.get("to")
                move = {"from": frm, "to": to, "capture": None}

                board  = session.engine.board
                legal  = get_all_legal_moves(board)
                valid  = any(m.get("from") == frm and m["to"] == to for m in legal)

                if not valid:
                    await _send(websocket, {"type": "error", "message": f"Illegal move to {to}"})
                    continue

                if session.engine.move_forms_mill(move):
                    session._pending      = move
                    session._board_before = board
                    caps = sorted(board.legal_captures(board.turn))
                    # Show the piece at its new position before waiting for capture choice
                    projected = board.apply_move({**move, "capture": None})
                    await _send(websocket, {
                        "type":            "capture_required",
                        "legal_captures":  caps,
                        "projected_board": dict(projected.positions),
                    })
                    continue

                board_before = board
                session._board_history.append(board)
                session.engine.apply_move(move)

                if session.coordinator and not session.vs_human:
                    await asyncio.to_thread(
                        session.coordinator.react_to_human_move,
                        board_before, session.engine.board, move,
                    )

                await _commentary(websocket, session)
                await _send(websocket, _state(session))

                if session.engine.finished:
                    await _game_over(websocket, session)
                elif _is_ai_turn(session):
                    await _ai_turn(websocket, session)

            # ── capture ───────────────────────────────────────────────────────
            elif kind == "capture" and session and session._pending:
                cap   = msg.get("position")
                board = session.engine.board
                caps  = board.legal_captures(board.turn)

                if cap not in caps:
                    await _send(websocket, {"type": "error", "message": f"Cannot capture {cap}"})
                    continue

                session._pending["capture"] = cap
                board_before    = session._board_before
                completed_move  = dict(session._pending)   # save before clearing
                session._board_history.append(board_before)
                session.engine.apply_move(completed_move)
                session._pending      = None
                session._board_before = None

                if session.coordinator and not session.vs_human:
                    await asyncio.to_thread(
                        session.coordinator.react_to_human_move,
                        board_before, session.engine.board, completed_move,
                    )

                await _commentary(websocket, session)
                await _send(websocket, _state(session))

                if session.engine.finished:
                    await _game_over(websocket, session)
                elif _is_ai_turn(session):
                    await _ai_turn(websocket, session)

            # ── undo ──────────────────────────────────────────────────────────
            elif kind == "undo" and session and session._board_history:
                session.engine.board    = session._board_history.pop()
                session.engine.finished = False
                session.engine.winner   = None
                session._pending        = None
                session._board_before   = None
                await _send(websocket, _state(session))

            # ── player_message ────────────────────────────────────────────────
            elif kind == "player_message" and session:
                text = str(msg.get("text", "")).strip()[:500]
                if not text:
                    continue
                llm = session.coordinator.mills_llm if session.coordinator else None
                if llm and llm._client:
                    board = session.engine.board
                    response = await asyncio.to_thread(llm.player_chat, text, board)
                    # Echo message and response through commentary
                    if session.coordinator:
                        session.coordinator.emit("Player", text)
                        if response:
                            session.coordinator.emit("MillsAI", response)
                    await _commentary(websocket, session)
                    # Persist in game record
                    session.engine.game_record.setdefault("player_chat", []).append(
                        {"turn": session.engine._turn_num, "message": text, "response": response}
                    )
                else:
                    await _send(websocket, {"type": "commentary",
                                            "text": "LLM not available — enable MillsAI commentary."})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await _send(websocket, {"type": "error", "message": str(exc)})
        except Exception:
            pass
