"""web/app.py — FastAPI server for Nine Men's Morris."""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_ROOT = Path(__file__).parent.parent
_WEB  = Path(__file__).parent

# ── Rotating debug log ────────────────────────────────────────────────────────
# 3 generations (server.log, server.log.1, server.log.2), 200 KB each.
# Every server restart begins a fresh log at server.log; old ones shift up.

_LOG_DIR = _ROOT / "data" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "server.log",
    maxBytes=200_000,
    backupCount=3,
    encoding="utf-8",
)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

log = logging.getLogger("nmm")
log.setLevel(logging.DEBUG)
log.addHandler(_handler)
log.addHandler(logging.StreamHandler())   # also print to console
log.info("=== Server started ===")

import sys
sys.path.insert(0, str(_ROOT))

from game.game_engine import GameEngine
from game.rules import get_all_legal_moves, get_game_phase
from ai.game_ai import GameAI
from ai.heuristics import HeuristicWeights
from ai.memory_manager import MemoryManager
from ai.mills_llm import MillsLLM
from ai.coordinator import Coordinator
from ai.opening_book import OpeningBook
from ai.opening_recognizer import OpeningRecognizer
from ai.endgame_recognizer import EndgameRecognizer
from ai.trajectory_db import TrajectoryDB

# Load trajectory DB once at startup — updated incrementally as games complete.
_trajectory_db    = TrajectoryDB(_ROOT / "data" / "games")
_BAD_MOVES_PATH   = _ROOT / "data" / "bad_moves.json"
try:
    _trajectory_db.load(bad_moves_path=_BAD_MOVES_PATH)
    log.info(
        "TrajectoryDB: %d games, %d prefix entries",
        _trajectory_db.game_count, _trajectory_db.entry_count,
    )
except Exception as _exc:
    log.warning("TrajectoryDB: load failed — %s", _exc)


def _load_settings() -> dict:
    try:
        return json.loads((_ROOT / "data" / "settings.json").read_text())
    except Exception:
        return {}


# ── Session ───────────────────────────────────────────────────────────────────

_HINT_CAP = 3

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
        self.hints_used: int = 0
        self._pending: Optional[dict] = None   # move awaiting a capture choice
        self._board_before = None              # board snapshot before human move
        self._board_history: list = []         # list of BoardState for undo
        # Snapshot captured just before the AI applies its move — used for undo
        self._pre_ai_board = None
        self._pre_ai_move_log: list = []
        self._pre_ai_game_record_len: int = 0
        self._pre_ai_post_placement: int = 0
        self._pre_ai_engine_turn: int = 1
        self._last_ai_move: Optional[dict] = None
        self._can_undo_ai: bool = False        # True only right after an AI move


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Nine Men's Morris")
app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")
templates = Jinja2Templates(directory=str(_WEB / "templates"))


_SETTINGS_PATH = _ROOT / "data" / "settings.json"


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/weights")
async def get_weights():
    from fastapi.responses import JSONResponse
    settings = _load_settings()
    return JSONResponse(settings.get("ai_weights", {}))


@app.post("/api/weights")
async def save_weights(request: Request):
    from fastapi.responses import JSONResponse
    body = await request.json()
    settings = _load_settings()
    settings["ai_weights"] = body
    _SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    return JSONResponse({"ok": True})


_PERSONALITIES_DIR = _ROOT / "data" / "personalities"
_VALID_PERSONALITIES = {"balanced", "aggressive", "defensive", "positional", "scholar", "chaos", "custom"}


@app.get("/api/personalities/{name}")
async def get_personality(name: str):
    from fastapi.responses import JSONResponse
    if name not in _VALID_PERSONALITIES:
        return JSONResponse({}, status_code=404)
    path = _PERSONALITIES_DIR / f"{name}.json"
    if path.exists():
        return JSONResponse(json.loads(path.read_text()))
    return JSONResponse({})


@app.post("/api/personalities/{name}")
async def save_personality(name: str, request: Request):
    from fastapi.responses import JSONResponse
    if name not in _VALID_PERSONALITIES:
        return JSONResponse({"error": "unknown personality"}, status_code=400)
    _PERSONALITIES_DIR.mkdir(parents=True, exist_ok=True)
    body = await request.json()
    (_PERSONALITIES_DIR / f"{name}.json").write_text(json.dumps(body, indent=2))
    return JSONResponse({"ok": True})


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
        "finished":              engine.finished,
        "winner":                engine.winner,
        "draw_reason":           engine.draw_reason,
        "post_placement_moves":  engine._post_placement_moves,
        "moves":            [
            {"color": m["color"], "notation": m["notation"]}
            for m in engine.game_record.get("moves", [])
        ],
    }


async def _send(ws: WebSocket, obj: dict) -> None:
    await ws.send_text(json.dumps(obj))


def _classify_commentary(line: str) -> tuple[str, str, str]:
    """Parse '[Speaker] text' and return (speaker, text, section).

    Section is 'human' (top box, LLM↔human) or 'ai' (bottom box, AI↔AI).
    """
    AI_SPEAKERS = {"GameAI", "Game"}
    if line.startswith("[") and "]" in line:
        end = line.index("]")
        speaker = line[1:end]
        text    = line[end + 2:]  # skip '] '
    else:
        speaker = "MillsAI"
        text    = line
    section = "ai" if speaker in AI_SPEAKERS else "human"
    return speaker, text, section


async def _commentary(ws: WebSocket, session: Session) -> None:
    if session.coordinator:
        for line in session.coordinator.flush_dialogue():
            speaker, text, section = _classify_commentary(line)
            await _send(ws, {
                "type": "commentary",
                "speaker": speaker,
                "text": text,
                "section": section,
            })


async def _game_over(ws: WebSocket, session: Session) -> None:
    winner      = session.engine.winner
    draw_reason = session.engine.draw_reason
    if winner:
        msg = f"{'White' if winner == 'W' else 'Black'} wins!"
    elif draw_reason:
        msg = f"Draw — {draw_reason}."
    else:
        msg = "Draw!"
    await _send(ws, {"type": "game_over", "winner": winner, "draw_reason": draw_reason, "message": msg})

    if session.coordinator:
        record = session.coordinator.build_game_record(
            winner=winner, human_color=session.human_color
        )
        await asyncio.to_thread(session.coordinator.on_game_end, record)
        await _commentary(ws, session)


def _expected_think_seconds(difficulty: int, total_pieces: int) -> float:
    if total_pieces < 10:
        return 4.0
    # Time-limited levels: return the actual budget so the UI countdown matches.
    budgets = {5: 15, 6: 24, 7: 36, 8: 60, 9: 60, 10: 90}
    if difficulty in budgets:
        return float(budgets[difficulty])
    # Fixed-depth levels (1–4): generous estimate so force_move doesn't fire mid-search.
    estimates = {1: 3, 2: 3, 3: 6, 4: 9}
    return float(estimates.get(difficulty, 5))


async def _ai_turn(ws: WebSocket, session: Session) -> None:
    import time as _time
    board = session.engine.board
    diff  = session.game_ai.difficulty if session.game_ai else 3
    total = sum(board.pieces_on_board.values())
    exp   = _expected_think_seconds(diff, total)
    log.info("AI turn start  color=%s diff=%s total_pieces=%s expected=%.1fs",
             board.turn, diff, total, exp)
    await _send(ws, {
        "type":             "thinking",
        "color":            board.turn,
        "expected_seconds": exp,
    })

    t0 = _time.time()
    try:
        if session.coordinator:
            move = await asyncio.to_thread(session.coordinator.deliberate, board)
        else:
            move = await asyncio.to_thread(session.game_ai.choose_move, board)
    except Exception as exc:
        log.error("AI deliberation failed: %s", exc, exc_info=True)
        raise

    elapsed = _time.time() - t0
    log.info("AI turn end    move=%s elapsed=%.2fs", move, elapsed)

    # Resignation: human dominated for 3 consecutive AI turns
    if session.coordinator and session.coordinator.resignation_offered:
        session.coordinator.resignation_offered = False
        session.engine.finished = True
        session.engine.winner   = session.human_color
        human_name = "White" if session.human_color == "W" else "Black"
        await _commentary(ws, session)
        await _send(ws, {
            "type":       "game_over",
            "winner":     session.human_color,
            "draw_reason": None,
            "result":     "ai_resignation",
            "message":    f"{human_name} wins — AI resigns!",
        })
        return

    # Snapshot state BEFORE applying so the human can mark this move as bad.
    session._pre_ai_board            = session.engine.board
    session._pre_ai_move_log         = list(session.engine._move_log)
    session._pre_ai_game_record_len  = len(session.engine.game_record["moves"])
    session._pre_ai_post_placement   = session.engine._post_placement_moves
    session._pre_ai_engine_turn      = session.engine._turn_num
    session._last_ai_move            = move
    session._can_undo_ai             = True

    session.engine.apply_move(move)

    await _send(ws, {
        "type":        "ai_move",
        "from":        move.get("from"),
        "to":          move.get("to"),
        "capture":     move.get("capture"),
        "was_blunder": bool(session.game_ai and session.game_ai.last_was_blunder),
        "can_mark_bad": True,
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
    ai_thinking: bool = False

    async def _run_ai_background() -> None:
        nonlocal ai_thinking
        # Compute expected duration so we can set a server-side safety deadline.
        if session and session.engine and session.game_ai:
            _board = session.engine.board
            _diff  = session.game_ai.difficulty
            _total = sum(_board.pieces_on_board.values())
            _exp   = _expected_think_seconds(_diff, _total)
        else:
            _exp = 10.0
        _grace = 5.0

        async def _auto_force():
            await asyncio.sleep(_exp + _grace)
            if ai_thinking and session and session.game_ai:
                log.info("Server auto-forcing AI move after %.1fs", _exp + _grace)
                session.game_ai.force_stop()

        auto_task = asyncio.create_task(_auto_force())
        try:
            await _ai_turn(websocket, session)
        except Exception as exc:
            log.error("AI background task failed: %s", exc, exc_info=True)
            try:
                await _send(websocket, {"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            auto_task.cancel()
            ai_thinking = False
            log.debug("AI background task finished, ai_thinking=False")

    def _maybe_start_ai() -> None:
        nonlocal ai_thinking
        if _is_ai_turn(session) and not ai_thinking:
            ai_thinking = True
            asyncio.create_task(_run_ai_background())

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            kind = msg.get("type")

            # ── force_move — interrupt AI; received while AI task runs ────────
            if kind == "force_move":
                if session and session.game_ai:
                    session.game_ai.force_stop()
                continue

            log.debug("WS msg kind=%s", kind)

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
                    _aw      = msg.get("ai_weights") or {}
                    def _w(key, default): return int(_aw.get(key, default))
                    _hw      = HeuristicWeights(
                        close_mill=_w("close_mill", 500),
                        cycling_mill=_w("cycling_mill", 300),
                        block_opponent_mill=_w("block_opponent_mill", 400),
                        stop_opponent_mills=_w("stop_opponent_mills", 450),
                        feeder_diamond=_w("feeder_diamond", 200),
                        mill_wrapping=_w("mill_wrapping", 150),
                        cardinal_block=_w("cardinal_block", 400),
                        scatter_placement=_w("scatter_placement", 100),
                        setup_mill=_w("setup_mill", 150),
                        mill_opening=_w("mill_opening", 200),
                        long_term_position=_w("long_term_position", 100),
                        mill_count_scale=_w("mill_count_scale", 100),
                        mobility_scale=_w("mobility_scale", 100),
                        blocked_scale=_w("blocked_scale", 100),
                        make_mistakes=_w("make_mistakes", 0),
                        opening_adherence=_w("opening_adherence", 50),
                    )
                    game_ai  = GameAI(
                        color=ai_color, difficulty=diff, weights=_hw,
                        blunder_probability=_hw.make_mistakes / 100.0,
                    )

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
                            trajectory_db=_trajectory_db,
                            vs_human=True,  # coordinator always faces a human in web games
                        )
                        await asyncio.to_thread(coord.on_game_start)

                session = Session(engine, game_ai, coord, hc, vs_human)
                log.info("New game  human=%s diff=%s vs_human=%s llm=%s", hc, diff, vs_human, use_llm)
                await _send(websocket, _state(session))
                await _commentary(websocket, session)
                _maybe_start_ai()

            # ── bad_move — mark last AI move as bad, undo it, re-deliberate ───
            elif kind == "bad_move" and session and not session.vs_human:
                if not session._can_undo_ai or session._last_ai_move is None:
                    await _send(websocket, {"type": "error", "message": "Nothing to undo"})
                    continue
                if ai_thinking:
                    # Can't undo while AI is currently computing
                    await _send(websocket, {"type": "error", "message": "AI is thinking — wait"})
                    continue

                bad_move_dict  = session._last_ai_move
                bad_notation   = (
                    f"{bad_move_dict['from']}-{bad_move_dict['to']}"
                    if bad_move_dict.get("from")
                    else bad_move_dict["to"]
                )
                if bad_move_dict.get("capture"):
                    bad_notation += f"x{bad_move_dict['capture']}"

                # Collect move history BEFORE the bad move (coordinator records both sides)
                prior_notations: list[str] = []
                if session.coordinator:
                    prior_notations = [
                        m.get("notation", "")
                        for m in session.coordinator._game_moves[:-1]
                        if m.get("notation")
                    ]
                else:
                    prior_notations = [
                        m.get("notation", "")
                        for m in session.engine.game_record["moves"][:-1]
                        if m.get("notation")
                    ]

                # Persist and apply the ban to the trajectory DB
                _trajectory_db.save_bad_move(_BAD_MOVES_PATH, prior_notations, bad_notation)
                log.info("Bad move marked: %r  prior_len=%d", bad_notation, len(prior_notations))

                # ── Restore engine to pre-AI-move state ──────────────────────
                session.engine.board                 = session._pre_ai_board
                session.engine._move_log             = session._pre_ai_move_log
                session.engine.game_record["moves"]  = (
                    session.engine.game_record["moves"][:session._pre_ai_game_record_len]
                )
                session.engine._post_placement_moves = session._pre_ai_post_placement
                session.engine._turn_num             = session._pre_ai_engine_turn
                session.engine.finished              = False
                session.engine.winner                = None
                session.engine.draw_reason           = None

                # ── Restore coordinator ──────────────────────────────────────
                if session.coordinator:
                    if session.coordinator._game_moves:
                        session.coordinator._game_moves.pop()
                    session.coordinator._turn_num = max(0, session.coordinator._turn_num - 1)
                    # Reset opening recognizer — slight context loss but board is correct
                    if session.coordinator.opening_recognizer:
                        session.coordinator.opening_recognizer.reset()

                session._can_undo_ai   = False
                session._last_ai_move  = None

                await _send(websocket, {"type": "bad_move_ack", "bad_notation": bad_notation})
                await _send(websocket, _state(session))
                _maybe_start_ai()

            # ── move ──────────────────────────────────────────────────────────
            elif kind == "move" and session:
                frm = msg.get("from")  # None for placement
                to  = msg.get("to")
                move = {"from": frm, "to": to, "capture": None}

                board  = session.engine.board
                legal  = get_all_legal_moves(board)
                valid  = any(m.get("from") == frm and m["to"] == to for m in legal)

                log.info("Human move from=%s to=%s turn=%s", frm, to, board.turn)
                if not valid:
                    log.warning("Illegal move from=%s to=%s legal=%s", frm, to,
                                [(m.get("from"), m["to"]) for m in legal])
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
                session._can_undo_ai = False   # human move committed — no more undo of AI
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
                else:
                    _maybe_start_ai()

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
                session._can_undo_ai = False   # human move committed
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
                else:
                    _maybe_start_ai()

            # ── undo ──────────────────────────────────────────────────────────
            elif kind == "undo" and session and session._board_history:
                session.engine.board       = session._board_history.pop()
                session.engine.finished    = False
                session.engine.winner      = None
                session.engine.draw_reason = None
                session._pending           = None
                session._board_before      = None
                await _send(websocket, _state(session))

            # ── hint_request ──────────────────────────────────────────────────
            elif kind == "hint_request" and session:
                if not session.game_ai:
                    await _send(websocket, {"type": "error", "message": "Hints require an AI opponent."})
                    continue
                if session.hints_used >= _HINT_CAP:
                    await _send(websocket, {"type": "error", "message": "No hints remaining this game."})
                    continue

                board = session.engine.board
                hint_move = await asyncio.to_thread(session.game_ai.choose_move, board)
                session.hints_used += 1
                hints_left = _HINT_CAP - session.hints_used

                explanation: Optional[str] = None
                if session.coordinator:
                    from_pos = hint_move.get("from")
                    to_pos   = hint_move.get("to")
                    move_str = f"{from_pos}-{to_pos}" if from_pos else to_pos
                    prompt   = (
                        f"In one or two sentences, why is {move_str} a strong move "
                        f"in this position? Be specific about the strategic reason."
                    )
                    explanation = await asyncio.to_thread(
                        session.coordinator.mills_llm.player_chat, prompt, board
                    )

                await _send(websocket, {
                    "type":        "hint",
                    "from":        hint_move.get("from"),
                    "to":          hint_move.get("to"),
                    "explanation": explanation,
                    "hints_left":  hints_left,
                })

            # ── force_aggressive ──────────────────────────────────────────────
            elif kind == "force_aggressive" and session:
                if session.game_ai:
                    session.game_ai.force_aggressive = bool(msg.get("active", False))

            # ── draw_offer ────────────────────────────────────────────────────
            elif kind == "draw_offer" and session:
                engine = session.engine
                if engine.finished:
                    await _send(websocket, {"type": "error", "message": "Game already over."})
                    continue
                if engine._post_placement_moves < engine.DRAW_OFFER_THRESHOLD:
                    await _send(websocket, {"type": "error", "message": "Draw offer not available yet."})
                    continue

                # AI decides: accept if it is not winning (eval < 0.15 from AI's perspective)
                accept = True
                if session.game_ai:
                    board = engine.board
                    try:
                        eval_from_human = session.game_ai.position_eval(board)
                        # eval_from_human is White-positive; adjust sign for AI color
                        if session.game_ai.color == "W":
                            ai_eval = eval_from_human
                        else:
                            ai_eval = -eval_from_human
                        accept = ai_eval < 0.15   # AI not clearly winning
                    except Exception:
                        accept = True

                if accept:
                    engine.finished   = True
                    engine.winner     = None
                    engine.draw_reason = "agreement"
                    engine.game_record["draw_reason"] = "agreement"
                    await _send(websocket, {"type": "draw_accepted"})
                    await _game_over(websocket, session)
                else:
                    await _send(websocket, {"type": "draw_rejected"})

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
                    await _send(websocket, {
                        "type": "commentary", "speaker": "Game",
                        "text": "LLM not available — enable MillsAI commentary.",
                        "section": "human",
                    })

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as exc:
        log.error("WebSocket error: %s", exc, exc_info=True)
        try:
            await _send(websocket, {"type": "error", "message": str(exc)})
        except Exception:
            pass
