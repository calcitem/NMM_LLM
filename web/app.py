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

from game.board import BoardState
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
from ai.endgame_db import EndgameDB
from ai.starting_play import combined_family_summary
from ai.player_profile import PlayerProfile, load_profile, save_profile, is_valid_name

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

# Load evolved weights if available — produced by tools/evolve_weights.py.
_WEIGHTS_DIR = _ROOT / "data" / "weights"

def _load_evolved_weights() -> dict:
    path = _WEIGHTS_DIR / "best.json"
    if path.exists():
        try:
            d = json.loads(path.read_text())
            log.info("Evolved weights loaded from %s", path)
            return d
        except Exception as exc:
            log.warning("Could not load evolved weights: %s", exc)
    return {}

_evolved_weights: dict = _load_evolved_weights()

# Load endgame DB once at startup — provides position-exact move guidance.
_endgame_db = EndgameDB(_ROOT / "data" / "games")
try:
    _endgame_db.load()
    log.info(
        "EndgameDB: %d games, %d position entries",
        _endgame_db.game_count, _endgame_db.position_count,
    )
except Exception as _exc:
    log.warning("EndgameDB: load failed — %s", _exc)


def _load_settings() -> dict:
    try:
        return json.loads((_ROOT / "data" / "settings.json").read_text())
    except Exception:
        return {}


# ── Library consolidation (Stage 5.27) ───────────────────────────────────────

_CONSOLIDATION_INTERVAL = 50  # games between automatic DB reloads
_GAME_COUNT_PATH = _ROOT / "data" / "game_count.json"


def _get_consolidated_count() -> int:
    try:
        return json.loads(_GAME_COUNT_PATH.read_text()).get("last_consolidated", 0)
    except Exception:
        return 0


async def _maybe_consolidate(ws: "WebSocket") -> None:
    games_dir = _ROOT / "data" / "games"
    count = len(list(games_dir.glob("*.jsonl")))
    last  = await asyncio.to_thread(_get_consolidated_count)
    if count > 0 and (count - last) >= _CONSOLIDATION_INTERVAL:
        asyncio.create_task(_consolidate_libraries(ws, count))


async def _consolidate_libraries(ws: "WebSocket", game_count: int) -> None:
    global _trajectory_db, _endgame_db
    try:
        log.info("Library consolidation: %d games", game_count)
        new_tdb = TrajectoryDB(_ROOT / "data" / "games")
        await asyncio.to_thread(new_tdb.load, bad_moves_path=_BAD_MOVES_PATH)
        _trajectory_db = new_tdb

        new_edb = EndgameDB(_ROOT / "data" / "games")
        await asyncio.to_thread(new_edb.load)
        _endgame_db = new_edb

        _GAME_COUNT_PATH.write_text(json.dumps(
            {"total": game_count, "last_consolidated": game_count}, indent=2
        ))
        log.info(
            "Library reload done: %d traj entries, %d endgame positions",
            _trajectory_db.entry_count, _endgame_db.position_count,
        )
        await _send(ws, {
            "type":              "library_reload",
            "game_count":        game_count,
            "traj_entries":      _trajectory_db.entry_count,
            "endgame_positions": _endgame_db.position_count,
        })
    except Exception as exc:
        log.error("Library consolidation failed: %s", exc, exc_info=True)


# ── Adaptive Difficulty Tracker ───────────────────────────────────────────────

class AdaptiveTracker:
    """
    Tracks human win/loss streaks within one WebSocket session and adjusts
    the AI difficulty automatically.

    Softening: after SOFTEN_AFTER consecutive losses, difficulty drops by 1
    and extra blunder rate is added — the AI starts making deliberate mistakes.

    Hardening: after HARDEN_SUGGEST consecutive wins at the current level,
    the tracker suggests the player try a harder difficulty.  If the player
    was previously softened, difficulty is gradually restored toward their
    original setting instead.
    """

    SOFTEN_AFTER   = 3     # consecutive losses → auto-soften
    HARDEN_SUGGEST = 3     # consecutive wins  → suggest harder / restore
    BLUNDER_BOOST  = 0.15  # extra blunder probability per softening step

    def __init__(self) -> None:
        self.base_difficulty: int    = 3
        self.current_difficulty: int = 3
        self.extra_blunder: float    = 0.0
        self.win_streak: int         = 0
        self.loss_streak: int        = 0
        self._ever_played: bool      = False  # True after first game completes

    # ── Property helpers ──────────────────────────────────────────────────────

    @property
    def is_softened(self) -> bool:
        return self.current_difficulty < self.base_difficulty

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_new_game(self, requested_difficulty: int) -> int:
        """
        Called at the start of each game with the player's chosen difficulty.
        If the player changed the difficulty manually, reset all streaks.
        Returns the effective difficulty to use for this game.
        """
        if requested_difficulty != self.base_difficulty:
            self.base_difficulty    = requested_difficulty
            self.current_difficulty = requested_difficulty
            self.extra_blunder      = 0.0
            self.win_streak         = 0
            self.loss_streak        = 0
        return self.current_difficulty

    def record(self, human_won: bool | None) -> dict:
        """
        Record a completed game result and update streaks.
        human_won=None means draw — streaks are not changed.

        Returns an action dict:
          {"action": "softened"|"restored"|"suggest_harder"|None,
           "difficulty": int, "streak": int}
        """
        self._ever_played = True
        if human_won is None:
            return {"action": None, "difficulty": self.current_difficulty}

        if human_won:
            self.win_streak  += 1
            self.loss_streak  = 0
            if self.win_streak >= self.HARDEN_SUGGEST:
                self.win_streak = 0
                if self.is_softened:
                    self.current_difficulty = min(
                        self.base_difficulty, self.current_difficulty + 1
                    )
                    self.extra_blunder = max(0.0, self.extra_blunder - self.BLUNDER_BOOST)
                    return {"action": "restored", "difficulty": self.current_difficulty}
                else:
                    return {
                        "action": "suggest_harder",
                        "difficulty": self.current_difficulty + 1,
                    }
        else:
            self.loss_streak += 1
            self.win_streak   = 0
            if self.loss_streak >= self.SOFTEN_AFTER and self.current_difficulty > 1:
                self.loss_streak        = 0
                self.current_difficulty -= 1
                self.extra_blunder       = min(0.35, self.extra_blunder + self.BLUNDER_BOOST)
                return {"action": "softened", "difficulty": self.current_difficulty}

        return {"action": None, "difficulty": self.current_difficulty}


# ── Personality weight presets (server-side canonical values) ─────────────────
# Used by tournament games so the server applies the correct style regardless
# of client slider state. Keys mirror the personality names in the UI.

_PERSONALITY_WEIGHTS: dict[str, dict] = {
    "balanced":   {},  # defaults only
    "aggressive": {
        "close_mill": 700, "block_opponent_mill": 250, "cycling_mill": 500,
        "feeder_diamond": 300, "mill_wrapping": 250, "setup_mill": 200,
        "cardinal_block": 150, "make_mistakes": 5,
    },
    "defensive": {
        "close_mill": 350, "block_opponent_mill": 600, "stop_opponent_mills": 700,
        "cardinal_block": 550, "cycling_mill": 150, "feeder_diamond": 100,
        "mill_wrapping": 80, "make_mistakes": 3,
    },
    "positional": {
        "long_term_position": 180, "feeder_diamond": 320, "setup_mill": 200,
        "mill_count_scale": 140, "mobility_scale": 130, "blocked_scale": 130,
        "close_mill": 400, "block_opponent_mill": 350,
    },
    "scholar": {
        "opening_adherence": 90, "setup_mill": 250, "feeder_diamond": 280,
        "scatter_placement": 180, "close_mill": 450, "cycling_mill": 350,
    },
    "chaos": {
        "make_mistakes": 25, "scatter_placement": 50, "cardinal_block": 50,
        "feeder_diamond": 50, "cycling_mill": 100,
    },
}


# ── Tournament ────────────────────────────────────────────────────────────────

class TournamentState:
    """Tracks one tournament run — 6 games vs the personality roster."""

    QUALIFY_GAMES = 0  # no qualification required — tournament always available

    ROSTER: list[dict] = [  # ordered weakest → strongest
        {"name": "chaos",      "label": "Chaos — The Trickster",      "diff": 2, "elo": 720},
        {"name": "aggressive", "label": "Aggressive — The Crusher",    "diff": 3, "elo": 850},
        {"name": "scholar",    "label": "Scholar — The Bookworm",      "diff": 3, "elo": 900},
        {"name": "balanced",   "label": "Balanced",                    "diff": 4, "elo": 960},
        {"name": "defensive",  "label": "Defensive — The Blocker",     "diff": 4, "elo": 1020},
        {"name": "positional", "label": "Positional — The Strategist", "diff": 5, "elo": 1080},
    ]
    _COLORS = ["W", "B", "W", "B", "W", "B"]  # alternate for fairness

    def __init__(self) -> None:
        self.results: list[dict] = []
        self.current_idx: int   = 0
        self.player_elo: int    = 1000

    @property
    def complete(self) -> bool:
        return self.current_idx >= len(self.ROSTER)

    @property
    def current(self) -> dict | None:
        if self.complete:
            return None
        entry = self.ROSTER[self.current_idx]
        return {
            **entry,
            "human_color": self._COLORS[self.current_idx],
            "game_idx":    self.current_idx,
        }

    def record(self, winner: str | None, human_color: str) -> None:
        entry     = self.ROSTER[self.current_idx]
        human_won = (winner == human_color) if winner else None
        pts       = 2 if human_won is True else (1 if human_won is None else 0)
        ai_color = "B" if human_color == "W" else "W"
        self.results.append({
            "personality":       entry["name"],
            "label":             entry["label"],
            "difficulty":        entry["diff"],
            "human_color":       human_color,
            "white_personality": "Human" if human_color == "W" else entry["label"],
            "black_personality": "Human" if human_color == "B" else entry["label"],
            "result":            "W" if human_won is True else ("D" if human_won is None else "L"),
            "points":            pts,
        })
        # K=32 Elo update
        expected        = 1.0 / (1.0 + 10.0 ** ((entry["elo"] - self.player_elo) / 400.0))
        actual          = 1.0 if human_won is True else (0.5 if human_won is None else 0.0)
        self.player_elo = max(100, int(self.player_elo + 32 * (actual - expected)))
        self.current_idx += 1

    def total_points(self) -> int:
        return sum(r["points"] for r in self.results)

    def rank_label(self) -> str:
        pct = self.total_points() / (len(self.ROSTER) * 2)
        if pct >= 0.80: return "Master"
        if pct >= 0.60: return "Advanced"
        if pct >= 0.40: return "Intermediate"
        if pct >= 0.20: return "Beginner"
        return "Apprentice"

    def summary(self) -> dict:
        return {
            "results":     self.results,
            "points":      self.total_points(),
            "max_points":  len(self.ROSTER) * 2,
            "player_elo":  self.player_elo,
            "complete":    self.complete,
            "current_idx": self.current_idx,
            "rank_label":  self.rank_label() if self.complete else "",
        }


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
        self.adaptive: Optional[AdaptiveTracker] = None
        self.is_tournament_game: bool = False


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
    # Evolved weights are the baseline; user-saved settings override them.
    merged = {**_evolved_weights, **settings.get("ai_weights", {})}
    return JSONResponse(merged)


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


@app.get("/api/profile/{name}")
async def get_profile(name: str):
    from fastapi.responses import JSONResponse
    if not is_valid_name(name):
        return JSONResponse({"error": "Invalid name"}, status_code=400)
    profile = await asyncio.to_thread(load_profile, name)
    return JSONResponse(profile.to_dict())


@app.post("/api/profile/{name}")
async def post_profile(name: str, request: Request):
    from fastapi.responses import JSONResponse
    if not is_valid_name(name):
        return JSONResponse({"error": "Invalid name"}, status_code=400)
    body = await request.json()
    profile = await asyncio.to_thread(load_profile, name)
    for field in ("elo", "wins", "losses", "draws", "current_difficulty",
                  "win_streak", "loss_streak", "extra_blunder"):
        if field in body:
            setattr(profile, field, type(getattr(profile, field))(body[field]))
    await asyncio.to_thread(save_profile, profile)
    return JSONResponse(profile.to_dict())


@app.get("/api/openings")
async def list_openings():
    from fastapi.responses import JSONResponse
    book = OpeningBook()
    result = []
    for op in sorted(book._index.values(), key=lambda o: o.name):
        stats = op.outcome_stats
        total = stats.get("W", 0) + stats.get("B", 0) + stats.get("D", 0)
        result.append({
            "id":       op.opening_id,
            "name":     op.name,
            "family":   op.family,
            "side":     op.side,
            "moves":    op.line_moves,
            "n_moves":  len(op.line_moves),
            "tags":     op.tags,
            "notes":    op.strategic_notes,
            "total_games": total,
            "w_wins":   stats.get("W", 0),
            "b_wins":   stats.get("B", 0),
            "draws":    stats.get("D", 0),
        })
    return JSONResponse(result)


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

    # Early starting-play family detection during placement phase
    early_families: dict = {}
    placed_w = board.pieces_placed.get("W", 0)
    placed_b = board.pieces_placed.get("B", 0)
    if phase == "place" and placed_w >= 3 and placed_b >= 2:
        try:
            early_families = combined_family_summary(board)
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
        "early_families":        early_families,
        "adaptive": (
            {
                "softened":    session.adaptive.is_softened,
                "difficulty":  session.adaptive.current_difficulty,
                "base":        session.adaptive.base_difficulty,
                "win_streak":  session.adaptive.win_streak,
                "loss_streak": session.adaptive.loss_streak,
            }
            if session.adaptive and not session.vs_human
            else None
        ),
        "moves":            [
            {
                "color":    m["color"],
                "notation": m["notation"],
                "fen":      m.get("board_fen_before", ""),
            }
            for m in engine.game_record.get("moves", [])
        ],
    }


async def _send(ws: WebSocket, obj: dict) -> None:
    await ws.send_text(json.dumps(obj))


def _classify_commentary(line: str) -> tuple[str, str, str]:
    """Parse '[Speaker] text' and return (speaker, text, section).

    Section is 'human' (top box, LLM↔human) or 'ai' (bottom box, AI↔AI).
    """
    AI_SPEAKERS = {"GameAI", "Game", "MillsLLM"}
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

    adaptive_action: dict | None = None
    if session.adaptive and not session.vs_human:
        human_won: bool | None = (
            None if not winner else winner == session.human_color
        )
        result = session.adaptive.record(human_won)
        if result.get("action"):
            adaptive_action = result

    payload: dict = {
        "type": "game_over", "winner": winner,
        "draw_reason": draw_reason, "message": msg,
    }
    if adaptive_action:
        payload["adaptive"] = adaptive_action
    await _send(ws, payload)

    if session.coordinator:
        record = session.coordinator.build_game_record(
            winner=winner, human_color=session.human_color
        )
        # Tag softened games so DB loaders can skip them (Bug 8-A protection).
        if session.adaptive and session.adaptive.extra_blunder > 0:
            record["adaptive_softened"] = True
        await asyncio.to_thread(session.coordinator.on_game_end, record)
        await _commentary(ws, session)
        asyncio.create_task(_maybe_consolidate(ws))


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
        await _send(ws, {
            "type":        "game_over",
            "winner":      session.human_color,
            "draw_reason": None,
            "result":      "ai_resignation",
            "message":     f"{human_name} wins — AI resigns!",
        })
        if session.coordinator:
            record = session.coordinator.build_game_record(
                winner=session.human_color, human_color=session.human_color
            )
            await asyncio.to_thread(session.coordinator.on_game_end, record)
            await _commentary(ws, session)
            asyncio.create_task(_maybe_consolidate(ws))
        return

    # Discard stale result if the board changed while we were computing
    # (e.g. the human pressed Undo mid-think).  GameEngine.apply_move always
    # replaces session.engine.board with a new object, so identity comparison
    # reliably detects any board change that happened during the await above.
    if session.engine.board is not board:
        log.warning("Stale AI move discarded — board changed during computation (undo race)")
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
    adaptive = AdaptiveTracker()   # persists across new_game messages on this connection
    session_games: int = 0
    tournament: Optional[TournamentState] = None
    player_name: str = ""          # set from new_game; persists for the connection

    async def _after_game_end() -> None:
        nonlocal session_games
        session_games += 1

        # Update player profile on every non-tournament human-vs-AI game
        if player_name and session and not session.vs_human and not session.is_tournament_game:
            human_won = (
                None if not session.engine.winner
                else session.engine.winner == session.human_color
            )
            diff = session.game_ai.difficulty if session.game_ai else 3
            profile = await asyncio.to_thread(load_profile, player_name)
            profile.record_result(human_won, diff)
            if session.adaptive:
                profile.sync_adaptive(session.adaptive)
            await asyncio.to_thread(save_profile, profile)
            await _send(websocket, {"type": "profile_update", **profile.to_dict()})

        if tournament is None or session is None or not session.is_tournament_game:
            return
        if tournament.complete:
            return
        tournament.record(session.engine.winner, session.human_color)
        if tournament.complete:
            await _send(websocket, {"type": "tournament_complete", **tournament.summary()})
        else:
            nxt = tournament.current
            await _send(websocket, {"type": "tournament_update", **tournament.summary()})
            await _send(websocket, {
                "type":        "tournament_next",
                "game_idx":    nxt["game_idx"],
                "personality": nxt["name"],
                "label":       nxt["label"],
                "difficulty":  nxt["diff"],
                "human_color": nxt["human_color"],
            })

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
            if session and session.engine.finished:
                await _after_game_end()
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

    def _maybe_start_ai(force: bool = False) -> None:
        nonlocal ai_thinking
        if (force or _is_ai_turn(session)) and not ai_thinking:
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
                is_tournament = (
                    bool(msg.get("tournament_game", False))
                    and tournament is not None
                    and not tournament.complete
                )

                # Player profile — load once per connection on first named game
                _name_in_msg = msg.get("player_name", "").strip()[:50]
                if _name_in_msg and is_valid_name(_name_in_msg):
                    player_name = _name_in_msg
                    if not adaptive._ever_played:
                        _profile = await asyncio.to_thread(load_profile, player_name)
                        # Restore only the difficulty level — streaks/blunder reset per session.
                        adaptive.base_difficulty    = _profile.current_difficulty
                        adaptive.current_difficulty = _profile.current_difficulty
                        log.info(
                            "Profile loaded: player=%r elo=%d diff=%d",
                            player_name, _profile.elo, _profile.current_difficulty,
                        )

                if is_tournament:
                    _tnxt   = tournament.current
                    hc      = _tnxt["human_color"]
                    diff    = _tnxt["diff"]
                    _t_pers = _tnxt["name"]
                else:
                    hc_raw  = msg.get("human_color", "W")
                    hc      = _random.choice(["W", "B"]) if hc_raw == "R" else hc_raw
                    diff    = max(1, min(10, int(msg.get("difficulty", adaptive.current_difficulty))))
                    _t_pers = ""
                vs_human  = bool(msg.get("vs_human", False))
                use_llm   = bool(msg.get("use_llm", True))
                settings  = _load_settings()

                engine   = GameEngine(human_color=hc)
                game_ai  = None
                coord    = None

                if not vs_human:
                    eff_diff = adaptive.on_new_game(diff)
                    ai_color = "B" if hc == "W" else "W"
                    # Merge: evolved < personality < user-saved < per-game weights
                    # Tournament games use only evolved + personality (no user overrides).
                    _p_w = _PERSONALITY_WEIGHTS.get(_t_pers, {}) if _t_pers else {}
                    if is_tournament:
                        _aw = {**_evolved_weights, **_p_w}
                    else:
                        _aw = {**_evolved_weights, **_p_w, **settings.get("ai_weights", {}),
                               **(msg.get("ai_weights") or {})}
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
                    base_blunder = _hw.make_mistakes / 100.0
                    game_ai  = GameAI(
                        color=ai_color, difficulty=eff_diff, weights=_hw,
                        blunder_probability=min(1.0, base_blunder + adaptive.extra_blunder),
                    )
                    log.info(
                        "Adaptive: requested diff=%d effective diff=%d extra_blunder=%.2f",
                        diff, eff_diff, adaptive.extra_blunder,
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
                            endgame_db=_endgame_db,
                            vs_human=True,  # coordinator always faces a human in web games
                            human_color=hc,
                        )
                        await asyncio.to_thread(coord.on_game_start)

                session = Session(engine, game_ai, coord, hc, vs_human)
                session.is_tournament_game = is_tournament
                if not vs_human:
                    session.adaptive = adaptive
                log.info("New game  human=%s diff=%s vs_human=%s llm=%s tournament=%s",
                         hc, diff, vs_human, use_llm, is_tournament)
                await _send(websocket, _state(session))
                await _commentary(websocket, session)
                _maybe_start_ai()

            # ── setup_game — start from an editor-supplied position ────────────
            elif kind == "setup_game":
                import random as _random
                hc_raw   = msg.get("human_color", "W")
                hc       = _random.choice(["W", "B"]) if hc_raw == "R" else hc_raw
                diff     = max(1, min(10, int(msg.get("difficulty", 3))))
                vs_human = bool(msg.get("vs_human", False))
                use_llm  = bool(msg.get("use_llm", True))
                setup_phase = msg.get("phase", "move")   # "place" | "move"
                setup_turn  = msg.get("turn", "W")        # "W" | "B"
                setup_pos   = msg.get("positions", {})    # {pos: "W"|"B"|""}

                setup_board = BoardState.from_setup(setup_pos, setup_turn, setup_phase)
                engine = GameEngine(human_color=hc)
                engine.board = setup_board
                engine.game_record["setup_game"] = True
                engine.game_record["setup_fen"]  = setup_board.to_fen_string()

                game_ai = None
                coord   = None
                settings = _load_settings()

                if not vs_human:
                    ai_color = "B" if hc == "W" else "W"
                    _aw = msg.get("ai_weights") or {}
                    def _w(key, default): return int(_aw.get(key, default))
                    _hw = HeuristicWeights(
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
                    game_ai = GameAI(
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
                            endgame_db=_endgame_db,
                            vs_human=True,
                            human_color=hc,
                        )
                        await asyncio.to_thread(coord.on_game_start)

                session = Session(engine, game_ai, coord, hc, vs_human)
                log.info(
                    "Setup game  human=%s diff=%s phase=%s turn=%s W=%d B=%d",
                    hc, diff, setup_phase, setup_turn,
                    setup_board.pieces_on_board["W"], setup_board.pieces_on_board["B"],
                )
                await _send(websocket, _state(session))
                await _commentary(websocket, session)
                _maybe_start_ai()

            # ── bad_move — mark last AI move as bad, undo it, re-deliberate ───
            elif kind == "bad_move" and session and session.game_ai:
                if not session._can_undo_ai or session._last_ai_move is None:
                    await _send(websocket, {"type": "error", "message": "Nothing to undo"})
                    continue
                if ai_thinking:
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

                # Ban the move for this exact board position only.
                # If any piece moves or is captured afterward, the FEN changes and
                # the move becomes legal again — matching the player's expectation
                # that the ban is contextual, not permanent.
                _ban_fen = session._pre_ai_board.to_fen_string()
                session.game_ai.ban_move(bad_notation, _ban_fen)
                log.info("Bad move banned: %r  at fen=%s", bad_notation, _ban_fen[:24])

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
                    if session.coordinator.opening_recognizer:
                        session.coordinator.opening_recognizer.reset()

                session._can_undo_ai   = False
                session._last_ai_move  = None

                await _send(websocket, {"type": "bad_move_ack", "bad_notation": bad_notation})
                await _send(websocket, _state(session))
                _maybe_start_ai(force=True)  # force=True so it fires in human vs AI mode too

            # ── handoff_to_ai — hand one side to the AI mid-game ─────────────
            elif kind == "handoff_to_ai" and session and not session.engine.finished:
                handoff_color = msg.get("color", "W")   # "W" or "B"
                if handoff_color not in ("W", "B"):
                    await _send(websocket, {"type": "error", "message": "Invalid color for handoff"})
                    continue

                human_color = "B" if handoff_color == "W" else "W"
                diff = max(1, min(10, int(msg.get("difficulty", adaptive.current_difficulty))))
                use_llm = bool(msg.get("use_llm", True))
                _aw = {**_evolved_weights, **settings.get("ai_weights", {}),
                       **(msg.get("ai_weights") or {})}
                def _w(key, default): return int(_aw.get(key, default))
                _hw = HeuristicWeights(
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
                new_ai = GameAI(color=handoff_color, difficulty=diff, weights=_hw)

                new_coord = None
                if use_llm:
                    _s = _load_settings()
                    url   = _s.get("ollama_url",   "http://localhost:11434")
                    model = _s.get("ollama_model", "llama3.1:8b")
                    mem   = MemoryManager(ollama_url=url, ollama_model=model)
                    llm   = MillsLLM(memory=mem, ollama_url=url, model=model)
                    book  = OpeningBook()
                    rec   = OpeningRecognizer(book)
                    egr   = EndgameRecognizer(
                        active_threshold=_s.get("endgame_active_threshold", 11),
                        deep_threshold=_s.get("endgame_deep_threshold", 8),
                        zugzwang_threshold=_s.get("endgame_zugzwang_threshold", 0.4),
                    )
                    new_coord = Coordinator(
                        game_ai=new_ai, mills_llm=llm, memory=mem,
                        poor_move_threshold=_s.get("poor_move_threshold", 0.3),
                        max_poor_move_comments=_s.get("max_poor_move_comments_per_game", 5),
                        opening_recognizer=rec, endgame_recognizer=egr,
                        trajectory_db=_trajectory_db,
                        endgame_db=_endgame_db,
                        vs_human=True,
                        human_color=human_color,
                    )
                    # Seed coordinator with all moves played so far so trajectory hints work
                    for m in session.engine.game_record.get("moves", []):
                        new_coord._game_moves.append({
                            "turn":            new_coord._turn_num,
                            "color":           m.get("color", ""),
                            "type":            "place",
                            "from":            m.get("from"),
                            "to":              m.get("to"),
                            "capture":         m.get("capture"),
                            "notation":        m.get("notation", ""),
                            "board_fen_before": m.get("board_fen_before", ""),
                            "opening_recognition": {"status": "inactive", "name": None, "confidence": 0.0},
                        })
                        new_coord._turn_num += 1
                    await asyncio.to_thread(new_coord.on_game_start)

                # Update the session — the handed-off side is now AI
                session.game_ai    = new_ai
                session.coordinator = new_coord
                session.human_color = human_color
                session.vs_human   = False
                session.adaptive   = adaptive

                # Mark the game record so it saves correctly
                session.engine.game_record["human_color"] = human_color
                session.engine.game_record["handoff_game"] = True
                session.engine.game_record["handoff_ply"]  = len(
                    session.engine.game_record.get("moves", [])
                )

                log.info("Handoff: %s → AI, human stays %s diff=%d", handoff_color, human_color, diff)
                await _send(websocket, {"type": "handoff_ack", "ai_color": handoff_color,
                                        "human_color": human_color})
                await _send(websocket, _state(session))
                await _commentary(websocket, session)
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
                session._board_history.append((
                    board,
                    len(session.engine.game_record.get("moves", [])),
                    list(session.engine._move_log),
                    session.engine._post_placement_moves,
                    session.engine._turn_num,
                ))
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
                    await _after_game_end()
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
                session._board_history.append((
                    board_before,
                    len(session.engine.game_record.get("moves", [])),
                    list(session.engine._move_log),
                    session.engine._post_placement_moves,
                    session.engine._turn_num,
                ))
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
                    await _after_game_end()
                else:
                    _maybe_start_ai()

            # ── undo ──────────────────────────────────────────────────────────
            elif kind == "undo" and session and session._board_history:
                # If the AI is mid-computation, interrupt it — the board is about to
                # change so whatever move it returns will be for the wrong position.
                if ai_thinking and session.game_ai:
                    session.game_ai.force_stop()
                snap = session._board_history.pop()
                (session.engine.board,
                 gr_len,
                 session.engine._move_log,
                 session.engine._post_placement_moves,
                 session.engine._turn_num) = snap
                session.engine.game_record["moves"] = session.engine.game_record["moves"][:gr_len]
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
                    _hint_hist = [m.get("notation", "") for m in session.coordinator._game_moves if m.get("notation")]
                    explanation = await asyncio.to_thread(
                        session.coordinator.mills_llm.player_chat, prompt, board, _hint_hist
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
                    await _after_game_end()
                else:
                    await _send(websocket, {"type": "draw_rejected"})

            # ── replay_opening ────────────────────────────────────────────────
            elif kind == "replay_opening":
                opening_id   = msg.get("opening_id", "")
                speed_ms     = int(msg.get("speed_ms", 800))
                continue_mode = msg.get("continue_mode", "practice")

                book    = OpeningBook()
                opening = book._index.get(opening_id)
                if opening is None:
                    await _send(websocket, {
                        "type": "error",
                        "message": f"Opening '{opening_id}' not found.",
                    })
                    continue

                # Create a fresh engine and session for the replay
                new_engine  = GameEngine(human_color="B")
                new_session = Session(
                    engine=new_engine,
                    game_ai=None,
                    coordinator=None,
                    human_color="B",
                    vs_human=(continue_mode == "practice"),
                )
                session = new_session  # noqa: F841 — reassign nonlocal

                # Send initial state to reset the client board
                await _send(websocket, _state(session))
                await _send(websocket, {
                    "type":    "commentary",
                    "speaker": "Game",
                    "text":    f"Replaying opening: {opening.name}",
                    "section": "ai",
                })

                # Replay each placement move
                for pos in opening.line_moves:
                    await asyncio.sleep(0)  # yield to event loop
                    move = {"from": None, "to": pos, "capture": None}

                    # Check legality
                    board = session.engine.board
                    legal = get_all_legal_moves(board)
                    valid = any(m.get("from") is None and m["to"] == pos for m in legal)
                    if not valid:
                        await _send(websocket, {
                            "type":    "commentary",
                            "speaker": "Game",
                            "text":    f"Opening replay stopped: move {pos} is not legal at this point.",
                            "section": "ai",
                        })
                        break

                    # Auto-capture when a mill is formed
                    if session.engine.move_forms_mill(move):
                        caps = sorted(board.legal_captures(board.turn))
                        if caps:
                            move["capture"] = caps[0]

                    session.engine.apply_move(move)

                    await asyncio.sleep(speed_ms / 1000)
                    await _send(websocket, {
                        "type":        "ai_move",
                        "from":        None,
                        "to":          pos,
                        "capture":     move.get("capture"),
                        "was_blunder": False,
                        "can_mark_bad": False,
                    })
                    await _send(websocket, _state(session))

                    if session.engine.finished:
                        break

                # Post-replay message
                if continue_mode == "practice":
                    await _send(websocket, {
                        "type":    "commentary",
                        "speaker": "Game",
                        "text":    "Opening complete — your turn to continue from this position.",
                        "section": "ai",
                    })
                else:
                    await _send(websocket, {
                        "type":    "commentary",
                        "speaker": "Game",
                        "text":    "Opening complete — now playing AI vs AI.",
                        "section": "ai",
                    })

            # ── player_message ────────────────────────────────────────────────
            elif kind == "player_message" and session:
                text = str(msg.get("text", "")).strip()[:500]
                if not text:
                    continue
                llm = session.coordinator.mills_llm if session.coordinator else None
                if llm and llm._client:
                    board = session.engine.board
                    _hist = [m.get("notation", "") for m in (session.coordinator._game_moves if session.coordinator else []) if m.get("notation")]
                    response = await asyncio.to_thread(llm.player_chat, text, board, _hist)
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

            # ── tournament_start ──────────────────────────────────────────────
            elif kind == "tournament_start":
                tournament = TournamentState()
                nxt = tournament.current
                await _send(websocket, {
                    "type":          "tournament_init",
                    "roster":        TournamentState.ROSTER,
                    "qualify_games": TournamentState.QUALIFY_GAMES,
                    "player_elo":    tournament.player_elo,
                })
                await _send(websocket, {
                    "type":        "tournament_next",
                    "game_idx":    nxt["game_idx"],
                    "personality": nxt["name"],
                    "label":       nxt["label"],
                    "difficulty":  nxt["diff"],
                    "human_color": nxt["human_color"],
                })

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as exc:
        log.error("WebSocket error: %s", exc, exc_info=True)
        try:
            await _send(websocket, {"type": "error", "message": str(exc)})
        except Exception:
            pass
