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
from ai.opening_book import Opening, OpeningBook
from ai.opening_recognizer import OpeningRecognizer
from ai.move_guidance import build_choose_move_kwargs, pick_target_opening
from ai.endgame_recognizer import EndgameRecognizer
from ai.trajectory_db import TrajectoryDB
from ai.endgame_db import EndgameDB
from ai.fullgame_db import FullGameDB
from ai.endgame_solved_db import EndgameSolvedDB
from ai.value_net import ValueNet
from ai.starting_play import combined_family_summary
from ai.ponder import PonderManager
from ai.player_profile import PlayerProfile, load_profile, save_profile, is_valid_name

_GAMES_PATH = _ROOT / "data" / "games"
_GAMES_PATH.mkdir(parents=True, exist_ok=True)


def _persist_game_record(record: dict) -> None:
    """Write a completed game record to the games JSONL folder (no LLM required)."""
    import uuid as _uuid
    from datetime import datetime as _dt
    session_id = record.get("session_id") or str(_uuid.uuid4())
    date_str = (record.get("date") or _dt.now().isoformat())[:10]
    fname = _GAMES_PATH / f"game_{date_str}_{session_id[:8]}.jsonl"
    with open(fname, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    log.info("Game saved (no-LLM): %s", fname.name)


# Load trajectory DB once at startup — updated incrementally as games complete.
_human_games_dir = _ROOT / "data" / "human_games"
_trajectory_db   = TrajectoryDB(
    _ROOT / "data" / "games",
    extra_dirs=[_human_games_dir] if _human_games_dir.exists() else [],
)
try:
    _trajectory_db.load()
    log.info(
        "TrajectoryDB: %d games, %d state entries",
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


# Load fullgame DB at startup — path configurable via settings.json "fullgame_db_path".
_raw_fgdb = _load_settings().get("fullgame_db_path") or ""
_fgdb_path = (
    Path(_raw_fgdb) if (_raw_fgdb and Path(_raw_fgdb).is_absolute())
    else (_ROOT / (_raw_fgdb or "data/fullgame.sqlite"))
)
_fullgame_db: "FullGameDB | None" = None
if _fgdb_path.exists():
    _fullgame_db = FullGameDB(_fgdb_path)
    if _fullgame_db.is_available():
        _fgdb_stats = _fullgame_db.stats()
        log.info(
            "FullGameDB: %d positions from %s",
            _fgdb_stats.get("positions", 0), _fgdb_path,
        )
    else:
        _fullgame_db = None
        log.warning("FullGameDB: file found but could not open — %s", _fgdb_path)
else:
    log.info("FullGameDB: not found at %s", _fgdb_path)

# Load endgame solved DB at startup — dir configurable via settings.json "endgame_solved_dir".
_raw_esdb = _load_settings().get("endgame_solved_dir") or ""
_esdb_dir = (
    Path(_raw_esdb) if (_raw_esdb and Path(_raw_esdb).is_absolute())
    else (_ROOT / (_raw_esdb or "data/endgame"))
)
_endgame_solved_db: "EndgameSolvedDB | None" = None
_esdb = EndgameSolvedDB(_esdb_dir)
if _esdb.is_available():
    log.info("EndgameSolvedDB: loaded from %s", _esdb_dir)
    _endgame_solved_db = _esdb
else:
    log.info("EndgameSolvedDB: not found at %s", _esdb_dir)

# Load value network — optional; has no effect unless value_net_blend > 0 in weights.
_value_net_path = _ROOT / "data" / "value_net.npz"
_value_net: "ValueNet | None" = ValueNet.load_if_exists(_value_net_path)
if _value_net is not None:
    _vn_size_kb = round(_value_net_path.stat().st_size / 1024, 1)
    log.info("ValueNet: loaded from %s (%s KB)", _value_net_path, _vn_size_kb)
else:
    log.info("ValueNet: not found at %s", _value_net_path)


# ── Sentinel overlay (optional — only loads if checkpoint exists) ─────────────
_sentinel_advisor = None
_sentinel_ckpt = None
try:
    from learned_ai.sentinel.infer import SentinelAdvisor, load_advisor
    from learned_ai.sentinel.config import load_config as _load_sentinel_config
    _sentinel_cfg = _load_sentinel_config()
    _sentinel_ckpt = Path(_ROOT) / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"
    if _sentinel_ckpt.exists():
        _sentinel_advisor = load_advisor(str(_sentinel_ckpt), _sentinel_cfg)
        if _sentinel_advisor is not None:
            log.info("Sentinel overlay loaded from %s", _sentinel_ckpt)
        else:
            log.warning("Sentinel checkpoint found but failed to load: %s", _sentinel_ckpt)
    else:
        log.info("Sentinel checkpoint not found at %s — overlay disabled", _sentinel_ckpt)
except Exception as _e:
    log.warning("Sentinel overlay unavailable: %s", _e)

# ── Malom perfect DB (ExternalSolvedDB) — used for DB Lines overlay and DB fallback ──
# Path is read from settings.json "malom_db_path" (user-configurable via Tools page);
# falls back to the sentinel config's external_db_path when the setting is absent.
_malom_db = None
try:
    from learned_ai.sentinel.db_teacher import ExternalSolvedDB as _ExternalSolvedDB
    _malom_path = _load_settings().get("malom_db_path") or ""
    if not _malom_path:
        from learned_ai.sentinel.config import load_config as _load_sentinel_config_malom
        _mcfg = _load_sentinel_config_malom()
        _malom_path = getattr(_mcfg, "external_db_path", "") or ""
    if _malom_path:
        _malom_db = _ExternalSolvedDB(_malom_path)
        if _malom_db.is_available():
            log.info("Malom perfect DB loaded from %s", _malom_path)
        else:
            log.warning("Malom DB path configured but unavailable: %s", _malom_path)
            _malom_db = None
    else:
        log.info("Malom DB not configured (no malom_db_path in settings)")
except Exception as _e:
    log.warning("Malom DB load failed (non-fatal): %s", _e)

# Probability that sentinel (or DB fallback) intervenes, by difficulty level.
SENTINEL_PROB_BY_DIFF: dict[int, float] = {
    1: 0.0, 2: 0.0, 3: 0.10, 4: 0.22, 5: 0.33,
    6: 0.50, 7: 0.65, 8: 0.80, 9: 0.90, 10: 1.0,
}

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
        new_tdb = TrajectoryDB(
            _ROOT / "data" / "games",
            extra_dirs=[_human_games_dir] if _human_games_dir.exists() else [],
        )
        await asyncio.to_thread(new_tdb.load)
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


# ── Auto-evolve weights after N games ────────────────────────────────────────

_games_since_evolve: int = 0  # resets on restart — auto-evolve is opportunistic
_AUTO_EVOLVE_LOG = _ROOT / "data" / "weights" / "auto_evolve.log"


async def _maybe_auto_evolve() -> None:
    global _games_since_evolve
    threshold = _load_settings().get("auto_evolve_after_games", 0)
    if threshold <= 0:
        return

    _games_since_evolve += 1
    if _games_since_evolve < threshold:
        return
    _games_since_evolve = 0

    if _TOOLS_LOCK.locked():
        log.info("Auto-evolve skipped — tool already running")
        return

    log.info("Auto-evolve triggered after %d games", threshold)

    async def _run():
        async with _TOOLS_LOCK:
            cmd = [
                sys.executable, "-u",
                str(_ROOT / "tools" / "evolve_weights_v2.py"),
                "--gauntlet",
                "--generations", "15",
                "--games-per-gen", "20",
                "--parallel", "4",
                "--sigma", "0.12",
            ]
            log.info("Auto-evolve: %s", " ".join(cmd[2:]))
            try:
                _AUTO_EVOLVE_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(_AUTO_EVOLVE_LOG, "a") as fh:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=fh, stderr=fh,
                        cwd=str(_ROOT),
                    )
                    await proc.wait()
                log.info("Auto-evolve finished (rc=%s)", proc.returncode)
            except Exception as exc:
                log.error("Auto-evolve failed: %s", exc)

    asyncio.create_task(_run())


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
        # Prioritises closing mills and cycling; accepts weaker blocking in exchange.
        # block_opponent_mill kept at floor (200 = 50% of default 400) — aggressive by design.
        "close_mill": 700, "block_opponent_mill": 200, "cycling_mill": 500,
        "feeder_diamond": 300, "mill_wrapping": 250, "setup_mill": 200,
        "cardinal_block": 150, "make_mistakes": 5,
    },
    "defensive": {
        # Prioritises blocking; still closes mills at a sane rate (floor: 250).
        # % multipliers capped at 133 to avoid distorting evaluate() calibration.
        "close_mill": 350, "block_opponent_mill": 600, "stop_opponent_mills": 700,
        "cardinal_block": 400, "cycling_mill": 150, "feeder_diamond": 100,
        "mill_wrapping": 80, "make_mistakes": 3,
    },
    "positional": {
        # Structure-first; % multipliers capped at 133.
        "long_term_position": 133, "feeder_diamond": 320, "setup_mill": 200,
        "mill_count_scale": 133, "mobility_scale": 130, "blocked_scale": 130,
        "close_mill": 400, "block_opponent_mill": 350,
    },
    "scholar": {
        # Book-heavy; tactical weights within safe range.
        "opening_adherence": 90, "setup_mill": 200, "feeder_diamond": 280,
        "scatter_placement": 150, "close_mill": 450, "cycling_mill": 350,
    },
    "chaos": {
        # Intentionally erratic — makes occasional mistakes.
        "make_mistakes": 15, "scatter_placement": 50, "cardinal_block": 100,
        "feeder_diamond": 100, "cycling_mill": 150,
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


def _hint_cap_for_elo(elo: int) -> int:
    """More hints for lower-rated players; experienced players get the baseline."""
    if elo < 900:  return 7
    if elo < 1100: return 5
    if elo < 1300: return 4
    return _HINT_CAP


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
        self.hint_cap:  int  = _HINT_CAP   # adjusted per player ELO after creation
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
        self._awaiting_guided_move: bool = False  # True while human is directing AI's move
        self._resignation_pending: bool = False   # True while waiting for player to accept/decline
        self.adaptive: Optional[AdaptiveTracker] = None
        self.is_tournament_game: bool = False
        self._last_game_record: Optional[dict] = None  # stored after game ends for good_game
        self.opening_recognizer: Optional["OpeningRecognizer"] = None  # no-LLM recognition
        self.target_opening: Optional[Opening] = None
        self.game_sym_idx: int = 0
        self.game_ply_offset: int = 0
        self.opening_active: bool = False  # True while an opening replay is streaming
        # ── AI-vs-AI fields ──────────────────────────────────────────────────────
        self.ai_vs_ai: bool = False
        self.save_to_library: bool = False
        # Per-side AI + coordinator for AI-vs-AI games
        self.game_ai_white: Optional[GameAI] = None
        self.game_ai_black: Optional[GameAI] = None
        self.coordinator_white: Optional[Coordinator] = None
        self.coordinator_black: Optional[Coordinator] = None
        self.white_personality: str = ""
        self.black_personality: str = ""
        # Background task handle for AI-vs-AI loop
        self._ava_task: Optional[asyncio.Task] = None
        # B-75: pondering during the opponent's turn (None when not applicable)
        self.ponder_manager: Optional[PonderManager] = None
        # Diagnostic overlay: board after move-but-before-capture, used by get_diagnostic
        self._proj_board: Optional[BoardState] = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Nine Men's Morris")
app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")
templates = Jinja2Templates(directory=str(_WEB / "templates"))


_SETTINGS_PATH = _ROOT / "data" / "settings.json"


def _static_ver() -> str:
    """Short content hash used as a cache-busting query string for JS/CSS."""
    import hashlib
    h = hashlib.md5()
    for name in ("game.js", "style.css", "board.js", "tools.js", "tools.css"):
        p = _WEB / "static" / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:8]


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"v": _static_ver()})


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


@app.get("/api/vn_status")
async def get_vn_status():
    from fastapi.responses import JSONResponse
    if _value_net is not None and _value_net_path.exists():
        size_kb = round(_value_net_path.stat().st_size / 1024, 1)
        return JSONResponse({"loaded": True, "size_kb": size_kb})
    return JSONResponse({"loaded": False})


@app.get("/api/sentinel_status")
async def sentinel_status():
    return {
        "available":    _sentinel_advisor is not None and _sentinel_advisor.is_loaded(),
        "checkpoint":   str(_sentinel_ckpt) if _sentinel_advisor else "",
        "malom_db":     _malom_db is not None and _malom_db.is_available(),
    }


_PERSONALITIES_DIR = _ROOT / "data" / "personalities"
_VALID_PERSONALITIES = {"balanced", "aggressive", "defensive", "positional", "scholar", "chaos"}


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
    for op in sorted(book._index.values(), key=lambda o: (o.family, o.name)):
        stats = op.outcome_stats
        total = stats.get("W", 0) + stats.get("B", 0) + stats.get("D", 0)
        penalty = book.get_penalty(op.opening_id)
        result.append({
            "id":             op.opening_id,
            "name":           op.name,
            "family":         op.family,
            "side":           op.side,
            "seed_source":    op.seed_source,
            "prunable":       book.is_prunable(op.opening_id),
            "needs_llm_name": op.needs_llm_name,
            "moves":          op.line_moves,
            "n_moves":        len(op.line_moves),
            "tags":           op.tags,
            "notes":          op.strategic_notes,
            "total_games":    total,
            "w_wins":         stats.get("W", 0),
            "b_wins":         stats.get("B", 0),
            "draws":          stats.get("D", 0),
            "score_w":        round(op.opening_score("W", penalty=penalty), 3),
            "score_b":        round(op.opening_score("B", penalty=penalty), 3),
            "penalty":        round(penalty, 3),
            "branches": [
                {
                    "id":             b.branch_id,
                    "name":           b.name,
                    "deviation_ply":  b.deviation_ply,
                    "deviation_move": b.deviation_move,
                    "seed_source":    b.seed_source,
                    "total_games":    sum(b.outcome_stats.get(k, 0) for k in ("W", "B", "D")),
                    "w_wins":         b.outcome_stats.get("W", 0),
                    "b_wins":         b.outcome_stats.get("B", 0),
                    "draws":          b.outcome_stats.get("D", 0),
                }
                for b in op.branch_moves
            ],
        })
    return JSONResponse(result)


# ── Tools page ────────────────────────────────────────────────────────────────

import collections as _collections
from fastapi.responses import JSONResponse as _JSONResponse

_TOOLS_LOCK = asyncio.Lock()
_tools_proc: "asyncio.subprocess.Process | None" = None
_tools_log: "_collections.deque[str]" = _collections.deque(maxlen=500)


@app.get("/tools")
async def tools_page(request: Request):
    return templates.TemplateResponse(request, "tools.html", {"v": _static_ver()})


def _db_file_info(path: "Path | None") -> dict:
    if path is None or not path.exists():
        return {"path": str(path) if path else None, "exists": False, "size_mb": 0, "mtime": None}
    stat = path.stat()
    from datetime import datetime as _dt
    return {
        "path": str(path),
        "exists": True,
        "size_mb": round(stat.st_size / 1_048_576, 2),
        "mtime": _dt.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
    }


@app.get("/api/tool_status")
async def tool_status():
    import glob as _glob
    from datetime import datetime as _dt

    # FullGame DB
    fgdb_path = _fgdb_path if _fgdb_path.exists() else None
    fgdb_info = _db_file_info(fgdb_path)
    if _fullgame_db and _fullgame_db.is_available():
        s = _fullgame_db.stats()
        fgdb_info["positions"] = s.get("positions", 0)
        # resolved count not pre-computed (would require scanning all records)
        fgdb_info["resolved"]  = None

    # Endgame solved DB — enumerate individual WDL tables
    esdb_info = _db_file_info(_esdb_dir if _esdb_dir.exists() else None)
    wdl_tables = []
    if _esdb_dir.exists():
        for wdl_path in sorted(_esdb_dir.glob("endgame_*.wdl")):
            st = wdl_path.stat()
            wdl_tables.append({
                "name": wdl_path.stem,
                "size_mb": round(st.st_size / 1_048_576, 2),
                "mtime": _dt.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    esdb_info["tables"] = wdl_tables
    esdb_info["table_count"] = len(wdl_tables)
    if _endgame_solved_db and _endgame_solved_db.is_available():
        s2 = _endgame_solved_db.stats() if hasattr(_endgame_solved_db, "stats") else {}
        esdb_info["positions"] = s2.get("positions", 0) if s2 else 0

    # Trajectory DB
    tdb_info = {
        "games":   _trajectory_db.game_count,
        "entries": _trajectory_db.entry_count,
    }

    # Endgame (in-memory from games) DB
    edb_info = {
        "games":     _endgame_db.game_count,
        "positions": _endgame_db.position_count,
    }

    # Evolved weights
    weights_path = _ROOT / "data" / "weights" / "best.json"
    weights_info = _db_file_info(weights_path if weights_path.exists() else None)

    # Value network
    vnet_path = _ROOT / "data" / "value_net.npz"
    vnet_info = _db_file_info(vnet_path if vnet_path.exists() else None)

    # Opening book
    book = OpeningBook()
    ob_total = len(list(book.values()))
    ob_named = sum(1 for o in book.values() if not o.needs_llm_name)

    # Games dir stats
    games_files = sorted(_glob.glob(str(_ROOT / "data" / "games" / "*.jsonl")))
    games_count = len(games_files)
    games_earliest = games_files[0].split("game_")[1][:10] if games_files else None
    games_latest   = games_files[-1].split("game_")[1][:10] if games_files else None

    # Human games dir stats
    hg_files = sorted(_glob.glob(str(_ROOT / "data" / "human_games" / "human_*.jsonl")))
    hg_count = len(hg_files)
    hg_players: set = set()
    hg_earliest: str | None = None
    hg_latest:   str | None = None
    if hg_files:
        for hgf in hg_files[:5] + hg_files[-5:]:
            try:
                rec = json.loads(Path(hgf).read_text(encoding="utf-8"))
                if rec.get("date") and (hg_earliest is None or rec["date"] < hg_earliest):
                    hg_earliest = rec["date"]
                if rec.get("date") and (hg_latest is None or rec["date"] > hg_latest):
                    hg_latest = rec["date"]
                if rec.get("white_player"): hg_players.add(rec["white_player"])
                if rec.get("black_player"): hg_players.add(rec["black_player"])
            except Exception:
                pass

    busy = _TOOLS_LOCK.locked()

    # Malom perfect DB
    malom_info = {"status": "not loaded", "path": str(_load_settings().get("malom_db_path", ""))}
    if _malom_db is not None and _malom_db.is_available():
        malom_info["status"] = "loaded"

    return _JSONResponse({
        "fullgame_db":    fgdb_info,
        "endgame_solved": esdb_info,
        "trajectory_db":  tdb_info,
        "endgame_db":     edb_info,
        "weights":        weights_info,
        "value_net":      vnet_info,
        "opening_book":   {"total": ob_total, "named": ob_named},
        "games":          {"count": games_count, "earliest": games_earliest, "latest": games_latest},
        "human_games":    {"count": hg_count, "earliest": hg_earliest, "latest": hg_latest, "players": len(hg_players)},
        "busy":           busy,
        "auto_evolve":    {
            "after_games": _load_settings().get("auto_evolve_after_games", 0),
            "games_since": _games_since_evolve,
        },
        "malom_db":       malom_info,
    })


@app.get("/api/auto_evolve")
async def get_auto_evolve():
    settings = _load_settings()
    return _JSONResponse({
        "after_games": settings.get("auto_evolve_after_games", 0),
        "games_since": _games_since_evolve,
    })


@app.post("/api/auto_evolve")
async def set_auto_evolve(request: Request):
    body = await request.json()
    after = int(body.get("after_games", 0))
    settings = _load_settings()
    settings["auto_evolve_after_games"] = after
    _SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    return _JSONResponse({"ok": True, "after_games": after})


@app.get("/api/db_settings")
async def get_db_settings():
    settings = _load_settings()
    return _JSONResponse({
        "fullgame_db_path":    settings.get("fullgame_db_path", ""),
        "endgame_solved_dir":  settings.get("endgame_solved_dir", ""),
        "malom_db_path":       settings.get("malom_db_path", ""),
        "playok_archive_path": settings.get("playok_archive_path", "~/playok_archive/games"),
    })


@app.post("/api/db_settings")
async def set_db_settings(request: Request):
    body = await request.json()
    settings = _load_settings()
    changed = False
    for key in ("fullgame_db_path", "endgame_solved_dir", "malom_db_path", "playok_archive_path"):
        if key in body:
            settings[key] = str(body[key]).strip()
            changed = True
    if changed:
        _SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    return _JSONResponse({"ok": True})


@app.websocket("/ws/tools")
async def ws_tools(websocket: WebSocket):
    global _tools_proc
    await websocket.accept()

    async def _send_line(text: str, kind: str = "log") -> None:
        try:
            await websocket.send_json({"type": kind, "text": text})
        except Exception:
            pass

    try:
        msg = await websocket.receive_json()
        tool = msg.get("tool", "")
        args = msg.get("args", [])

        # Validate tool name — only allow known scripts
        _ALLOWED = {
            "build_fullgame_db", "build_endgame_db", "self_play",
            "evolve_weights", "evolve_weights_v2", "name_openings", "purge_ai_learning",
            "endgame_play", "train_value_net", "import_playok",
        }
        if tool not in _ALLOWED:
            await _send_line(f"Unknown tool: {tool!r}", "error")
            return

        # Purge requires confirmed=True in the message
        if tool == "purge_ai_learning" and not msg.get("confirmed"):
            await _send_line("confirmation_required", "confirm")
            return

        if _TOOLS_LOCK.locked():
            await _send_line("Another tool is already running. Please wait.", "error")
            return

        async with _TOOLS_LOCK:
            script = str(_ROOT / "tools" / f"{tool}.py")
            cmd = [sys.executable, "-u", script, *[str(a) for a in args]]
            log.info("Tools: running %s", cmd)
            await _send_line(f"$ {' '.join(cmd[2:])}", "cmd")

            _tools_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(_ROOT),
            )
            proc = _tools_proc

            async def _stream():
                assert proc.stdout
                async for raw in proc.stdout:
                    line = raw.decode(errors="replace").rstrip()
                    _tools_log.append(line)
                    await _send_line(line)

            stream_task = asyncio.create_task(_stream())
            await asyncio.wait_for(stream_task, timeout=3600)
            await proc.wait()
            rc = proc.returncode
            await _send_line(f"[exited {rc}]", "done" if rc == 0 else "error")
            _tools_proc = None

    except asyncio.TimeoutError:
        await _send_line("[timed out after 1 hour]", "error")
    except WebSocketDisconnect:
        log.info("Tools WebSocket disconnected — process keeps running")
    except Exception as exc:
        log.error("Tools WebSocket error: %s", exc, exc_info=True)
        await _send_line(f"[error: {exc}]", "error")


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
        "is_human_turn":    (not session.ai_vs_ai) and (session.vs_human or color == session.human_color),
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
        "hints_left":            max(0, session.hint_cap - session.hints_used),
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
        "opening_active":   session.opening_active,
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


def _sentinel_payload(adv) -> dict:
    """Serialise a move-level SentinelAdvice into the ai_move 'sentinel' dict."""
    return {
        "player":                 adv.player,
        "played_move_quality":    round(adv.played_move_quality, 3),
        "best_available_quality": round(adv.best_available_quality, 3),
        "opportunity_gap":        round(adv.opportunity_gap, 3),
        "advisory_message":       adv.advisory_message,
        "intervention":           getattr(adv, "intervention_applied", None),
        "intervention_detail":    getattr(adv, "intervention_detail", None),
    }


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

    # ── AI-vs-AI persistence ─────────────────────────────────────────────────────
    if session.ai_vs_ai:
        if session.save_to_library:
            record = dict(session.engine.game_record)
            record["winner"]            = winner
            record["ai_vs_ai"]          = True
            record["white_personality"] = session.white_personality
            record["black_personality"] = session.black_personality
            if draw_reason:
                record["draw_reason"] = draw_reason
            session._last_game_record = record
            await asyncio.to_thread(_persist_game_record, record)
            if _trajectory_db is not None:
                await asyncio.to_thread(_trajectory_db.add_game, record)
            if _endgame_db is not None:
                await asyncio.to_thread(_endgame_db.add_game, record)
            asyncio.create_task(_maybe_consolidate(ws))
            log.info("AI-vs-AI game saved  winner=%s white=%s black=%s",
                     winner, session.white_personality, session.black_personality)
        else:
            log.info("AI-vs-AI game not saved (save_to_library=False)")
        return

    if session.coordinator:
        record = session.coordinator.build_game_record(
            winner=winner, human_color=session.human_color
        )
        # Tag softened games so DB loaders can skip them (Bug 8-A protection).
        if session.adaptive and session.adaptive.extra_blunder > 0:
            record["adaptive_softened"] = True
        session._last_game_record = record
        await asyncio.to_thread(session.coordinator.on_game_end, record)
        await _commentary(ws, session)
        asyncio.create_task(_maybe_consolidate(ws))
        asyncio.create_task(_maybe_auto_evolve())

        # Prompt user to name a newly saved unnamed opening
        novel_id = session.coordinator._last_novel_id
        if novel_id:
            opening = session.coordinator.opening_recognizer.book.get_by_id(novel_id)
            if opening:
                await _send(ws, {"type": "openings_updated"})
                await _send(ws, {
                    "type":       "name_opening_prompt",
                    "opening_id": novel_id,
                    "auto_name":  opening.name,
                    "moves":      opening.line_moves[:8],
                })
            session.coordinator._last_novel_id = None

    elif not session.vs_human and session.game_ai is not None:
        # No coordinator (LLM disabled) but still an AI-vs-human game — persist
        # the engine's own game_record so TrajectoryDB and EndgameDB learn from it.
        record = dict(session.engine.game_record)
        record["winner"] = winner
        if draw_reason:
            record["draw_reason"] = draw_reason
        if session.adaptive and session.adaptive.extra_blunder > 0:
            record["adaptive_softened"] = True

        # Populate opening recognition fields from the standalone recognizer.
        if session.opening_recognizer:
            final_rec = session.opening_recognizer.get_current_result()
            record["recognised_opening_id"]    = final_rec.opening_id
            record["recognised_opening_name"]  = final_rec.name
            record["opening_recognition_status"] = final_rec.status

        session._last_game_record = record
        await asyncio.to_thread(_persist_game_record, record)
        if _trajectory_db is not None:
            await asyncio.to_thread(_trajectory_db.add_game, record)
        if _endgame_db is not None:
            await asyncio.to_thread(_endgame_db.add_game, record)

        # Save novel / unmatched openings and notify the frontend.
        novel_id = None
        if session.opening_recognizer:
            final_rec = session.opening_recognizer.get_current_result()
            if final_rec.status in ("novel", "inactive"):
                placement_moves = [
                    m["to"] for m in record.get("moves", [])
                    if m.get("type") == "place"
                ]
                if len(placement_moves) >= 6:
                    book = session.opening_recognizer.book
                    first3 = "-".join(placement_moves[:3])
                    auto_name = f"Novel — {first3}"
                    sigs = Coordinator._compute_fen_signatures(placement_moves)
                    similar = book.find_similar(placement_moves, min_common=4)
                    if similar:
                        canonical = max(
                            similar,
                            key=lambda o: sum(o.outcome_stats.get(k, 0) for k in ("W","B","D")),
                        )
                        if winner in ("W", "B", "D"):
                            canonical.outcome_stats[winner] = canonical.outcome_stats.get(winner, 0) + 1
                        book.save_opening(canonical)
                        novel_id = canonical.opening_id
                    else:
                        novel = book.save_novel_opening(
                            placement_moves, sigs,
                            outcome=winner,
                            needs_llm_name=True,
                        )
                        novel.name = auto_name
                        book.save_opening(novel)
                        novel_id = novel.opening_id
                    # Notify frontend; /api/openings reads from disk each call so
                    # the client will get the new entry on its next fetch.
                    await _send(ws, {"type": "openings_updated"})

        if novel_id:
            book = session.opening_recognizer.book  # type: ignore[union-attr]
            opening = book.get_by_id(novel_id)
            if opening:
                await _send(ws, {
                    "type":       "name_opening_prompt",
                    "opening_id": novel_id,
                    "auto_name":  opening.name,
                    "moves":      opening.line_moves[:8],
                })

        asyncio.create_task(_maybe_consolidate(ws))
        asyncio.create_task(_maybe_auto_evolve())


def _make_game_ai_for_personality(color: str, personality: str, difficulty: int) -> GameAI:
    """Build a GameAI instance using the server-canonical personality weights."""
    _p_w = _PERSONALITY_WEIGHTS.get(personality, {})
    _aw  = {**_evolved_weights, **_p_w}
    def _w(key, default): return int(_aw.get(key, default))
    hw = HeuristicWeights(
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
        value_net_blend=_w("value_net_blend", 0),
        cross_mill_cycling=_w("cross_mill_cycling", 300),
    )
    return GameAI(
        color=color, difficulty=difficulty, weights=hw,
        blunder_probability=hw.make_mistakes / 100.0,
        fullgame_db=_fullgame_db,
        endgame_solved_db=_endgame_solved_db,
        malom_db=_malom_db,
        value_net=_value_net,
    )


async def _run_ai_vs_ai_loop(ws: WebSocket, session: Session) -> None:
    """Drive an AI-vs-AI game: alternate moves for W and B until the game ends."""
    import time as _time
    log.info("AI-vs-AI loop started  white=%s black=%s save=%s",
             session.white_personality, session.black_personality, session.save_to_library)
    try:
        while not session.engine.finished:
            board = session.engine.board
            color = board.turn
            game_ai    = session.game_ai_white if color == "W" else session.game_ai_black
            coord      = session.coordinator_white if color == "W" else session.coordinator_black
            opp_coord  = session.coordinator_black if color == "W" else session.coordinator_white

            if game_ai is None:
                log.error("AI-vs-AI: no game_ai for color=%s — aborting", color)
                break

            total = sum(board.pieces_on_board.values())
            diff  = game_ai.difficulty
            exp   = _expected_think_seconds(diff, total)

            await _send(ws, {
                "type":             "thinking",
                "color":            color,
                "expected_seconds": exp,
            })

            t0 = _time.time()
            try:
                if coord:
                    move = await asyncio.to_thread(coord.deliberate, board)
                else:
                    move = await asyncio.to_thread(game_ai.choose_move, board)
            except Exception as exc:
                log.error("AI-vs-AI deliberation failed: %s", exc, exc_info=True)
                await _send(ws, {"type": "error", "message": f"AI error: {exc}"})
                return

            elapsed = _time.time() - t0
            log.info("AI-vs-AI move  color=%s move=%s elapsed=%.2fs", color, move, elapsed)

            # Check board identity (shouldn't change, but guard anyway)
            if session.engine.board is not board:
                log.warning("AI-vs-AI: stale move discarded")
                break

            session.engine.apply_move(move)

            _avai_move_msg = {
                "type":        "ai_move",
                "from":        move.get("from"),
                "to":          move.get("to"),
                "capture":     move.get("capture"),
                "was_blunder": bool(game_ai.last_was_blunder),
                "can_mark_bad": False,
            }
            if game_ai and game_ai.last_sentinel_advice is not None:
                _avai_move_msg["sentinel"] = _sentinel_payload(game_ai.last_sentinel_advice)
                game_ai.last_sentinel_advice = None  # consume after sending
            await _send(ws, _avai_move_msg)

            # Flush commentary from the active coordinator
            if coord:
                for line in coord.flush_dialogue():
                    speaker, text, section = _classify_commentary(line)
                    await _send(ws, {
                        "type": "commentary",
                        "speaker": speaker,
                        "text": text,
                        "section": section,
                    })

            await _send(ws, _state(session))

            if session.engine.finished:
                break

            # Brief pause so the board animates move-by-move
            await asyncio.sleep(1.0)

        # Game ended — send game_over and handle persistence
        await _game_over(ws, session)
    except asyncio.CancelledError:
        log.info("AI-vs-AI loop cancelled")
    except Exception as exc:
        log.error("AI-vs-AI loop error: %s", exc, exc_info=True)
        try:
            await _send(ws, {"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        session._ava_task = None
        log.info("AI-vs-AI loop finished")


def _expected_think_seconds(difficulty: int, total_pieces: int) -> float:
    if total_pieces < 10:
        return 4.0
    # Time-limited levels: return the actual budget so the UI countdown matches.
    budgets = {5: 12, 6: 18, 7: 20, 8: 30, 9: 40, 10: 60}
    if difficulty in budgets:
        return float(budgets[difficulty])
    # Fixed-depth levels (1–4): generous estimate so force_move doesn't fire mid-search.
    estimates = {1: 3, 2: 3, 3: 6, 4: 9}
    return float(estimates.get(difficulty, 5))


def _nollm_choose_move(session: Session, board: BoardState) -> dict:
    """Choose a move on the no-LLM path with opening + trajectory guidance (B-65)."""
    game_ai = session.game_ai
    if game_ai is None:
        return {}
    game_moves = session.engine.game_record.get("moves", [])
    kwargs = build_choose_move_kwargs(
        board,
        game_ai,
        game_moves,
        opening_recognizer=session.opening_recognizer,
        target_opening=session.target_opening,
        game_sym_idx=session.game_sym_idx,
        ply_offset=session.game_ply_offset,
        trajectory_db=_trajectory_db,
    )

    # Re-target when the recognizer has found a FEN transposition to a different opening.
    # The current ply is handled correctly by the recognition book_move; the retarget
    # ensures the new opening guides subsequent plies if recognition drops to "novel".
    _rec = kwargs.get("recognition")
    if (
        _rec is not None
        and _rec.status == "transposition"
        and session.opening_recognizer is not None
        and _rec.opening_id != (session.target_opening.opening_id if session.target_opening else None)
    ):
        new_opening = session.opening_recognizer.book.get_by_id(_rec.opening_id)
        if new_opening is not None:
            session.target_opening = new_opening
            session.game_sym_idx = 0
            session.game_ply_offset = _rec.matched_ply
            log.info(
                "Re-targeted opening on transposition: %s (ply_offset=%d)",
                new_opening.name, _rec.matched_ply,
            )

    trajectory_context = kwargs.pop("trajectory_context", "")
    if trajectory_context:
        log.info("No-LLM trajectory: %s", trajectory_context)
    return game_ai.choose_move(board, **kwargs)


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

    # B-75: stop any running ponder; check if we pre-computed this position.
    _ponder_hit: dict | None = None
    if session.ponder_manager is not None and not session.coordinator:
        session.ponder_manager.stop()
        _ponder_hit = session.ponder_manager.get_result(board)

    t0 = _time.time()
    try:
        if _ponder_hit is not None:
            move = _ponder_hit
            if session.game_ai:
                session.game_ai.last_was_blunder = False
                session.game_ai.last_thinking    = "pondered"
        elif session.coordinator:
            move = await asyncio.to_thread(session.coordinator.deliberate, board)
        else:
            move = await asyncio.to_thread(_nollm_choose_move, session, board)
    except Exception as exc:
        log.error("AI deliberation failed: %s", exc, exc_info=True)
        raise

    elapsed = _time.time() - t0
    log.info("AI turn end    move=%s elapsed=%.2fs ponder_hit=%s", move, elapsed, _ponder_hit is not None)

    # Latch whether the coordinator wants to resign (cleared so it only fires once).
    resignation_offered = bool(
        session.coordinator and session.coordinator.resignation_offered
    )
    if resignation_offered and session.coordinator:
        session.coordinator.resignation_offered = False

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

    # Advance no-LLM opening recognizer for AI placement moves.
    if session.opening_recognizer and not session.coordinator and move.get("from") is None:
        session.opening_recognizer.update(move.get("to", ""), session.engine.board)

    _ai_move_msg: dict = {
        "type":        "ai_move",
        "from":        move.get("from"),
        "to":          move.get("to"),
        "capture":     move.get("capture"),
        "was_blunder": bool(session.game_ai and session.game_ai.last_was_blunder),
        "can_mark_bad": True,
    }
    if session.coordinator and session.coordinator.last_thinking:
        _ai_move_msg["thinking"] = session.coordinator.last_thinking
    if session.game_ai and session.game_ai.last_sentinel_advice is not None:
        _ai_move_msg["sentinel"] = _sentinel_payload(session.game_ai.last_sentinel_advice)
        session.game_ai.last_sentinel_advice = None  # consume after sending
    await _send(ws, _ai_move_msg)
    await _commentary(ws, session)
    await _send(ws, _state(session))

    if session.engine.finished:
        await _game_over(ws, session)
        return

    # B-75: start pondering from the predicted opponent reply.
    # Only when no coordinator (no-LLM path) and ponder_manager is set.
    if session.ponder_manager is not None and session.game_ai is not None:
        _game_moves = session.engine.game_record.get("moves", [])
        _game_notations = [m["notation"] for m in _game_moves]
        session.ponder_manager.start(
            board=session.engine.board,
            game_ai=session.game_ai,
            game_notations=_game_notations,
            trajectory_db=_trajectory_db,
        )

    # Offer resignation after move + commentary — player decides whether to accept.
    if resignation_offered and not session.engine.finished:
        session._resignation_pending = True
        await _send(ws, {"type": "resignation_offer"})


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
    player_elo:  int = 1000        # updated when profile is loaded; used for hint cap

    async def _after_game_end() -> None:
        nonlocal session_games
        session_games += 1

        # Update player profile on every non-tournament human-vs-AI game
        if player_name and session and not session.vs_human and not session.is_tournament_game and not session.ai_vs_ai:
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
                if session and session.ai_vs_ai:
                    # Stop whichever side is currently computing
                    color = session.engine.board.turn if session.engine else "W"
                    ai_to_stop = session.game_ai_white if color == "W" else session.game_ai_black
                    if ai_to_stop:
                        ai_to_stop.force_stop()
                elif session and session.game_ai:
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
                        player_elo = _profile.elo
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
                use_sentinel   = bool(msg.get("use_sentinel", False))
                sentinel_mode  = msg.get("sentinel_mode", "advisory")  # "advisory"|"score_adjust"|"reconsider"
                use_perfect_db = bool(msg.get("use_perfect_db", False))
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
                        value_net_blend=_w("value_net_blend", 0),
                        cross_mill_cycling=_w("cross_mill_cycling", 300),
                    )
                    base_blunder = _hw.make_mistakes / 100.0
                    game_ai  = GameAI(
                        color=ai_color, difficulty=eff_diff, weights=_hw,
                        blunder_probability=min(1.0, base_blunder + adaptive.extra_blunder),
                        fullgame_db=_fullgame_db,
                        endgame_solved_db=_endgame_solved_db,
                        malom_db=_malom_db,
                        value_net=_value_net,
                    )
                    log.info(
                        "Adaptive: requested diff=%d effective diff=%d extra_blunder=%.2f",
                        diff, eff_diff, adaptive.extra_blunder,
                    )

                    _sent_prob = SENTINEL_PROB_BY_DIFF.get(eff_diff, 0.0)
                    if use_perfect_db:
                        game_ai.use_perfect_db = True
                        game_ai.sentinel_mode = "score_adjust"
                        game_ai._sentinel_activation_prob = _sent_prob
                        log.info("Malom perfect DB guidance enabled (diff=%d, prob=%.0f%%)", eff_diff, _sent_prob * 100)
                    elif (_sent_prob > 0.0 or use_sentinel) and _sentinel_advisor is not None and _sentinel_advisor.is_loaded():
                        _sent_mode = sentinel_mode if use_sentinel else "score_adjust"
                        game_ai.set_sentinel(_sentinel_advisor, mode=_sent_mode)
                        game_ai._sentinel_activation_prob = _sent_prob if not use_sentinel else 1.0
                        log.info("Sentinel attached (diff=%d, prob=%.0f%%, mode=%s)", eff_diff, game_ai._sentinel_activation_prob * 100, _sent_mode)
                    elif _sent_prob > 0.0 and _sentinel_advisor is None:
                        game_ai.sentinel_mode = "score_adjust"
                        game_ai._sentinel_activation_prob = _sent_prob
                        log.info("Sentinel unavailable — DB fallback (diff=%d, prob=%.0f%%)", eff_diff, _sent_prob * 100)

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
                    session.hint_cap = _hint_cap_for_elo(player_elo)
                    # Always run opening recognition, even without LLM.
                    # The coordinator owns its own recognizer; only wire up
                    # the standalone one when no coordinator is present.
                    if coord is None and game_ai is not None:
                        _nollm_book = OpeningBook()
                        session.opening_recognizer = OpeningRecognizer(_nollm_book)
                        target, sym_idx = pick_target_opening(_nollm_book, game_ai.color)
                        session.target_opening = target
                        session.game_sym_idx = sym_idx
                        session.game_ply_offset = 0
                        if target:
                            log.info(
                                "No-LLM targeting opening: %s sym=%d score=%.2f",
                                target.name, sym_idx,
                                target.opening_score(game_ai.color),
                            )
                        # B-75: enable pondering at difficulty >= 3 (where search depth
                        # is deep enough for pre-computation to provide real benefit).
                        if game_ai.difficulty >= 3:
                            session.ponder_manager = PonderManager()
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
                use_llm        = bool(msg.get("use_llm", True))
                use_sentinel   = bool(msg.get("use_sentinel", False))
                sentinel_mode  = msg.get("sentinel_mode", "advisory")
                use_perfect_db = bool(msg.get("use_perfect_db", False))
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
                        value_net_blend=_w("value_net_blend", 0),
                        cross_mill_cycling=_w("cross_mill_cycling", 300),
                    )
                    game_ai = GameAI(
                        color=ai_color, difficulty=diff, weights=_hw,
                        blunder_probability=_hw.make_mistakes / 100.0,
                        fullgame_db=_fullgame_db,
                        endgame_solved_db=_endgame_solved_db,
                        malom_db=_malom_db,
                        value_net=_value_net,
                    )

                    _sent_prob_s = SENTINEL_PROB_BY_DIFF.get(diff, 0.0)
                    if use_perfect_db:
                        game_ai.use_perfect_db = True
                        game_ai.sentinel_mode = "score_adjust"
                        game_ai._sentinel_activation_prob = _sent_prob_s
                        log.info("Malom perfect DB guidance enabled (diff=%d, prob=%.0f%%)", diff, _sent_prob_s * 100)
                    elif (_sent_prob_s > 0.0 or use_sentinel) and _sentinel_advisor is not None and _sentinel_advisor.is_loaded():
                        _sent_mode_s = sentinel_mode if use_sentinel else "score_adjust"
                        game_ai.set_sentinel(_sentinel_advisor, mode=_sent_mode_s)
                        game_ai._sentinel_activation_prob = _sent_prob_s if not use_sentinel else 1.0
                        log.info("Sentinel attached (diff=%d, prob=%.0f%%, mode=%s)", diff, game_ai._sentinel_activation_prob * 100, _sent_mode_s)
                    elif _sent_prob_s > 0.0 and _sentinel_advisor is None:
                        game_ai.sentinel_mode = "score_adjust"
                        game_ai._sentinel_activation_prob = _sent_prob_s
                        log.info("Sentinel unavailable — DB fallback (diff=%d, prob=%.0f%%)", diff, _sent_prob_s * 100)

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
                if not vs_human and coord is None and game_ai is not None:
                    _nollm_book = OpeningBook()
                    session.opening_recognizer = OpeningRecognizer(_nollm_book)
                    target, sym_idx = pick_target_opening(_nollm_book, game_ai.color)
                    session.target_opening = target
                    session.game_sym_idx = sym_idx
                    session.game_ply_offset = 0
                log.info(
                    "Setup game  human=%s diff=%s phase=%s turn=%s W=%d B=%d",
                    hc, diff, setup_phase, setup_turn,
                    setup_board.pieces_on_board["W"], setup_board.pieces_on_board["B"],
                )
                await _send(websocket, _state(session))
                await _commentary(websocket, session)
                _maybe_start_ai()

            # ── override_ai — undo last AI move; human will direct it manually ─
            elif kind == "override_ai" and session and session.game_ai:
                if session.opening_active:
                    await _send(websocket, {"type": "error", "message": "Cannot override during opening replay"})
                    continue
                if not session._can_undo_ai or session._last_ai_move is None:
                    await _send(websocket, {"type": "error", "message": "Nothing to override"})
                    continue
                if ai_thinking:
                    await _send(websocket, {"type": "error", "message": "AI is thinking — wait"})
                    continue

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

                session._can_undo_ai           = False
                session._last_ai_move          = None
                session._awaiting_guided_move  = True

                log.info("override_ai: board restored to pre-AI state; awaiting guided move")
                await _send(websocket, {"type": "override_ready"})
                await _send(websocket, _state(session))

            # ── guided_move — human directs the AI's move after override ─────
            elif kind == "guided_move" and session and session._awaiting_guided_move:
                frm = msg.get("from")   # None for placement
                to  = msg.get("to")
                if not to:
                    await _send(websocket, {"type": "error", "message": "guided_move missing 'to'"})
                    continue

                board  = session.engine.board
                legal  = get_all_legal_moves(board)
                valid  = any(m.get("from") == frm and m["to"] == to for m in legal)
                if not valid:
                    await _send(websocket, {"type": "error", "message": f"Illegal guided move to {to}"})
                    continue

                move = {"from": frm, "to": to, "capture": None}

                # Auto-pick capture if the guided move closes a mill.
                if session.engine.move_forms_mill(move):
                    caps = board.legal_captures(board.turn)
                    move["capture"] = caps[0] if caps else None

                session._awaiting_guided_move = False
                session.engine.apply_move(move)

                if session.opening_recognizer and not session.coordinator and frm is None:
                    session.opening_recognizer.update(to, session.engine.board)

                log.info("guided_move applied: from=%s to=%s capture=%s", frm, to, move.get("capture"))
                await _send(websocket, {
                    "type":        "ai_move",
                    "from":        frm,
                    "to":          to,
                    "capture":     move.get("capture"),
                    "was_blunder": False,
                    "can_mark_bad": False,
                })
                await _commentary(websocket, session)
                await _send(websocket, _state(session))

                if session.engine.finished:
                    await _game_over(websocket, session)
                    await _after_game_end()

            # ── accept_resignation — player accepts the AI's offer to resign ────
            elif kind == "accept_resignation" and session and session._resignation_pending:
                session._resignation_pending = False
                session.engine.finished = True
                session.engine.winner   = session.human_color
                human_name = "White" if session.human_color == "W" else "Black"
                await _send(websocket, {
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
                    await _commentary(websocket, session)
                    asyncio.create_task(_maybe_consolidate(websocket))
                await _after_game_end()

            # ── decline_resignation — player forces the AI to keep playing ──────
            elif kind == "decline_resignation" and session:
                session._resignation_pending = False
                if session.coordinator:
                    session.coordinator._dominant_turn_streak = 0
                log.info("Resignation declined — game continues.")

            # ── get_diagnostic — score all legal moves for the overlay ──────────
            elif kind == "get_diagnostic" and session:
                from fastapi.responses import JSONResponse as _JR
                from ai.heuristics import (
                    tactical_move_bonus as _tac_bonus,
                    evaluate as _heval,
                    DEFAULT_WEIGHTS as _DW,
                )
                diag_mode   = msg.get("mode", "static")  # "static" | "negamax" | "capture"
                diag_depth  = max(1, min(5, int(msg.get("depth", 3))))
                diag_seq    = msg.get("seq", 0)
                fen_override = msg.get("fen")             # replay positions

                # Server-side guard: skip negamax diagnostics while AI search is running
                # to prevent race conditions on shared GameAI state (TT, killers, history)
                if ai_thinking and diag_mode == "negamax":
                    continue

                # Determine board to analyse
                if diag_mode == "capture" and session._proj_board is not None:
                    diag_board = session._proj_board
                elif fen_override:
                    try:
                        diag_board = BoardState.from_fen_string(fen_override)
                    except Exception:
                        diag_board = session.engine.board
                else:
                    diag_board = session.engine.board

                color   = diag_board.turn
                weights = session.game_ai._weights if session.game_ai else _DW

                eval_w = int(_heval(diag_board, "W"))
                eval_b = int(_heval(diag_board, "B"))

                if diag_mode == "capture":
                    # Score each legal capture from the perspective of the player who just moved.
                    # The just-moved player's color = opponent of diag_board.turn.
                    mover = "B" if diag_board.turn == "W" else "W"
                    caps = diag_board.legal_captures(mover)
                    moves_out = []
                    for cap_pos in caps:
                        after_cap = diag_board.apply_move({"from": None, "to": None, "capture": cap_pos})
                        moves_out.append({
                            "from": None, "to": cap_pos, "capture": None,
                            "score": int(_heval(after_cap, mover)),
                        })
                    moves_out.sort(key=lambda x: x["score"], reverse=True)

                elif diag_mode == "negamax" and session.game_ai:
                    game_notations = [
                        m.get("notation", "")
                        for m in session.engine.game_record.get("moves", [])
                    ]
                    if fen_override:
                        prefix_json = msg.get("prefix", [])
                        game_notations = list(prefix_json) if prefix_json else []
                    moves_out = await asyncio.to_thread(
                        session.game_ai.diagnostic_scores,
                        diag_board, diag_depth, game_notations,
                    )

                else:
                    # Static: tac_bonus + evaluate for each legal move
                    legal = get_all_legal_moves(diag_board)
                    moves_out = []
                    for mv in legal:
                        after = diag_board.apply_move(mv)
                        tac   = _tac_bonus(diag_board, after, color, weights,
                                           return_breakdown=True)
                        ev    = int(_heval(after, color))
                        moves_out.append({
                            "from":      mv.get("from"),
                            "to":        mv["to"],
                            "capture":   mv.get("capture"),
                            "tac_total": int(tac["total"]),
                            "tac_terms": [[lbl, val] for lbl, val in tac.get("top_terms", [])],
                            "eval_score": ev,
                            "score":     int(tac["total"]) + ev,
                        })
                    moves_out.sort(key=lambda x: x["score"], reverse=True)

                # ── Merge DB data into every move entry ─────────────────────
                def _diag_ntn(mv_entry):
                    frm = mv_entry.get("from")
                    to  = mv_entry.get("to") or ""
                    cap = mv_entry.get("capture")
                    s = f"{frm}-{to}" if frm else to
                    if cap:
                        s += f"x{cap}"
                    return s

                # Trajectory DB: per-move relative frequency at this board state
                traj_freqs: dict = {}
                if _trajectory_db:
                    try:
                        traj_freqs = _trajectory_db.query_all_frequencies(diag_board)
                    except Exception:
                        pass

                # FullGame DB: per-move WIN/LOSS/NEUTRAL delta
                db_deltas: dict = {}
                if _fullgame_db and _fullgame_db.is_available():
                    try:
                        db_deltas = _fullgame_db.score_delta(diag_board, color)
                    except Exception:
                        pass

                # Endgame DB: probe each resulting position for WDL
                eg_flags: dict = {}
                if _endgame_solved_db:
                    total_pc = sum(diag_board.pieces_on_board.values())
                    all_placed = (diag_board.pieces_placed.get("W", 0) >= 9
                                  and diag_board.pieces_placed.get("B", 0) >= 9)
                    if all_placed and total_pc <= 8:
                        if diag_mode == "capture":
                            mover_eg = "B" if diag_board.turn == "W" else "W"
                            for mv_e in moves_out:
                                cap_pos = mv_e.get("to")
                                if cap_pos:
                                    after_eg = diag_board.apply_move(
                                        {"from": None, "to": None, "capture": cap_pos})
                                    res = _endgame_solved_db.query(after_eg)
                                    if res:
                                        # res is from after_eg.turn (opponent) POV; flip for us
                                        eg_flags[cap_pos] = (
                                            "W" if res == "L" else
                                            "L" if res == "W" else "D"
                                        )
                        else:
                            try:
                                legal_eg = get_all_legal_moves(diag_board)
                            except Exception:
                                legal_eg = []
                            for mv_eg in legal_eg:
                                try:
                                    after_eg = diag_board.apply_move(mv_eg)
                                    res = _endgame_solved_db.query(after_eg)
                                    if res:
                                        ntn_eg = _diag_ntn(mv_eg)
                                        eg_flags[ntn_eg] = (
                                            "W" if res == "L" else
                                            "L" if res == "W" else "D"
                                        )
                                except Exception:
                                    pass

                # Malom perfect DB: fill eg_flags for any move not covered above
                if _malom_db is not None and _malom_db.is_available():
                    _flip = {"W": "L", "L": "W", "D": "D"}
                    if diag_mode == "capture":
                        for mv_e in moves_out:
                            cap_pos = mv_e.get("to")
                            if cap_pos and not eg_flags.get(cap_pos):
                                try:
                                    after_ml = diag_board.apply_move(
                                        {"from": None, "to": None, "capture": cap_pos})
                                    res_ml = _malom_db.query(after_ml)
                                    if res_ml:
                                        eg_flags[cap_pos] = _flip.get(res_ml)
                                except Exception:
                                    pass
                    else:
                        try:
                            legal_ml = get_all_legal_moves(diag_board)
                        except Exception:
                            legal_ml = []
                        for mv_ml in legal_ml:
                            ntn_ml = _diag_ntn(mv_ml)
                            if eg_flags.get(ntn_ml):
                                continue  # already set by endgame_solved_db
                            try:
                                after_ml = diag_board.apply_move(mv_ml)
                                res_ml = _malom_db.query(after_ml)
                                if res_ml:
                                    eg_flags[ntn_ml] = _flip.get(res_ml)
                            except Exception:
                                pass

                for mv_e in moves_out:
                    ntn = _diag_ntn(mv_e)
                    mv_e["traj_freq"] = round(traj_freqs.get(ntn, 0.0), 3)
                    mv_e["db_delta"]  = db_deltas.get(ntn)   # float or None
                    # Capture mode eg_flags keyed by captured square
                    cap_pos = mv_e.get("to") if diag_mode == "capture" else None
                    mv_e["eg_flag"] = eg_flags.get(cap_pos or ntn)  # "W"/"L"/"D"/None

                # ── Sentinel overlay: score each legal move ───────────────────
                if _sentinel_advisor is not None and _sentinel_advisor.is_loaded():
                    try:
                        candidates = [
                            {"from": mv_e.get("from"), "to": mv_e.get("to"),
                             "capture": mv_e.get("capture")}
                            for mv_e in moves_out
                        ]
                        if candidates:
                            sent_advice = await asyncio.to_thread(
                                _sentinel_advisor.advise,
                                diag_board, candidates, color, 0,
                            )
                            if sent_advice is not None:
                                for i, mv_e in enumerate(moves_out):
                                    if i < len(sent_advice.move_scores):
                                        mv_e["sentinel_score"] = round(sent_advice.move_scores[i], 3)
                    except Exception as _se:
                        log.debug("Sentinel diagnostic scoring failed: %s", _se)
                for mv_e in moves_out:
                    if "sentinel_score" not in mv_e:
                        mv_e["sentinel_score"] = None

                await _send(websocket, {
                    "type":    "diagnostic",
                    "seq":     diag_seq,
                    "mode":    diag_mode,
                    "color":   color,
                    "eval_w":  eval_w,
                    "eval_b":  eval_b,
                    "moves":   moves_out,
                    "fen":     fen_override or diag_board.to_fen_string(),
                })

            # ── good_game — elevate a draw to win-like status in trajectory ────
            elif kind == "good_game" and session:
                rec = session._last_game_record
                if rec is None or rec.get("winner") is not None:
                    await _send(websocket, {"type": "error",
                        "message": "Good Game only available after a draw."})
                else:
                    ai_color = "B" if session.human_color == "W" else "W"
                    boosted = dict(rec, winner=ai_color)
                    if session.coordinator and session.coordinator.trajectory_db is not None:
                        await asyncio.to_thread(
                            session.coordinator.trajectory_db.add_game, boosted)
                    if session.coordinator and session.coordinator.endgame_db is not None:
                        await asyncio.to_thread(
                            session.coordinator.endgame_db.add_game, boosted)
                    session._last_game_record = None  # prevent double-boosting
                    await _send(websocket, {"type": "good_game_ack"})

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
                    value_net_blend=_w("value_net_blend", 0),
                    cross_mill_cycling=_w("cross_mill_cycling", 300),
                )
                new_ai = GameAI(color=handoff_color, difficulty=diff, weights=_hw, fullgame_db=_fullgame_db, endgame_solved_db=_endgame_solved_db, malom_db=_malom_db, value_net=_value_net)

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
                # B-75: stop ponder early so the thread doesn't fight the upcoming search.
                if session.ponder_manager is not None and session.ponder_manager.is_running():
                    session.ponder_manager.stop()

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
                    session._proj_board = projected
                    await _send(websocket, {
                        "type":            "capture_required",
                        "legal_captures":  caps,
                        "projected_board": dict(projected.positions),
                        "projected_fen":   projected.to_fen_string(),
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
                elif session.opening_recognizer and move.get("from") is None:
                    session.opening_recognizer.update(move.get("to", ""), session.engine.board)

                await _commentary(websocket, session)
                await _send(websocket, _state(session))

                if session.engine.finished:
                    await _game_over(websocket, session)
                    await _after_game_end()
                else:
                    _maybe_start_ai()

            # ── capture ───────────────────────────────────────────────────────
            elif kind == "capture" and session and session._pending:
                # B-75: stop ponder — the capture completes the human's move.
                if session.ponder_manager is not None and session.ponder_manager.is_running():
                    session.ponder_manager.stop()

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
                if session.hints_used >= session.hint_cap:
                    await _send(websocket, {"type": "error", "message": "No hints remaining this game."})
                    continue

                board = session.engine.board
                hint_move = await asyncio.to_thread(session.game_ai.choose_move, board)
                session.hints_used += 1
                hints_left = session.hint_cap - session.hints_used

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
                # B: mark opening replay active so override_ai requests are ignored
                new_session.opening_active = True
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

                # B: opening replay complete — clear flag
                session.opening_active = False

                # Post-replay: set up continuation
                if continue_mode == "practice":
                    await _send(websocket, {
                        "type":    "commentary",
                        "speaker": "Game",
                        "text":    "Opening complete — your turn to continue from this position.",
                        "section": "ai",
                    })
                else:
                    # D/C: Watch mode — start AI vs AI from the current position.
                    # Build a simple GameAI for each side (no coordinator needed);
                    # _run_ai_vs_ai_loop safely handles None coordinators.
                    _ava_diff = 3
                    _ava_hw = HeuristicWeights()
                    _ai_w = GameAI(color="W", difficulty=_ava_diff, weights=_ava_hw, fullgame_db=_fullgame_db, endgame_solved_db=_endgame_solved_db, malom_db=_malom_db, value_net=_value_net)
                    _ai_b = GameAI(color="B", difficulty=_ava_diff, weights=_ava_hw, fullgame_db=_fullgame_db, endgame_solved_db=_endgame_solved_db, malom_db=_malom_db, value_net=_value_net)
                    session.ai_vs_ai = True
                    session.vs_human = False
                    session.game_ai_white = _ai_w
                    session.game_ai_black = _ai_b
                    session.game_ai = _ai_w if session.engine.board.turn == "W" else _ai_b
                    session.coordinator_white = None   # C: safe — loop checks before deliberate
                    session.coordinator_black = None
                    log.info(
                        "Watch mode: starting AI-vs-AI from opening '%s'", opening_id
                    )
                    await _send(websocket, {
                        "type":    "commentary",
                        "speaker": "Game",
                        "text":    "Opening complete — now playing AI vs AI.",
                        "section": "ai",
                    })
                    # Cancel any stale AI-vs-AI task before starting a fresh one
                    if session._ava_task and not session._ava_task.done():
                        session._ava_task.cancel()
                    session._ava_task = asyncio.create_task(
                        _run_ai_vs_ai_loop(websocket, session)
                    )

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

            # ── prune_opening — remove a learned opening ──────────────────────
            elif kind == "prune_opening":
                opening_id = msg.get("opening_id", "")
                if not opening_id:
                    await _send(websocket, {"type": "error", "message": "Missing opening_id"})
                else:
                    _book = (
                        session.coordinator.opening_recognizer.book
                        if session and session.coordinator and session.coordinator.opening_recognizer
                        else OpeningBook()
                    )
                    if _book.prune_opening(opening_id):
                        await _send(websocket, {
                            "type": "prune_opening_ack",
                            "opening_id": opening_id,
                        })
                    else:
                        await _send(websocket, {
                            "type": "error",
                            "message": f"Cannot prune '{opening_id}' — not found or protected.",
                        })

            # ── rename_opening — set a display name on any opening ────────────
            elif kind == "rename_opening":
                opening_id = msg.get("opening_id", "")
                name = str(msg.get("name", "")).strip()
                if not opening_id or not name:
                    await _send(websocket, {"type": "error", "message": "Missing opening_id or name"})
                else:
                    _book = (
                        session.coordinator.opening_recognizer.book
                        if session and session.coordinator and session.coordinator.opening_recognizer
                        else OpeningBook()
                    )
                    if _book.set_name(opening_id, name, needs_llm_name=False):
                        await _send(websocket, {
                            "type": "rename_opening_ack",
                            "opening_id": opening_id,
                            "name": name,
                        })
                    else:
                        await _send(websocket, {
                            "type": "error",
                            "message": f"Opening '{opening_id}' not found.",
                        })

            # ── suggest_opening_name — ask LLM to propose a name ──────────────
            elif kind == "suggest_opening_name":
                opening_id = msg.get("opening_id", "")
                _book = (
                    session.coordinator.opening_recognizer.book
                    if session and session.coordinator and session.coordinator.opening_recognizer
                    else OpeningBook()
                )
                _opening = _book.get_by_id(opening_id) if opening_id else None
                if _opening is None:
                    await _send(websocket, {
                        "type": "error",
                        "message": f"Opening '{opening_id}' not found.",
                    })
                else:
                    _llm = (
                        session.coordinator.mills_llm
                        if session and session.coordinator
                        else None
                    )
                    suggested = None
                    if _llm and _llm._client:
                        suggested = await asyncio.to_thread(
                            _llm.name_novel_opening, _opening.line_moves
                        )
                    await _send(websocket, {
                        "type":       "opening_name_suggestion",
                        "opening_id": opening_id,
                        "suggested":  suggested or _opening.name,
                        "llm_used":   bool(_llm and _llm._client and suggested),
                    })

            # ── start_ai_vs_ai ────────────────────────────────────────────────
            elif kind == "start_ai_vs_ai":
                white_personality = str(msg.get("white_personality", "balanced"))
                black_personality = str(msg.get("black_personality", "aggressive"))
                save_flag         = bool(msg.get("save_to_library", False))
                diff_w = max(1, min(10, int(msg.get("difficulty_white", 3))))
                diff_b = max(1, min(10, int(msg.get("difficulty_black", 3))))

                # Normalise personality names
                if white_personality not in _PERSONALITY_WEIGHTS:
                    white_personality = "balanced"
                if black_personality not in _PERSONALITY_WEIGHTS:
                    black_personality = "balanced"

                # Cancel any running AI-vs-AI loop
                if session and session._ava_task and not session._ava_task.done():
                    session._ava_task.cancel()

                engine = GameEngine(human_color="W")  # human_color irrelevant for AI-vs-AI
                gai_w  = _make_game_ai_for_personality("W", white_personality, diff_w)
                gai_b  = _make_game_ai_for_personality("B", black_personality, diff_b)

                use_llm   = bool(msg.get("use_llm", True))
                _s        = _load_settings()
                coord_w: Optional[Coordinator] = None
                coord_b: Optional[Coordinator] = None

                if use_llm:
                    url   = _s.get("ollama_url",   "http://localhost:11434")
                    model = _s.get("ollama_model", "llama3.1:8b")
                    for _ai_inst, _ai_color, _opp_color in [
                        (gai_w, "W", "B"), (gai_b, "B", "W"),
                    ]:
                        _mem  = MemoryManager(ollama_url=url, ollama_model=model)
                        _llm  = MillsLLM(memory=_mem, ollama_url=url, model=model)
                        _book = OpeningBook()
                        _rec  = OpeningRecognizer(_book)
                        _egr  = EndgameRecognizer(
                            active_threshold=_s.get("endgame_active_threshold", 11),
                            deep_threshold=_s.get("endgame_deep_threshold", 8),
                            zugzwang_threshold=_s.get("endgame_zugzwang_threshold", 0.4),
                        )
                        _coord = Coordinator(
                            game_ai=_ai_inst, mills_llm=_llm, memory=_mem,
                            poor_move_threshold=_s.get("poor_move_threshold", 0.3),
                            max_poor_move_comments=_s.get("max_poor_move_comments_per_game", 5),
                            opening_recognizer=_rec, endgame_recognizer=_egr,
                            trajectory_db=_trajectory_db,
                            endgame_db=_endgame_db,
                            vs_human=False,
                            human_color=_opp_color,  # each AI's "opponent" is the other color
                        )
                        await asyncio.to_thread(_coord.on_game_start)
                        if _ai_color == "W":
                            coord_w = _coord
                        else:
                            coord_b = _coord

                new_session = Session(engine, gai_w, coord_w, "W", False)
                new_session.ai_vs_ai          = True
                new_session.save_to_library   = save_flag
                new_session.game_ai_white     = gai_w
                new_session.game_ai_black     = gai_b
                new_session.coordinator_white = coord_w
                new_session.coordinator_black = coord_b
                new_session.white_personality = white_personality
                new_session.black_personality = black_personality
                session = new_session

                log.info(
                    "AI-vs-AI started  white=%s(diff=%d) black=%s(diff=%d) save=%s llm=%s",
                    white_personality, diff_w, black_personality, diff_b, save_flag, use_llm,
                )
                await _send(websocket, _state(session))
                await _send(websocket, {
                    "type": "commentary", "speaker": "Game",
                    "text": f"AI vs AI — {white_personality.capitalize()} (White) vs {black_personality.capitalize()} (Black)",
                    "section": "ai",
                })
                session._ava_task = asyncio.create_task(
                    _run_ai_vs_ai_loop(websocket, session)
                )

            # ── toggle_save_library — flip save flag mid AI-vs-AI game ────────
            elif kind == "toggle_save_library" and session and session.ai_vs_ai:
                session.save_to_library = bool(msg.get("save", False))
                await _send(websocket, {
                    "type": "save_library_ack",
                    "save": session.save_to_library,
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
