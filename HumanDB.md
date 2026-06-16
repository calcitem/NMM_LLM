# HumanDB — Human-Game Position Database

*Replaces the 23 k-file startup scan with a single pre-built SQLite file, adds Malom WDL annotation for every human-reached position, and powers a 3D interactive game explorer.*

---

## Problem

`TrajectoryDB.load()` scans all `data/human_games/*.jsonl` at server startup (currently 23 013 files, growing). This adds several seconds of blocking I/O every boot and grows with the dataset. `MalomDB` is a separate 77 GB lookup that must be queried position by position at runtime.

**HumanDB solves both** by pre-computing everything offline once and writing a single indexed SQLite file that opens in milliseconds.

---

## Scope

HumanDB annotates the positions humans actually reach. It does **not** replace Malom for positions outside the human corpus — the engine's alpha-beta search explores many positions humans never play, and those still need Malom or the internal retrograde DB. HumanDB is a drop-in replacement for:

- `TrajectoryDB` — startup file scan → `query()` / `query_all_frequencies()`
- `MalomDB` — runtime per-position WDL lookup **for positions in the human corpus only**

For OOD positions encountered during search, the existing Malom / `EndgameSolvedDB` paths remain active.

---

## Architecture

```
tools/build_human_db.py         ← one-shot offline builder
    ↓
data/human_db.sqlite            ← the artifact (ship or gitignore as preferred)
    ↓
ai/human_db.py (HumanDB class)  ← runtime read-only adapter
    ↓
web/app.py                      ← loads HumanDB instead of TrajectoryDB at startup
game_ai.py                      ← prefers HumanDB.trajectory_score_delta()
```

---

## SQLite Schema

```sql
-- One row per canonical board state (D4-normalised, using make_board_state_key).
CREATE TABLE positions (
    state_key            TEXT PRIMARY KEY,
    total_games          INTEGER NOT NULL DEFAULT 0,
    wins                 INTEGER NOT NULL DEFAULT 0,
    losses               INTEGER NOT NULL DEFAULT 0,
    draws                INTEGER NOT NULL DEFAULT 0,
    malom_wdl            TEXT,      -- 'W' | 'L' | 'D' | NULL (Malom unavailable at build time)
    malom_dtw            INTEGER,   -- depth-to-win; NULL when not WIN
    canonical_winning_move TEXT     -- most-played notation among winning-mover games; NULL if no wins
);

-- One row per (canonical state, next-move notation).
-- notation is stored in D4-canonical space and de-normalised at query time.
CREATE TABLE moves (
    state_key            TEXT NOT NULL,
    notation             TEXT NOT NULL,
    wins                 INTEGER NOT NULL DEFAULT 0,
    losses               INTEGER NOT NULL DEFAULT 0,
    draws                INTEGER NOT NULL DEFAULT 0,
    total                INTEGER NOT NULL DEFAULT 0,
    moves_to_end_sum     REAL    NOT NULL DEFAULT 0.0,  -- sum of plies remaining; avg = sum/total
    malom_wdl_after      TEXT,    -- Malom WDL of the successor position from the NEXT mover's perspective
    malom_dtw_after      INTEGER, -- Malom depth-to-win/loss (positive=win, negative=loss) for successor
    PRIMARY KEY (state_key, notation)
);

-- Tracks which files have been processed so --update only scans new/changed files.
CREATE TABLE IF NOT EXISTS processed_files (
    file_path   TEXT PRIMARY KEY,
    mtime       REAL NOT NULL,
    games_found INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_moves_state ON moves(state_key);

-- Build metadata.
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- INSERT INTO meta VALUES ('build_date', '2026-06-16');
-- INSERT INTO meta VALUES ('game_count', '23013');
-- INSERT INTO meta VALUES ('malom_annotated', 'true');
-- INSERT INTO meta VALUES ('schema_version', '1');
```

**Key reuse:** `make_board_state_key(board)` from `ai/trajectory_db.py:50` produces the canonical key and the D4 `sym_idx` needed to de-normalise stored notations back to actual-game coordinates. No new canonicalisation code needed.

---

## Build Pipeline (`tools/build_human_db.py`)

### Inputs
- `data/human_games/*.jsonl` — 23 013 PlayOK human-vs-human games
- Malom DB at configured path (`configs/sentinel_default.yaml` → `external_db_path`) — optional; build proceeds without it, leaving `malom_wdl` NULL

### Steps

1. **Scan games** — `rglob("*.jsonl")`, filter `source_type == "human_vs_human"`, skip `adaptive_softened`.
2. **Replay each game** — re-use `BoardState.from_fen_string()` (already done in `TrajectoryDB._index_game`). For each ply:
   - Compute `state_key, sym_idx = make_board_state_key(board)`
   - Store `canon_notation = transform_notation(notation, sym_idx)`
   - Record `(state_key, canon_notation, color, winner, game_length - current_ply)`
3. **Aggregate** — in-memory: `{state_key: {canon_notation: {wins, losses, draws, total, move_to_end_sum}}}`.
4. **Malom annotation** — for each unique `state_key`, reconstruct a representative board and call `MalomDB.query(board)`. Reuse `ExternalSolvedDB` from `learned_ai/sentinel/db_teacher.py` to avoid re-implementing the Malom query loop (that module already wraps `ai/malom_db.MalomDB` gracefully with `is_available()` guards).
5. **Canonical winning move** — for each `state_key`, pick the `canon_notation` with the highest `wins` count (ties broken by `total`). This is the top of the winning-line chain.
6. **Write SQLite** — open `data/human_db.sqlite`, insert positions and moves in one transaction. Log progress every 1 000 games.

### CLI

```bash
# First build (or full rebuild):
.venv/bin/python tools/build_human_db.py \
    --games-dir data/human_games \
    --output data/human_db.sqlite \
    --malom-db /mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted

# Incremental update (only new/changed files; skips files already in processed_files):
.venv/bin/python tools/build_human_db.py --update

# Skip Malom annotation (faster; malom_wdl/dtw columns stay NULL):
.venv/bin/python tools/build_human_db.py --update --no-malom

# Force full rebuild, ignoring processed_files:
.venv/bin/python tools/build_human_db.py --rebuild
```

Estimated build time: ~3–5 min for 23 k games without Malom; ~30–60 min with Malom annotation (one sector-file lookup per unique next-position).

**Incremental updates** are also triggered at runtime: `HumanDB.add_game(record)` is called from `web/app.py` whenever a human game completes, immediately upsetting the SQLite rows for positions seen in that game. No rebuild needed as the library grows.

---

## Runtime API (`ai/human_db.py`)

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class MoveStats:
    notation: str
    wins: int
    losses: int
    draws: int
    total: int
    win_pct: float                # wins / total
    avg_moves_to_end: float       # human-game average plies remaining after this move
    malom_wdl_after: str | None   # 'W'|'L'|'D' for successor position (next mover's perspective)
    malom_dtw_after: int | None   # Malom DTW for successor (positive=win, negative=loss)

@dataclass
class PositionStats:
    total_games: int
    wins: int; losses: int; draws: int
    malom_wdl: str | None    # 'W'|'L'|'D' for current mover
    malom_dtw: int | None
    canonical_winning_move: str | None  # best next notation

class HumanDB:
    def __init__(self, db_path: Path) -> None: ...
    def is_available(self) -> bool: ...

    # Drop-in for MalomDB.query()
    def get_malom_wdl(self, board) -> dict | None:
        """Returns {"outcome": "W"|"L"|"D", "dtw": int|None} or None."""

    # Drop-in for TrajectoryDB.query()
    def trajectory_score_delta(self, board, min_samples=3) -> dict[str, float]:
        """Per-notation score delta [-0.5, +0.5], confidence-weighted. Empty when no data."""

    # Drop-in for TrajectoryDB.query_all_frequencies()
    def move_frequencies(self, board, min_samples=5) -> dict[str, float]:
        """Per-notation relative frequency [0, 1]."""

    # Visualization / explorer
    def query_position(self, board) -> PositionStats | None: ...
    def query_moves(self, board) -> list[MoveStats]: ...

    # Follow canonical_winning_move chain; depth-limited to avoid cycles
    def canonical_winning_line(self, board, depth=10) -> list[str]:
        """Returns list of notations following the most-played winning path."""
```

The confidence-weighting formula in `trajectory_score_delta` mirrors `TrajectoryDB.query()` exactly so existing callers (`game_ai.py`, `ponder.py`) need no changes to the scoring logic.

---

## Integration

### `web/app.py`

```python
# At startup, prefer HumanDB if the file exists.
_human_db: HumanDB | None = None
_human_db_path = Path("data/human_db.sqlite")
if _human_db_path.exists():
    from ai.human_db import HumanDB
    _human_db = HumanDB(_human_db_path)
    logger.info("HumanDB loaded from %s", _human_db_path)
else:
    # Fallback: original file-scan path
    _trajectory_db = TrajectoryDB(...)
    _trajectory_db.load()
```

Pass `_human_db` (or `_trajectory_db`) to `GameAI` constructors; the engine's call sites use the common method names so either object works.

### `game_ai.py`

Replace calls to `self._trajectory_db.query(board)` with a helper that checks for `_human_db` first:

```python
def _trajectory_hints(self, board) -> dict[str, float]:
    if self._human_db and self._human_db.is_available():
        return self._human_db.trajectory_score_delta(board)
    if self._trajectory_db:
        return self._trajectory_db.query(board, self.color)
    return {}
```

### Malom fallback

`game_ai._consult_malom()` keeps its existing path. `HumanDB.get_malom_wdl()` is used only for positions the human corpus covers; the engine still calls `MalomDB` directly for positions it needs during search. Long-term, once `build_human_db.py` has annotated every human position, the Malom call frequency at runtime drops significantly.

---

## Phase 2 — 3D Game Explorer

*Separable from Phase 1. HumanDB must be built before this is wired up.*

### Route

```
GET /tools/explorer                       → explorer.html
GET /api/explorer/position?fen=<fen>      → JSON: PositionStats + MoveStats[] + winning_line[]
GET /api/explorer/move?fen=<fen>&move=<n> → JSON: resulting PositionStats + MoveStats[] (opponent view)
```

### 3D Board (Three.js)

- **Board geometry**: the 24 NMM squares rendered as flat hexagonal pads in 3D space at their topological positions (outer/middle/inner ring concentric squares projected onto a plane).
- **Pieces**: cylinders (White = ivory, Black = obsidian).
- **Move frequency bars**: for each legal next-move target square, a vertical bar with:
  - **Height** proportional to `win_pct` (0 → flush with board, 1.0 → tallest bar)
  - **Colour**: green = Malom WDL after move is WIN for mover; red = LOSS; grey = DRAW or unknown
  - **Label**: `win_pct %` floating above the bar
- **Interaction**:
  - Hover a bar → tooltip showing `wins / losses / draws / total / avg_moves_to_end`
  - Click a bar → commit the move; board updates; opponent's bars are now shown from the opponent's perspective (own bars turn the other colour)
  - **Back button** — undo one move; re-queries the parent position
  - **Forward / best line button** — follows `canonical_winning_move` chain automatically; animates step by step
- **FEN input** — paste any position FEN to jump directly to it

### `web/templates/explorer.html`

Standalone page linked from the Tools panel. Loads Three.js from CDN (or bundled). All position data fetched via the JSON API above; no game state on the server.

### `web/static/explorer.js`

~600–900 lines. Key components:
- `BoardScene` — Three.js scene, camera, lighting, board geometry
- `PieceLayer` — manages White/Black cylinder meshes
- `BarLayer` — manages win-% bar meshes, raycasting for hover/click
- `ExplorerController` — fetches API, updates layers, manages navigation history (stack of FEN strings for Back)

---

## Deliverables

| Phase | File | Description |
|---|---|---|
| 1a | `tools/build_human_db.py` | Offline builder; CLI with `--games-dir`, `--output`, `--malom-db` |
| 1b | `ai/human_db.py` | `HumanDB` runtime adapter |
| 1c | `web/app.py` (edit) | Load HumanDB at startup; fall back to TrajectoryDB |
| 1d | `game_ai.py` (edit) | `_trajectory_hints()` helper prefers HumanDB |
| 2a | `web/app.py` (edit) | `/tools/explorer` route + `/api/explorer/*` endpoints |
| 2b | `web/templates/explorer.html` | Explorer page shell |
| 2c | `web/static/explorer.js` | Three.js 3D board + bar overlay + navigation |

---

## Open Questions / Risks

- **Malom availability at build time**: the 77 GB DB may not always be mounted. `--no-malom` flag builds without WDL annotation; `malom_wdl` columns remain NULL. Explorer still works — bars show only human win% without colour-coding by DB quality.
- **Cycle detection in `canonical_winning_line`**: NMM games can cycle (mill-open / mill-close). The chain follower must track visited `state_key` values and stop if revisited, capping at `depth` moves regardless.
- **3D framework size**: Three.js minified is ~600 KB. If CDN is preferred, no bundle step needed. If the project goes offline-first, bundle it under `web/static/vendor/`.
- **B-85 (endgame DB symmetry bug)**: does not affect HumanDB because we only store positions humans actually played, not the full retrograde table. HumanDB is unaffected by B-85.
